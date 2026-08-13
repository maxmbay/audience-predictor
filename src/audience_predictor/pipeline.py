"""End-to-end experiments: load catalog-value targets, train Equinox encoders, evaluate."""

from __future__ import annotations

import json
from pathlib import Path

import jax
import numpy as np
import polars as pl
from sklearn.decomposition import PCA

from audience_predictor.config import Config
from audience_predictor.data.bridge import (
    atlas_path,
    catalog_value_root,
    movies_path,
    ratings_path,
    titles_path,
)
from audience_predictor.data.coverage import write_coverage_report
from audience_predictor.data.genome import (
    content_features,
    genome_bucket_matrix,
    parse_year,
    side_features,
)
from audience_predictor.data.hasher import hash_embed
from audience_predictor.data.infer import blocks_from_document
from audience_predictor.data.moviesum import (
    gold_movie_rows,
    ingest_moviesum,
    load_moviesum_blocks,
    provenance_path,
)
from audience_predictor.data.narrative import (
    NarrativeBatch,
    pooled_block_features,
    tag_blocks_for_movies,
)
from audience_predictor.data.subtitles import (
    ingest_sublens,
    load_sublens_blocks,
    silver_movie_rows,
)
from audience_predictor.data.splits import movie_split
from audience_predictor.data.tags import load_tags, movie_tag_text
from audience_predictor.data.targets import polarization_matrix, polarization_table
from audience_predictor.data.encoder import get_encoder
from audience_predictor.eval.comps import neighbor_table, neighbors_from_vector
from audience_predictor.eval.metrics import cosine, recovery_metrics
from audience_predictor.models.mlp import AudienceMLP
from audience_predictor.paths import output_dir
from audience_predictor.train.loop import (
    load_checkpoint,
    predict_mlp,
    predict_set_encoder,
    predict_tag_encoder,
    predict_tag_mean_encoder,
    save_checkpoint,
    train_mlp,
    train_set_encoder,
    train_tag_encoder,
    train_tag_mean_encoder,
)
from audience_predictor.viz.figures import (
    plot_ablation,
    plot_atlas,
    plot_coverage_funnel,
    plot_loss,
    plot_neighbors,
    plot_polarization,
    plot_reconstruction,
    plot_residual,
    plot_scaling,
    plot_uncertainty,
)
from audience_predictor.viz.style import figures_dir


def _load_spine(config: Config):
    root = catalog_value_root(config.data.catalog_value_root)
    movies = pl.read_parquet(movies_path(root)).sort("movie_row")
    ratings = pl.read_parquet(ratings_path(root))
    titles = np.load(titles_path(root))
    z = titles["z"].astype(np.float32)
    bias = titles["bias"].astype(np.float32)
    n_ratings = movies["n_ratings"].to_numpy()
    aux_table = polarization_table(ratings, z.shape[0])
    aux = polarization_matrix(aux_table)
    split = movie_split(
        n_ratings,
        min_train_ratings=config.train.min_train_ratings,
        holdout_frac=config.train.holdout_frac,
        seed=config.seed,
    )
    years = np.array([parse_year(t) or 0 for t in movies["title"].to_list()], dtype=np.int32)
    return root, movies, z, bias, n_ratings, aux, split, years


def _tag_bags(movies: pl.DataFrame, root, hash_dim: int) -> np.ndarray:
    text = movie_tag_text(load_tags(root))
    joined = movies.join(text, on="movieId", how="left").sort("movie_row")
    bags = np.zeros((joined.height, hash_dim), dtype=np.float32)
    for row in joined.iter_rows(named=True):
        blob = row.get("tag_text") or ""
        bags[int(row["movie_row"])] = hash_embed(str(blob), hash_dim)
    return bags


def run_coverage(config: Config) -> Path:
    return write_coverage_report(config.data.catalog_value_root)


def run_ingest_moviesum(config: Config, encoder: str = "hash") -> Path:
    ingest_moviesum(
        hash_dim=config.data.hash_dim,
        max_blocks=config.data.max_blocks,
        catalog_root=config.data.catalog_value_root,
        encoder=encoder,
    )
    return write_coverage_report(config.data.catalog_value_root)


def run_ingest_sublens(config: Config, encoder: str = "hash") -> Path:
    ingest_sublens(
        hash_dim=config.data.hash_dim,
        max_blocks=config.data.max_blocks,
        catalog_root=config.data.catalog_value_root,
        encoder=encoder,
    )
    return write_coverage_report(config.data.catalog_value_root)


MODALITIES = (
    "metadata",
    "genome",
    "tags",
    "hierarchical",
    "genome_gold",
    "screenplay",
    "dialogue",
    "screenplay_bge",
    "dialogue_bge",
    "screenplay_mean",
    "dialogue_mean",
    "screenplay_pma",
    "dialogue_pma",
    "subtitle",
    "subtitle_mean",
    "genome_script",
    "genome_production",
    "genome_reception",
    "script_tags",
    "script_tags_mean",
)

GENOME_BUCKET_MODALITIES = {
    "genome_script": "script",
    "genome_production": "production",
    "genome_reception": "reception",
}


def _gold_split(split, gold_rows: np.ndarray, mask: np.ndarray | None = None):
    rows = gold_rows
    if mask is not None:
        rows = np.intersect1d(rows, np.flatnonzero(np.asarray(mask).any(axis=1)))
    train = np.intersect1d(split.train_idx, rows)
    hold = np.intersect1d(split.holdout_idx, rows)
    if train.size < 20:
        raise ValueError(f"gold train set too small: {train.size}")
    print(f"gold titles train={train.size:,}  holdout={hold.size:,}  eligible={rows.size:,}")
    return train, hold


def _holdout_mask(n: int, hold_idx: np.ndarray) -> np.ndarray:
    mask = np.zeros(n, dtype=bool)
    mask[hold_idx] = True
    return mask


def _slice_metrics(pred: dict, z, aux, idx: np.ndarray) -> dict:
    return recovery_metrics(
        pred["mu"][idx],
        z[idx],
        pred["log_var"][idx],
        pred["aux"][idx],
        aux[idx],
    )


def run_train(config: Config, modality: str) -> Path:
    root, movies, z, bias, n_ratings, aux, split, _years = _load_spine(config)
    out = output_dir() / modality
    out.mkdir(parents=True, exist_ok=True)
    cfg = config.train
    rep = config.representation

    used_train = split.train_idx
    used_hold = split.holdout_idx

    if modality == "metadata":
        x = side_features(movies)
        model, history = train_mlp(
            x, z, bias, aux, split.train_idx,
            hidden=rep.hidden, epochs=cfg.epochs, lr=cfg.lr, weight_decay=cfg.weight_decay,
            nll_weight=cfg.nll_weight, aux_weight=cfg.aux_weight, bias_weight=cfg.bias_weight,
            seed=config.seed,
        )
        pred = predict_mlp(model, x)
        save_checkpoint(out / "model.eqx", model)
    elif modality == "genome":
        x = content_features(movies, root=root)
        model, history = train_mlp(
            x, z, bias, aux, split.train_idx,
            hidden=rep.hidden, epochs=cfg.epochs, lr=cfg.lr, weight_decay=cfg.weight_decay,
            nll_weight=cfg.nll_weight, aux_weight=cfg.aux_weight, bias_weight=cfg.bias_weight,
            seed=config.seed,
        )
        pred = predict_mlp(model, x)
        save_checkpoint(out / "model.eqx", model)
    elif modality == "genome_gold":
        x = content_features(movies, root=root)
        gold = gold_movie_rows()
        used_train, used_hold = _gold_split(split, gold)
        model, history = train_mlp(
            x, z, bias, aux, used_train,
            hidden=rep.hidden, epochs=cfg.epochs, lr=cfg.lr, weight_decay=cfg.weight_decay,
            nll_weight=cfg.nll_weight, aux_weight=cfg.aux_weight, bias_weight=cfg.bias_weight,
            seed=config.seed,
        )
        pred = predict_mlp(model, x)
        save_checkpoint(out / "model.eqx", model)
    elif modality in GENOME_BUCKET_MODALITIES:
        bucket = GENOME_BUCKET_MODALITIES[modality]
        x = genome_bucket_matrix(movies, bucket, root=root)
        print(f"{modality}: {x.shape[1]} {bucket} tags (no genre/year)")
        model, history = train_mlp(
            x, z, bias, aux, split.train_idx,
            hidden=rep.hidden, epochs=cfg.epochs, lr=cfg.lr, weight_decay=cfg.weight_decay,
            nll_weight=cfg.nll_weight, aux_weight=cfg.aux_weight, bias_weight=cfg.bias_weight,
            seed=config.seed,
        )
        pred = predict_mlp(model, x)
        save_checkpoint(out / "model.eqx", model)
    elif modality == "script_tags":
        batch = load_sublens_blocks("bge")
        silver = silver_movie_rows()
        used_train, used_hold = _gold_split(split, silver, batch.mask)
        y = genome_bucket_matrix(movies, "script", root=root)
        tag_model, tag_history = train_tag_encoder(
            batch.dialogue,
            batch.mask,
            y,
            used_train,
            model_dim=rep.hidden,
            n_heads=rep.n_heads,
            n_inducing=rep.n_inducing,
            epochs=cfg.epochs,
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
            seed=config.seed,
            pool="isab",
        )
        tags_hat = predict_tag_encoder(tag_model, batch.dialogue, batch.mask)
        save_checkpoint(out / "tag_encoder.eqx", tag_model)
        (out / "tag_spec.json").write_text(
            json.dumps(
                {
                    "in_dim": int(batch.dialogue.shape[-1]),
                    "model_dim": int(rep.hidden),
                    "n_tags": int(y.shape[1]),
                    "n_heads": int(rep.n_heads),
                    "n_inducing": int(rep.n_inducing),
                    "pool": "isab",
                    "bucket": "script",
                    "mlp_hidden": int(rep.hidden),
                    "out_dim": int(z.shape[1]),
                },
                indent=2,
            )
        )
        np.savez_compressed(out / "predicted_tags.npz", tags=tags_hat, history=np.asarray(tag_history))
        model, history = train_mlp(
            tags_hat, z, bias, aux, used_train,
            hidden=rep.hidden, epochs=cfg.epochs, lr=cfg.lr, weight_decay=cfg.weight_decay,
            nll_weight=cfg.nll_weight, aux_weight=cfg.aux_weight, bias_weight=cfg.bias_weight,
            seed=config.seed,
        )
        pred = predict_mlp(model, tags_hat)
        save_checkpoint(out / "model.eqx", model)
    elif modality == "script_tags_mean":
        batch = load_sublens_blocks("bge")
        silver = silver_movie_rows()
        used_train, used_hold = _gold_split(split, silver, batch.mask)
        y = genome_bucket_matrix(movies, "script", root=root)
        pooled = pooled_block_features(batch, include_action=False)
        tag_model, tag_history = train_tag_mean_encoder(
            pooled, y, used_train,
            hidden=rep.hidden, epochs=cfg.epochs, lr=cfg.lr, weight_decay=cfg.weight_decay,
            seed=config.seed,
        )
        tags_hat = predict_tag_mean_encoder(tag_model, pooled)
        save_checkpoint(out / "tag_encoder.eqx", tag_model)
        (out / "tag_spec.json").write_text(
            json.dumps(
                {
                    "kind": "mean",
                    "in_dim": int(pooled.shape[-1]),
                    "hidden": int(rep.hidden),
                    "n_tags": int(y.shape[1]),
                    "bucket": "script",
                    "mlp_hidden": int(rep.hidden),
                    "out_dim": int(z.shape[1]),
                },
                indent=2,
            )
        )
        np.savez_compressed(out / "predicted_tags.npz", tags=tags_hat, history=np.asarray(tag_history))
        oracle_path = output_dir() / "genome_script" / "model.eqx"
        if oracle_path.exists():
            like = AudienceMLP(in_dim=y.shape[1], out_dim=z.shape[1], hidden=rep.hidden, key=jax.random.key(0))
            model = load_checkpoint(oracle_path, like)
            history = list(tag_history)
            print("script_tags_mean: decoding tags with frozen genome_script MLP")
        else:
            model, history = train_mlp(
                tags_hat, z, bias, aux, used_train,
                hidden=rep.hidden, epochs=cfg.epochs, lr=cfg.lr, weight_decay=cfg.weight_decay,
                nll_weight=cfg.nll_weight, aux_weight=cfg.aux_weight, bias_weight=cfg.bias_weight,
                seed=config.seed,
            )
        pred = predict_mlp(model, tags_hat)
        save_checkpoint(out / "model.eqx", model)
    elif modality == "tags":
        x = _tag_bags(movies, root, config.data.hash_dim)
        model, history = train_mlp(
            x, z, bias, aux, split.train_idx,
            hidden=rep.hidden, epochs=cfg.epochs, lr=cfg.lr, weight_decay=cfg.weight_decay,
            nll_weight=cfg.nll_weight, aux_weight=cfg.aux_weight, bias_weight=cfg.bias_weight,
            seed=config.seed,
        )
        pred = predict_mlp(model, x)
        save_checkpoint(out / "model.eqx", model)
    elif modality == "hierarchical":
        tags = load_tags(root)
        batch = tag_blocks_for_movies(
            movies, tags, hash_dim=config.data.hash_dim, max_blocks=config.data.max_blocks
        )
        model, history = train_set_encoder(
            batch.dialogue, batch.action, batch.mask, batch.structure,
            z, bias, aux, split.train_idx,
            model_dim=rep.model_dim, out_dim=rep.embedding_dim, n_heads=rep.n_heads,
            n_inducing=rep.n_inducing, epochs=cfg.epochs, lr=cfg.lr, weight_decay=cfg.weight_decay,
            nll_weight=cfg.nll_weight, aux_weight=cfg.aux_weight, bias_weight=cfg.bias_weight,
            seed=config.seed,
        )
        pred = predict_set_encoder(model, batch)
        save_checkpoint(out / "model.eqx", model)
        np.savez_compressed(out / "residual.npz", z_dialogue=pred["z_dialogue"], delta=pred["delta"])
    elif modality in {
        "screenplay",
        "dialogue",
        "screenplay_bge",
        "dialogue_bge",
        "screenplay_pma",
        "dialogue_pma",
    }:
        encoder = "hash" if modality in {"screenplay", "dialogue"} else "bge"
        pool = "pma" if modality.endswith("_pma") else "isab"
        batch = load_moviesum_blocks(encoder)
        gold = gold_movie_rows()
        used_train, used_hold = _gold_split(split, gold, batch.mask)
        full = modality.startswith("screenplay")
        action = batch.action if full else np.zeros_like(batch.action)
        structure = batch.structure if full else np.zeros_like(batch.structure)
        model, history = train_set_encoder(
            batch.dialogue, action, batch.mask, structure,
            z, bias, aux, used_train,
            model_dim=rep.model_dim, out_dim=rep.embedding_dim, n_heads=rep.n_heads,
            n_inducing=rep.n_inducing, pool=pool, epochs=cfg.epochs, lr=cfg.lr,
            weight_decay=cfg.weight_decay, nll_weight=cfg.nll_weight,
            aux_weight=cfg.aux_weight, bias_weight=cfg.bias_weight,
            seed=config.seed,
        )
        eval_batch = NarrativeBatch(
            dialogue=batch.dialogue,
            action=action,
            mask=batch.mask,
            structure=structure,
            document_type=batch.document_type,
        )
        pred = predict_set_encoder(model, eval_batch)
        save_checkpoint(out / "model.eqx", model)
        np.savez_compressed(out / "residual.npz", z_dialogue=pred["z_dialogue"], delta=pred["delta"])
    elif modality in {"screenplay_mean", "dialogue_mean"}:
        batch = load_moviesum_blocks("bge")
        gold = gold_movie_rows()
        used_train, used_hold = _gold_split(split, gold, batch.mask)
        x = pooled_block_features(batch, include_action=modality.startswith("screenplay"))
        model, history = train_mlp(
            x, z, bias, aux, used_train,
            hidden=rep.hidden, epochs=cfg.epochs, lr=cfg.lr, weight_decay=cfg.weight_decay,
            nll_weight=cfg.nll_weight, aux_weight=cfg.aux_weight, bias_weight=cfg.bias_weight,
            seed=config.seed,
        )
        pred = predict_mlp(model, x)
        save_checkpoint(out / "model.eqx", model)
    elif modality in {"subtitle", "subtitle_mean"}:
        encoder = "bge" if modality == "subtitle_mean" else "hash"
        batch = load_sublens_blocks(encoder)
        silver = silver_movie_rows()
        used_train, used_hold = _gold_split(split, silver, batch.mask)
        x = pooled_block_features(batch, include_action=False)
        model, history = train_mlp(
            x, z, bias, aux, used_train,
            hidden=rep.hidden, epochs=cfg.epochs, lr=cfg.lr, weight_decay=cfg.weight_decay,
            nll_weight=cfg.nll_weight, aux_weight=cfg.aux_weight, bias_weight=cfg.bias_weight,
            seed=config.seed,
        )
        pred = predict_mlp(model, x)
        save_checkpoint(out / "model.eqx", model)
    else:
        raise ValueError(f"unknown modality {modality}")

    np.savez_compressed(
        out / "predictions.npz",
        mu=pred["mu"],
        log_var=pred["log_var"],
        bias=pred["bias"],
        aux=pred["aux"],
        train_idx=used_train,
        holdout_idx=used_hold,
        holdout_mask=_holdout_mask(z.shape[0], used_hold),
        full_train_idx=split.train_idx,
        full_holdout_idx=split.holdout_idx,
        n_ratings=n_ratings,
        history=np.asarray(history),
    )
    metrics = {
        "train": _slice_metrics(pred, z, aux, used_train),
        "holdout": _slice_metrics(pred, z, aux, used_hold),
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps({"modality": modality, **{f"{k}_{m}": v for m, block in metrics.items() for k, v in block.items() if k in {"cosine_mean", "rmse", "nll", "n"}}}, indent=2))
    return out


def run_evaluate(config: Config) -> Path:
    _root, _movies, z, _bias, _n_ratings, aux, split, _years = _load_spine(config)
    gold = gold_movie_rows() if provenance_path().exists() else np.zeros(0, dtype=np.int64)
    gold_train = np.intersect1d(split.train_idx, gold)
    gold_hold = np.intersect1d(split.holdout_idx, gold)
    summary = {}
    for modality in MODALITIES:
        path = output_dir() / modality / "metrics.json"
        pred_path = output_dir() / modality / "predictions.npz"
        if not path.exists():
            continue
        block = json.loads(path.read_text())
        if pred_path.exists() and gold_hold.size:
            pred = np.load(pred_path)
            block["gold_train"] = _slice_metrics(pred, z, aux, gold_train) if gold_train.size else {}
            block["gold_holdout"] = _slice_metrics(pred, z, aux, gold_hold)
        summary[modality] = block
    dest = output_dir() / "ablation.json"
    dest.write_text(json.dumps(summary, indent=2))
    print(dest.read_text())
    return dest


def run_figures(config: Config) -> Path:
    root, movies, z, _bias, n_ratings, aux, _split, _years = _load_spine(config)
    figs = figures_dir()
    ablation_rows: list[tuple[str, float, float]] = []
    primary = None
    for modality, label in (
        ("metadata", "metadata only"),
        ("genome", "genome + genre + year"),
        ("genome_gold", "genome on gold titles only"),
        ("tags", "user tags (hashed bag)"),
        ("hierarchical", "tag-set transformer"),
        ("dialogue", "hashed dialogue"),
        ("screenplay", "hashed screenplay"),
        ("dialogue_bge", "BGE dialogue"),
        ("screenplay_bge", "BGE screenplay"),
        ("dialogue_mean", "BGE dialogue mean-pool"),
        ("screenplay_mean", "BGE screenplay mean-pool"),
        ("dialogue_pma", "BGE dialogue PMA"),
        ("screenplay_pma", "BGE screenplay PMA"),
        ("subtitle", "hashed subtitles"),
        ("subtitle_mean", "BGE subtitle mean-pool"),
        ("genome_script", "script-visible genome (oracle)"),
        ("genome_production", "production genome (oracle)"),
        ("genome_reception", "reception genome (oracle)"),
        ("script_tags", "predicted script-visible tags (ISAB)"),
        ("script_tags_mean", "predicted script-visible tags (mean-pool)"),
    ):
        pred_path = output_dir() / modality / "predictions.npz"
        if not pred_path.exists():
            continue
        pred = np.load(pred_path)
        metrics = json.loads((output_dir() / modality / "metrics.json").read_text())
        ablation_rows.append(
            (label, metrics["train"]["cosine_mean"], metrics["holdout"]["cosine_mean"])
        )
        if modality == "genome" or primary is None:
            primary = (modality, pred)

    if primary is None:
        raise FileNotFoundError("No trained modalities. Run `train --modality genome` first.")

    modality, pred = primary
    mu = pred["mu"]
    holdout = pred["holdout_mask"].astype(bool)
    cos = cosine(mu, z)
    plot_reconstruction(n_ratings, cos, holdout, figs / "reconstruction.png")
    plot_loss(list(pred["history"]), figs / f"loss_{modality}.png")
    plot_uncertainty(
        1.0 - cos[holdout], np.exp(0.5 * pred["log_var"][holdout]), figs / "uncertainty.png"
    )
    plot_polarization(pred["aux"][holdout], aux[holdout], figs / "polarization.png")
    if ablation_rows:
        plot_ablation(ablation_rows, figs / "ablation.png")

    xy = np.load(atlas_path(root))["xy"]
    pca = PCA(n_components=2, random_state=config.seed).fit(z)
    pred_xy = pca.transform(mu)
    title_list = movies.sort("movie_row")["title"].to_list()
    probes = []
    for name in config.eval.probes:
        hit = movies.filter(pl.col("title") == name)
        if not hit.is_empty():
            probes.append(int(hit["movie_row"][0]))
    plot_atlas(
        xy,
        pred_xy,
        holdout,
        np.exp(0.5 * pred["log_var"]),
        title_list,
        probes[:5],
        figs / "atlas.png",
    )

    panels = []
    for name in config.eval.probes:
        collab = neighbor_table(movies, z, name)
        content = neighbor_table(movies, mu, name)
        if collab is not None and content is not None:
            panels.append((name, collab, content))
    if panels:
        plot_neighbors(panels, figs / "comps.png")

    tags = movie_tag_text(load_tags(root))
    n_tags = int(movies.join(tags, on="movieId", how="left")["n_tags"].fill_null(0).gt(0).sum())
    n_gold = int(gold_movie_rows().size) if provenance_path().exists() else 0
    n_silver = int(silver_movie_rows().size)
    plot_coverage_funnel(
        [
            ("MovieLens core", int(movies.height)),
            ("IMDb-linked", int(movies["imdbId"].is_not_null().sum())),
            ("≥1,000 ratings", int((n_ratings >= 1000).sum())),
            ("user tags (silver text)", n_tags),
            ("silver subtitles ∩ core", n_silver),
            ("gold screenplays ∩ core", n_gold),
        ],
        figs / "coverage_funnel.png",
    )

    residual_path = output_dir() / "screenplay_bge" / "residual.npz"
    screen_pred = output_dir() / "screenplay_bge" / "predictions.npz"
    if not residual_path.exists():
        residual_path = output_dir() / "screenplay" / "residual.npz"
        screen_pred = output_dir() / "screenplay" / "predictions.npz"
    if residual_path.exists() and screen_pred.exists():
        residual = np.load(residual_path)
        spred = np.load(screen_pred)
        s_hold = spred["holdout_mask"].astype(bool)
        full_cos = cosine(spred["mu"], z)
        dial_cos = cosine(residual["z_dialogue"], z)
        plot_residual(
            np.linalg.norm(residual["delta"][s_hold], axis=1),
            full_cos[s_hold] - dial_cos[s_hold],
            figs / "residual.png",
        )
    else:
        residual_path = output_dir() / "hierarchical" / "residual.npz"
        hier_pred = output_dir() / "hierarchical" / "predictions.npz"
        if residual_path.exists() and hier_pred.exists():
            residual = np.load(residual_path)
            hpred = np.load(hier_pred)
            h_hold = hpred["holdout_mask"].astype(bool)
            full_cos = cosine(hpred["mu"], z)
            dial_cos = cosine(residual["z_dialogue"], z)
            plot_residual(
                np.linalg.norm(residual["delta"][h_hold], axis=1),
                full_cos[h_hold] - dial_cos[h_hold],
                figs / "residual.png",
            )

    scaling = output_dir() / "scaling.json"
    if scaling.exists():
        payload = json.loads(scaling.read_text())
        plot_scaling(payload["sizes"], payload["holdout_cosine"], figs / "scaling.png")

    print(f"Wrote figures under {figs}")
    return figs


def run_predict(config: Config, text: str, *, k: int = 8) -> dict:
    """Drop in any narrative text; print comps and polarization.

    Fingerprint comes from frozen-BGE mean-pool → audience MLP (subtitle_mean),
    the only narrative map that currently beats chance. Predicted genome tags
    are not used for z until they beat this baseline.
    """
    mean_ckpt = output_dir() / "subtitle_mean" / "model.eqx"
    if not mean_ckpt.exists():
        raise FileNotFoundError("Train subtitle_mean before predict.")
    encoder = get_encoder("bge")
    dialogue, _action, mask, _struct, kind = blocks_from_document(
        text, max_blocks=config.data.max_blocks, encoder=encoder
    )
    weights = mask.astype(np.float32)[..., None]
    pooled = (dialogue * weights).sum(axis=1) / np.clip(weights.sum(axis=1), 1.0, None)
    key = jax.random.key(0)
    mlp_like = AudienceMLP(
        in_dim=pooled.shape[-1],
        out_dim=config.representation.embedding_dim,
        hidden=config.representation.hidden,
        key=key,
    )
    mlp = load_checkpoint(mean_ckpt, mlp_like)
    pred = predict_mlp(mlp, pooled)
    mu = pred["mu"][0]
    aux = pred["aux"][0]
    root, movies, z, _bias, _n_ratings, _aux, _split, _years = _load_spine(config)
    comps = neighbors_from_vector(movies, z, mu, k=k)
    result = {
        "path": "subtitle_mean",
        "document_type": kind,
        "n_blocks": int(mask.sum()),
        "polarization": {
            "mean_rating": float(aux[0]),
            "rating_std": float(aux[1]),
            "p_love": float(aux[2]),
            "p_hate": float(aux[3]),
        },
        "comps": comps.to_dicts() if comps is not None else [],
    }
    print(json.dumps(result, indent=2))
    return result


def run_scaling(config: Config) -> Path:
    root, movies, z, bias, n_ratings, aux, split, _ = _load_spine(config)
    x = content_features(movies, root=root)
    rng = np.random.default_rng(config.seed)
    train = split.train_idx.copy()
    rng.shuffle(train)
    sizes = [s for s in (500, 1000, 2000, 4000) if s <= len(train)]
    if sizes[-1] != len(train):
        sizes.append(int(len(train)))
    scores = []
    for n in sizes:
        idx = np.sort(train[:n])
        model, _ = train_mlp(
            x, z, bias, aux, idx,
            hidden=config.representation.hidden,
            epochs=config.train.epochs,
            lr=config.train.lr,
            weight_decay=config.train.weight_decay,
            nll_weight=config.train.nll_weight,
            aux_weight=config.train.aux_weight,
            bias_weight=config.train.bias_weight,
            seed=config.seed,
        )
        pred = predict_mlp(model, x)
        metrics = recovery_metrics(pred["mu"][split.holdout_idx], z[split.holdout_idx])
        scores.append(metrics["cosine_mean"])
        print(f"  n={n}  holdout cosine={metrics['cosine_mean']:.3f}")
    dest = output_dir() / "scaling.json"
    dest.write_text(json.dumps({"sizes": sizes, "holdout_cosine": scores}, indent=2))
    plot_scaling(sizes, scores, figures_dir() / "scaling.png")
    return dest
