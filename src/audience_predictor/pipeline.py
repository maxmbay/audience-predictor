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
from audience_predictor.data.encoder import get_encoder
from audience_predictor.data.genome import (
    content_features,
    genome_bucket_matrix,
    genome_matrix,
    genome_tag_names,
    parse_year,
    side_features,
)
from audience_predictor.data.genome_buckets import bucket_tag
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
from audience_predictor.data.splits import movie_split
from audience_predictor.data.subtitles import (
    ingest_sublens,
    load_sublens_blocks,
    silver_movie_rows,
)
from audience_predictor.data.tags import load_tags, movie_tag_text
from audience_predictor.data.targets import polarization_matrix, polarization_table
from audience_predictor.eval.comps import (
    COMP_K,
    COMP_MIN_RATINGS,
    neighbor_table,
    neighbors_from_vector,
    popular_rows,
    product_metrics,
)
from audience_predictor.eval.heads import format_predict_card, predict_head_for_document
from audience_predictor.eval.metrics import cosine, recovery_metrics
from audience_predictor.models.mlp import AudienceMLP
from audience_predictor.paths import output_dir, reports_dir
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
from audience_predictor.viz.model_card import (
    plot_architecture_dag,
    plot_baseline_atlas,
    plot_honesty_ladder,
    plot_product_recall,
    plot_tag_constellation,
    plot_two_step,
    plot_usefulness_plane,
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


def _slice_metrics(
    pred: dict,
    z,
    aux,
    idx: np.ndarray,
    *,
    n_ratings: np.ndarray | None = None,
    popular: np.ndarray | None = None,
) -> dict:
    out = recovery_metrics(
        pred["mu"][idx],
        z[idx],
        pred["log_var"][idx],
        pred["aux"][idx],
        aux[idx],
    )
    if n_ratings is not None and popular is not None and np.asarray(idx).size:
        out.update(product_metrics(z, pred["mu"], idx, n_ratings, popular, k=COMP_K))
    return out


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
    _root, movies, z, _bias, n_ratings, aux, split, _years = _load_spine(config)
    popular = popular_rows(movies, COMP_MIN_RATINGS)
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
        if pred_path.exists():
            pred = np.load(pred_path)
            hold_idx = pred["holdout_idx"]
            block["holdout"] = {
                **block.get("holdout", {}),
                **product_metrics(z, pred["mu"], hold_idx, n_ratings, popular, k=COMP_K),
            }
            if gold_hold.size:
                block["gold_train"] = (
                    _slice_metrics(pred, z, aux, gold_train) if gold_train.size else {}
                )
                block["gold_holdout"] = _slice_metrics(
                    pred, z, aux, gold_hold, n_ratings=n_ratings, popular=popular
                )
        summary[modality] = block
    dest = output_dir() / "ablation.json"
    dest.write_text(json.dumps(summary, indent=2))
    print(f"Wrote {dest}")
    print(
        f"{'modality':<22} {'cos':>7} {'R@10':>7} {'obscure':>8} {'love':>6} {'hate':>6} {'n':>6}"
    )
    for name, block in summary.items():
        ho = block.get("holdout") or {}
        print(
            f"{name:<22} {ho.get('cosine_mean', 0):7.3f} "
            f"{ho.get('neighbor_recall_at_10', 0):7.3f} "
            f"{ho.get('neighbor_recall_at_10_obscure', 0):8.3f} "
            f"{ho.get('spearman_p_love', 0):6.2f} "
            f"{ho.get('spearman_p_hate', 0):6.2f} "
            f"{int(ho.get('n', 0)):6d}"
        )
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
    run_model_card(config)
    return figs


def _holdout_block(data: dict, name: str) -> dict:
    return (data.get(name) or {}).get("holdout") or {}


def run_model_card(config: Config) -> Path:
    """Write the signature baseline figures used by docs/MODEL.md."""
    root, movies, z, _bias, _n_ratings, _aux, _split, _years = _load_spine(config)
    figs = figures_dir() / "model"
    figs.mkdir(parents=True, exist_ok=True)
    ablation_path = output_dir() / "ablation.json"
    data = json.loads(ablation_path.read_text()) if ablation_path.exists() else {}

    def cos(name: str) -> float:
        return float(_holdout_block(data, name).get("cosine_mean", 0.0))

    def spear(name: str) -> float:
        return float(_holdout_block(data, name).get("spearman_mean_rating", 0.0))

    plot_architecture_dag(figs / "dag.png")
    plot_honesty_ladder(
        [
            ("BGE screenplay mean-pool", cos("screenplay_mean"), "script"),
            ("BGE dialogue mean-pool", cos("dialogue_mean"), "script"),
            ("BGE subtitle mean-pool  ← parked baseline", cos("subtitle_mean"), "script"),
            ("oracle script-visible tags", cos("genome_script"), "script"),
            ("oracle production tags", cos("genome_production"), "production"),
            ("oracle reception tags (Oscars, masterpiece)", cos("genome_reception"), "release"),
            ("full genome + genre + year", cos("genome"), "release"),
            ("genre + year", cos("metadata"), "baseline"),
            ("hashed subtitles", cos("subtitle"), "baseline"),
            ("predicted tags (ISAB)", cos("script_tags"), "baseline"),
        ],
        figs / "honesty_ladder.png",
    )
    plot_usefulness_plane(
        [
            ("genome", cos("genome"), spear("genome"), "leak"),
            ("reception tags", cos("genome_reception"), spear("genome_reception"), "leak"),
            ("script-visible oracle", cos("genome_script"), spear("genome_script"), "ceiling"),
            ("production tags", cos("genome_production"), spear("genome_production"), "ceiling"),
            ("metadata", cos("metadata"), spear("metadata"), "fail"),
            ("subtitle BGE", cos("subtitle_mean"), spear("subtitle_mean"), "narrative"),
            ("dialogue BGE", cos("dialogue_mean"), spear("dialogue_mean"), "narrative"),
            ("hashed subtitles", cos("subtitle"), spear("subtitle"), "fail"),
            ("ISAB → tags", cos("script_tags"), spear("script_tags"), "fail"),
        ],
        figs / "usefulness_plane.png",
    )
    names = genome_tag_names(root)
    buckets = [bucket_tag(n) for n in names]
    plot_tag_constellation(
        genome_matrix(movies, root=root),
        buckets,
        names,
        figs / "tag_constellation.png",
    )
    plot_two_step(
        cos("genome_script"),
        cos("script_tags"),
        cos("script_tags_mean"),
        cos("subtitle_mean"),
        figs / "two_step.png",
    )

    def recall(name: str) -> float:
        return float(_holdout_block(data, name).get("neighbor_recall_at_10", 0.0))

    def recall_obscure(name: str) -> float:
        return float(_holdout_block(data, name).get("neighbor_recall_at_10_obscure", 0.0))

    chance = float(_holdout_block(data, "subtitle_mean").get("neighbor_recall_chance", 0.0))
    plot_product_recall(
        [
            ("hashed subtitles", recall("subtitle"), recall_obscure("subtitle"), "fail"),
            ("ISAB → tags", recall("script_tags"), recall_obscure("script_tags"), "fail"),
            ("metadata", recall("metadata"), recall_obscure("metadata"), "baseline"),
            (
                "BGE subtitle mean-pool  ← parked baseline",
                recall("subtitle_mean"),
                recall_obscure("subtitle_mean"),
                "script",
            ),
            ("BGE dialogue mean-pool", recall("dialogue_mean"), recall_obscure("dialogue_mean"), "script"),
            (
                "oracle script-visible tags",
                recall("genome_script"),
                recall_obscure("genome_script"),
                "ceiling",
            ),
            ("full genome", recall("genome"), recall_obscure("genome"), "leak"),
        ],
        chance,
        figs / "product_recall.png",
    )

    xy = np.load(atlas_path(root))["xy"]
    pca = PCA(n_components=2, random_state=config.seed).fit(z)
    genome_pred = output_dir() / "genome" / "predictions.npz"
    narr_pred = output_dir() / "subtitle_mean" / "predictions.npz"
    if genome_pred.exists() and narr_pred.exists():
        g = np.load(genome_pred)
        n = np.load(narr_pred)
        title_list = movies.sort("movie_row")["title"].to_list()
        probes = []
        for name in config.eval.probes:
            hit = movies.filter(pl.col("title") == name)
            if not hit.is_empty():
                probes.append(int(hit["movie_row"][0]))
        plot_baseline_atlas(
            xy,
            pca.transform(g["mu"]),
            pca.transform(n["mu"]),
            g["holdout_mask"].astype(bool),
            n["holdout_mask"].astype(bool),
            title_list,
            probes[:5],
            figs / "atlas_dual.png",
        )
    print(f"Wrote model-card figures under {figs}")
    return figs


def _pool_for_head(
    dialogue: np.ndarray,
    action: np.ndarray,
    mask: np.ndarray,
    structure: np.ndarray,
    head: str,
) -> np.ndarray:
    weights = mask.astype(np.float32)[..., None]
    denom = np.clip(weights.sum(axis=1), 1.0, None)
    pooled = (dialogue * weights).sum(axis=1) / denom
    if head == "screenplay_mean":
        action_mean = (action * weights).sum(axis=1) / denom
        return np.concatenate([pooled, action_mean, structure], axis=1)
    return pooled


def run_predict(config: Config, text: str, *, k: int = 8, as_json: bool = False) -> dict:
    """Drop in any narrative text; print comps and polarization.

    Screenplays use the gold dialogue_mean head when it exists. Everything
    else uses the parked subtitle_mean baseline. Predicted genome tags are
    not used for z until they beat this baseline on comps and polarization.
    """
    encoder = get_encoder("bge")
    dialogue, action, mask, structure, kind = blocks_from_document(
        text, max_blocks=config.data.max_blocks, encoder=encoder
    )
    head = predict_head_for_document(kind)
    ckpt = output_dir() / head / "model.eqx"
    if not ckpt.exists():
        raise FileNotFoundError(f"Train {head} before predict.")
    pooled = _pool_for_head(dialogue, action, mask, structure, head)
    key = jax.random.key(0)
    mlp_like = AudienceMLP(
        in_dim=pooled.shape[-1],
        out_dim=config.representation.embedding_dim,
        hidden=config.representation.hidden,
        key=key,
    )
    mlp = load_checkpoint(ckpt, mlp_like)
    pred = predict_mlp(mlp, pooled)
    mu = pred["mu"][0]
    aux = pred["aux"][0]
    sigma = float(np.exp(0.5 * pred["log_var"][0]))
    _root, movies, z, _bias, _n_ratings, _aux, _split, _years = _load_spine(config)
    comps = neighbors_from_vector(movies, z, mu, k=k)
    result = {
        "path": head,
        "document_type": kind,
        "n_blocks": int(mask.sum()),
        "sigma": sigma,
        "polarization": {
            "mean_rating": float(aux[0]),
            "rating_std": float(aux[1]),
            "p_love": float(aux[2]),
            "p_hate": float(aux[3]),
        },
        "comps": comps.to_dicts() if comps is not None else [],
    }
    print(json.dumps(result, indent=2) if as_json else format_predict_card(result))
    return result


def run_probes(config: Config, *, k: int = 6, n_each: int = 5) -> Path:
    """Write holdout title comps for a human 'useful / not' pass. No script text."""
    _root, movies, z, _bias, n_ratings, _aux, split, _years = _load_spine(config)
    gold = gold_movie_rows() if provenance_path().exists() else np.zeros(0, dtype=np.int64)
    hold = np.intersect1d(split.holdout_idx, gold)
    dest = reports_dir() / "probes.md"
    if hold.size == 0:
        dest.write_text("# Qualitative probes\n\nNo gold holdout titles yet.\n")
        return dest
    counts = n_ratings[hold]
    order = np.argsort(counts)
    pick = np.unique(np.concatenate([hold[order[:n_each]], hold[order[-n_each:]]]))
    pick = pick[np.argsort(n_ratings[pick])]
    spaces: dict[str, np.ndarray | None] = {
        "collaborative": z,
        "subtitle_mean": None,
        "dialogue_mean": None,
        "genome": None,
    }
    for name in ("subtitle_mean", "dialogue_mean", "genome"):
        path = output_dir() / name / "predictions.npz"
        if path.exists():
            spaces[name] = np.load(path)["mu"]
    titles = dict(zip(movies["movie_row"].to_list(), movies["title"].to_list()))
    lines = [
        "# Qualitative probes",
        "",
        "Held-out MovieSum titles, obscure then famous. Neighborhoods only —",
        "no screenplay text. Mark each card useful / not against the collaborative",
        "column. Later branches have to beat `subtitle_mean` here, not genome cosine.",
        "",
    ]
    median = float(np.median(counts))
    for idx in pick:
        i = int(idx)
        name = titles.get(i, f"movie_row={i}")
        band = "famous" if n_ratings[i] >= median else "obscure"
        lines.append(f"## {name}")
        lines.append("")
        lines.append(f"{band}  ·  {int(n_ratings[i]):,} MovieLens ratings  ·  holdout")
        lines.append("")
        tables: dict[str, list[str]] = {}
        for label, space in spaces.items():
            if space is None:
                continue
            table = neighbor_table(movies, space, name, k=k)
            if table is not None:
                tables[label] = table["title"].to_list()
        labels = [
            lab
            for lab in ("collaborative", "subtitle_mean", "dialogue_mean", "genome")
            if lab in tables
        ]
        if not labels:
            continue
        header = "| rank | " + " | ".join(labels) + " |"
        sep = "|-----:|" + "|".join(["---"] * len(labels)) + "|"
        lines.append(header)
        lines.append(sep)
        depth = max(len(tables[lab]) for lab in labels)
        for rank in range(depth):
            cells = [str(rank + 1)]
            for lab in labels:
                col = tables[lab]
                cells.append(col[rank] if rank < len(col) else "")
            lines.append("| " + " | ".join(cells) + " |")
        lines.append("")
    dest.write_text("\n".join(lines) + "\n")
    return dest


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
