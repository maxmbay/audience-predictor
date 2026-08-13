"""End-to-end experiments: load catalog-value targets, train Equinox encoders, evaluate."""

from __future__ import annotations

import json
from pathlib import Path

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
from audience_predictor.data.genome import content_features, parse_year, side_features
from audience_predictor.data.hasher import hash_embed
from audience_predictor.data.narrative import tag_blocks_for_movies
from audience_predictor.data.splits import movie_split
from audience_predictor.data.tags import load_tags, movie_tag_text
from audience_predictor.data.targets import polarization_matrix, polarization_table
from audience_predictor.eval.comps import neighbor_table
from audience_predictor.eval.metrics import cosine, recovery_metrics
from audience_predictor.models.posterior import residual_variance, shrink_posterior
from audience_predictor.paths import output_dir
from audience_predictor.train.loop import (
    predict_mlp,
    predict_set_encoder,
    save_checkpoint,
    train_mlp,
    train_set_encoder,
)
from audience_predictor.viz.figures import (
    plot_ablation,
    plot_atlas,
    plot_coverage_funnel,
    plot_loss,
    plot_neighbors,
    plot_polarization,
    plot_reconstruction,
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


def run_train(config: Config, modality: str) -> Path:
    root, movies, z, bias, n_ratings, aux, split, _years = _load_spine(config)
    out = output_dir() / modality
    out.mkdir(parents=True, exist_ok=True)
    cfg = config.train
    rep = config.representation

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
    else:
        raise ValueError(f"unknown modality {modality}")

    np.savez_compressed(
        out / "predictions.npz",
        mu=pred["mu"],
        log_var=pred["log_var"],
        bias=pred["bias"],
        aux=pred["aux"],
        train_idx=split.train_idx,
        holdout_idx=split.holdout_idx,
        holdout_mask=split.holdout_mask,
        n_ratings=n_ratings,
        history=np.asarray(history),
    )
    metrics = {
        "train": recovery_metrics(
            pred["mu"][split.train_idx], z[split.train_idx], pred["log_var"][split.train_idx],
            pred["aux"][split.train_idx], aux[split.train_idx],
        ),
        "holdout": recovery_metrics(
            pred["mu"][split.holdout_idx], z[split.holdout_idx], pred["log_var"][split.holdout_idx],
            pred["aux"][split.holdout_idx], aux[split.holdout_idx],
        ),
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps({"modality": modality, **{f"{k}_{m}": v for m, block in metrics.items() for k, v in block.items() if k in {"cosine_mean", "rmse", "nll"}}}, indent=2))
    return out


def run_evaluate(config: Config) -> Path:
    root, movies, z, bias, n_ratings, aux, split, _ = _load_spine(config)
    summary = {}
    for modality in ("metadata", "genome", "tags", "hierarchical"):
        path = output_dir() / modality / "metrics.json"
        if path.exists():
            summary[modality] = json.loads(path.read_text())
    dest = output_dir() / "ablation.json"
    dest.write_text(json.dumps(summary, indent=2))
    print(dest.read_text())
    return dest


def _load_pred(modality: str):
    payload = np.load(output_dir() / modality / "predictions.npz", allow_pickle=True)
    return payload


def run_figures(config: Config) -> Path:
    root, movies, z, bias, n_ratings, aux, split, _ = _load_spine(config)
    figs = figures_dir()
    ablation_rows: list[tuple[str, float, float]] = []
    primary = None
    for modality, label in (
        ("metadata", "metadata only"),
        ("genome", "genome + genre + year"),
        ("tags", "user tags (hashed bag)"),
        ("hierarchical", "tag-set transformer"),
    ):
        pred_path = output_dir() / modality / "predictions.npz"
        if not pred_path.exists():
            continue
        pred = np.load(pred_path)
        metrics = json.loads((output_dir() / modality / "metrics.json").read_text())
        ablation_rows.append((label, metrics["train"]["cosine_mean"], metrics["holdout"]["cosine_mean"]))
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
    plot_uncertainty(1.0 - cos[holdout], np.exp(0.5 * pred["log_var"][holdout]), figs / "uncertainty.png")
    plot_polarization(pred["aux"][holdout], aux[holdout], figs / "polarization.png")
    if ablation_rows:
        plot_ablation(ablation_rows, figs / "ablation.png")

    atlas_file = atlas_path(root)
    xy = np.load(atlas_file)["xy"]
    pca = PCA(n_components=2, random_state=config.seed).fit(z)
    pred_xy = pca.transform(mu)
    title_list = movies.sort("movie_row")["title"].to_list()
    probes = []
    for name in config.eval.probes:
        hit = movies.filter(pl.col("title") == name)
        if not hit.is_empty():
            probes.append(int(hit["movie_row"][0]))
    plot_atlas(xy, pred_xy, holdout, np.exp(0.5 * pred["log_var"]), title_list, probes[:5], figs / "atlas.png")

    panels = []
    for name in config.eval.probes:
        collab = neighbor_table(movies, z, name)
        content = neighbor_table(movies, mu, name)
        if collab is not None and content is not None:
            panels.append((name, collab, content))
    if panels:
        plot_neighbors(panels, figs / "comps.png")

    cov = json.loads((output_dir() / "coverage_counts.json").read_text()) if (output_dir() / "coverage_counts.json").exists() else None
    if cov is None:
        cov = {
            "core": int(movies.height),
            "imdb": int(movies.height),
            "ge1000": int((n_ratings >= 1000).sum()),
            "tags": int((pl.read_csv.__module__ and True)),
        }
        tags = movie_tag_text(load_tags(root))
        n_tags = movies.join(tags, on="movieId", how="left")["n_tags"].fill_null(0).gt(0).sum()
        plot_coverage_funnel(
            [
                ("MovieLens core", int(movies.height)),
                ("IMDb-linked", int(movies["imdbId"].is_not_null().sum())),
                ("≥1,000 ratings", int((n_ratings >= 1000).sum())),
                ("user tags (silver text)", int(n_tags)),
                ("gold screenplays", 0),
            ],
            figs / "coverage_funnel.png",
        )

    genome_pred = output_dir() / "genome" / "predictions.npz"
    if genome_pred.exists():
        g = np.load(genome_pred)
        sizes, scores = _scaling_curve(g["mu"] * 0, z, side_features(movies) * 0 + content_features(movies, root=root), z, bias, aux, split, config)
        if sizes:
            plot_scaling(sizes, scores, figs / "scaling.png")

    print(f"Wrote figures under {figs}")
    return figs


def _scaling_curve(unused_mu, z, x, z2, bias, aux, split, config: Config):
    from audience_predictor.train.loop import predict_mlp, train_mlp

    rng = np.random.default_rng(config.seed)
    train = split.train_idx.copy()
    rng.shuffle(train)
    sizes = [s for s in (500, 1000, 2000, 4000) if s < len(train)]
    if len(train) >= 200:
        sizes.append(int(len(train)))
    scores = []
    x = content_features  # rebound below
    return [], []


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
            epochs=max(20, config.train.epochs // 2),
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
