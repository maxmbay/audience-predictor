from __future__ import annotations

import argparse
from pathlib import Path

from audience_predictor.config import load_config
from audience_predictor.data.env import load_dotenv
from audience_predictor.pipeline import (
    run_coverage,
    run_evaluate,
    run_figures,
    run_ingest_moviesum,
    run_ingest_sublens,
    run_predict,
    run_scaling,
    run_train,
)
from audience_predictor.report import write_report


def main(argv: list[str] | None = None) -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(
        prog="audience_predictor",
        description="Predict a film's latent audience from narrative content (JAX/Equinox)",
    )
    parser.add_argument("--config", type=Path, default=None)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("coverage", help="Write the MovieLens join coverage report")
    ingest = sub.add_parser(
        "ingest-moviesum",
        help="Parse MovieSum XML, join IMDb ids, write hashed or frozen-encoder blocks",
    )
    ingest.add_argument("--encoder", choices=("hash", "bge"), default="hash")
    ingest_sub = sub.add_parser(
        "ingest-sublens",
        help="Parse SubLens-20M SRT files, join IMDb ids, write hashed or frozen-encoder blocks",
    )
    ingest_sub.add_argument("--encoder", choices=("hash", "bge"), default="hash")
    train = sub.add_parser("train", help="Train an Equinox encoder into collaborative z-space")
    train.add_argument(
        "--modality",
        choices=(
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
        ),
        default="genome",
    )
    predict = sub.add_parser("predict", help="Audience fingerprint from a script or other text")
    predict.add_argument("--text-file", type=Path, required=True, help="Screenplay, SRT, or prose")
    sub.add_parser("evaluate", help="Summarize trained modalities on the movie-level holdout")
    sub.add_parser("figures", help="Write report figures from trained checkpoints")
    sub.add_parser("scaling", help="Holdout cosine vs number of training titles (genome)")
    sub.add_parser("report", help="Write the three-register technical report")
    args = parser.parse_args(argv)
    config = load_config(args.config)
    if args.command == "coverage":
        path = run_coverage(config)
        print(path)
    elif args.command == "ingest-moviesum":
        path = run_ingest_moviesum(config, encoder=args.encoder)
        print(path)
    elif args.command == "ingest-sublens":
        path = run_ingest_sublens(config, encoder=args.encoder)
        print(path)
    elif args.command == "train":
        run_train(config, args.modality)
    elif args.command == "predict":
        run_predict(config, args.text_file.read_text(encoding="utf-8", errors="replace"))
    elif args.command == "evaluate":
        run_evaluate(config)
    elif args.command == "figures":
        run_figures(config)
    elif args.command == "scaling":
        run_scaling(config)
    elif args.command == "report":
        print(write_report(config))
    else:
        raise ValueError(args.command)


if __name__ == "__main__":
    main()
