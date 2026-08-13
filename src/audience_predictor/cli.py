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
    sub.add_parser("ingest-moviesum", help="Parse MovieSum XML, join IMDb ids, write hashed blocks")
    train = sub.add_parser("train", help="Train an Equinox encoder into collaborative z-space")
    train.add_argument(
        "--modality",
        choices=("metadata", "genome", "tags", "hierarchical", "genome_gold", "screenplay", "dialogue"),
        default="genome",
    )
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
        path = run_ingest_moviesum(config)
        print(path)
    elif args.command == "train":
        run_train(config, args.modality)
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
