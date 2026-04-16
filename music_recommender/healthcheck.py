from __future__ import annotations

import argparse
from pathlib import Path

from .service import MusicRecommenderService


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate the music recommender artifacts and print a summary."
    )
    parser.add_argument(
        "--base-dir",
        default=".",
        help="Directory containing songs_df.joblib, feature_matrix.joblib, and item_users_dict.joblib.",
    )
    args = parser.parse_args()

    base_dir = Path(args.base_dir).resolve()
    service = MusicRecommenderService.from_base_dir(base_dir)
    summary = service.health_summary()

    print("Health check passed.")
    print(f"base_dir: {base_dir}")
    print(f"songs: {summary['song_count']}")
    print(f"features: {summary['feature_count']}")
    print(f"collaborative_items: {summary['collaborative_item_count']}")


if __name__ == "__main__":
    main()
