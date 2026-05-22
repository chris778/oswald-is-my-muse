import argparse
from pathlib import Path
import sys
import tempfile

import pandas as pd

repo_root = Path(__file__).resolve().parent
sys.path.insert(0, str(repo_root / "src"))

from lorcana_analyzer.report import run_full_analysis


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Lorcana replay analysis")
    parser.add_argument(
        "--sync-api",
        action="store_true",
        help="Sync match history, replays, and gamelogs via duels.ink API before analysis.",
    )
    parser.add_argument(
        "--token-file",
        type=Path,
        default=repo_root / "api" / "access",
        help="Path to API bearer token file (default: api/access).",
    )
    parser.add_argument(
        "--api-base-url",
        default="https://duels.ink",
        help="Base URL for API requests.",
    )
    parser.add_argument(
        "--from-ts",
        default=None,
        help="Optional inclusive lower bound (ISO 8601) for match-history sync.",
    )
    parser.add_argument(
        "--to-ts",
        default=None,
        help="Optional exclusive upper bound (ISO 8601) for match-history sync.",
    )
    parser.add_argument(
        "--skip-gamelogs",
        action="store_true",
        help="Skip downloading gamelog files during API sync.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    results_dir = repo_root / "results"
    replays_root = repo_root / "replays"
    gamelogs_root = repo_root / "gamelogs"
    output_root = repo_root / "analysis_output"

    if args.sync_api:
        from lorcana_analyzer.api_client import sync_from_api

        summary = sync_from_api(
            token_file=args.token_file,
            results_dir=results_dir,
            replays_root=replays_root,
            gamelogs_root=gamelogs_root,
            base_url=args.api_base_url,
            from_ts=args.from_ts,
            to_ts=args.to_ts,
            download_gamelogs=not args.skip_gamelogs,
        )
        print(f"API sync rows fetched: {summary.rows_fetched}")
        if summary.history_csv:
            print(f"Wrote match history CSV: {summary.history_csv}")
        print(
            "Replays downloaded/skipped/missing: "
            f"{summary.replay_downloaded}/{summary.replay_skipped}/{summary.replay_missing}"
        )
        if not args.skip_gamelogs:
            print(
                "Gamelogs downloaded/skipped/missing: "
                f"{summary.gamelog_downloaded}/{summary.gamelog_skipped}/{summary.gamelog_missing}"
            )

    # Merge all game-history CSVs and deduplicate with whichever key columns are available.
    csv_files = sorted(results_dir.glob("game-history-*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No game-history CSVs found in {results_dir}")

    frames = [pd.read_csv(f) for f in csv_files]
    combined = pd.concat(frames, ignore_index=True)

    dedupe_candidates = [
        ["game_id"],
        ["gameId"],
        ["Started At", "Opponent"],
        ["started_at", "opponent"],
    ]
    dedupe_subset: list[str] | None = None
    for candidate in dedupe_candidates:
        if all(col in combined.columns for col in candidate):
            dedupe_subset = candidate
            break
    if dedupe_subset:
        combined = combined.drop_duplicates(subset=dedupe_subset)
    else:
        combined = combined.drop_duplicates()

    sort_col = next((c for c in ["Started At", "started_at", "updated_at", "updatedAt"] if c in combined.columns), None)
    if sort_col:
        combined = combined.sort_values(sort_col, ascending=False)
    combined = combined.reset_index(drop=True)
    print(f"Combined {len(combined)} unique games from {len(csv_files)} file(s).")

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".csv", delete=False, encoding="utf-8"
    ) as tmp:
        combined.to_csv(tmp, index=False)
        tmp_path = Path(tmp.name)

    try:
        run_full_analysis(results_csv=tmp_path, replays_root=replays_root, output_root=output_root)
    finally:
        tmp_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()

