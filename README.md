# Lorcana Loss Pattern Analyzer

Python MVP pipeline for extracting Disney Lorcana replay data and finding patterns correlated with losses.

## What it does

- Loads and merges all `results/game-history-*.csv` exports
- Decompresses and parses replay `.replay.gz` JSON files from `replays/`
- Builds cleaned game and event datasets
- Produces loss-correlation analysis for cards and event patterns
- Splits key metrics by turn order (`OTP` vs `OTD`)
- Produces prioritized win-rate improvement recommendations
- Exports analysis tables and plots to `analysis_output/`

## Quick start

1. Create and activate a Python environment.
2. Install dependencies:

```powershell
pip install -r requirements.txt
```

3. Run the analyzer:

```powershell
python analyze.py
```

4. Run the dashboard:

```powershell
python -m streamlit run dashboard.py
```

## API sync mode

If you have a duels.ink API token in `api/access`, you can sync match history + replay files directly before analysis:

```powershell
python analyze.py --sync-api
```

Optional filters:

```powershell
python analyze.py --sync-api --from-ts 2026-05-01T00:00:00Z --to-ts 2026-05-22T00:00:00Z
```

Skip gamelog downloads (replay analysis still works):

```powershell
python analyze.py --sync-api --skip-gamelogs
```

The sync step writes/updates:

- `results/game-history-YYYY-MM-DD.csv`
- `replays/YYYY-MM-DD/*.replay.gz`
- `gamelogs/YYYY-MM-DD/*.logs.gz` (unless skipped)

## Deploy (Streamlit Community Cloud)

This app can be deployed as a shared Streamlit site.

1. Push this repo to GitHub.
2. Make sure `analysis_output/` contains the CSVs the dashboard reads (for example `clean_games.csv`, `clean_events.csv`, and other generated analysis CSVs).
3. In Streamlit Community Cloud, create a new app from your GitHub repo.
4. Set the main file path to:

```text
dashboard.py
```

5. Deploy.

### Notes

- Keep `api/access` local only (already gitignored).
- If you later need API sync in hosted environments, use platform secrets instead of committing tokens.

## Outputs

Generated under `analysis_output/`:

- `clean_games.csv`
- `clean_events.csv`
- `card_loss_correlation.csv`
- `event_type_loss_correlation.csv`
- `opening_hand_loss_correlation.csv`
- `summary_metrics.csv`
- `win_rate_by_turn_order.csv`
- `analysis_summary.md`
- `winrate_recommendations.csv`
- `winrate_improvement_plan.md`
- `loss_model_feature_importance.csv` (when enough data is available)
