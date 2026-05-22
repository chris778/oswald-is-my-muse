from __future__ import annotations

import hashlib

import pandas as pd


def _deck_version_hash(decklist_map: object) -> str:
    if not isinstance(decklist_map, dict) or not decklist_map:
        return "unknown"
    sorted_entries = sorted(f"{qty}x{name}" for name, qty in decklist_map.items())
    return hashlib.md5("|".join(sorted_entries).encode()).hexdigest()[:8]


def add_archetype_columns(games_df: pd.DataFrame) -> pd.DataFrame:
    df = games_df.copy()
    df["archetype"] = df["my_colors"].fillna("Unknown").astype(str)
    df["deck_version"] = df["decklist_map"].map(_deck_version_hash)
    df["archetype_version"] = df["archetype"] + " [" + df["deck_version"] + "]"
    return df


def archetype_win_rates(games_df: pd.DataFrame, min_games: int = 5) -> pd.DataFrame:
    if "archetype" not in games_df.columns:
        return pd.DataFrame()

    rows = []
    for arch, group in games_df.groupby("archetype"):
        if len(group) < min_games:
            continue
        for turn_order in ("OTP", "OTD", "All"):
            subset = group if turn_order == "All" else group[group["turn_order"] == turn_order]
            if subset.empty:
                continue
            rows.append(
                {
                    "archetype": arch,
                    "turn_order": turn_order,
                    "games": len(subset),
                    "wins": int((~subset["is_loss"]).sum()),
                    "losses": int(subset["is_loss"].sum()),
                    "win_rate": 1.0 - subset["is_loss"].mean(),
                }
            )

    return pd.DataFrame(rows).sort_values(["archetype", "turn_order"])


def deck_version_win_rates(games_df: pd.DataFrame, min_games: int = 5) -> pd.DataFrame:
    if "archetype_version" not in games_df.columns:
        return pd.DataFrame()

    rows = []
    for version, group in games_df.groupby("archetype_version"):
        if len(group) < min_games:
            continue
        for turn_order in ("OTP", "OTD", "All"):
            subset = group if turn_order == "All" else group[group["turn_order"] == turn_order]
            if subset.empty:
                continue
            rows.append(
                {
                    "archetype_version": version,
                    "turn_order": turn_order,
                    "games": len(subset),
                    "wins": int((~subset["is_loss"]).sum()),
                    "losses": int(subset["is_loss"].sum()),
                    "win_rate": 1.0 - subset["is_loss"].mean(),
                }
            )

    return pd.DataFrame(rows).sort_values(["archetype_version", "turn_order"])
