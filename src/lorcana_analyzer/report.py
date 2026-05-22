from __future__ import annotations

from pathlib import Path
from functools import lru_cache
import json

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import requests

from .analytics import (
    card_loss_correlation,
    card_pair_correlation,
    card_seen_vs_played_correlation,
    compute_summary_metrics,
    early_ink_rates,
    event_type_loss_correlation,
    first_quest_turn,
    lore_curves,
    matchup_win_rates,
    mulligan_impact,
    opening_hand_loss_correlation,
    opponent_card_loss_correlation,
    tempo_curves,
)
from .advisor import build_recommendations, save_recommendation_artifacts
from .decks import add_archetype_columns, archetype_win_rates, deck_version_win_rates
from .pipeline import build_replay_tables, link_results_to_replays, load_results_csv
from .simulator import train_loss_model


_ALLOWED_COLORS = ["amber", "ruby", "sapphire", "amethyst", "steel", "emerald"]
_COLOR_DISPLAY = {
    "amber": "Amber",
    "ruby": "Ruby",
    "sapphire": "Sapphire",
    "amethyst": "Amethyst",
    "steel": "Steel",
    "emerald": "Emerald",
}


def _format_color_pair(colors: list[str]) -> str:
    normalized = [c.strip().lower() for c in colors if c and c.strip().lower() in _ALLOWED_COLORS]
    if not normalized:
        return "Unknown"

    seen: list[str] = []
    for c in normalized:
        if c not in seen:
            seen.append(c)

    if len(seen) >= 2:
        order = {c: i for i, c in enumerate(_ALLOWED_COLORS)}
        pair = sorted(seen[:2], key=lambda c: order[c])
        return f"{_COLOR_DISPLAY[pair[0]]}/{_COLOR_DISPLAY[pair[1]]}"
    return _COLOR_DISPLAY[seen[0]]


def _parse_colors_from_text(value: object) -> list[str]:
    text = str(value or "").strip().lower()
    if not text or text in {"unknown", "none", "nan", "null"}:
        return []

    text = text.replace("/", " ").replace(",", " ").replace("-", " ")
    tokens = [t for t in text.split() if t in _ALLOWED_COLORS]

    deduped: list[str] = []
    for token in tokens:
        if token not in deduped:
            deduped.append(token)
    return deduped


@lru_cache(maxsize=1)
def _card_id_to_color_map() -> dict[str, str]:
    """Fetch and cache card_id -> color mapping from public Lorcana API."""
    try:
        # This endpoint currently returns the full card list.
        resp = requests.get("https://api.lorcana-api.com/cards/fetch/?search=oswald", timeout=30)
        resp.raise_for_status()
        payload = resp.json()
    except Exception:
        return {}

    mapping: dict[str, str] = {}
    if not isinstance(payload, list):
        return mapping

    for row in payload:
        if not isinstance(row, dict):
            continue
        set_num = str(row.get("Set_Num") or "").strip()
        card_num = str(row.get("Card_Num") or "").strip()
        color = str(row.get("Color") or "").strip().lower()
        if not set_num or not card_num or color not in _ALLOWED_COLORS:
            continue
        mapping[f"{set_num}-{card_num}".lower()] = color
    return mapping


def _parse_opp_decklist_ids(value: object) -> list[tuple[str, int]]:
    if value is None:
        return []
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return []
    if not text.startswith("["):
        return []
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return []
    if not isinstance(parsed, list):
        return []

    out: list[tuple[str, int]] = []
    for item in parsed:
        if not isinstance(item, dict):
            continue
        card_id = str(item.get("cardId") or item.get("card_id") or "").strip().lower()
        try:
            qty = int(item.get("count") or 0)
        except (TypeError, ValueError):
            qty = 0
        if card_id and qty > 0:
            out.append((card_id, qty))
    return out


def _infer_opponent_colors(games_df: pd.DataFrame, events_df: pd.DataFrame) -> pd.Series:
    """Infer opponent color pairs from decklist/event card_ids and backfill Unknown values."""
    card_color = _card_id_to_color_map()
    if games_df.empty:
        return pd.Series(dtype="object")

    inferred = pd.Series("Unknown", index=games_df.index, dtype="object")

    # 1) Prefer opp_decklist from API history where available.
    if "opp_decklist" in games_df.columns and card_color:
        for idx, val in games_df["opp_decklist"].items():
            id_qty = _parse_opp_decklist_ids(val)
            if not id_qty:
                continue
            counts: dict[str, int] = {}
            for card_id, qty in id_qty:
                c = card_color.get(card_id)
                if not c:
                    continue
                counts[c] = counts.get(c, 0) + qty
            if len(counts) >= 2:
                ordered = sorted(counts.items(), key=lambda t: t[1], reverse=True)
                inferred.at[idx] = _format_color_pair([name for name, _ in ordered[:2]])

    # 2) Fallback from opponent CARD_PLAYED events.
    if not events_df.empty and {"csv_row_id", "card_id", "event_type", "is_my_event"}.issubset(events_df.columns) and card_color:
        opp_events = events_df[
            (events_df["is_my_event"] == False)
            & (events_df["event_type"] == "CARD_PLAYED")
            & events_df["card_id"].notna()
        ].copy()
        if not opp_events.empty:
            opp_events["card_id"] = opp_events["card_id"].astype(str).str.strip().str.lower()
            for csv_id, group in opp_events.groupby("csv_row_id"):
                counts: dict[str, int] = {}
                for cid in group["card_id"]:
                    c = card_color.get(cid)
                    if not c:
                        continue
                    counts[c] = counts.get(c, 0) + 1
                if len(counts) < 2:
                    continue
                ordered = sorted(counts.items(), key=lambda t: t[1], reverse=True)
                pair = _format_color_pair([name for name, _ in ordered[:2]])
                mask = (games_df.get("csv_row_id") == csv_id)
                inferred = inferred.where(~mask | (inferred != "Unknown"), pair)

    # 3) Fallback to normalized existing values only when inference is unavailable.
    existing = pd.Series("Unknown", index=games_df.index, dtype="object")
    for col in ["opp_deck_colors", "opponent_colors", "Opponent Colors"]:
        if col in games_df.columns:
            parsed = games_df[col].map(_parse_colors_from_text)
            normalized = parsed.map(_format_color_pair)
            existing = existing.where(existing != "Unknown", normalized)

    return inferred.where(inferred != "Unknown", existing)


def _save_turn_order_chart(games_df: pd.DataFrame, output_root: Path) -> None:
    turn_df = (
        games_df.groupby("turn_order", as_index=False)
        .agg(loss_rate=("is_loss", "mean"), games=("csv_row_id", "count"))
    )
    turn_df["win_rate"] = 1.0 - turn_df["loss_rate"]
    turn_df.to_csv(output_root / "win_rate_by_turn_order.csv", index=False)

    plt.figure(figsize=(7, 4))
    ax = sns.barplot(data=turn_df, x="turn_order", y="win_rate", color="#2c7fb8")
    ax.set_title("Win Rate by Turn Order")
    ax.set_xlabel("Turn Order")
    ax.set_ylabel("Win Rate")
    for i, row in turn_df.reset_index(drop=True).iterrows():
        ax.text(i, row["win_rate"] + 0.01, f"n={int(row['games'])}", ha="center")
    plt.ylim(0, 1.1)
    plt.tight_layout()
    plt.savefig(output_root / "win_rate_by_turn_order.png", dpi=150)
    plt.close()


def _save_top_loss_cards_chart(card_df: pd.DataFrame, output_root: Path) -> None:
    if card_df.empty:
        return
    top = card_df.head(15).sort_values("loss_lift_vs_baseline", ascending=True)

    plt.figure(figsize=(9, 6))
    ax = sns.barplot(data=top, x="loss_lift_vs_baseline", y="card_name", color="#d95f0e")
    ax.set_title("Top Cards Correlated with Higher Loss Rate")
    ax.set_xlabel("Loss Lift vs Baseline")
    ax.set_ylabel("Card")
    for i, row in top.reset_index(drop=True).iterrows():
        ax.text(row["loss_lift_vs_baseline"], i, f" n={int(row['games_with_card'])}")
    plt.tight_layout()
    plt.savefig(output_root / "top_loss_cards.png", dpi=150)
    plt.close()


def _save_top_opening_hand_chart(opening_df: pd.DataFrame, output_root: Path) -> None:
    if opening_df.empty:
        return
    top = opening_df.head(15).sort_values("loss_lift_vs_baseline", ascending=True)

    plt.figure(figsize=(9, 6))
    ax = sns.barplot(data=top, x="loss_lift_vs_baseline", y="card_name", color="#756bb1")
    ax.set_title("Top Opening-Hand Cards Correlated with Higher Loss Rate")
    ax.set_xlabel("Loss Lift vs Baseline")
    ax.set_ylabel("Card")
    for i, row in top.reset_index(drop=True).iterrows():
        ax.text(row["loss_lift_vs_baseline"], i, f" n={int(row['games_with_card'])}")
    plt.tight_layout()
    plt.savefig(output_root / "top_opening_hand_loss_cards.png", dpi=150)
    plt.close()


def _save_archetype_chart(arch_df: pd.DataFrame, output_root: Path) -> None:
    if arch_df.empty:
        return

    pivot = arch_df[arch_df["turn_order"].isin(("OTP", "OTD"))].pivot_table(
        index="archetype", columns="turn_order", values="win_rate"
    ).reset_index()
    if pivot.empty:
        return

    archetypes = pivot["archetype"].tolist()
    x = range(len(archetypes))
    width = 0.35

    fig, ax = plt.subplots(figsize=(max(8, len(archetypes) * 1.2), 5))
    if "OTP" in pivot.columns:
        otp_vals = pivot["OTP"].fillna(0).tolist()
        bars_otp = ax.bar([i - width / 2 for i in x], otp_vals, width, label="OTP", color="#2c7fb8")
        for bar in bars_otp:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{bar.get_height():.0%}", ha="center", va="bottom", fontsize=8)
    if "OTD" in pivot.columns:
        otd_vals = pivot["OTD"].fillna(0).tolist()
        bars_otd = ax.bar([i + width / 2 for i in x], otd_vals, width, label="OTD", color="#d95f0e")
        for bar in bars_otd:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{bar.get_height():.0%}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(list(x))
    ax.set_xticklabels(archetypes, rotation=30, ha="right")
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Win Rate")
    ax.set_title("Win Rate by Deck Archetype and Turn Order")
    ax.legend()
    ax.axhline(0.5, linestyle="--", color="gray", linewidth=0.8)
    plt.tight_layout()
    plt.savefig(output_root / "archetype_win_rates.png", dpi=150)
    plt.close()


def _save_matchup_heatmap(matchup_df: pd.DataFrame, output_root: Path) -> None:
    if matchup_df.empty:
        return

    pivot = matchup_df.pivot_table(
        index="archetype", columns="opponent_colors", values="win_rate"
    )
    if pivot.empty:
        return

    fig, ax = plt.subplots(figsize=(max(8, pivot.shape[1] * 1.2), max(4, pivot.shape[0] * 0.8)))
    sns.heatmap(
        pivot, annot=True, fmt=".0%", cmap="RdYlGn", vmin=0, vmax=1,
        linewidths=0.5, ax=ax, cbar_kws={"label": "Win Rate"}
    )
    ax.set_title("Win Rate by My Archetype vs Opponent Colors")
    ax.set_xlabel("Opponent Colors")
    ax.set_ylabel("My Archetype")
    plt.tight_layout()
    plt.savefig(output_root / "matchup_heatmap.png", dpi=150)
    plt.close()


def _save_tempo_charts(tempo_df: pd.DataFrame, output_root: Path) -> None:
    if tempo_df.empty:
        return

    event_labels = {
        "CARD_PLAYED": "Cards Played",
        "CARD_QUEST": "Quests",
        "CARD_ATTACK": "Attacks",
        "CARD_INKED": "Cards Inked",
    }
    event_types = tempo_df["event_type"].unique()

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()

    for i, event_type in enumerate(["CARD_PLAYED", "CARD_QUEST", "CARD_ATTACK", "CARD_INKED"]):
        ax = axes[i]
        subset = tempo_df[tempo_df["event_type"] == event_type]
        if subset.empty:
            ax.set_visible(False)
            continue
        for outcome, color in (("Win", "#2c7fb8"), ("Loss", "#d95f0e")):
            data = subset[subset["outcome"] == outcome].sort_values("turn_number")
            if data.empty:
                continue
            ax.plot(data["turn_number"], data["avg_count"], marker="o", color=color, label=outcome)
        ax.set_title(event_labels.get(event_type, event_type))
        ax.set_xlabel("Turn")
        ax.set_ylabel("Avg per game")
        ax.legend()
        ax.set_xticks(range(1, 13))

    plt.suptitle("Action Tempo Curves: Win vs Loss", fontsize=13)
    plt.tight_layout()
    plt.savefig(output_root / "tempo_curves.png", dpi=150)
    plt.close()


def _save_first_quest_chart(fq_df: pd.DataFrame, output_root: Path) -> None:
    if fq_df.empty:
        return

    relevant = fq_df[fq_df["first_quest_turn"] > 0].copy()
    if relevant.empty:
        return

    fig, ax = plt.subplots(figsize=(9, 4))
    for outcome, color in (("Win", "#2c7fb8"), ("Loss", "#d95f0e")):
        data = relevant[relevant["outcome"] == outcome].sort_values("first_quest_turn")
        if data.empty:
            continue
        ax.bar(
            data["first_quest_turn"] + (0.2 if outcome == "Win" else -0.2),
            data["pct"], width=0.38, color=color, label=outcome, alpha=0.85
        )
    ax.set_xlabel("First Quest Turn")
    ax.set_ylabel("% of Games")
    ax.set_title("First Quest Turn Distribution: Win vs Loss")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax.set_xticks(sorted(relevant["first_quest_turn"].unique()))
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_root / "first_quest_turn.png", dpi=150)
    plt.close()


def _save_ink_rate_chart(ink_df: pd.DataFrame, output_root: Path) -> None:
    if ink_df.empty:
        return

    fig, ax = plt.subplots(figsize=(7, 4))
    for outcome, color in (("Win", "#2c7fb8"), ("Loss", "#d95f0e")):
        data = ink_df[ink_df["outcome"] == outcome].sort_values("turn")
        if data.empty:
            continue
        ax.plot(data["turn"], data["ink_rate"], marker="o", color=color, label=outcome)
    ax.set_xlabel("Turn")
    ax.set_ylabel("Rate of Inking")
    ax.set_title("Inking Rate per Turn (Turns 1-4): Win vs Loss")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax.set_xticks([1, 2, 3, 4])
    ax.set_ylim(0, 1.05)
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_root / "early_ink_rates.png", dpi=150)
    plt.close()


def _save_markdown_summary(
    output_root: Path,
    games_df: pd.DataFrame,
    replay_errors_df: pd.DataFrame,
    card_df: pd.DataFrame,
    opening_df: pd.DataFrame,
    arch_df: pd.DataFrame,
    matchup_df: pd.DataFrame,
) -> None:
    total_games = len(games_df)
    matched = int(games_df["join_status"].eq("matched").sum()) if "join_status" in games_df.columns else 0
    unmatched = total_games - matched

    lines = [
        "# Lorcana Loss Analysis Summary",
        "",
        f"- Total CSV games: {total_games}",
        f"- Replays matched: {matched}",
        f"- Replays unmatched: {unmatched}",
        f"- Replay parse errors: {len(replay_errors_df)}",
        f"- Overall win rate: {(1.0 - games_df['is_loss'].mean()):.2%}" if total_games else "- Overall win rate: N/A",
        "",
        "## Top Card Correlations (Loss Lift)",
    ]

    if card_df.empty:
        lines.append("- No card-level signals met the minimum sample threshold.")
    else:
        for _, row in card_df.head(10).iterrows():
            lines.append(
                f"- {row['card_name']}: loss lift {row['loss_lift_vs_baseline']:+.2%} across {int(row['games_with_card'])} games"
            )

    lines.append("")
    lines.append("## Top Opening Hand Correlations")
    if opening_df.empty:
        lines.append("- No opening-hand signals met the minimum sample threshold.")
    else:
        for _, row in opening_df.head(10).iterrows():
            lines.append(
                f"- {row['card_name']}: opening-hand loss lift {row['loss_lift_vs_baseline']:+.2%} across {int(row['games_with_card'])} games"
            )

    lines.append("")
    lines.append("## Archetype Win Rates")
    if arch_df.empty:
        lines.append("- Insufficient data for archetype analysis.")
    else:
        all_rows = arch_df[arch_df["turn_order"] == "All"].sort_values("win_rate", ascending=False)
        for _, row in all_rows.iterrows():
            lines.append(
                f"- {row['archetype']}: {row['win_rate']:.0%} win rate ({int(row['games'])} games)"
            )

    lines.append("")
    lines.append("## Matchup Win Rates")
    if matchup_df.empty:
        lines.append("- Insufficient data for matchup analysis.")
    else:
        for _, row in matchup_df.sort_values("games", ascending=False).head(15).iterrows():
            lines.append(
                f"- {row['archetype']} vs {row['opponent_colors']}: "
                f"{row['win_rate']:.0%} win rate ({int(row['games'])} games)"
            )

    lines.append("")
    lines.append("## Notes")
    lines.append("- These are correlation metrics, not causal proof.")
    lines.append("- Prioritize insights with larger sample sizes and stable lift over time.")

    (output_root / "analysis_summary.md").write_text("\n".join(lines), encoding="utf-8")


def run_full_analysis(results_csv: Path, replays_root: Path, output_root: Path) -> None:
    output_root.mkdir(parents=True, exist_ok=True)

    results_df = load_results_csv(results_csv)
    replay_games_df, replay_events_df, replay_errors_df = build_replay_tables(replays_root)
    linked_games_df = link_results_to_replays(results_df, replay_games_df)

    # Add archetype columns before any archetype-dependent analytics
    linked_games_df = add_archetype_columns(linked_games_df)

    if not replay_events_df.empty:
        replay_events_df = replay_events_df.merge(
            linked_games_df[["csv_row_id", "game_id"]].dropna(), on="game_id", how="inner"
        )

    # Enrich opponent colors at generation time so downstream dashboards avoid live inference.
    linked_games_df["opponent_colors"] = _infer_opponent_colors(linked_games_df, replay_events_df)

    summary_df = compute_summary_metrics(linked_games_df)
    card_df = card_loss_correlation(linked_games_df, replay_events_df)
    event_type_df = event_type_loss_correlation(linked_games_df, replay_events_df)
    opening_df = opening_hand_loss_correlation(linked_games_df)
    arch_df = archetype_win_rates(linked_games_df)
    deck_ver_df = deck_version_win_rates(linked_games_df)
    matchup_df = matchup_win_rates(linked_games_df)
    tempo_df = tempo_curves(linked_games_df, replay_events_df)
    ink_df = early_ink_rates(linked_games_df, replay_events_df)
    fq_df = first_quest_turn(linked_games_df, replay_events_df)
    opp_card_df = opponent_card_loss_correlation(linked_games_df, replay_events_df)
    pair_df = card_pair_correlation(linked_games_df, replay_events_df)
    seen_played_df = card_seen_vs_played_correlation(linked_games_df, replay_events_df)
    lore_df = lore_curves(linked_games_df, replay_events_df)
    mulligan_df = mulligan_impact(linked_games_df)

    # Optional predictive model used for feature importance guidance.
    model_info = train_loss_model(linked_games_df, replay_events_df, card_df)

    recommendations_df = build_recommendations(
        games_df=linked_games_df,
        card_df=card_df,
        opening_df=opening_df,
        event_type_df=event_type_df,
        ink_df=ink_df,
        fq_df=fq_df,
        matchup_df=matchup_df,
        seen_played_df=seen_played_df,
    )

    linked_games_df.to_csv(output_root / "clean_games.csv", index=False)
    replay_events_df.to_csv(output_root / "clean_events.csv", index=False)
    replay_errors_df.to_csv(output_root / "replay_parse_errors.csv", index=False)
    summary_df.to_csv(output_root / "summary_metrics.csv", index=False)
    card_df.to_csv(output_root / "card_loss_correlation.csv", index=False)
    event_type_df.to_csv(output_root / "event_type_loss_correlation.csv", index=False)
    opening_df.to_csv(output_root / "opening_hand_loss_correlation.csv", index=False)
    arch_df.to_csv(output_root / "archetype_win_rates.csv", index=False)
    deck_ver_df.to_csv(output_root / "deck_version_win_rates.csv", index=False)
    matchup_df.to_csv(output_root / "matchup_win_rates.csv", index=False)
    tempo_df.to_csv(output_root / "sequencing_tempo_curves.csv", index=False)
    ink_df.to_csv(output_root / "sequencing_early_ink.csv", index=False)
    fq_df.to_csv(output_root / "sequencing_first_quest.csv", index=False)
    opp_card_df.to_csv(output_root / "opponent_card_loss_correlation.csv", index=False)
    pair_df.to_csv(output_root / "card_pair_correlation.csv", index=False)
    seen_played_df.to_csv(output_root / "card_seen_vs_played.csv", index=False)
    lore_df.to_csv(output_root / "lore_curves.csv", index=False)
    mulligan_df.to_csv(output_root / "mulligan_impact.csv", index=False)

    save_recommendation_artifacts(
        output_root=output_root,
        games_df=linked_games_df,
        recommendations_df=recommendations_df,
        model_info=model_info,
    )

    _save_turn_order_chart(linked_games_df, output_root)
    _save_top_loss_cards_chart(card_df, output_root)
    _save_top_opening_hand_chart(opening_df, output_root)
    _save_archetype_chart(arch_df, output_root)
    _save_matchup_heatmap(matchup_df, output_root)
    _save_tempo_charts(tempo_df, output_root)
    _save_first_quest_chart(fq_df, output_root)
    _save_ink_rate_chart(ink_df, output_root)
    _save_markdown_summary(output_root, linked_games_df, replay_errors_df, card_df, opening_df, arch_df, matchup_df)
