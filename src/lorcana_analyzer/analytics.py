from __future__ import annotations

import pandas as pd


RELEVANT_CARD_EVENT_TYPES = {
    "CARD_PLAYED",
    "CARD_INKED",
    "CARD_QUEST",
    "CARD_ATTACK",
    "ABILITY_ACTIVATED",
    "ABILITY_TRIGGERED",
}


def compute_summary_metrics(games_df: pd.DataFrame) -> pd.DataFrame:
    total_games = len(games_df)
    matched_games = int(games_df["game_id"].notna().sum()) if "game_id" in games_df.columns else 0
    overall_win_rate = (1.0 - games_df["is_loss"].mean()) if total_games else 0.0

    by_turn_order = (
        games_df.groupby("turn_order", dropna=False)
        .agg(games=("csv_row_id", "count"), loss_rate=("is_loss", "mean"))
        .reset_index()
    )
    by_turn_order["win_rate"] = 1.0 - by_turn_order["loss_rate"]

    summary_rows = [
        {"metric": "total_games", "value": total_games},
        {"metric": "matched_replay_games", "value": matched_games},
        {"metric": "overall_win_rate", "value": overall_win_rate},
    ]
    for _, row in by_turn_order.iterrows():
        summary_rows.append({"metric": f"win_rate_{row['turn_order']}", "value": row["win_rate"]})

    return pd.DataFrame(summary_rows)


def card_loss_correlation(games_df: pd.DataFrame, events_df: pd.DataFrame, min_games: int = 5) -> pd.DataFrame:
    if events_df.empty:
        return pd.DataFrame()

    my_events = events_df[(events_df["is_my_event"]) & (events_df["event_type"].isin(RELEVANT_CARD_EVENT_TYPES))]
    my_events = my_events[my_events["card_name"].notna()]
    if my_events.empty:
        return pd.DataFrame()

    game_card_presence = my_events[["csv_row_id", "card_name"]].drop_duplicates()

    total_games = games_df["csv_row_id"].nunique()
    total_losses = games_df["is_loss"].sum()
    baseline_loss_rate = total_losses / total_games if total_games else 0.0

    merged = game_card_presence.merge(games_df[["csv_row_id", "is_loss"]], on="csv_row_id", how="left")

    grouped = (
        merged.groupby("card_name", as_index=False)
        .agg(games_with_card=("csv_row_id", "nunique"), losses_with_card=("is_loss", "sum"))
    )
    grouped = grouped[grouped["games_with_card"] >= min_games].copy()
    if grouped.empty:
        return grouped

    grouped["loss_rate_with_card"] = grouped["losses_with_card"] / grouped["games_with_card"]
    grouped["loss_lift_vs_baseline"] = grouped["loss_rate_with_card"] - baseline_loss_rate
    grouped["loss_odds_ratio_approx"] = (grouped["loss_rate_with_card"] + 1e-6) / (baseline_loss_rate + 1e-6)
    grouped = grouped.sort_values(["loss_lift_vs_baseline", "games_with_card"], ascending=[False, False])
    return grouped


def event_type_loss_correlation(games_df: pd.DataFrame, events_df: pd.DataFrame, min_games: int = 5) -> pd.DataFrame:
    if events_df.empty:
        return pd.DataFrame()

    my_events = events_df[events_df["is_my_event"]].copy()
    if my_events.empty:
        return pd.DataFrame()

    counts = (
        my_events.groupby(["csv_row_id", "event_type"], as_index=False)
        .size()
        .rename(columns={"size": "event_count"})
    )

    merged = counts.merge(games_df[["csv_row_id", "is_loss"]], on="csv_row_id", how="left")
    grouped = (
        merged.groupby("event_type", as_index=False)
        .agg(
            games_with_event=("csv_row_id", "nunique"),
            avg_count_in_losses=("event_count", lambda s: s[merged.loc[s.index, "is_loss"]].mean()),
            avg_count_in_wins=("event_count", lambda s: s[~merged.loc[s.index, "is_loss"]].mean()),
        )
    )
    grouped = grouped[grouped["games_with_event"] >= min_games].copy()
    grouped["avg_count_in_losses"] = grouped["avg_count_in_losses"].fillna(0.0)
    grouped["avg_count_in_wins"] = grouped["avg_count_in_wins"].fillna(0.0)
    grouped["loss_minus_win_count"] = grouped["avg_count_in_losses"] - grouped["avg_count_in_wins"]
    grouped = grouped.sort_values(["loss_minus_win_count", "games_with_event"], ascending=[False, False])
    return grouped


def opening_hand_loss_correlation(games_df: pd.DataFrame, min_games: int = 5) -> pd.DataFrame:
    if "opening_hand_cards" not in games_df.columns:
        return pd.DataFrame()

    working = games_df[["csv_row_id", "is_loss", "opening_hand_cards"]].copy()
    working["opening_hand_cards"] = working["opening_hand_cards"].fillna("")
    working["opening_hand_cards"] = working["opening_hand_cards"].map(
        lambda x: [card.strip() for card in str(x).split("|") if card.strip()]
    )

    exploded = working.explode("opening_hand_cards")
    exploded = exploded[exploded["opening_hand_cards"].notna()]
    exploded = exploded.rename(columns={"opening_hand_cards": "card_name"})
    exploded = exploded[["csv_row_id", "card_name", "is_loss"]].drop_duplicates()
    if exploded.empty:
        return pd.DataFrame()

    total_games = games_df["csv_row_id"].nunique()
    baseline_loss_rate = games_df["is_loss"].mean() if total_games else 0.0

    grouped = (
        exploded.groupby("card_name", as_index=False)
        .agg(games_with_card=("csv_row_id", "nunique"), losses_with_card=("is_loss", "sum"))
    )
    grouped = grouped[grouped["games_with_card"] >= min_games].copy()
    grouped["loss_rate_with_card"] = grouped["losses_with_card"] / grouped["games_with_card"]
    grouped["loss_lift_vs_baseline"] = grouped["loss_rate_with_card"] - baseline_loss_rate
    grouped = grouped.sort_values(["loss_lift_vs_baseline", "games_with_card"], ascending=[False, False])
    return grouped


# ---------------------------------------------------------------------------
# Matchup analysis
# ---------------------------------------------------------------------------


def matchup_win_rates(games_df: pd.DataFrame, min_games: int = 5) -> pd.DataFrame:
    """Win rate per (my archetype, opponent colors) pair."""
    required = {"archetype", "opponent_colors", "is_loss", "csv_row_id"}
    if not required.issubset(games_df.columns):
        return pd.DataFrame()

    grouped = (
        games_df.groupby(["archetype", "opponent_colors"], as_index=False)
        .agg(games=("csv_row_id", "count"), wins=("is_loss", lambda s: (~s).sum()), losses=("is_loss", "sum"))
    )
    grouped = grouped[grouped["games"] >= min_games].copy()
    grouped["win_rate"] = grouped["wins"] / grouped["games"]
    return grouped.sort_values(["archetype", "win_rate"], ascending=[True, True])


# ---------------------------------------------------------------------------
# Sequencing / tempo analysis
# ---------------------------------------------------------------------------


def _my_action_events(events_df: pd.DataFrame) -> pd.DataFrame:
    return events_df[
        events_df["is_my_event"]
        & events_df["event_type"].isin(
            {"CARD_PLAYED", "CARD_INKED", "CARD_QUEST", "CARD_ATTACK", "ABILITY_ACTIVATED"}
        )
    ].copy()


def tempo_curves(games_df: pd.DataFrame, events_df: pd.DataFrame, max_turn: int = 12) -> pd.DataFrame:
    """Average action counts per turn for wins vs losses, capped at max_turn."""
    if events_df.empty:
        return pd.DataFrame()

    actions = _my_action_events(events_df)
    actions = actions[actions["turn_number"].notna() & (actions["turn_number"] <= max_turn)]

    counts = (
        actions.groupby(["csv_row_id", "turn_number", "event_type"])
        .size()
        .reset_index(name="count")
    )
    counts = counts.merge(games_df[["csv_row_id", "is_loss"]], on="csv_row_id", how="left")

    rows = []
    for event_type in ("CARD_PLAYED", "CARD_QUEST", "CARD_ATTACK", "CARD_INKED"):
        subset = counts[counts["event_type"] == event_type]
        for is_loss_val, label in ((True, "Loss"), (False, "Win")):
            group = subset[subset["is_loss"] == is_loss_val]
            if group.empty:
                continue
            avg_per_turn = group.groupby("turn_number")["count"].mean().reset_index()
            avg_per_turn.columns = ["turn_number", "avg_count"]
            avg_per_turn["event_type"] = event_type
            avg_per_turn["outcome"] = label
            rows.append(avg_per_turn)

    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def early_ink_rates(games_df: pd.DataFrame, events_df: pd.DataFrame, turns: int = 4) -> pd.DataFrame:
    """Rate at which the player inked on each of the first N turns, split by win/loss."""
    if events_df.empty:
        return pd.DataFrame()

    ink_events = events_df[
        events_df["is_my_event"] & (events_df["event_type"] == "CARD_INKED")
    ].copy()
    ink_events = ink_events[ink_events["turn_number"].notna() & (ink_events["turn_number"] <= turns)]

    inked_per_turn = (
        ink_events.groupby(["csv_row_id", "turn_number"])
        .size()
        .reset_index(name="inked")
    )
    inked_per_turn["inked"] = True

    all_games = games_df[["csv_row_id", "is_loss"]].copy()
    rows = []
    for turn in range(1, turns + 1):
        turn_subset = inked_per_turn[inked_per_turn["turn_number"] == turn][["csv_row_id", "inked"]]
        merged = all_games.merge(turn_subset, on="csv_row_id", how="left")
        merged["inked"] = merged["inked"].fillna(False)
        for is_loss_val, label in ((True, "Loss"), (False, "Win")):
            group = merged[merged["is_loss"] == is_loss_val]
            if group.empty:
                continue
            rows.append(
                {
                    "turn": turn,
                    "outcome": label,
                    "games": len(group),
                    "inked_count": int(group["inked"].sum()),
                    "ink_rate": group["inked"].mean(),
                }
            )

    return pd.DataFrame(rows)


def first_quest_turn(games_df: pd.DataFrame, events_df: pd.DataFrame) -> pd.DataFrame:
    """Distribution of the first turn a quest was played, split by win/loss."""
    if events_df.empty:
        return pd.DataFrame()

    quest_events = events_df[
        events_df["is_my_event"] & (events_df["event_type"] == "CARD_QUEST") & events_df["turn_number"].notna()
    ]
    first_quest = quest_events.groupby("csv_row_id")["turn_number"].min().reset_index()
    first_quest.columns = ["csv_row_id", "first_quest_turn"]

    merged = games_df[["csv_row_id", "is_loss"]].merge(first_quest, on="csv_row_id", how="left")
    merged["first_quest_turn"] = merged["first_quest_turn"].fillna(-1).astype(int)

    rows = []
    for is_loss_val, label in ((True, "Loss"), (False, "Win")):
        group = merged[merged["is_loss"] == is_loss_val].copy()
        dist = group["first_quest_turn"].value_counts().sort_index().reset_index()
        dist.columns = ["first_quest_turn", "game_count"]
        dist["outcome"] = label
        dist["pct"] = dist["game_count"] / len(group)
        rows.append(dist)

    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


# ---------------------------------------------------------------------------
# Opponent card analysis
# ---------------------------------------------------------------------------


def opponent_card_loss_correlation(
    games_df: pd.DataFrame, events_df: pd.DataFrame, min_games: int = 5
) -> pd.DataFrame:
    """Cards opponent played most often in games you lost."""
    if events_df.empty:
        return pd.DataFrame()

    opp_events = events_df[
        (~events_df["is_my_event"]) & (events_df["event_type"].isin(RELEVANT_CARD_EVENT_TYPES))
    ]
    opp_events = opp_events[opp_events["card_name"].notna()]
    if opp_events.empty:
        return pd.DataFrame()

    game_card_presence = opp_events[["csv_row_id", "card_name"]].drop_duplicates()

    total_games = games_df["csv_row_id"].nunique()
    baseline_loss_rate = games_df["is_loss"].mean() if total_games else 0.0

    merged = game_card_presence.merge(games_df[["csv_row_id", "is_loss"]], on="csv_row_id", how="left")
    grouped = (
        merged.groupby("card_name", as_index=False)
        .agg(games_with_card=("csv_row_id", "nunique"), losses_with_card=("is_loss", "sum"))
    )
    grouped = grouped[grouped["games_with_card"] >= min_games].copy()
    if grouped.empty:
        return grouped

    grouped["loss_rate_with_card"] = grouped["losses_with_card"] / grouped["games_with_card"]
    grouped["loss_lift_vs_baseline"] = grouped["loss_rate_with_card"] - baseline_loss_rate
    grouped["loss_odds_ratio_approx"] = (grouped["loss_rate_with_card"] + 1e-6) / (baseline_loss_rate + 1e-6)
    return grouped.sort_values(["loss_lift_vs_baseline", "games_with_card"], ascending=[False, False])


# ---------------------------------------------------------------------------
# Card pair / synergy analysis
# ---------------------------------------------------------------------------


def card_pair_correlation(
    games_df: pd.DataFrame, events_df: pd.DataFrame, min_games: int = 5
) -> pd.DataFrame:
    """Win rate uplift for pairs of cards played together in the same game."""
    if events_df.empty:
        return pd.DataFrame()

    my_cards = events_df[
        events_df["is_my_event"] & (events_df["event_type"] == "CARD_PLAYED") & events_df["card_name"].notna()
    ][["csv_row_id", "card_name"]].drop_duplicates()

    if my_cards.empty:
        return pd.DataFrame()

    baseline_win_rate = 1.0 - games_df["is_loss"].mean() if len(games_df) else 0.0

    pairs = my_cards.merge(my_cards, on="csv_row_id", suffixes=("_a", "_b"))
    pairs = pairs[pairs["card_name_a"] < pairs["card_name_b"]]

    if pairs.empty:
        return pd.DataFrame()

    pairs = pairs.merge(games_df[["csv_row_id", "is_loss"]], on="csv_row_id", how="left")
    grouped = (
        pairs.groupby(["card_name_a", "card_name_b"], as_index=False)
        .agg(games=("csv_row_id", "nunique"), wins=("is_loss", lambda s: (~s).sum()))
    )
    grouped = grouped[grouped["games"] >= min_games].copy()
    grouped["win_rate"] = grouped["wins"] / grouped["games"]
    grouped["win_lift_vs_baseline"] = grouped["win_rate"] - baseline_win_rate
    return grouped.sort_values(["win_lift_vs_baseline", "games"], ascending=[False, False])


# ---------------------------------------------------------------------------
# Lore curve analysis
# ---------------------------------------------------------------------------


def lore_curves(games_df: pd.DataFrame, events_df: pd.DataFrame, max_turn: int = 15) -> pd.DataFrame:
    """Per-turn and cumulative lore gained, split by win/loss."""
    if events_df.empty:
        return pd.DataFrame()

    lore_events = events_df[
        events_df["is_my_event"]
        & (events_df["event_type"] == "LORE_GAINED")
        & events_df["turn_number"].notna()
        & (events_df["turn_number"] <= max_turn)
    ].copy()

    if lore_events.empty:
        return pd.DataFrame()

    if "lore_amount" not in lore_events.columns:
        lore_events["lore_amount"] = 1

    per_turn = (
        lore_events.groupby(["csv_row_id", "turn_number"], as_index=False)["lore_amount"].sum()
    )
    per_turn = per_turn.merge(games_df[["csv_row_id", "is_loss"]], on="csv_row_id", how="left")

    rows = []
    for is_loss_val, label in ((True, "Loss"), (False, "Win")):
        subset = per_turn[per_turn["is_loss"] == is_loss_val]
        avg = subset.groupby("turn_number")["lore_amount"].mean().reset_index()
        avg.columns = ["turn_number", "avg_lore_per_turn"]
        avg["outcome"] = label
        avg = avg.sort_values("turn_number")
        avg["cumulative_avg_lore"] = avg["avg_lore_per_turn"].cumsum()
        rows.append(avg)

    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


# ---------------------------------------------------------------------------
# Mulligan impact
# ---------------------------------------------------------------------------


def mulligan_impact(games_df: pd.DataFrame) -> pd.DataFrame:
    """Win rate by number of mulligans taken, split by turn order."""
    if "mulligan_count_my" not in games_df.columns:
        return pd.DataFrame()

    df = games_df[games_df["mulligan_count_my"].notna()].copy()
    df["mulligan_count_my"] = df["mulligan_count_my"].astype(int)

    rows = []
    for n_mull, group in df.groupby("mulligan_count_my"):
        for turn_order in ("All", "OTP", "OTD"):
            subset = group if turn_order == "All" else group[group["turn_order"] == turn_order]
            if subset.empty:
                continue
            rows.append(
                {
                    "mulligans": int(n_mull),
                    "turn_order": turn_order,
                    "games": len(subset),
                    "wins": int((~subset["is_loss"]).sum()),
                    "losses": int(subset["is_loss"].sum()),
                    "win_rate": 1.0 - subset["is_loss"].mean(),
                }
            )

    return pd.DataFrame(rows).sort_values(["turn_order", "mulligans"])


# ---------------------------------------------------------------------------
# Seen vs Played breakdown
# ---------------------------------------------------------------------------

_SEEN_EVENT_TYPES = {"INITIAL_HAND", "CARD_DRAWN"}
_INKED_EVENT_TYPES = {"CARD_INKED", "CARD_PUT_INTO_INKWELL"}


def card_seen_vs_played_correlation(
    games_df: pd.DataFrame, events_df: pd.DataFrame, min_games: int = 5
) -> pd.DataFrame:
    """
    For each card, compare loss rates across three interactions:
      - Drawn/Seen  : card appeared in INITIAL_HAND or CARD_DRAWN
      - Played      : card was played as a character/action (CARD_PLAYED)
      - Inked       : card was used as ink (CARD_INKED / CARD_PUT_INTO_INKWELL)

    Also computes 'play_through_rate' = games_played / games_drawn so you can
    see cards you draw but rarely actually use.
    """
    if events_df.empty:
        return pd.DataFrame()

    baseline_loss_rate = games_df["is_loss"].mean() if len(games_df) else 0.0
    game_outcome = games_df[["csv_row_id", "is_loss"]]

    def _loss_rate_for(ev_types: set) -> pd.DataFrame:
        sub = events_df[events_df["event_type"].isin(ev_types) & events_df["card_name"].notna()]
        presence = sub[["csv_row_id", "card_name"]].drop_duplicates()
        merged = presence.merge(game_outcome, on="csv_row_id", how="left")
        return (
            merged.groupby("card_name", as_index=False)
            .agg(games=("csv_row_id", "nunique"), losses=("is_loss", "sum"))
        )

    drawn_df = _loss_rate_for(_SEEN_EVENT_TYPES).rename(
        columns={"games": "games_drawn", "losses": "losses_drawn"}
    )
    played_df = _loss_rate_for({"CARD_PLAYED"}).rename(
        columns={"games": "games_played", "losses": "losses_played"}
    )
    inked_df = _loss_rate_for(_INKED_EVENT_TYPES).rename(
        columns={"games": "games_inked", "losses": "losses_inked"}
    )

    result = (
        drawn_df.merge(played_df, on="card_name", how="outer")
        .merge(inked_df, on="card_name", how="outer")
    ).fillna(0)

    result[["games_drawn", "games_played", "games_inked"]] = (
        result[["games_drawn", "games_played", "games_inked"]].astype(int)
    )
    result[["losses_drawn", "losses_played", "losses_inked"]] = (
        result[["losses_drawn", "losses_played", "losses_inked"]].astype(int)
    )

    result = result[result["games_drawn"] >= min_games].copy()
    if result.empty:
        return result

    result["loss_rate_drawn"] = result["losses_drawn"] / result["games_drawn"].replace(0, float("nan"))
    result["loss_rate_played"] = result["losses_played"] / result["games_played"].replace(0, float("nan"))
    result["loss_rate_inked"] = result["losses_inked"] / result["games_inked"].replace(0, float("nan"))
    result["play_through_rate"] = result["games_played"] / result["games_drawn"].replace(0, float("nan"))
    result["loss_lift_when_played"] = result["loss_rate_played"] - baseline_loss_rate

    return result.sort_values("loss_lift_when_played", ascending=False, na_position="last")
