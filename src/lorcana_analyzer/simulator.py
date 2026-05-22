"""
Lorcana deck simulator and stress-tester.

Uses a two-layer approach:
  1. Heuristic Monte Carlo  — fast, interpretable, always available.
     Draws random opening hands from the deck, weights each hand by the
     observed card-level loss correlations from your game history, and
     samples win/loss outcomes stochastically.

  2. Logistic regression model (numpy-only, no sklearn required) trained on
     your actual game data.  Features are aggregated per-game statistics
     (avg deck loss lift, opening-hand quality, turn order, mulligan count)
     so the model is well-conditioned even with ~200 samples.

All estimates are directional indicators, not precise predictions.
The models are only as good as the correlation data they are built from.
"""

from __future__ import annotations

import ast
import math
import random
from typing import Optional

import numpy as np
import pandas as pd


# ──────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ──────────────────────────────────────────────────────────────────────────────

def _deck_to_list(decklist: dict[str, int]) -> list[str]:
    """Expand {card: qty} into a full 60-card list."""
    cards: list[str] = []
    for name, qty in decklist.items():
        cards.extend([name] * int(qty))
    return cards


def _draw_hand(deck_list: list[str], size: int = 7) -> list[str]:
    if len(deck_list) <= size:
        return list(deck_list)
    return random.sample(deck_list, size)


def _build_corr_lookup(
    card_corr_df: pd.DataFrame,
    seen_played_df: pd.DataFrame,
    baseline_loss: float,
) -> dict[str, float]:
    """Build a lowercase card→loss_lift lookup, preferring played-rate data."""
    lookup: dict[str, float] = {}
    if not card_corr_df.empty and "card_name" in card_corr_df.columns:
        for _, row in card_corr_df.iterrows():
            lookup[row["card_name"].lower()] = float(row.get("loss_lift_vs_baseline", 0.0))

    # Override with seen/played data: use a blend of drawn and played loss rates
    if not seen_played_df.empty and "card_name" in seen_played_df.columns:
        for _, row in seen_played_df.iterrows():
            k = row["card_name"].lower()
            lr_played = row.get("loss_rate_played", None)
            lr_drawn = row.get("loss_rate_drawn", None)
            if pd.notna(lr_played) and pd.notna(lr_drawn):
                blended = 0.6 * float(lr_played) + 0.4 * float(lr_drawn)
                lookup[k] = blended - baseline_loss
            elif pd.notna(lr_played):
                lookup[k] = float(lr_played) - baseline_loss

    return lookup


def _parse_decklist_map(raw: str) -> dict[str, int]:
    """Safely parse the decklist_map string stored in clean_games.csv."""
    if not isinstance(raw, str) or not raw.strip():
        return {}
    try:
        return ast.literal_eval(raw)
    except Exception:
        return {}


# ──────────────────────────────────────────────────────────────────────────────
# 1. Monte Carlo simulation
# ──────────────────────────────────────────────────────────────────────────────

def simulate_deck(
    decklist: dict[str, int],
    games_df: pd.DataFrame,
    card_corr_df: pd.DataFrame,
    seen_played_df: pd.DataFrame,
    n_sims: int = 2000,
    seed: int = 42,
) -> dict:
    """
    Monte Carlo win-rate estimator for a given decklist.

    Returns a dict with:
        win_rate            — projected win rate (0–1)
        ci_low / ci_high    — 95% confidence interval
        baseline_win_rate   — your historical baseline for comparison
        n_sims              — simulations run
        per_card_df         — per-card impact DataFrame
    """
    random.seed(seed)
    baseline_loss = float(games_df["is_loss"].mean()) if not games_df.empty else 0.5
    lookup = _build_corr_lookup(card_corr_df, seen_played_df, baseline_loss)

    deck_list = _deck_to_list(decklist)
    total_cards = len(deck_list)

    wins = 0
    for _ in range(n_sims):
        hand = _draw_hand(deck_list, 7)
        lifts = [lookup.get(c.lower(), 0.0) for c in hand]
        avg_lift = sum(lifts) / max(len(lifts), 1)
        adj_loss = max(0.02, min(0.98, baseline_loss + avg_lift * 0.8))
        if random.random() >= adj_loss:
            wins += 1

    win_rate = wins / n_sims
    se = math.sqrt(win_rate * (1 - win_rate) / n_sims)
    ci_low = max(0.0, win_rate - 1.96 * se)
    ci_high = min(1.0, win_rate + 1.96 * se)

    per_card_rows = []
    for name, qty in decklist.items():
        lift = lookup.get(name.lower(), 0.0)
        weight = qty / max(total_cards, 1)
        per_card_rows.append(
            {
                "card_name": name,
                "qty": qty,
                "loss_lift": lift,
                "deck_weight": round(weight, 3),
                "projected_wr_drag": round(-lift * weight, 4),  # positive = hurts win rate
            }
        )

    per_card_df = pd.DataFrame(per_card_rows).sort_values(
        "projected_wr_drag", ascending=False
    )

    return {
        "win_rate": win_rate,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "baseline_win_rate": 1.0 - baseline_loss,
        "n_sims": n_sims,
        "per_card_df": per_card_df,
    }


def compare_decklists(
    decklist_a: dict[str, int],
    decklist_b: dict[str, int],
    games_df: pd.DataFrame,
    card_corr_df: pd.DataFrame,
    seen_played_df: pd.DataFrame,
    n_sims: int = 2000,
) -> tuple[pd.DataFrame, dict, dict]:
    """Run simulation for both decks and return a comparison table + raw results."""
    result_a = simulate_deck(decklist_a, games_df, card_corr_df, seen_played_df, n_sims, seed=42)
    result_b = simulate_deck(decklist_b, games_df, card_corr_df, seen_played_df, n_sims, seed=42)

    summary = pd.DataFrame(
        [
            {
                "Deck": "Deck A",
                "Projected Win Rate": result_a["win_rate"],
                "95% CI": f"{result_a['ci_low']:.1%} – {result_a['ci_high']:.1%}",
                "vs Baseline": result_a["win_rate"] - result_a["baseline_win_rate"],
            },
            {
                "Deck": "Deck B",
                "Projected Win Rate": result_b["win_rate"],
                "95% CI": f"{result_b['ci_low']:.1%} – {result_b['ci_high']:.1%}",
                "vs Baseline": result_b["win_rate"] - result_b["baseline_win_rate"],
            },
        ]
    )
    return summary, result_a, result_b


# ──────────────────────────────────────────────────────────────────────────────
# 2. Opponent stress test
# ──────────────────────────────────────────────────────────────────────────────

def stress_test_deck(
    decklist: dict[str, int],
    games_df: pd.DataFrame,
    events_df: pd.DataFrame,
    card_corr_df: pd.DataFrame,
    pair_df: pd.DataFrame,
    n_sims_per_game: int = 300,
    max_early_turn: int = 6,
    seed: int = 42,
) -> pd.DataFrame:
    """
    For each historical loss, simulate how the given deck would perform
    against that specific opponent sequence.

    Uses:
      - Card correlation to score your opening hand quality
      - Pair win-lift to reward deck cards that historically beat the
        opponent's cards

    Returns one row per historical loss, sorted by projected win probability.
    """
    random.seed(seed)
    losses_df = games_df[games_df["is_loss"] == True].copy()
    if losses_df.empty or events_df.empty:
        return pd.DataFrame()

    baseline_loss = float(games_df["is_loss"].mean())
    lookup = _build_corr_lookup(card_corr_df, pd.DataFrame(), baseline_loss)

    # Pair lookup: (my_card_lower, opp_card_lower) → win_lift
    pair_lookup: dict[tuple[str, str], float] = {}
    if not pair_df.empty:
        for _, row in pair_df.iterrows():
            a = str(row.get("card_name_a", "")).lower()
            b = str(row.get("card_name_b", "")).lower()
            lift = float(row.get("win_lift_vs_baseline", 0.0))
            pair_lookup[(a, b)] = lift
            pair_lookup[(b, a)] = lift

    deck_list = _deck_to_list(decklist)
    deck_lower = {n.lower() for n in decklist}

    rows = []
    for _, game in losses_df.iterrows():
        rid = game["csv_row_id"]

        opp_plays = events_df[
            (events_df["csv_row_id"] == rid)
            & (~events_df["is_my_event"])
            & (events_df["event_type"] == "CARD_PLAYED")
            & events_df["card_name"].notna()
            & events_df["turn_number"].notna()
            & (events_df["turn_number"] <= max_early_turn)
        ]["card_name"].dropna().tolist()

        if not opp_plays:
            continue

        opp_lower = [c.lower() for c in opp_plays]

        # Simulate n_sims_per_game hands vs this opponent sequence
        sim_wins = 0
        for _ in range(n_sims_per_game):
            hand = _draw_hand(deck_list, 7)
            hand_lower = [c.lower() for c in hand]

            # Hand quality from card correlations
            hand_lift = sum(lookup.get(c, 0.0) for c in hand_lower) / max(len(hand_lower), 1)

            # Counter bonus: how many of our hand cards have good pair synergy
            # against the opponent's sequence (positive win_lift = good matchup)
            counter_bonus = 0.0
            for my_c in hand_lower:
                for opp_c in opp_lower:
                    bonus = pair_lookup.get((my_c, opp_c), 0.0)
                    if bonus > 0:
                        counter_bonus += bonus * 0.05  # scale down to avoid dominating

            adj_loss = max(0.02, min(0.98, baseline_loss + hand_lift * 0.8 - counter_bonus))
            if random.random() >= adj_loss:
                sim_wins += 1

        win_prob = sim_wins / n_sims_per_game
        rows.append(
            {
                "csv_row_id": int(rid),
                "opponent_name": str(game.get("opponent_name", "Unknown")),
                "opponent_colors": str(game.get("opponent_colors", "?")),
                "turn_order": str(game.get("turn_order", "?")),
                "opp_early_plays": ", ".join(opp_plays[:5]),
                "n_opp_cards": len(opp_plays),
                "projected_win_prob": round(win_prob, 3),
            }
        )

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows).sort_values("projected_win_prob", ascending=True)


# ──────────────────────────────────────────────────────────────────────────────
# 3. Logistic regression model (numpy-only)
# ──────────────────────────────────────────────────────────────────────────────

def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def _train_logistic(
    X: np.ndarray,
    y: np.ndarray,
    lr: float = 0.05,
    epochs: int = 500,
    l2: float = 0.1,
) -> np.ndarray:
    """Simple logistic regression with L2 regularisation, gradient descent."""
    n, d = X.shape
    w = np.zeros(d)
    for _ in range(epochs):
        pred = _sigmoid(X @ w)
        grad = X.T @ (pred - y) / n + l2 * w
        w -= lr * grad
    return w


def build_game_features(
    games_df: pd.DataFrame,
    events_df: pd.DataFrame,
    card_corr_df: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Build a feature matrix from game history for logistic regression.

    Features per game:
      - otp              (1 = on the play, 0 = on the draw)
      - mulligan_count   (normalised by 7)
      - deck_avg_lift    (average loss_lift of all cards in decklist)
      - deck_max_lift    (worst single card in deck)
      - deck_n_risky     (fraction of deck slots with loss_lift > 0.08)
      - opener_avg_lift  (average loss_lift of opening hand cards)
      - opener_max_lift  (worst single card in opening hand)
      - n_played_t1_3    (cards played on turns 1-3 normalised)

    Returns X (n_games × n_features), y (n_games,), feature_names
    """
    if card_corr_df.empty or games_df.empty:
        return np.empty((0, 0)), np.array([]), []

    corr_lookup: dict[str, float] = {}
    for _, row in card_corr_df.iterrows():
        corr_lookup[row["card_name"].lower()] = float(row.get("loss_lift_vs_baseline", 0.0))

    feature_names = [
        "otp",
        "mulligan_count",
        "deck_avg_lift",
        "deck_max_lift",
        "deck_n_risky",
        "opener_avg_lift",
        "opener_max_lift",
        "n_played_t1_3",
    ]

    # Pre-build opening hand and early play data from events
    oh_lifts: dict[int, list[float]] = {}
    early_play_counts: dict[int, int] = {}
    if not events_df.empty:
        oh_events = events_df[
            events_df["is_my_event"]
            & (events_df["event_type"] == "INITIAL_HAND")
            & events_df["card_name"].notna()
        ]
        for _, ev in oh_events.iterrows():
            rid = int(ev["csv_row_id"])
            lift = corr_lookup.get(str(ev["card_name"]).lower(), 0.0)
            oh_lifts.setdefault(rid, []).append(lift)

        early_plays = events_df[
            events_df["is_my_event"]
            & (events_df["event_type"] == "CARD_PLAYED")
            & events_df["turn_number"].notna()
            & (events_df["turn_number"] <= 3)
        ]
        for _, ev in early_plays.iterrows():
            rid = int(ev["csv_row_id"])
            early_play_counts[rid] = early_play_counts.get(rid, 0) + 1

    X_rows, y_rows = [], []
    for _, row in games_df.iterrows():
        rid = int(row["csv_row_id"])

        # Turn order
        otp = 1.0 if str(row.get("turn_order", "")).upper() == "OTP" else 0.0

        # Mulligan
        mulligan = float(row.get("mulligan_count_my", 3)) / 7.0

        # Deck features
        deck_map = _parse_decklist_map(str(row.get("decklist_map", "")))
        if not deck_map:
            continue
        deck_lifts = [
            corr_lookup.get(name.lower(), 0.0) * qty
            for name, qty in deck_map.items()
        ]
        total_deck = sum(deck_map.values())
        deck_avg = sum(deck_lifts) / max(total_deck, 1)
        deck_max = max(
            (corr_lookup.get(name.lower(), 0.0) for name in deck_map), default=0.0
        )
        deck_risky = sum(
            qty for name, qty in deck_map.items()
            if corr_lookup.get(name.lower(), 0.0) > 0.08
        ) / max(total_deck, 1)

        # Opening hand features
        oh = oh_lifts.get(rid, [])
        opener_avg = float(np.mean(oh)) if oh else 0.0
        opener_max = float(max(oh)) if oh else 0.0

        # Early play count (normalised)
        n_early = early_play_counts.get(rid, 0) / 3.0

        X_rows.append([otp, mulligan, deck_avg, deck_max, deck_risky, opener_avg, opener_max, n_early])
        y_rows.append(1.0 if row["is_loss"] else 0.0)

    if not X_rows:
        return np.empty((0, 0)), np.array([]), feature_names

    X = np.array(X_rows, dtype=float)
    y = np.array(y_rows, dtype=float)

    # Replace any NaN/inf with column medians
    for col in range(X.shape[1]):
        col_data = X[:, col]
        bad = ~np.isfinite(col_data)
        if bad.any():
            median = np.nanmedian(col_data)
            X[bad, col] = median if np.isfinite(median) else 0.0

    return X, y, feature_names


def train_loss_model(
    games_df: pd.DataFrame,
    events_df: pd.DataFrame,
    card_corr_df: pd.DataFrame,
) -> dict | None:
    """
    Train a logistic regression model predicting game loss probability.

    Returns dict with:
        weights         — np.ndarray of model coefficients
        feature_names   — list of feature names
        feature_means   — normalisation means
        feature_stds    — normalisation stds
        train_accuracy  — accuracy on training data
        feature_importance_df — DataFrame of feature importances
    Or None if not enough data.
    """
    X, y, feat_names = build_game_features(games_df, events_df, card_corr_df)
    if X.shape[0] < 30:
        return None

    means = X.mean(axis=0)
    stds = X.std(axis=0)
    stds[stds == 0] = 1.0
    X_norm = (X - means) / stds

    w = _train_logistic(X_norm, y, lr=0.05, epochs=800, l2=0.15)

    preds = (_sigmoid(X_norm @ w) >= 0.5).astype(float)
    accuracy = float((preds == y).mean())

    importance_df = pd.DataFrame(
        {"Feature": feat_names, "Coefficient": w, "Abs Importance": np.abs(w)}
    ).sort_values("Abs Importance", ascending=False)

    return {
        "weights": w,
        "feature_names": feat_names,
        "feature_means": means,
        "feature_stds": stds,
        "train_accuracy": accuracy,
        "n_games": int(X_norm.shape[0]),
        "feature_importance_df": importance_df,
    }


def predict_deck_loss_prob(
    decklist: dict[str, int],
    model: dict,
    card_corr_df: pd.DataFrame,
    turn_order: str = "OTP",
    mulligan_count: float = 3.0,
) -> float:
    """
    Use the trained logistic model to estimate loss probability for a deck.
    Returns loss probability (lower = better deck).
    """
    corr_lookup: dict[str, float] = {}
    if not card_corr_df.empty and "card_name" in card_corr_df.columns:
        for _, row in card_corr_df.iterrows():
            corr_lookup[row["card_name"].lower()] = float(row.get("loss_lift_vs_baseline", 0.0))

    deck_lifts = [
        corr_lookup.get(name.lower(), 0.0) * qty for name, qty in decklist.items()
    ]
    total = sum(decklist.values())
    deck_avg = sum(deck_lifts) / max(total, 1)
    deck_max = max((corr_lookup.get(name.lower(), 0.0) for name in decklist), default=0.0)
    deck_risky = sum(
        qty for name, qty in decklist.items()
        if corr_lookup.get(name.lower(), 0.0) > 0.08
    ) / max(total, 1)

    otp = 1.0 if turn_order.upper() == "OTP" else 0.0
    mull = mulligan_count / 7.0

    x = np.array([otp, mull, deck_avg, deck_max, deck_risky, deck_avg, deck_max, 1.0])
    x_norm = (x - model["feature_means"]) / model["feature_stds"]
    return float(_sigmoid(np.dot(model["weights"], x_norm)))
