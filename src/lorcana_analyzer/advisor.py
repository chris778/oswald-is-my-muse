from __future__ import annotations

from pathlib import Path

import pandas as pd


def _sample_score(games: int) -> float:
    return max(0.0, min(1.0, games / 80.0))


def _effect_score(effect: float, cap: float = 12.0) -> float:
    return max(0.0, min(1.0, abs(effect) / cap))


def _priority(effect: float, games: int) -> float:
    # Weight effect size slightly more than sample size for coaching value.
    return round(100.0 * (0.6 * _effect_score(effect) + 0.4 * _sample_score(games)), 1)


def _add_action(
    rows: list[dict],
    *,
    category: str,
    action: str,
    rationale: str,
    signal_score: float,
    games: int,
    confidence: str,
) -> None:
    rows.append(
        {
            "category": category,
            "action": action,
            "rationale": rationale,
            "signal_score": round(signal_score, 2),
            "games": int(games),
            "confidence": confidence,
            "priority": _priority(signal_score, games),
        }
    )


def build_recommendations(
    games_df: pd.DataFrame,
    card_df: pd.DataFrame,
    opening_df: pd.DataFrame,
    event_type_df: pd.DataFrame,
    ink_df: pd.DataFrame,
    fq_df: pd.DataFrame,
    matchup_df: pd.DataFrame,
    seen_played_df: pd.DataFrame,
    min_card_games: int = 15,
) -> pd.DataFrame:
    rows: list[dict] = []

    if games_df.empty:
        return pd.DataFrame(columns=["category", "action", "rationale", "signal_score", "games", "confidence", "priority"])

    total_games = len(games_df)

    # 1) Turn order planning
    if "turn_order" in games_df.columns and "is_loss" in games_df.columns:
        otp = games_df[games_df["turn_order"] == "OTP"]
        otd = games_df[games_df["turn_order"] == "OTD"]
        if not otp.empty and not otd.empty:
            otp_wr = 1.0 - otp["is_loss"].mean()
            otd_wr = 1.0 - otd["is_loss"].mean()
            gap_pp = (otp_wr - otd_wr) * 100.0
            if abs(gap_pp) >= 3.0:
                if gap_pp > 0:
                    _add_action(
                        rows,
                        category="Turn Order",
                        action="Adopt an OTD-specific mulligan plan and lower your OTD curve by 2-4 cards.",
                        rationale=f"OTP outperforms OTD by {gap_pp:.1f} percentage points.",
                        signal_score=gap_pp,
                        games=min(len(otp), len(otd)),
                        confidence="High" if min(len(otp), len(otd)) >= 100 else "Medium",
                    )
                else:
                    _add_action(
                        rows,
                        category="Turn Order",
                        action="On OTP, keep higher-pressure openers and avoid passive keeps.",
                        rationale=f"OTD outperforms OTP by {abs(gap_pp):.1f} percentage points.",
                        signal_score=gap_pp,
                        games=min(len(otp), len(otd)),
                        confidence="High" if min(len(otp), len(otd)) >= 100 else "Medium",
                    )

    # 2) Early ink discipline (turns 1-4)
    if not ink_df.empty and {"turn", "outcome", "ink_rate", "games"}.issubset(ink_df.columns):
        turns = ink_df[ink_df["turn"].between(1, 4)]
        if not turns.empty:
            p = turns.pivot_table(index="turn", columns="outcome", values="ink_rate", aggfunc="mean")
            g = turns.groupby("outcome")["games"].max()
            if {"Win", "Loss"}.issubset(p.columns):
                avg_diff = float((p["Win"] - p["Loss"]).mean())
                diff_pp = avg_diff * 100.0
                if abs(diff_pp) >= 2.0:
                    action = (
                        "Increase early ink consistency with more flexible inkables and cleaner turn-1 to turn-3 keeps."
                        if diff_pp > 0
                        else "Stop auto-inking every turn; preserve key cards in matchups where tempo fights matter."
                    )
                    _add_action(
                        rows,
                        category="Sequencing",
                        action=action,
                        rationale=f"Average turn 1-4 ink rate differs by {diff_pp:.1f} points in wins vs losses.",
                        signal_score=diff_pp,
                        games=int(min(g.get("Win", 0), g.get("Loss", 0))),
                        confidence="Medium",
                    )

    # 3) First quest timing
    if not fq_df.empty and {"first_quest_turn", "game_count", "outcome", "pct"}.issubset(fq_df.columns):
        valid = fq_df[fq_df["first_quest_turn"] > 0].copy()
        if not valid.empty:
            means = {}
            counts = {}
            for outcome in ("Win", "Loss"):
                sub = valid[valid["outcome"] == outcome]
                if sub.empty:
                    continue
                means[outcome] = float((sub["first_quest_turn"] * sub["pct"]).sum() / sub["pct"].sum())
                counts[outcome] = int(sub["game_count"].sum())
            if {"Win", "Loss"}.issubset(means):
                delta = means["Loss"] - means["Win"]
                if delta >= 0.35:
                    _add_action(
                        rows,
                        category="Sequencing",
                        action="Prioritize establishing an early quest line by turn 2-3 before taking low-value challenges.",
                        rationale=f"First quest happens {delta:.2f} turns later in losses than wins.",
                        signal_score=delta * 5.0,
                        games=min(counts.get("Win", 0), counts.get("Loss", 0)),
                        confidence="High" if min(counts.get("Win", 0), counts.get("Loss", 0)) >= 100 else "Medium",
                    )

    # 4) Event mix signals
    if not event_type_df.empty and {"event_type", "loss_minus_win_count", "games_with_event"}.issubset(event_type_df.columns):
        signals = {
            "CARD_ATTACK": ("Reduce low-value challenges; challenge only to protect high lore pressure.", 0.15),
            "CARD_QUEST": ("Find more lines that keep characters safe to quest repeatedly.", 0.15),
            "CARD_PLAYED": ("Increase early board development and reduce clunky opening hands.", 0.12),
        }
        for event_type, (action, min_abs) in signals.items():
            sub = event_type_df[event_type_df["event_type"] == event_type]
            if sub.empty:
                continue
            row = sub.iloc[0]
            delta = float(row["loss_minus_win_count"])
            games = int(row["games_with_event"])
            if abs(delta) < min_abs or games < 80:
                continue
            if event_type == "CARD_ATTACK" and delta > 0:
                signal_score = delta * 2.5
                rationale = f"Losses average {delta:.2f} more attacks per game than wins."
            elif event_type in {"CARD_QUEST", "CARD_PLAYED"} and delta < 0:
                signal_score = abs(delta) * 2.5
                rationale = f"Wins average {abs(delta):.2f} more {event_type.lower().replace('card_', '')} events per game."
            else:
                continue
            _add_action(
                rows,
                category="Play Pattern",
                action=action,
                rationale=rationale,
                signal_score=signal_score,
                games=games,
                confidence="Medium",
            )

    # 5) Card-level cuts and keeps
    if not card_df.empty and {"card_name", "games_with_card", "loss_lift_vs_baseline"}.issubset(card_df.columns):
        risky = card_df[(card_df["games_with_card"] >= min_card_games) & (card_df["loss_lift_vs_baseline"] >= 0.08)]
        for _, row in risky.sort_values("loss_lift_vs_baseline", ascending=False).head(4).iterrows():
            lift_pp = float(row["loss_lift_vs_baseline"]) * 100.0
            _add_action(
                rows,
                category="Decklist",
                action=f"Run a 10-game experiment cutting 1-2 copies of {row['card_name']}.",
                rationale=f"Card shows +{lift_pp:.1f} loss-rate lift vs baseline.",
                signal_score=lift_pp,
                games=int(row["games_with_card"]),
                confidence="High" if int(row["games_with_card"]) >= 25 else "Medium",
            )

    if not seen_played_df.empty and {"card_name", "games_played", "loss_lift_when_played"}.issubset(seen_played_df.columns):
        keeper_games = max(min_card_games, 20)
        keepers = seen_played_df[(seen_played_df["games_played"] >= keeper_games) & (seen_played_df["loss_lift_when_played"] <= -0.04)]
        for _, row in keepers.sort_values("loss_lift_when_played", ascending=True).head(3).iterrows():
            lift_pp = abs(float(row["loss_lift_when_played"])) * 100.0
            _add_action(
                rows,
                category="Decklist",
                action=f"Treat {row['card_name']} as a core keep and prioritize lines that deploy it on curve.",
                rationale=f"When played, this card improves results by {lift_pp:.1f} points vs baseline loss rate.",
                signal_score=lift_pp,
                games=int(row["games_played"]),
                confidence="High" if int(row["games_played"]) >= 30 else "Medium",
            )

    # 6) Opening hand traps
    if not opening_df.empty and {"card_name", "games_with_card", "loss_lift_vs_baseline"}.issubset(opening_df.columns):
        opener_traps = opening_df[(opening_df["games_with_card"] >= 10) & (opening_df["loss_lift_vs_baseline"] >= 0.10)]
        for _, row in opener_traps.sort_values("loss_lift_vs_baseline", ascending=False).head(4).iterrows():
            lift_pp = float(row["loss_lift_vs_baseline"]) * 100.0
            _add_action(
                rows,
                category="Mulligan",
                action=f"Mulligan {row['card_name']} more aggressively in your opener unless hand context is premium.",
                rationale=f"Opening with this card has +{lift_pp:.1f} loss-rate lift.",
                signal_score=lift_pp,
                games=int(row["games_with_card"]),
                confidence="Medium",
            )

    # 7) Matchup prep
    if not matchup_df.empty and {"archetype", "opponent_colors", "games", "win_rate"}.issubset(matchup_df.columns):
        bad = matchup_df[matchup_df["games"] >= 12].sort_values("win_rate", ascending=True).head(3)
        for _, row in bad.iterrows():
            miss_pp = max(0.0, (0.5 - float(row["win_rate"])) * 100.0)
            if miss_pp < 4.0:
                continue
            _add_action(
                rows,
                category="Matchup",
                action=f"Create a specific plan for {row['archetype']} vs {row['opponent_colors']} and track the next 15 games separately.",
                rationale=f"Current win rate is {float(row['win_rate']) * 100.0:.1f}% across {int(row['games'])} games.",
                signal_score=miss_pp,
                games=int(row["games"]),
                confidence="Medium",
            )

    if not rows:
        return pd.DataFrame(columns=["category", "action", "rationale", "signal_score", "games", "confidence", "priority"])

    out = pd.DataFrame(rows)
    out = out.sort_values(["priority", "games"], ascending=[False, False]).reset_index(drop=True)
    out.insert(0, "rank", range(1, len(out) + 1))
    return out


def render_recommendations_markdown(
    recommendations_df: pd.DataFrame,
    games_df: pd.DataFrame,
    model_info: dict | None = None,
) -> str:
    lines: list[str] = []
    total_games = len(games_df)
    win_rate = (1.0 - games_df["is_loss"].mean()) if total_games else 0.0

    lines.append("# Win Rate Improvement Plan")
    lines.append("")
    lines.append(f"- Games analyzed: {total_games}")
    lines.append(f"- Baseline win rate: {win_rate:.1%}")
    lines.append("")

    if recommendations_df.empty:
        lines.append("## Top Actions")
        lines.append("No high-confidence actions were found with the current thresholds.")
    else:
        lines.append("## Top Actions")
        for _, row in recommendations_df.head(12).iterrows():
            lines.append(
                f"{int(row['rank'])}. [{row['category']}] {row['action']} "
                f"(signal score: {row['signal_score']:+.1f}, games: {int(row['games'])}, confidence: {row['confidence']}, priority: {row['priority']:.1f})"
            )
            lines.append(f"   - Why: {row['rationale']}")

    lines.append("")
    lines.append("## 4-Week Research Loop")
    lines.append("1. Pick the top 2 decklist actions and top 1 sequencing action from this report.")
    lines.append("2. Run two 10-15 game blocks: Block A with current deck, Block B with one targeted change.")
    lines.append("3. Keep mulligan policy fixed during each block to avoid confounding.")
    lines.append("4. Re-run analyze.py after each block and compare only the changed metrics.")
    lines.append("5. Keep changes that improve win rate and also improve at least one process metric (first quest turn, early ink, or opener loss lift).")

    lines.append("")
    lines.append("## Simulation and Modeling")
    lines.append("- Use Simulation Lab for pre-screening candidate cuts/adds before spending games on them.")
    lines.append("- Use stress tests on historical losses to spot recurring matchup failures.")
    if model_info is None:
        lines.append("- Loss model: not enough stable samples yet for feature-importance guidance.")
    else:
        n_games = int(model_info.get("n_games", 0))
        acc = float(model_info.get("train_accuracy", 0.0))
        lines.append(f"- Loss model trained on {n_games} games, train accuracy {acc:.1%}.")
        top = model_info.get("feature_importance_df", pd.DataFrame())
        if isinstance(top, pd.DataFrame) and not top.empty:
            lines.append("- Top model drivers:")
            for _, r in top.head(5).iterrows():
                lines.append(f"  - {r['Feature']}: coefficient {float(r['Coefficient']):+.3f}")

    lines.append("")
    lines.append("## Interpretation Guardrails")
    lines.append("- Correlation is directional, not proof of causation.")
    lines.append("- Prefer high-priority actions with larger sample sizes.")
    lines.append("- Revert changes quickly if they fail to improve both win rate and process metrics.")

    return "\n".join(lines)


def save_recommendation_artifacts(
    output_root: Path,
    games_df: pd.DataFrame,
    recommendations_df: pd.DataFrame,
    model_info: dict | None = None,
) -> None:
    output_root.mkdir(parents=True, exist_ok=True)

    recommendations_df.to_csv(output_root / "winrate_recommendations.csv", index=False)

    md = render_recommendations_markdown(recommendations_df, games_df, model_info=model_info)
    (output_root / "winrate_improvement_plan.md").write_text(md, encoding="utf-8")

    if model_info is not None:
        fi = model_info.get("feature_importance_df")
        if isinstance(fi, pd.DataFrame) and not fi.empty:
            fi.to_csv(output_root / "loss_model_feature_importance.csv", index=False)
