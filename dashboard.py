"""
Lorcana Analyzer – Streamlit Dashboard
Run with:  C:/Python311/python.exe -m streamlit run dashboard.py
"""

from __future__ import annotations

import sys
import json
import re
import random
import html
from pathlib import Path

import pandas as pd
import streamlit as st
import requests

# ── path setup ──────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent
ANALYSIS_DIR = ROOT / "analysis_output"
sys.path.insert(0, str(ROOT / "src"))

# ── page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Lorcana Analyzer",
    page_icon="🃏",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── helpers ─────────────────────────────────────────────────────────────────

@st.cache_data(ttl=300)
def load(filename: str) -> pd.DataFrame:
    path = ANALYSIS_DIR / filename
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def pct(val: float) -> str:
    return f"{val:.1%}"


def fmt_lift(val: float) -> str:
    return f"{val:+.1%}"


def fmt_multiplier(val: float) -> str:
    return f"{val:.2f}×"


def first_present(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for name in candidates:
        if name in df.columns:
            return name
    return None


def best_populated_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    present = [c for c in candidates if c in df.columns]
    if not present:
        return None

    counts = []
    for col in present:
        non_empty = (
            df[col]
            .fillna("")
            .astype(str)
            .str.strip()
            .ne("")
            .sum()
        )
        counts.append((int(non_empty), col))
    counts.sort(reverse=True)
    return counts[0][1] if counts and counts[0][0] > 0 else None


def parse_deck_cell(cell: object) -> dict[str, int]:
    if cell is None or (isinstance(cell, float) and pd.isna(cell)):
        return {}

    if isinstance(cell, list):
        out: dict[str, int] = {}
        for item in cell:
            if not isinstance(item, dict):
                continue
            card_id = str(item.get("cardId") or item.get("card_id") or "").strip()
            try:
                count = int(item.get("count") or 0)
            except (TypeError, ValueError):
                continue
            if card_id and count > 0:
                out[card_id] = out.get(card_id, 0) + count
        return out

    if isinstance(cell, str):
        text = cell.strip()
        if not text:
            return {}
        if text.startswith("["):
            try:
                parsed = json.loads(text)
                return parse_deck_cell(parsed)
            except json.JSONDecodeError:
                return {}
    return {}


def parse_card_list(cell: object) -> list[str]:
    """Parse card list values stored as pipe-delimited text, JSON arrays, or plain text."""
    if cell is None or (isinstance(cell, float) and pd.isna(cell)):
        return []

    if isinstance(cell, list):
        return [str(x).strip() for x in cell if str(x).strip()]

    text = str(cell).strip()
    if not text:
        return []

    if text.startswith("["):
        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                return [str(x).strip() for x in parsed if str(x).strip()]
        except json.JSONDecodeError:
            pass

    if "|" in text:
        return [part.strip() for part in text.split("|") if part.strip()]

    return [text]


def remove_first_occurrence(cards: list[str], card_name: str) -> None:
    """Remove only one matching copy to preserve duplicates in hand reconstruction."""
    for idx, value in enumerate(cards):
        if value == card_name:
            cards.pop(idx)
            return


def _compress_repeated_sequence(values: list[str], expected_len: int) -> list[str]:
    """If parser duplicated a full card sequence, keep a single copy."""
    if expected_len <= 0:
        return values
    if len(values) >= expected_len * 2 and values[:expected_len] == values[expected_len:expected_len * 2]:
        return values[:expected_len]
    return values


def build_mulligan_snapshot(csv_row_id: int) -> dict[str, list[str]]:
    """Build opening hand, removed cards, and post-mulligan hand for one game."""
    if events_df.empty or "csv_row_id" not in events_df.columns:
        return {"opening": [], "removed": [], "post_mulligan": []}

    e = events_df[
        (events_df["csv_row_id"] == csv_row_id)
        & (events_df["is_my_event"] == True)
        & (events_df["event_type"].isin(["INITIAL_HAND", "MULLIGAN"]))
    ].copy()

    if e.empty:
        return {"opening": [], "removed": [], "post_mulligan": []}

    e["timestamp"] = pd.to_datetime(e["timestamp"], errors="coerce", utc=True)
    e = e.sort_values(["timestamp", "event_id"], na_position="last")

    initial_rows = e[e["event_type"] == "INITIAL_HAND"].copy()
    opening: list[str] = []
    if not initial_rows.empty:
        first_hand_event = initial_rows.iloc[0]["event_id"]
        first_group = initial_rows[initial_rows["event_id"] == first_hand_event]
        opening_raw = first_group["card_name"].dropna().astype(str).tolist()
        msg_text = str(first_group["message"].dropna().astype(str).iloc[0]) if first_group["message"].notna().any() else ""
        expected_open = len(re.findall(r"\{card:\d+\}", msg_text))
        opening = _compress_repeated_sequence(opening_raw, expected_open)
        if expected_open > 0:
            opening = opening[:expected_open]

    current_hand = opening.copy()
    removed_cards: list[str] = []

    mulligans = e[e["event_type"] == "MULLIGAN"].copy()
    if not mulligans.empty:
        for event_id, grp in mulligans.groupby("event_id", sort=False):
            rows = grp.copy()
            card_names = rows["card_name"].dropna().astype(str).tolist()
            msg = rows["message"].dropna().astype(str)
            msg_text = msg.iloc[0] if not msg.empty else ""

            mull_count = 0
            m = re.search(r"mulliganed\s+(\d+)\s+cards?", msg_text, flags=re.IGNORECASE)
            if m:
                mull_count = int(m.group(1))

            expected_total = 2 * mull_count if mull_count > 0 else 0
            card_names = _compress_repeated_sequence(card_names, expected_total)

            if mull_count > 0 and len(card_names) >= (2 * mull_count):
                removed = card_names[:mull_count]
                drawn = card_names[mull_count:(2 * mull_count)]
            elif mull_count > 0 and len(card_names) >= mull_count:
                removed = card_names[:mull_count]
                drawn = []
            else:
                removed = card_names
                drawn = []

            for card in removed:
                remove_first_occurrence(current_hand, card)
                removed_cards.append(card)
            for card in drawn:
                current_hand.append(card)

    return {
        "opening": opening,
        "removed": removed_cards,
        "post_mulligan": current_hand,
    }


def read_guide_notes() -> str:
    """Load editable deck-guide notes from disk if present."""
    notes_path = ANALYSIS_DIR / "deck_play_guide_notes.md"
    if not notes_path.exists():
        return (
            "### OTP Keeps\n"
            "- Add OTP keep notes here.\n\n"
            "### OTD Keeps\n"
            "- Add OTD keep notes here.\n\n"
            "### Important Combos\n"
            "- Add combo notes here."
        )
    return notes_path.read_text(encoding="utf-8")


@st.cache_data(ttl=86400)
def load_card_image_map() -> dict[str, str]:
    """Load card-name -> image URL map from the public Lorcana API."""
    try:
        resp = requests.get("https://api.lorcana-api.com/cards/fetch/?search=oswald", timeout=20)
        resp.raise_for_status()
        cards = resp.json()
    except Exception:
        return {}

    image_map: dict[str, str] = {}
    if not isinstance(cards, list):
        return image_map

    for card in cards:
        if not isinstance(card, dict):
            continue
        name = str(card.get("Name") or "").strip()
        image = str(card.get("Image") or "").strip()
        if name and image:
            image_map[name.lower()] = image
    return image_map


def to_replay_url(url: str, replay_id: str | None = None) -> str:
    """Convert API download links (/r/) into shareable replay links (/replay/)."""
    text = (url or "").strip()
    if text:
        return text.replace("/r/", "/replay/")
    if replay_id:
        rid = str(replay_id).strip()
        if rid:
            return f"https://duels.ink/replay/{rid}"
    return ""


def top_card_ids(df: pd.DataFrame, col_name: str, top_n: int = 20) -> pd.DataFrame:
    counts: dict[str, int] = {}
    if col_name not in df.columns:
        return pd.DataFrame(columns=["card_id", "copies_seen"])

    for value in df[col_name].dropna():
        parsed = parse_deck_cell(value)
        for card_id, qty in parsed.items():
            counts[card_id] = counts.get(card_id, 0) + qty

    if not counts:
        return pd.DataFrame(columns=["card_id", "copies_seen"])

    out = pd.DataFrame(
        [{"card_id": cid, "copies_seen": qty} for cid, qty in counts.items()]
    ).sort_values("copies_seen", ascending=False)
    return out.head(top_n)


def _friendly_card_table(df: pd.DataFrame) -> pd.DataFrame:
    """Rename raw analytics columns to human-friendly labels for display."""
    rename = {
        "card_name": "Card",
        "games_with_card": "Games",
        "loss_rate_with_card": "Loss Rate",
        "loss_lift_vs_baseline": "vs Baseline ↑",
        "loss_odds_ratio_approx": "Loss Multiplier",
        "wins_with_card": "Wins",
        "losses_with_card": "Losses",
    }
    d = df.rename(columns={k: v for k, v in rename.items() if k in df.columns}).copy()
    if "Loss Rate" in d.columns:
        d["Loss Rate"] = d["Loss Rate"].map(pct)
    if "vs Baseline ↑" in d.columns:
        d["vs Baseline ↑"] = d["vs Baseline ↑"].map(fmt_lift)
    if "Loss Multiplier" in d.columns:
        d["Loss Multiplier"] = d["Loss Multiplier"].map(fmt_multiplier)
    return d


@st.cache_data(ttl=300)
def _recompute_card_corr(csv_row_ids_tuple: tuple, min_games: int) -> pd.DataFrame:
    """Recompute card loss correlation for a subset of games (for archetype filter)."""
    from src.lorcana_analyzer.analytics import card_loss_correlation
    ids = set(csv_row_ids_tuple)
    g = games_df[games_df["csv_row_id"].isin(ids)].copy()
    e = events_df[events_df["csv_row_id"].isin(ids)].copy() if not events_df.empty else events_df
    return card_loss_correlation(g, e, min_games=min_games)


@st.cache_data(ttl=300)
def _recompute_seen_played(csv_row_ids_tuple: tuple, min_games: int) -> pd.DataFrame:
    from src.lorcana_analyzer.analytics import card_seen_vs_played_correlation
    ids = set(csv_row_ids_tuple)
    g = games_df[games_df["csv_row_id"].isin(ids)].copy()
    e = events_df[events_df["csv_row_id"].isin(ids)].copy() if not events_df.empty else events_df
    return card_seen_vs_played_correlation(g, e, min_games=min_games)


@st.cache_data(ttl=300)
def _auto_recommendations(csv_row_ids_tuple: tuple, min_card_games: int = 12) -> pd.DataFrame:
    """Compute top recommendations automatically for current filtered games."""
    from src.lorcana_analyzer.advisor import build_recommendations
    from src.lorcana_analyzer.analytics import (
        card_loss_correlation,
        card_seen_vs_played_correlation,
        early_ink_rates,
        event_type_loss_correlation,
        first_quest_turn,
        matchup_win_rates,
        opening_hand_loss_correlation,
    )

    if not csv_row_ids_tuple:
        return pd.DataFrame(columns=["rank", "category", "action", "rationale", "signal_score", "games", "confidence", "priority"])

    ids = set(csv_row_ids_tuple)
    g = games_df[games_df["csv_row_id"].isin(ids)].copy()
    e = events_df[events_df["csv_row_id"].isin(ids)].copy() if not events_df.empty else events_df

    card_corr = card_loss_correlation(g, e, min_games=min_card_games)
    seen_played = card_seen_vs_played_correlation(g, e, min_games=min_card_games)
    opening_corr = opening_hand_loss_correlation(g, min_games=min_card_games)
    event_corr = event_type_loss_correlation(g, e, min_games=5)
    ink_rates = early_ink_rates(g, e, turns=4)
    first_quest = first_quest_turn(g, e)
    matchup_rates = matchup_win_rates(g, min_games=8)

    return build_recommendations(
        games_df=g,
        card_df=card_corr,
        opening_df=opening_corr,
        event_type_df=event_corr,
        ink_df=ink_rates,
        fq_df=first_quest,
        matchup_df=matchup_rates,
        seen_played_df=seen_played,
        min_card_games=min_card_games,
    )


def styled_metric(label: str, value: str, delta: str | None = None) -> None:
    st.metric(label, value, delta)


# ── data ────────────────────────────────────────────────────────────────────
games_df = load("clean_games.csv")
events_df = load("clean_events.csv")
summary_df = load("summary_metrics.csv")
card_df = load("card_loss_correlation.csv")
opening_df = load("opening_hand_loss_correlation.csv")
arch_df = load("archetype_win_rates.csv")
matchup_df = load("matchup_win_rates.csv")
deck_ver_df = load("deck_version_win_rates.csv")
tempo_df = load("sequencing_tempo_curves.csv")
ink_df = load("sequencing_early_ink.csv")
fq_df = load("sequencing_first_quest.csv")
event_type_df = load("event_type_loss_correlation.csv")
opp_card_df = load("opponent_card_loss_correlation.csv")
pair_df = load("card_pair_correlation.csv")
seen_played_df = load("card_seen_vs_played.csv")
lore_df = load("lore_curves.csv")
mulligan_df = load("mulligan_impact.csv")

# ── sidebar: global filters ──────────────────────────────────────────────────
st.sidebar.title("🃏 Lorcana Analyzer")
st.sidebar.markdown("---")

pages = [
    "📊 Overview",
    "📘 Deck Guide",
    "🔌 API Insights",
    "🃏 Card Analysis",
    "🏰 Archetypes & Matchups",
    "⚔️ Sequencing & Tempo",
    "📈 Lore & Mulligans",
    "🧙 Deck Advisor",
    "🎲 Simulation Lab",
    "🗂️ Raw Data",
]
page = st.sidebar.radio("Navigate", pages, label_visibility="collapsed")

# Date filter (supports both legacy exports and API sync fields)
if not games_df.empty:
    date_col = best_populated_column(games_df, ["started_at", "date", "Started At", "created_at"])
    if date_col:
        parsed_dates = pd.to_datetime(games_df[date_col], errors="coerce", utc=True)
        valid_dates = parsed_dates.dropna()
        if not valid_dates.empty:
            min_d = valid_dates.min().date()
            max_d = valid_dates.max().date()
            date_range = st.sidebar.date_input("Date range", value=(min_d, max_d), key="date_range")
            if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
                selected_start = pd.Timestamp(date_range[0]).date()
                selected_end = pd.Timestamp(date_range[1]).date()
                # Keep all rows when the default full range is selected.
                if selected_start != min_d or selected_end != max_d:
                    start = pd.Timestamp(selected_start).tz_localize("UTC")
                    end = pd.Timestamp(selected_end).tz_localize("UTC") + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
                    games_df = games_df[(parsed_dates >= start) & (parsed_dates <= end)]

# Archetype filter
if not games_df.empty and "archetype" in games_df.columns:
    archetypes = sorted(games_df["archetype"].dropna().unique())
    selected_arch = st.sidebar.multiselect("My archetype(s)", archetypes, default=archetypes)
    games_df = games_df[games_df["archetype"].isin(selected_arch)]

# Turn order filter
if not games_df.empty and ("turn_order" in games_df.columns or "went_first" in games_df.columns):
    current_to = (
        games_df["turn_order"].fillna("").astype(str)
        if "turn_order" in games_df.columns
        else pd.Series("", index=games_df.index, dtype="object")
    )
    current_to = current_to.str.strip().str.upper()
    current_to = current_to.replace({"TRUE": "OTP", "FALSE": "OTD", "1": "OTP", "0": "OTD"})
    invalid_mask = current_to.isin({"", "NAN", "NONE", "NULL"})

    if "went_first" in games_df.columns:
        went_first = games_df["went_first"].astype(str).str.strip().str.lower()
        inferred_to = went_first.map({"true": "OTP", "false": "OTD", "1": "OTP", "0": "OTD"})
        current_to = current_to.where(~invalid_mask, inferred_to.fillna("Unknown"))
    else:
        current_to = current_to.where(~invalid_mask, "Unknown")

    games_df["turn_order"] = current_to
    turn_orders = sorted(games_df["turn_order"].dropna().unique())
    selected_to = st.sidebar.multiselect("Turn order", turn_orders, default=turn_orders)
    games_df = games_df[games_df["turn_order"].isin(selected_to)]

# Match format filter (supports legacy `Match Format` and API `match_format`)
if not games_df.empty:
    fmt_col = best_populated_column(games_df, ["match_format", "Match Format"])
    if fmt_col:
        fmt_series = games_df[fmt_col].fillna("Unknown").astype(str).str.strip()
        fmt_series = fmt_series.replace("", "Unknown")
        formats = sorted(fmt_series.unique())
        selected_fmt = st.sidebar.multiselect("Match format", formats, default=formats)
        games_df = games_df[fmt_series.isin(selected_fmt)]

# Queue filter (very useful for Infinity BO3 focus)
if not games_df.empty:
    queue_col = best_populated_column(games_df, ["queue_name", "Queue", "queue_id"])
    if queue_col:
        queue_series = games_df[queue_col].fillna("Unknown").astype(str).str.strip().replace("", "Unknown")
        queue_values = sorted(queue_series.unique())
        selected_queue = st.sidebar.multiselect("Queue", queue_values, default=queue_values)
        games_df = games_df[queue_series.isin(selected_queue)]

# Deck-focused filters for API data
if not games_df.empty:
    deck_id_col = best_populated_column(games_df, ["your_deck_id"])
    deck_colors_col = best_populated_column(games_df, ["your_deck_colors", "my_colors", "archetype"])

    if deck_id_col or deck_colors_col:
        st.sidebar.markdown("---")
        st.sidebar.markdown("**Deck Focus**")

        if deck_id_col:
            deck_id_series = games_df[deck_id_col].fillna("Unknown").astype(str).str.strip().replace("", "Unknown")
            deck_id_counts = deck_id_series.value_counts()
            deck_id_labels = [f"{deck_id} ({count})" for deck_id, count in deck_id_counts.items()]
            selected_deck_labels = st.sidebar.multiselect("Deck ID", deck_id_labels, default=deck_id_labels)
            selected_deck_ids = {label.rsplit(" (", 1)[0] for label in selected_deck_labels}
            games_df = games_df[deck_id_series.isin(selected_deck_ids)]

        if deck_colors_col:
            deck_color_series = games_df[deck_colors_col].fillna("Unknown").astype(str).str.strip().replace("", "Unknown")
            deck_colors = sorted(deck_color_series.unique())
            selected_deck_colors = st.sidebar.multiselect("Deck colors", deck_colors, default=deck_colors)
            games_df = games_df[deck_color_series.isin(selected_deck_colors)]

        # No real deck-title column is currently present in clean_games, so provide keyword fallback.
        deck_keyword = st.sidebar.text_input("Deck keyword", value="", help="Searches deck id/colors/decklist text")
        if deck_keyword:
            search_cols = [
                c for c in ["your_deck_id", "your_deck_colors", "your_decklist", "Decklist"] if c in games_df.columns
            ]
            if search_cols:
                mask = pd.Series(False, index=games_df.index)
                for col in search_cols:
                    mask = mask | games_df[col].astype(str).str.contains(deck_keyword, case=False, na=False)
                games_df = games_df[mask]

st.sidebar.markdown("---")
if not games_df.empty:
    st.sidebar.caption(f"Showing **{len(games_df)}** games after filters")

# ════════════════════════════════════════════════════════════════════════════
# PAGE: Overview
# ════════════════════════════════════════════════════════════════════════════
if page == "📊 Overview":
    st.title("📊 Overview")

    if games_df.empty:
        st.warning("No game data found. Run `analyze.py` first to generate outputs.")
        st.stop()

    total = len(games_df)
    wins = int((~games_df["is_loss"]).sum())
    losses = int(games_df["is_loss"].sum())
    win_rate = wins / total if total else 0

    filtered_otp = games_df[games_df["turn_order"] == "OTP"]
    filtered_otd = games_df[games_df["turn_order"] == "OTD"]
    otp_wr = (1 - filtered_otp["is_loss"].mean()) if len(filtered_otp) else None
    otd_wr = (1 - filtered_otd["is_loss"].mean()) if len(filtered_otd) else None

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Games", total)
    c2.metric("Wins", wins)
    c3.metric("Losses", losses)
    c4.metric("Win Rate", pct(win_rate))
    c5.metric("OTP vs OTD", f"{pct(otp_wr) if otp_wr is not None else 'N/A'} / {pct(otd_wr) if otd_wr is not None else 'N/A'}")

    st.markdown("---")

    # 🎯 Quick Recommendations Section
    st.subheader("🎯 Quick Wins")
    st.caption("Auto-generated recommendations based on your current filters. Updates as you change queue, deck, date range, etc.")

    if "csv_row_id" in games_df.columns and not games_df.empty:
        rec_ids = tuple(sorted(games_df["csv_row_id"].dropna().astype(int).unique()))
        recommendations = _auto_recommendations(rec_ids, min_card_games=12)

        if recommendations.empty:
            st.info("✓ No strong signals yet. Your current strategy seems solid!")
        else:
            top_recs = recommendations.head(5)
            
            # Display as a set of cards with icons
            cols = st.columns(min(len(top_recs), 2))
            for idx, (_, rec) in enumerate(top_recs.iterrows()):
                col = cols[idx % len(cols)]
                with col:
                    # Category icons
                    icons = {
                        "Decklist": "🎯",
                        "Mulligan": "🔄",
                        "Sequencing": "⚔️",
                        "Turn Order": "🔀",
                        "Play Pattern": "🎪",
                        "Matchup": "🏰",
                    }
                    icon = icons.get(rec.get("category"), "💡")
                    
                    with st.container(border=True):
                        st.markdown(f"**{icon} {rec.get('category')}** (Priority: {rec.get('priority', 0):.1f})")
                        st.markdown(f"_{rec.get('action')}_")
                        
                        # Compact details
                        detail_cols = st.columns([1, 1, 1])
                        with detail_cols[0]:
                            st.caption(f"📊 Games: {int(rec.get('games', 0))}")
                        with detail_cols[1]:
                            confidence = rec.get("confidence", "Medium")
                            confidence_color = "🟢" if confidence == "High" else "🟡" if confidence == "Medium" else "🔴"
                            st.caption(f"{confidence_color} {confidence}")
                        with detail_cols[2]:
                            signal = rec.get("signal_score", 0)
                            st.caption(f"📈 Signal: {signal:+.1f}")
                        
                        with st.expander("Why?"):
                            st.caption(rec.get("rationale", ""))
    else:
        st.info("No recommendations available for the current filter selection.")

    st.markdown("---")

    # Win rate by turn order bar
    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Win Rate by Turn Order")
        to_data = (
            games_df.groupby("turn_order", as_index=False)
            .agg(games=("is_loss", "count"), win_rate=("is_loss", lambda s: 1 - s.mean()))
        )
        if not to_data.empty:
            to_data["win_rate_pct"] = (to_data["win_rate"] * 100).round(1)
            st.bar_chart(to_data.set_index("turn_order")["win_rate_pct"], height=280)

    with col_b:
        st.subheader("Games Over Time")
        if "date" in games_df.columns and games_df["date"].notna().any():
            timeline = (
                games_df.groupby(games_df["date"].dt.date)
                .agg(games=("is_loss", "count"), win_rate=("is_loss", lambda s: 1 - s.mean()))
                .reset_index()
            )
            timeline["date"] = pd.to_datetime(timeline["date"])
            timeline["rolling_wr"] = timeline["win_rate"].rolling(7, min_periods=1).mean()
            st.line_chart(timeline.set_index("date")["rolling_wr"], height=280)
        else:
            st.info("No date column available for timeline.")

    st.markdown("---")
    st.subheader("Recent Games")
    display_cols = [c for c in ["date", "opponent_name", "archetype", "opponent_colors", "turn_order", "is_loss", "turn_count", "mulligan_count_my"] if c in games_df.columns]
    recent = games_df[display_cols].copy()
    if "is_loss" in recent.columns:
        recent["result"] = recent["is_loss"].map({True: "❌ Loss", False: "✅ Win"})
        recent = recent.drop(columns=["is_loss"])
    st.dataframe(recent.tail(20).iloc[::-1].reset_index(drop=True), use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════
# PAGE: Card Analysis
# ════════════════════════════════════════════════════════════════════════════
elif page == "📘 Deck Guide":
    st.title("📘 Deck Guide")
    st.caption(
        "Shareable coaching view for your deck: notes, mulligan examples, and replay library by color/turn/result."
    )

    if games_df.empty:
        st.info("No game data available. Run `analyze.py` first.")
        st.stop()

    st.subheader("1) Deck Notes (OTP / OTD / Combos)")
    st.caption(
        "These notes are loaded from `analysis_output/deck_play_guide_notes.md` so you can edit them later."
    )
    st.markdown(read_guide_notes())

    card_image_map = load_card_image_map()

    def _render_card_tiles(cards: list[str]) -> None:
        if not cards:
            st.info("No cards found for this view.")
            return

        tiles: list[str] = []
        for card in cards:
            safe_card = html.escape(str(card))
            img_url = card_image_map.get(str(card).lower(), "")
            if img_url:
                safe_img = html.escape(img_url)
                tiles.append(
                    "<div style='display:flex;flex-direction:column;gap:4px;'>"
                    f"<img src='{safe_img}' style='width:100%;max-width:180px;border-radius:8px;display:block;'/>"
                    f"<div style='font-size:12px;line-height:1.2;color:#111;'>{safe_card}</div>"
                    "</div>"
                )
            else:
                tiles.append(
                    "<div style='width:100%;max-width:180px;border:1px dashed #CFCFCF;border-radius:8px;"
                    "padding:8px;min-height:44px;background:#F7F7F7;'>"
                    f"<div style='font-size:12px;font-weight:600;line-height:1.2;color:#111;'>{safe_card}</div>"
                    "</div>"
                )

        st.markdown(
            (
                "<div style='display:grid;grid-template-columns:repeat(auto-fill,minmax(150px,1fr));"
                "gap:8px;align-items:start;'>"
                + "".join(tiles)
                + "</div>"
            ),
            unsafe_allow_html=True,
        )

    # Guide scope: Oswald decks only. Use explicit Oswald card IDs as primary signal.
    oswald_card_ids = {"6-142", "6-D23-6"}
    oswald_ids: set[int] = set()
    if not events_df.empty and {"csv_row_id", "card_id", "is_my_event"}.issubset(events_df.columns):
        card_id_series = events_df["card_id"].fillna("").astype(str).str.strip()
        mask = (
            events_df["is_my_event"].fillna(False).astype(bool)
            & card_id_series.isin(oswald_card_ids)
        )
        oswald_ids = set(events_df.loc[mask, "csv_row_id"].dropna().astype(int).tolist())

    # Fallback when card IDs are unavailable or sparse in a subset.
    if not oswald_ids and not events_df.empty and {"csv_row_id", "card_name", "is_my_event"}.issubset(events_df.columns):
        name_mask = (
            events_df["is_my_event"].fillna(False).astype(bool)
            & events_df["card_name"].fillna("").astype(str).str.contains("oswald", case=False, na=False)
        )
        oswald_ids = set(events_df.loc[name_mask, "csv_row_id"].dropna().astype(int).tolist())

    # Backup signal when event names are sparse.
    if "opening_hand_cards" in games_df.columns:
        open_mask = games_df["opening_hand_cards"].fillna("").astype(str).str.contains("oswald", case=False, na=False)
        oswald_ids.update(games_df.loc[open_mask, "csv_row_id"].dropna().astype(int).tolist())

    # If still empty, derive backup signal from full games opening hand.
    full_games = load("clean_games.csv")
    if not oswald_ids and "opening_hand_cards" in full_games.columns:
        full_open_mask = full_games["opening_hand_cards"].fillna("").astype(str).str.contains("oswald", case=False, na=False)
        oswald_ids.update(full_games.loc[full_open_mask, "csv_row_id"].dropna().astype(int).tolist())

    if "turn_order" not in full_games.columns and "went_first" in full_games.columns:
        inferred = full_games["went_first"].astype(str).str.strip().str.lower().map(
            {"true": "OTP", "false": "OTD", "1": "OTP", "0": "OTD"}
        )
        full_games["turn_order"] = inferred.fillna("Unknown")

    guide_games = games_df[games_df["csv_row_id"].isin(oswald_ids)].copy() if "csv_row_id" in games_df.columns else pd.DataFrame()

    using_fallback = False
    if guide_games.empty and not full_games.empty:
        fallback_games = full_games[full_games["csv_row_id"].isin(oswald_ids)].copy() if "csv_row_id" in full_games.columns else pd.DataFrame()
        if not fallback_games.empty:
            guide_games = fallback_games
            using_fallback = True

    if guide_games.empty:
        st.info("No Oswald deck games found. Try re-running analysis with replay data that includes Oswald decks.")
        st.stop()

    if using_fallback:
        st.info("No Oswald games in current sidebar filters, so Deck Guide is showing Oswald examples from all games.")

    st.markdown("---")
    st.subheader("2) How To Alter Your Hand")
    st.caption(
        "Training mode: evaluate the hand first, then reveal result after deciding your mulligan plan."
    )

    replay_col = best_populated_column(guide_games, ["replay_url"])
    opp_col = best_populated_column(guide_games, ["opp_display_name", "opponent", "Opponent", "opponent_name"])
    started_col = best_populated_column(guide_games, ["started_at", "date", "Started At", "created_at"])
    color_col = best_populated_column(guide_games, ["opponent_colors"])
    series_col = best_populated_column(guide_games, ["match_game_number"])

    if not color_col:
        st.info("`opponent_colors` is missing. Re-run `analyze.py` to regenerate clean data with inferred opponent colors.")
        st.stop()

    hand_pool = guide_games.copy()
    if not full_games.empty and "csv_row_id" in hand_pool.columns and "csv_row_id" in full_games.columns:
        # Backfill user-facing info fields from canonical clean_games rows keyed by csv_row_id.
        backfill_cols = [
            c
            for c in [
                "opp_display_name",
                "opponent",
                "Opponent",
                "opponent_name",
                "started_at",
                "date",
                "Started At",
                "created_at",
                "turn_order",
                "match_game_number",
                "opponent_colors",
                "replay_url",
                "replay_id",
            ]
            if c in full_games.columns
        ]
        if backfill_cols:
            backfill_df = full_games[["csv_row_id"] + backfill_cols].copy()
            hand_pool = hand_pool.merge(backfill_df, on="csv_row_id", how="left", suffixes=("", "_full"))
            for c in backfill_cols:
                full_c = f"{c}_full"
                if full_c not in hand_pool.columns:
                    continue
                if c in hand_pool.columns:
                    left = hand_pool[c].fillna("").astype(str).str.strip()
                    right = hand_pool[full_c].fillna("").astype(str).str.strip()
                    hand_pool[c] = left.where(left.ne(""), right)
                else:
                    hand_pool[c] = hand_pool[full_c]
                hand_pool = hand_pool.drop(columns=[full_c])

    if "opening_hand_cards" in hand_pool.columns:
        hand_pool = hand_pool[hand_pool["opening_hand_cards"].fillna("").astype(str).str.strip().ne("")]
    if replay_col:
        hand_pool = hand_pool[hand_pool[replay_col].fillna("").astype(str).str.strip().ne("")]

    if hand_pool.empty:
        st.info("No games with opening-hand data found in the current filters.")
    else:
        st.caption("Optional: filter examples before selecting or randomizing a game.")
        filter_col1, filter_col2 = st.columns(2)

        color_options = (
            sorted(hand_pool[color_col].fillna("Unknown").astype(str).str.strip().replace("", "Unknown").unique())
            if color_col in hand_pool.columns
            else []
        )
        turn_options = (
            sorted(hand_pool["turn_order"].fillna("Unknown").astype(str).str.strip().replace("", "Unknown").unique())
            if "turn_order" in hand_pool.columns
            else ["OTP", "OTD", "Unknown"]
        )

        with filter_col1:
            selected_hand_colors = st.multiselect(
                "Color combo (optional)",
                color_options,
                default=color_options,
                key="guide_hand_filter_colors",
            )
        with filter_col2:
            selected_hand_turns = st.multiselect(
                "Turn order (optional)",
                turn_options,
                default=turn_options,
                key="guide_hand_filter_turns",
            )

        filtered_hand_pool = hand_pool.copy()
        if color_col in filtered_hand_pool.columns:
            normalized_color = filtered_hand_pool[color_col].fillna("Unknown").astype(str).str.strip().replace("", "Unknown")
            filtered_hand_pool = filtered_hand_pool[normalized_color.isin(selected_hand_colors)]
        if "turn_order" in filtered_hand_pool.columns:
            normalized_to = filtered_hand_pool["turn_order"].fillna("Unknown").astype(str).str.strip().replace("", "Unknown")
            filtered_hand_pool = filtered_hand_pool[normalized_to.isin(selected_hand_turns)]

        label_opp_cols = [c for c in ["opp_display_name", "opponent", "Opponent", "opponent_name"] if c in hand_pool.columns]
        label_date_cols = [c for c in ["started_at", "date", "Started At", "created_at"] if c in hand_pool.columns]
        hand_options = []
        for _, row in filtered_hand_pool.tail(200).iloc[::-1].iterrows():
            csv_id = int(row.get("csv_row_id", 0))

            opp = "Unknown"
            for c in label_opp_cols:
                v = str(row.get(c, "")).strip()
                if v and v.lower() not in {"nan", "none", "null"}:
                    opp = v
                    break

            date_txt = ""
            for c in label_date_cols:
                v = str(row.get(c, "")).strip()
                if v and v.lower() not in {"nan", "none", "null"}:
                    date_txt = v[:19]
                    break

            to = str(row.get("turn_order", "Unknown") or "Unknown").strip() or "Unknown"
            color_txt = str(row.get(color_col, "Unknown") or "Unknown").strip() if color_col else "Unknown"
            if not color_txt or color_txt.lower() in {"nan", "none", "null"}:
                color_txt = "Unknown"

            series_txt = "G?"
            if series_col and pd.notna(row.get(series_col)):
                try:
                    series_txt = f"G{int(float(row.get(series_col)))}"
                except (TypeError, ValueError):
                    series_txt = "G?"

            label = f"{date_txt} | {opp} | {color_txt} | {to} | {series_txt} | csv_row_id={csv_id}"
            hand_options.append((label, csv_id))

        option_labels = [x[0] for x in hand_options]
        if not option_labels:
            st.info("No games match the optional color/turn filters.")
            st.stop()

        if st.session_state.get("guide_game_pick") not in option_labels:
            seeded = random.choice(option_labels)
            st.session_state["guide_game_pick"] = seeded
            st.session_state["guide_random_pick"] = seeded

        if st.button("Pick random game", key="guide_random_button"):
            picked = random.choice(option_labels)
            st.session_state["guide_game_pick"] = picked
            st.session_state["guide_random_pick"] = picked
            st.rerun()

        selected_label = st.selectbox("Replay game", option_labels, key="guide_game_pick")
        st.session_state["guide_random_pick"] = selected_label
        selected_csv_id = dict(hand_options)[selected_label]

        selected_game = hand_pool[hand_pool["csv_row_id"] == selected_csv_id].iloc[0]
        opening_from_game = parse_card_list(selected_game.get("opening_hand_cards", ""))
        snapshot = build_mulligan_snapshot(selected_csv_id)
        opening_cards = snapshot["opening"] if snapshot["opening"] else opening_from_game
        removed_cards = snapshot["removed"]
        post_cards = snapshot["post_mulligan"] if snapshot["post_mulligan"] else opening_cards

        if replay_col or "replay_id" in selected_game.index:
            replay_link = to_replay_url(
                str(selected_game.get(replay_col, "")) if replay_col else "",
                replay_id=str(selected_game.get("replay_id", "")),
            )
            if replay_link:
                st.markdown(f"Replay URL: {replay_link}")

        st.markdown("**First Screen (Opening Hand)**")
        _render_card_tiles(opening_cards)

        show_solution = st.toggle(
            "Show my mulligan changes (cuts + post hand)",
            value=False,
            key=f"show_solution_{selected_csv_id}",
        )
        if show_solution:
            col_removed, col_post = st.columns(2)
            with col_removed:
                st.markdown("**Cards You Got Rid Of**")
                _render_card_tiles(removed_cards)
            with col_post:
                st.markdown("**Second Screen (Post-Mulligan Hand)**")
                _render_card_tiles(post_cards)
        else:
            st.info("Your mulligan changes are hidden. Toggle above to reveal cuts and post-mulligan hand.")

        st.markdown("**Before revealing result:** decide what you would mulligan and why.")
        reveal_key = f"reveal_result_{selected_csv_id}"
        if reveal_key not in st.session_state:
            st.session_state[reveal_key] = False
        if st.button("Reveal result for this example", key=f"reveal_button_{selected_csv_id}"):
            st.session_state[reveal_key] = True

        if st.session_state[reveal_key]:
            st.success(f"Actual result: {'Loss' if bool(selected_game.get('is_loss', False)) else 'Win'}")
        else:
            st.info("Result hidden. Click reveal when you are ready.")

    st.markdown("---")
    st.subheader("3) Replay URLs By Color + OTP/OTD + Result")

    replay_view = guide_games.copy()
    if replay_col:
        replay_view = replay_view[replay_view[replay_col].fillna("").astype(str).str.strip().ne("")]
    replay_view["opponent_colors"] = replay_view[color_col].fillna("Unknown").astype(str).str.strip().replace("", "Unknown")

    replay_view["result_label"] = replay_view["is_loss"].map({True: "Loss", False: "Win"})
    replay_view["turn_order"] = replay_view["turn_order"].fillna("Unknown").astype(str)

    color_values = sorted(replay_view["opponent_colors"].dropna().unique()) if not replay_view.empty else []
    selected_colors = st.multiselect("Colors", color_values, default=color_values)
    selected_turn = st.multiselect("Turn order", ["OTP", "OTD", "Unknown"], default=["OTP", "OTD", "Unknown"])
    selected_results = st.multiselect("Result", ["Win", "Loss"], default=["Win", "Loss"])

    filtered_replays = replay_view[
        replay_view["opponent_colors"].isin(selected_colors)
        & replay_view["turn_order"].isin(selected_turn)
        & replay_view["result_label"].isin(selected_results)
    ].copy()

    filtered_replays["replay_url_user"] = filtered_replays.apply(
        lambda r: to_replay_url(
            str(r.get(replay_col, "")) if replay_col else "",
            replay_id=str(r.get("replay_id", "")) if "replay_id" in filtered_replays.columns else "",
        ),
        axis=1,
    ) if not filtered_replays.empty else ""

    show_cols = [
        col
        for col in [started_col, opp_col, "opponent_colors", series_col, "turn_order", "result_label", replay_col]
        if col and col in filtered_replays.columns
    ]

    if filtered_replays.empty:
        st.info("No replay URLs match this filter.")
    else:
        show_cols_user = [
            col if col != replay_col else "replay_url_user"
            for col in show_cols
        ]

        st.caption(f"{len(filtered_replays)} replay links")
        st.dataframe(
            filtered_replays[show_cols_user]
            .rename(
                columns={
                    "opponent_colors": "Color",
                    series_col: "Series Game",
                    "turn_order": "Turn Order",
                    "result_label": "Result",
                    "replay_url_user": "Replay URL",
                }
            )
            .reset_index(drop=True),
            use_container_width=True,
        )

        replay_csv = filtered_replays[show_cols_user].to_csv(index=False).encode()
        st.download_button(
            label="Download filtered replay links CSV",
            data=replay_csv,
            file_name="deck_guide_replay_links.csv",
            mime="text/csv",
        )

    st.markdown("---")
    st.subheader("4) Share Deck Guide")
    st.caption("Export a standalone HTML snapshot with your notes and the current replay filter results.")

    export_replays = filtered_replays.copy()
    if export_replays.empty:
        st.info("No replay rows in the current filter to include in HTML export.")
    else:
        export_columns = [
            c
            for c in [started_col, opp_col, "opponent_colors", series_col, "turn_order", "result_label", "replay_url_user"]
            if c and c in export_replays.columns
        ]
        export_view = export_replays[export_columns].rename(
            columns={
                started_col: "Date",
                opp_col: "Opponent",
                "opponent_colors": "Color",
                series_col: "Series Game",
                "turn_order": "Turn Order",
                "result_label": "Result",
                "replay_url_user": "Replay URL",
            }
        )

        notes_text = read_guide_notes()
        notes_html = "<br>".join(html.escape(notes_text).splitlines())

        rows_html: list[str] = []
        for _, r in export_view.iterrows():
            replay_url = str(r.get("Replay URL", "") or "").strip()
            replay_cell = f"<a href='{html.escape(replay_url)}' target='_blank'>{html.escape(replay_url)}</a>" if replay_url else ""
            rows_html.append(
                "<tr>"
                f"<td>{html.escape(str(r.get('Date', '')))}</td>"
                f"<td>{html.escape(str(r.get('Opponent', '')))}</td>"
                f"<td>{html.escape(str(r.get('Color', '')))}</td>"
                f"<td>{html.escape(str(r.get('Turn Order', '')))}</td>"
                f"<td>{html.escape(str(r.get('Series Game', '')))}</td>"
                f"<td>{html.escape(str(r.get('Result', '')))}</td>"
                f"<td>{replay_cell}</td>"
                "</tr>"
            )

        html_doc = (
            "<!doctype html><html><head><meta charset='utf-8'>"
            "<title>Lorcana Deck Guide</title>"
            "<style>"
            "body{font-family:Segoe UI,Arial,sans-serif;max-width:1100px;margin:24px auto;padding:0 12px;color:#1b1b1b;}"
            "h1,h2{margin:0 0 10px 0;}"
            "p{margin:0 0 12px 0;color:#555;}"
            ".notes{border:1px solid #ddd;border-radius:8px;padding:12px;background:#fafafa;white-space:normal;line-height:1.45;margin-bottom:18px;}"
            "table{width:100%;border-collapse:collapse;font-size:13px;}"
            "th,td{border:1px solid #ddd;padding:8px;vertical-align:top;}"
            "th{background:#f4f4f4;text-align:left;}"
            "a{color:#0b5cab;text-decoration:none;}a:hover{text-decoration:underline;}"
            "</style></head><body>"
            "<h1>Lorcana Deck Guide</h1>"
            "<p>Exported from Deck Guide with current color/turn/result filters.</p>"
            "<h2>Deck Notes</h2>"
            f"<div class='notes'>{notes_html}</div>"
            "<h2>Replay URLs</h2>"
            "<table><thead><tr><th>Date</th><th>Opponent</th><th>Color</th><th>Turn Order</th><th>Series Game</th><th>Result</th><th>Replay URL</th></tr></thead>"
            f"<tbody>{''.join(rows_html)}</tbody></table>"
            "</body></html>"
        )

        st.download_button(
            label="Download Deck Guide HTML",
            data=html_doc.encode("utf-8"),
            file_name="deck_guide_export.html",
            mime="text/html",
        )


# ════════════════════════════════════════════════════════════════════════════
# PAGE: Card Analysis
# ════════════════════════════════════════════════════════════════════════════
elif page == "🃏 Card Analysis":
    st.title("🃏 Card Analysis")

    # Live archetype filter: recompute if sidebar filters are active
    filtered_row_ids = tuple(sorted(games_df["csv_row_id"].dropna().astype(int).tolist())) if not games_df.empty else ()
    is_filtered = not events_df.empty and len(filtered_row_ids) < len(load("clean_games.csv"))

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
        ["My Cards", "Seen vs Played", "Opponent Cards", "Card Pairs", "Opening Hand", "Event Types"]
    )

    with tab1:
        st.subheader("Your Cards Correlated with Losses")
        st.caption(
            "Shows cards you interacted with (played, quested, attacked, or inked) in games you lost more "
            "than your average. **vs Baseline** = how much your loss rate rises when this card is in play. "
            "**Loss Multiplier** = e.g. 1.5× means you lose 50% more often when this card is around."
        )

        min_games_c = st.slider("Minimum games", 3, 20, 5, key="card_min")
        n_show = st.slider("Top N cards", 5, 30, 15, key="card_n")

        active_card_df = _recompute_card_corr(filtered_row_ids, min_games_c) if is_filtered else card_df[card_df["games_with_card"] >= min_games_c].copy()

        if active_card_df.empty:
            st.info("No card data for current filter. Try lowering the minimum games or broadening the archetype filter.")
        else:
            top = active_card_df.head(n_show)
            chart_data = top.set_index("card_name")[["loss_lift_vs_baseline"]].rename(
                columns={"loss_lift_vs_baseline": "Loss Rate vs Baseline"}
            )
            st.bar_chart(chart_data, height=350)

            with st.expander("Full table"):
                st.dataframe(_friendly_card_table(active_card_df), use_container_width=True)

        st.markdown("---")
        st.subheader("Search a Card")
        search = st.text_input("Card name contains...", key="card_search")
        if search and not active_card_df.empty:
            result = active_card_df[active_card_df["card_name"].str.contains(search, case=False, na=False)]
            if result.empty:
                st.write("No matching cards found.")
            else:
                st.dataframe(_friendly_card_table(result), use_container_width=True)

    with tab2:
        st.subheader("Seen vs Played Breakdown")
        st.caption(
            "Compares what happens based on **how** you interact with each card:\n"
            "- **Drawn** = it appeared in your hand (opening or drawn mid-game)\n"
            "- **Played** = you played it as a character or action card\n"
            "- **Inked** = you used it as an ink resource instead of playing it\n\n"
            "**Play-through Rate** = how often you play the card when you draw it. "
            "Low rate = you're drawing it but not playing it (holding dead cards or forced to ink them). "
            "High loss rate when played = the card itself may be hurting you."
        )

        min_games_sp = st.slider("Minimum games drawn", 3, 20, 5, key="sp_min")
        n_sp = st.slider("Top N cards", 5, 30, 20, key="sp_n")

        active_sp_df = _recompute_seen_played(filtered_row_ids, min_games_sp) if is_filtered else seen_played_df[seen_played_df["games_drawn"] >= min_games_sp].copy() if not seen_played_df.empty else pd.DataFrame()

        if active_sp_df.empty:
            st.info("No seen/played data available. Run `analyze.py` first.")
        else:
            view_sp = st.radio(
                "Sort by",
                ["Highest loss rate when played", "Lowest play-through rate (cards you draw but don't play)"],
                key="sp_sort",
            )

            if "Lowest play-through" in view_sp:
                sorted_sp = active_sp_df.sort_values("play_through_rate", ascending=True, na_position="last").head(n_sp)
            else:
                sorted_sp = active_sp_df.sort_values("loss_lift_when_played", ascending=False, na_position="last").head(n_sp)

            display_sp = sorted_sp[["card_name", "games_drawn", "games_played", "games_inked",
                                     "loss_rate_drawn", "loss_rate_played", "loss_rate_inked",
                                     "play_through_rate", "loss_lift_when_played"]].copy()
            display_sp = display_sp.rename(columns={
                "card_name": "Card",
                "games_drawn": "Times Drawn",
                "games_played": "Times Played",
                "games_inked": "Times Inked",
                "loss_rate_drawn": "Loss Rate (Drawn)",
                "loss_rate_played": "Loss Rate (Played)",
                "loss_rate_inked": "Loss Rate (Inked)",
                "play_through_rate": "Play-Through Rate",
                "loss_lift_when_played": "Loss Lift When Played",
            })
            for col in ["Loss Rate (Drawn)", "Loss Rate (Played)", "Loss Rate (Inked)", "Play-Through Rate", "Loss Lift When Played"]:
                if col in display_sp.columns:
                    fmt = fmt_lift if "Lift" in col else pct
                    display_sp[col] = display_sp[col].map(lambda v, f=fmt: f(v) if pd.notna(v) else "—")

            st.dataframe(display_sp.reset_index(drop=True), use_container_width=True)

    with tab3:
        st.subheader("Opponent Cards Correlated with Your Losses")
        st.caption(
            "When the opponent played these cards, you lost more often than your baseline. "
            "**vs Baseline** = extra loss rate. **Loss Multiplier** = how many times more likely you are to "
            "lose when they play this card (1.0 = no effect, 1.5 = 50% more likely to lose)."
        )

        if opp_card_df.empty:
            st.info("No opponent card data available.")
        else:
            min_go = st.slider("Minimum games", 3, 20, 5, key="opp_min")
            n_opp = st.slider("Top N", 5, 30, 15, key="opp_n")
            filtered_opp = opp_card_df[opp_card_df["games_with_card"] >= min_go].head(n_opp)
            chart_opp = filtered_opp.set_index("card_name")[["loss_lift_vs_baseline"]].rename(
                columns={"loss_lift_vs_baseline": "Loss Rate vs Baseline"}
            )
            st.bar_chart(chart_opp, height=350)

            with st.expander("Full table"):
                st.dataframe(_friendly_card_table(opp_card_df[opp_card_df["games_with_card"] >= min_go]), use_container_width=True)

    with tab4:
        st.subheader("Card Pair Synergy")
        st.caption(
            "Pairs of cards you play together in the same game. "
            "**Win Rate Boost** = how much higher your win rate is with both cards vs your baseline. "
            "Positive = this combo correlates with winning. Negative = playing these together correlates with losing."
        )

        if pair_df.empty:
            st.info("No card pair data available.")
        else:
            min_gp = st.slider("Minimum games", 3, 20, 5, key="pair_min")
            n_pair = st.slider("Top N pairs", 5, 30, 15, key="pair_n")
            filtered_pairs = pair_df[pair_df["games"] >= min_gp]

            col_pos, col_neg = st.columns(2)
            with col_pos:
                st.markdown("**Best combos (highest win boost)**")
                top_pos = filtered_pairs.head(n_pair).copy()
                top_pos["Pair"] = top_pos["card_name_a"].str[:22] + " + " + top_pos["card_name_b"].str[:22]
                top_pos["Win Rate"] = top_pos["win_rate"].map(pct)
                top_pos["Win Rate Boost"] = top_pos["win_lift_vs_baseline"].map(fmt_lift)
                st.dataframe(top_pos[["Pair", "games", "Win Rate", "Win Rate Boost"]].rename(columns={"games": "Games"}).reset_index(drop=True), use_container_width=True)
            with col_neg:
                st.markdown("**Worst combos (biggest win rate drag)**")
                top_neg = filtered_pairs.tail(n_pair).iloc[::-1].copy()
                top_neg["Pair"] = top_neg["card_name_a"].str[:22] + " + " + top_neg["card_name_b"].str[:22]
                top_neg["Win Rate"] = top_neg["win_rate"].map(pct)
                top_neg["Win Rate Boost"] = top_neg["win_lift_vs_baseline"].map(fmt_lift)
                st.dataframe(top_neg[["Pair", "games", "Win Rate", "Win Rate Boost"]].rename(columns={"games": "Games"}).reset_index(drop=True), use_container_width=True)

    with tab5:
        st.subheader("Opening Hand Loss Correlation")
        st.caption(
            "Cards that were in your starting hand in games you lost more than average. "
            "**vs Baseline** = extra loss rate when this card is in your opener. "
            "Consider whether to mulligan aggressively when you see these cards."
        )

        if opening_df.empty:
            st.info("No opening hand data available.")
        else:
            min_g = st.slider("Minimum games", 3, 20, 5, key="oh_min")
            n_oh = st.slider("Top N", 5, 30, 15, key="oh_n")
            filtered_oh = opening_df[opening_df["games_with_card"] >= min_g].head(n_oh)
            chart_oh = filtered_oh.set_index("card_name")[["loss_lift_vs_baseline"]].rename(
                columns={"loss_lift_vs_baseline": "Loss Rate vs Baseline"}
            )
            st.bar_chart(chart_oh, height=300)

            with st.expander("Full table"):
                st.dataframe(_friendly_card_table(opening_df[opening_df["games_with_card"] >= min_g]), use_container_width=True)

    with tab6:
        st.subheader("Event Type Patterns: Wins vs Losses")
        st.caption(
            "Difference in how often each action type occurs in losses vs wins. "
            "Positive = you do this more in games you lose. Negative = you do it more in games you win."
        )

        if event_type_df.empty:
            st.info("No event type data available.")
        else:
            et = event_type_df.copy()
            et["diff"] = et["avg_count_in_losses"] - et["avg_count_in_wins"]
            et = et.sort_values("diff", ascending=False)
            chart_et = et.set_index("event_type")[["diff"]].rename(columns={"diff": "Avg Count Difference (Loss − Win)"})
            st.bar_chart(chart_et, height=300)
            with st.expander("Full table"):
                et_display = et.rename(columns={
                    "event_type": "Event Type",
                    "games_with_event": "Games",
                    "avg_count_in_losses": "Avg in Losses",
                    "avg_count_in_wins": "Avg in Wins",
                    "loss_minus_win_count": "Difference",
                })
                st.dataframe(et_display, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════
# PAGE: Archetypes & Matchups
# ════════════════════════════════════════════════════════════════════════════
elif page == "🏰 Archetypes & Matchups":
    st.title("🏰 Archetypes & Matchups")

    tab1, tab2, tab3 = st.tabs(["Archetype Win Rates", "Matchup Heatmap", "Deck Versions"])

    with tab1:
        st.subheader("Win Rate by Archetype and Turn Order")

        if arch_df.empty:
            st.info("No archetype data. Run `analyze.py` first.")
        else:
            min_g = st.slider("Min games (All)", 3, 20, 5, key="arch_min")
            filtered_arch = arch_df.copy()

            to_show = st.selectbox("Turn order", ["All", "OTP", "OTD"], key="arch_to")
            subset = filtered_arch[filtered_arch["turn_order"] == to_show]
            subset = subset[subset["games"] >= min_g]

            if subset.empty:
                st.info("No archetypes meet the minimum game threshold.")
            else:
                subset = subset.sort_values("win_rate", ascending=False)
                chart_data = subset.set_index("archetype")[["win_rate"]].copy()
                chart_data["win_rate"] = (chart_data["win_rate"] * 100).round(1)
                st.bar_chart(chart_data, height=300)
                st.dataframe(
                    subset[["archetype", "games", "wins", "losses", "win_rate"]]
                    .assign(win_rate=lambda d: d["win_rate"].map(pct)),
                    use_container_width=True,
                )

    with tab2:
        st.subheader("Matchup Win Rate Heatmap")
        st.caption("Win rate for each (my archetype × opponent colors) matchup.")

        if matchup_df.empty:
            st.info("No matchup data available.")
        else:
            try:
                import matplotlib.pyplot as plt
                import seaborn as sns

                min_g_m = st.slider("Min games per matchup", 3, 15, 5, key="matchup_min")
                filtered_m = matchup_df[matchup_df["games"] >= min_g_m]

                if filtered_m.empty:
                    st.info("No matchups meet the minimum game threshold.")
                else:
                    pivot = filtered_m.pivot_table(
                        index="archetype", columns="opponent_colors", values="win_rate"
                    )
                    fig, ax = plt.subplots(figsize=(max(8, pivot.shape[1] * 1.5), max(4, pivot.shape[0])))
                    sns.heatmap(
                        pivot, annot=True, fmt=".0%", cmap="RdYlGn",
                        vmin=0, vmax=1, linewidths=0.5, ax=ax,
                        cbar_kws={"label": "Win Rate"},
                    )
                    ax.set_title("Matchup Win Rate")
                    ax.set_xlabel("Opponent Colors")
                    ax.set_ylabel("My Archetype")
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()

                with st.expander("Raw matchup table"):
                    d = matchup_df.copy()
                    d["win_rate"] = d["win_rate"].map(pct)
                    st.dataframe(d.sort_values("games", ascending=False), use_container_width=True)
            except ImportError:
                st.warning("matplotlib/seaborn not available. Showing table instead.")
                st.dataframe(matchup_df, use_container_width=True)

    with tab3:
        st.subheader("Win Rate by Deck Version")
        st.caption("Each unique decklist hash gets its own row — useful for tracking changes over time.")

        if deck_ver_df.empty:
            st.info("No deck version data available.")
        else:
            to_dv = st.selectbox("Turn order", ["All", "OTP", "OTD"], key="dv_to")
            subset_dv = deck_ver_df[deck_ver_df["turn_order"] == to_dv].sort_values("win_rate", ascending=False)
            subset_dv = subset_dv.copy()
            subset_dv["win_rate"] = subset_dv["win_rate"].map(pct)
            st.dataframe(subset_dv, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════
# PAGE: Sequencing & Tempo
# ════════════════════════════════════════════════════════════════════════════
elif page == "⚔️ Sequencing & Tempo":
    st.title("⚔️ Sequencing & Tempo")

    tab1, tab2, tab3 = st.tabs(["Action Tempo Curves", "Early Inking", "First Quest Turn"])

    with tab1:
        st.subheader("Average Actions per Turn: Win vs Loss")
        st.caption("Higher is generally better — but look for where wins and losses diverge.")

        if tempo_df.empty:
            st.info("No tempo data. Run `analyze.py` first.")
        else:
            event_choices = sorted(tempo_df["event_type"].unique())
            selected_event = st.selectbox("Event type", event_choices, key="tempo_event")
            subset_t = tempo_df[tempo_df["event_type"] == selected_event]

            win_t = subset_t[subset_t["outcome"] == "Win"].set_index("turn_number")["avg_count"]
            loss_t = subset_t[subset_t["outcome"] == "Loss"].set_index("turn_number")["avg_count"]

            combined = pd.DataFrame({"Win": win_t, "Loss": loss_t}).fillna(0)
            st.line_chart(combined, height=320)
            st.caption("X axis = turn number, Y axis = avg count per game")

    with tab2:
        st.subheader("Inking Rate by Turn (Turns 1–4)")
        st.caption("How often did you ink on each of the first four turns?")

        if ink_df.empty:
            st.info("No early ink data available.")
        else:
            win_ink = ink_df[ink_df["outcome"] == "Win"].set_index("turn")["ink_rate"]
            loss_ink = ink_df[ink_df["outcome"] == "Loss"].set_index("turn")["ink_rate"]
            combined_ink = pd.DataFrame({"Win": win_ink, "Loss": loss_ink}).fillna(0)
            st.bar_chart(combined_ink, height=300)

            with st.expander("Raw data"):
                d = ink_df.copy()
                d["ink_rate"] = d["ink_rate"].map(pct)
                st.dataframe(d, use_container_width=True)

    with tab3:
        st.subheader("First Quest Turn Distribution")
        st.caption("Which turn did you first quest a character? Earlier quests are generally better.")

        if fq_df.empty:
            st.info("No first quest data available.")
        else:
            relevant = fq_df[fq_df["first_quest_turn"] > 0].copy()
            if relevant.empty:
                st.info("No quest events found in replay data.")
            else:
                win_fq = relevant[relevant["outcome"] == "Win"].set_index("first_quest_turn")["pct"]
                loss_fq = relevant[relevant["outcome"] == "Loss"].set_index("first_quest_turn")["pct"]
                combined_fq = pd.DataFrame({"Win": win_fq, "Loss": loss_fq}).fillna(0)
                st.bar_chart(combined_fq, height=300)
                st.caption("Y axis = % of games where first quest happened on that turn")

                with st.expander("Raw data"):
                    d = relevant.copy()
                    d["pct"] = d["pct"].map(pct)
                    st.dataframe(d, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════
# PAGE: Lore & Mulligans
# ════════════════════════════════════════════════════════════════════════════
elif page == "📈 Lore & Mulligans":
    st.title("📈 Lore & Mulligans")

    tab1, tab2 = st.tabs(["Lore Curves", "Mulligan Impact"])

    with tab1:
        st.subheader("Lore Gained per Turn: Win vs Loss")
        st.caption("How much lore you average per turn and cumulatively. A gap between Win and Loss lines shows where the game was decided.")

        if lore_df.empty:
            st.info("No lore data found in replays. LORE_GAINED events may not be present.")
        else:
            view = st.radio("View", ["Per-turn", "Cumulative"], horizontal=True, key="lore_view")
            col = "avg_lore_per_turn" if view == "Per-turn" else "cumulative_avg_lore"

            win_l = lore_df[lore_df["outcome"] == "Win"].set_index("turn_number")[col]
            loss_l = lore_df[lore_df["outcome"] == "Loss"].set_index("turn_number")[col]
            combined_lore = pd.DataFrame({"Win": win_l, "Loss": loss_l}).fillna(0)
            st.line_chart(combined_lore, height=320)

            with st.expander("Raw data"):
                st.dataframe(lore_df, use_container_width=True)

    with tab2:
        st.subheader("Win Rate by Number of Mulligans")
        st.caption("Did mulliganing help or hurt? 0 = kept opening hand.")

        if mulligan_df.empty:
            st.info("No mulligan data available (requires matched replays).")
        else:
            to_mull = st.selectbox("Turn order", ["All", "OTP", "OTD"], key="mull_to")
            subset_m = mulligan_df[mulligan_df["turn_order"] == to_mull].sort_values("mulligans")

            if subset_m.empty:
                st.info("No data for selected turn order.")
            else:
                chart_m = subset_m.set_index("mulligans")[["win_rate"]].copy()
                chart_m["win_rate"] = (chart_m["win_rate"] * 100).round(1)
                st.bar_chart(chart_m, height=280)
                st.caption("X = mulligans taken, Y = win rate %")

                with st.expander("Full table"):
                    d = subset_m.copy()
                    d["win_rate"] = d["win_rate"].map(pct)
                    st.dataframe(d, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════
# PAGE: Deck Advisor
# ════════════════════════════════════════════════════════════════════════════
elif page == "🧙 Deck Advisor":
    st.title("🧙 Deck Advisor")
    st.caption(
        "Paste your decklist below to get personalised recommendations based on your game history. "
        "Supported formats: **Dreamborn** (4x Card Name) or **plain copy-paste** (4 Card Name). "
        "Recommendations are only as good as your data — cards need at least 3 games to show up."
    )

    # ── decklist input ──────────────────────────────────────────────────────
    sample_hint = "4x Oswald the Lucky Rabbit - Favorite Character\n2x Be Our Guest\n4x Genie - The Ever Impressive"
    raw_input = st.text_area(
        "Paste your decklist here",
        height=220,
        placeholder=sample_hint,
        key="decklist_input",
    )

    def parse_paste(text: str) -> dict[str, int]:
        """Parse Dreamborn / plain-text decklists into {card_name: qty}."""
        import re
        cards: dict[str, int] = {}
        for line in text.strip().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            # "4x Card Name" or "4 Card Name"
            m = re.match(r"^(\d+)\s*[xX]?\s+(.+)$", line)
            if m:
                qty, name = int(m.group(1)), m.group(2).strip()
                cards[name] = cards.get(name, 0) + qty
        return cards

    if not raw_input.strip():
        st.info("Paste your decklist above to get started.")
        st.stop()

    decklist = parse_paste(raw_input)
    if not decklist:
        st.warning("Could not parse the decklist. Make sure each line looks like: `4x Card Name`")
        st.stop()

    deck_names = set(decklist.keys())
    total_cards = sum(decklist.values())
    st.success(f"Parsed **{len(decklist)} unique cards** ({total_cards} total) from your decklist.")

    # ── filter analytics to decklist cards ─────────────────────────────────
    # Case-insensitive fuzzy match so minor spacing differences don't break things
    def match_cards(df: pd.DataFrame, name_col: str = "card_name") -> pd.DataFrame:
        if df.empty or name_col not in df.columns:
            return df.iloc[0:0]
        mask = df[name_col].str.lower().isin({n.lower() for n in deck_names})
        return df[mask].copy()

    dl_card = match_cards(card_df)
    dl_sp = match_cards(seen_played_df)
    dl_oh = match_cards(opening_df)

    # overall baseline
    baseline_loss = games_df["is_loss"].mean() if not games_df.empty and "is_loss" in games_df.columns else 0.5

    # ── turn-cost data from events ──────────────────────────────────────────
    @st.cache_data(ttl=300)
    def _deck_turn_data(deck_names_tuple: tuple) -> pd.DataFrame:
        """Return raw play events for deck cards. Join game metadata OUTSIDE to respect sidebar filters."""
        if events_df.empty:
            return pd.DataFrame()
        played = events_df[
            events_df["is_my_event"]
            & (events_df["event_type"] == "CARD_PLAYED")
            & events_df["card_name"].notna()
            & events_df["turn_number"].notna()
        ].copy()
        deck_set_lower = {n.lower() for n in deck_names_tuple}
        played = played[played["card_name"].str.lower().isin(deck_set_lower)].copy()
        return played

    _raw_turn_data = _deck_turn_data(tuple(sorted(deck_names)))
    # Join game metadata from the *current filtered* games_df — outside the cache so sidebar filters apply
    if not _raw_turn_data.empty:
        _meta_cols = [c for c in ["csv_row_id", "is_loss", "turn_order"] if c in games_df.columns]
        turn_data = _raw_turn_data.merge(games_df[_meta_cols], on="csv_row_id", how="inner")
    else:
        turn_data = pd.DataFrame()

    # ── Section 0: Research Assistant (ranked next steps) ──────────────────
    st.markdown("---")
    st.subheader("🧠 Research Assistant: Ranked Next Steps")
    st.caption(
        "Automatically prioritizes decklist, mulligan, sequencing, and matchup actions "
        "using your currently filtered games and this pasted decklist."
    )

    @st.cache_data(ttl=300)
    def _deck_recommendations(
        csv_row_ids_tuple: tuple,
        deck_names_tuple: tuple,
        min_card_games: int,
    ) -> pd.DataFrame:
        from src.lorcana_analyzer.advisor import build_recommendations
        from src.lorcana_analyzer.analytics import (
            card_loss_correlation,
            card_seen_vs_played_correlation,
            early_ink_rates,
            event_type_loss_correlation,
            first_quest_turn,
            matchup_win_rates,
            opening_hand_loss_correlation,
        )

        ids = set(csv_row_ids_tuple)
        g = games_df[games_df["csv_row_id"].isin(ids)].copy()
        e = events_df[events_df["csv_row_id"].isin(ids)].copy() if not events_df.empty else events_df

        card_corr = card_loss_correlation(g, e, min_games=min_card_games)
        seen_played = card_seen_vs_played_correlation(g, e, min_games=min_card_games)
        opening_corr = opening_hand_loss_correlation(g, min_games=min_card_games)
        event_corr = event_type_loss_correlation(g, e, min_games=5)
        ink_rates = early_ink_rates(g, e, turns=4)
        first_quest = first_quest_turn(g, e)
        matchup_rates = matchup_win_rates(g, min_games=8)

        deck_lower = {n.lower() for n in deck_names_tuple}

        def _filter_deck_cards(df: pd.DataFrame, col: str = "card_name") -> pd.DataFrame:
            if df.empty or col not in df.columns:
                return df.iloc[0:0]
            return df[df[col].str.lower().isin(deck_lower)].copy()

        return build_recommendations(
            games_df=g,
            card_df=_filter_deck_cards(card_corr),
            opening_df=_filter_deck_cards(opening_corr),
            event_type_df=event_corr,
            ink_df=ink_rates,
            fq_df=first_quest,
            matchup_df=matchup_rates,
            seen_played_df=_filter_deck_cards(seen_played),
            min_card_games=min_card_games,
        )

    rec_min_games = st.slider("Recommendation minimum card sample", 5, 30, 12, key="rec_min_games")

    if games_df.empty or "csv_row_id" not in games_df.columns:
        st.info("No filtered games available for recommendations.")
    else:
        rec_ids = tuple(sorted(games_df["csv_row_id"].dropna().astype(int).unique()))
        recommendations = _deck_recommendations(rec_ids, tuple(sorted(deck_names)), rec_min_games)

        if recommendations.empty:
            st.info("Not enough signal yet for ranked recommendations. Try lowering the minimum card sample.")
        else:
            top_n = st.slider("Show top recommendations", 5, 20, 10, key="rec_top_n")
            rec_disp = recommendations.head(top_n).copy()
            rec_disp = rec_disp.rename(
                columns={
                    "rank": "#",
                    "category": "Category",
                    "action": "Action",
                    "games": "Games",
                    "confidence": "Confidence",
                    "signal_score": "Signal",
                    "priority": "Priority",
                }
            )
            rec_disp["Signal"] = rec_disp["Signal"].map(lambda v: f"{v:+.1f}")
            rec_disp["Priority"] = rec_disp["Priority"].map(lambda v: f"{v:.1f}")
            st.dataframe(
                rec_disp[["#", "Category", "Action", "Games", "Confidence", "Signal", "Priority"]].reset_index(drop=True),
                use_container_width=True,
            )

            st.markdown("#### ⚗️ Experiment Builder")

            ids_set = set(rec_ids)
            events_filtered = events_df[events_df["csv_row_id"].isin(ids_set)].copy() if not events_df.empty else pd.DataFrame()

            def _avg_early_ink_rate(turn_max: int = 3) -> float | None:
                if events_filtered.empty:
                    return None
                ink = events_filtered[
                    events_filtered["is_my_event"]
                    & (events_filtered["event_type"] == "CARD_INKED")
                    & events_filtered["turn_number"].notna()
                    & (events_filtered["turn_number"] <= turn_max)
                ][["csv_row_id", "turn_number"]].drop_duplicates()
                if ink.empty:
                    return None
                game_count = max(len(games_df), 1)
                hits = len(ink)
                return hits / (game_count * turn_max)

            def _avg_first_quest_turn() -> float | None:
                if events_filtered.empty:
                    return None
                fq = events_filtered[
                    events_filtered["is_my_event"]
                    & (events_filtered["event_type"] == "CARD_QUEST")
                    & events_filtered["turn_number"].notna()
                ]
                if fq.empty:
                    return None
                first_q = fq.groupby("csv_row_id")["turn_number"].min()
                if first_q.empty:
                    return None
                return float(first_q.mean())

            def _avg_cards_played_t1_3() -> float | None:
                if events_filtered.empty:
                    return None
                plays = events_filtered[
                    events_filtered["is_my_event"]
                    & (events_filtered["event_type"] == "CARD_PLAYED")
                    & events_filtered["turn_number"].notna()
                    & (events_filtered["turn_number"] <= 3)
                ]
                if plays.empty:
                    return None
                counts = plays.groupby("csv_row_id").size()
                if counts.empty:
                    return None
                return float(counts.mean())

            def _parse_swaps() -> tuple[list[str], list[str]]:
                cuts: list[str] = []
                adds: list[str] = []
                for _, row in recommendations.iterrows():
                    if row.get("category") != "Decklist":
                        continue
                    action = str(row.get("action", ""))
                    if action.startswith("Run a 10-game experiment cutting 1-2 copies of "):
                        cuts.append(action.replace("Run a 10-game experiment cutting 1-2 copies of ", "").rstrip("."))
                    if action.startswith("Treat ") and " as a core keep" in action:
                        adds.append(action.replace("Treat ", "").split(" as a core keep", 1)[0].strip())
                # Deduplicate while preserving order
                cuts = list(dict.fromkeys(cuts))
                adds = list(dict.fromkeys(adds))
                return cuts[:2], adds[:2]

            cuts, adds = _parse_swaps()
            baseline_wr = 1 - games_df["is_loss"].mean() if ("is_loss" in games_df.columns and not games_df.empty) else None
            base_ink = _avg_early_ink_rate(3)
            base_fq = _avg_first_quest_turn()
            base_t13 = _avg_cards_played_t1_3()

            c_plan, c_swaps, c_score = st.columns(3)

            with c_plan:
                if st.button("Create A/B Checklist", key="build_ab_checklist", use_container_width=True):
                    lines = [
                        "### A/B Block Checklist",
                        "",
                        "**Block A (10-15 games):**",
                        "- Current list, no card changes.",
                        "- Keep mulligan policy fixed.",
                        "- Track matchup notes and OTP/OTD.",
                        "",
                        "**Block B (10-15 games):**",
                        "- Apply exactly one decklist change and one sequencing focus.",
                        "- Keep everything else unchanged.",
                        "- Re-run analysis and compare to Block A.",
                        "",
                        "**Decision rule:** Keep changes only if win rate improves and at least one process metric improves.",
                    ]
                    st.session_state["ab_checklist_md"] = "\n".join(lines)

            with c_swaps:
                if st.button("Create Suggested Swaps", key="build_swaps", use_container_width=True):
                    lines = ["### Suggested Test Swaps", ""]
                    if cuts:
                        lines.append("**Potential cuts (start with -1 copy each):**")
                        for name in cuts:
                            lines.append(f"- {name}")
                    else:
                        lines.append("- No high-confidence cut candidates yet.")

                    lines.append("")
                    if adds:
                        lines.append("**Potential keeps/add priorities:**")
                        for name in adds:
                            lines.append(f"- {name}")
                    else:
                        lines.append("- No high-confidence keep priorities yet.")

                    lines.append("")
                    lines.append("**Test format:** 10-15 games with only one swap package, then evaluate.")
                    st.session_state["swap_plan_md"] = "\n".join(lines)

            with c_score:
                if st.button("Create Scorecard Template", key="build_scorecard", use_container_width=True):
                    lines = [
                        "### Post-Block Scorecard",
                        "",
                        "| Metric | Baseline | Block Result | Change |",
                        "|---|---:|---:|---:|",
                        f"| Win rate | {pct(baseline_wr) if baseline_wr is not None else 'N/A'} |  |  |",
                        f"| Avg first quest turn (lower better) | {f'{base_fq:.2f}' if base_fq is not None else 'N/A'} |  |  |",
                        f"| Early ink rate T1-3 | {pct(base_ink) if base_ink is not None else 'N/A'} |  |  |",
                        f"| Avg cards played by turn 3 | {f'{base_t13:.2f}' if base_t13 is not None else 'N/A'} |  |  |",
                        "",
                        "**Keep/Reject rule:** Keep the experiment only if win rate increases and at least one process metric improves.",
                    ]
                    st.session_state["scorecard_md"] = "\n".join(lines)

            if st.session_state.get("ab_checklist_md"):
                st.markdown(st.session_state["ab_checklist_md"])
            if st.session_state.get("swap_plan_md"):
                st.markdown(st.session_state["swap_plan_md"])
            if st.session_state.get("scorecard_md"):
                st.markdown(st.session_state["scorecard_md"])

            with st.expander("How to use these in your next matches"):
                st.markdown("- Pick 1 decklist change + 1 sequencing change from the top 5.")
                st.markdown("- Play a 10-15 game block with those changes only.")
                st.markdown("- Re-run analysis and keep changes only if both win rate and process metrics improve.")

    # ── Section 1: Cards to consider removing ──────────────────────────────
    st.markdown("---")
    st.subheader("❌ Cards to Consider Removing")
    st.caption(
        "Sorted by loss correlation — cards you interact with most often in games you lose. "
        "Also considers low play-through rate (drawing but not playing) as a dead-card signal."
    )

    if dl_card.empty:
        st.info("Not enough data yet for these deck cards (need ≥ 3 games per card).")
    else:
        remove_candidates = dl_card.copy()
        remove_candidates = remove_candidates.sort_values("loss_lift_vs_baseline", ascending=False)

        # merge in seen/played info
        if not dl_sp.empty:
            remove_candidates = remove_candidates.merge(
                dl_sp[["card_name", "play_through_rate", "loss_rate_played", "loss_lift_when_played"]],
                on="card_name", how="left"
            )

        for _, row in remove_candidates.head(8).iterrows():
            lift = row.get("loss_lift_vs_baseline", 0)
            lr = row.get("loss_rate_with_card", baseline_loss)
            games = int(row.get("games_with_card", 0))
            ptr = row.get("play_through_rate", None)
            lift_played = row.get("loss_lift_when_played", None)

            if lift <= 0 and (pd.isna(ptr) or ptr > 0.5):
                continue  # skip neutral/positive cards

            reasons = []
            if lift > 0.08:
                reasons.append(f"loss rate {pct(lr)} (+{pct(lift)} vs your average)")
            if not pd.isna(ptr) and ptr < 0.4:
                reasons.append(f"only played {pct(ptr)} of the time when drawn (dead card in hand)")
            if not pd.isna(lift_played) and lift_played > 0.1:
                reasons.append(f"loss rate jumps {fmt_lift(lift_played)} specifically when you play it")

            if not reasons:
                continue

            qty = decklist.get(row["card_name"], "?")
            with st.expander(f"**{qty}x {row['card_name']}** — {games} games tracked"):
                for r in reasons:
                    st.markdown(f"- {r}")

    # ── Section 2: Cards to keep (strong performers) ────────────────────────
    st.markdown("---")
    st.subheader("✅ Cards to Keep")
    st.caption("Cards in your deck that correlate with wins (negative loss lift = you win more with them).")

    if dl_card.empty:
        st.info("Not enough data yet.")
    else:
        keepers = dl_card[dl_card["loss_lift_vs_baseline"] < -0.02].sort_values("loss_lift_vs_baseline", ascending=True)

        if keepers.empty:
            st.info("No cards with a strong win correlation yet — keep playing!")
        else:
            keep_display = keepers[["card_name", "games_with_card", "loss_rate_with_card", "loss_lift_vs_baseline"]].copy()
            keep_display = keep_display.rename(columns={
                "card_name": "Card",
                "games_with_card": "Games",
                "loss_rate_with_card": "Loss Rate",
                "loss_lift_vs_baseline": "vs Baseline",
            })
            keep_display["Loss Rate"] = keep_display["Loss Rate"].map(pct)
            keep_display["vs Baseline"] = keep_display["vs Baseline"].map(fmt_lift)
            st.dataframe(keep_display.reset_index(drop=True), use_container_width=True)

    # ── Section 3: OTP vs OTD tuning ────────────────────────────────────────
    st.markdown("---")
    st.subheader("🔄 OTP vs OTD: Cards that Play Differently")
    st.caption(
        "Some cards are great on the play (OTP) but weak on the draw (OTD), or vice versa. "
        "This compares your loss rate with each card split by turn order."
    )

    if events_df.empty or games_df.empty:
        st.info("Need replay data to compute OTP/OTD breakdowns.")
    elif not turn_data.empty and "turn_order" in games_df.columns:
        otp_otd_rows = []
        for name in sorted(deck_names):
            name_lower = name.lower()
            sub = turn_data[turn_data["card_name"].str.lower() == name_lower]
            if sub.empty:
                continue
            sub2 = sub.merge(games_df[["csv_row_id", "turn_order"]], on="csv_row_id", how="left")
            for to in ["OTP", "OTD"]:
                g = sub2[sub2["turn_order"] == to]
                if len(g["csv_row_id"].unique()) < 3:
                    continue
                lr = g.groupby("csv_row_id")["is_loss"].first().mean()
                otp_otd_rows.append({"Card": name, "Turn Order": to, "Games": len(g["csv_row_id"].unique()), "Loss Rate": lr})

        if not otp_otd_rows:
            st.info("Not enough per-turn-order data yet (need ≥ 3 games per card per turn order).")
        else:
            otp_df_disp = pd.DataFrame(otp_otd_rows)
            pivot = otp_df_disp.pivot(index="Card", columns="Turn Order", values="Loss Rate")
            if "OTP" in pivot.columns and "OTD" in pivot.columns:
                pivot["OTP vs OTD gap"] = (pivot["OTP"] - pivot["OTD"]).abs()
                pivot = pivot.sort_values("OTP vs OTD gap", ascending=False).dropna(subset=["OTP", "OTD"])
                pivot_disp = pivot.copy()
                for col in ["OTP", "OTD", "OTP vs OTD gap"]:
                    if col in pivot_disp.columns:
                        pivot_disp[col] = pivot_disp[col].map(pct)
                st.dataframe(pivot_disp, use_container_width=True)
                st.caption(
                    "Large gap = this card matters more in one mode. "
                    "High OTP loss rate → consider mulliganing it when going first. "
                    "High OTD loss rate → might be too slow when behind on tempo."
                )
            else:
                st.dataframe(otp_df_disp, use_container_width=True)
    else:
        st.info("Need turn order data in replays.")

    # ── Section 4: Ideal turn 1-6 curve ─────────────────────────────────────
    st.markdown("---")
    st.subheader("📅 Ideal Turn 1–6 Play Curve")
    st.caption(
        "Shows which cards in your deck you actually played on each turn 1–6, "
        "and whether playing each card earlier or later is associated with winning. "
        "Win games vs loss games are compared to find timing signals."
    )

    if turn_data.empty:
        st.info("No turn-by-turn play data available for this deck.")
    else:
        # Card timing: for each deck card, avg turn first played in wins vs losses
        first_play = (
            turn_data.groupby(["csv_row_id", "card_name", "is_loss"])["turn_number"]
            .min()
            .reset_index()
            .rename(columns={"turn_number": "first_play_turn"})
        )
        timing = (
            first_play.groupby(["card_name", "is_loss"])
            .agg(avg_turn=("first_play_turn", "mean"), games=("csv_row_id", "nunique"))
            .reset_index()
        )
        wins_timing = timing[~timing["is_loss"]].rename(columns={"avg_turn": "avg_turn_wins", "games": "games_wins"}).drop(columns="is_loss")
        loss_timing = timing[timing["is_loss"]].rename(columns={"avg_turn": "avg_turn_losses", "games": "games_losses"}).drop(columns="is_loss")
        card_timing = wins_timing.merge(loss_timing, on="card_name", how="outer")
        card_timing["turn_diff"] = card_timing["avg_turn_wins"] - card_timing["avg_turn_losses"]
        # Require at least 5 games in either wins or losses to show
        min_games_curve = st.slider("Minimum games played (wins + losses)", 3, 20, 5, key="curve_min")
        card_timing = card_timing[
            (card_timing["games_wins"].fillna(0) + card_timing["games_losses"].fillna(0)) >= min_games_curve
        ].sort_values("turn_diff", ascending=True)

        if card_timing.empty:
            st.info("Not enough data for card timing. Try lowering the minimum games threshold.")
        else:
            st.caption(
                "**Turn diff = avg turn (wins) − avg turn (losses).** "
                "Negative = you play this card *earlier* in games you win → strong signal to enable early. "
                "Positive = you play it later in wins → less time-sensitive. "
                "Large negative values are the most actionable."
            )
            disp = card_timing.copy()
            disp["Avg Turn (Wins)"] = disp["avg_turn_wins"].map(lambda v: f"{v:.1f}" if pd.notna(v) else "—")
            disp["Avg Turn (Losses)"] = disp["avg_turn_losses"].map(lambda v: f"{v:.1f}" if pd.notna(v) else "—")
            disp["Turn Diff"] = disp["turn_diff"].map(lambda v: f"{v:+.1f}" if pd.notna(v) else "—")
            disp["Games (W/L)"] = disp.apply(
                lambda r: f"{int(r['games_wins']) if pd.notna(r['games_wins']) else 0} / {int(r['games_losses']) if pd.notna(r['games_losses']) else 0}",
                axis=1
            )
            st.dataframe(
                disp[["card_name", "Avg Turn (Wins)", "Avg Turn (Losses)", "Turn Diff", "Games (W/L)"]]
                .rename(columns={"card_name": "Card"})
                .reset_index(drop=True),
                use_container_width=True,
            )

            # Bar chart of turn diff
            chart_td = disp.set_index("card_name")[["turn_diff"]].dropna().rename(columns={"turn_diff": "Turn Diff (Wins − Losses)"})
            st.bar_chart(chart_td, height=320)

        st.markdown("---")
        st.caption(
            "Cards where you play them earlier in wins are the highest-value early plays in your deck. "
            "Cards with near-zero diff are fine at any point. "
            "Remember: correlation, not causation."
        )

    # ── Section 5: Opening hand signals ──────────────────────────────────────
    st.markdown("---")
    st.subheader("🃏 Opening Hand: Keep or Mulligan?")
    st.caption("Based on your opening hand loss correlation, which deck cards hurt you most in the opener.")

    if dl_oh.empty:
        st.info("Not enough opening hand data for this deck yet.")
    else:
        oh_disp = dl_oh.sort_values("loss_lift_vs_baseline", ascending=False).copy()
        oh_disp["Loss Rate (Opener)"] = oh_disp["loss_rate_with_card"].map(pct)
        oh_disp["vs Baseline"] = oh_disp["loss_lift_vs_baseline"].map(fmt_lift)
        oh_disp["Recommendation"] = oh_disp["loss_lift_vs_baseline"].apply(
            lambda x: "🔴 Consider mulliganing" if x > 0.10 else ("🟢 Safe to keep" if x < -0.05 else "⚪ Neutral")
        )
        st.dataframe(
            oh_disp[["card_name", "games_with_card", "Loss Rate (Opener)", "vs Baseline", "Recommendation"]].rename(
                columns={"card_name": "Card", "games_with_card": "Games"}
            ).reset_index(drop=True),
            use_container_width=True,
        )


# ════════════════════════════════════════════════════════════════════════════
# PAGE: Simulation Lab
# ════════════════════════════════════════════════════════════════════════════
elif page == "🎲 Simulation Lab":
    st.title("🎲 Simulation Lab")
    st.caption(
        "Estimate win rates for any decklist, compare two builds, and stress-test against "
        "the opponent sequences that beat you in real games. "
        "**How it works:** a Monte Carlo engine draws thousands of random opening hands from your deck "
        "and weights each hand by the card-level loss correlations measured from your game history. "
        "A logistic regression model trained on your actual games adds a second estimate. "
        "With ~200 games, treat outputs as **directional signals**, not precise numbers."
    )

    # ── Load simulator lazily ──────────────────────────────────────────────
    @st.cache_resource
    def _get_model():
        from src.lorcana_analyzer.simulator import train_loss_model
        g = load("clean_games.csv")
        e = load("clean_events.csv")
        c = load("card_loss_correlation.csv")
        return train_loss_model(g, e, c)

    sim_model = _get_model()

    # ── Shared decklist parser (same as Deck Advisor) ─────────────────────
    import re as _re

    def _parse_sim_deck(text: str) -> dict[str, int]:
        cards: dict[str, int] = {}
        for line in text.strip().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            m = _re.match(r"^(\d+)\s*[xX]?\s+(.+)$", line)
            if m:
                qty, name = int(m.group(1)), m.group(2).strip()
                cards[name] = cards.get(name, 0) + qty
        return cards

    tab1, tab2, tab3 = st.tabs(
        ["🎯 Win Rate Estimator", "⚖️ A vs B Comparison", "🏹 Opponent Stress Test"]
    )

    # ── Tab 1: Win Rate Estimator ─────────────────────────────────────────
    with tab1:
        st.subheader("Win Rate Estimator")
        st.caption("Paste a decklist and run simulations to estimate projected win rate.")

        deck_text_sim = st.text_area(
            "Decklist (4x Card Name format)",
            height=200,
            key="sim_deck_input",
            placeholder="4x Oswald the Lucky Rabbit - Favorite Character\n2x Be Our Guest",
        )

        col_sims, col_to = st.columns(2)
        n_sims = col_sims.select_slider("Simulations", [500, 1000, 2000, 5000], value=2000)
        turn_order_sim = col_to.radio("Turn order assumption", ["OTP", "OTD", "Both"], horizontal=True, key="sim_to")
        run_sim = st.button("▶ Run Simulation", type="primary", key="run_sim")

        if run_sim:
            deck_sim = _parse_sim_deck(deck_text_sim)
            if not deck_sim:
                st.warning("Could not parse decklist.")
            else:
                from src.lorcana_analyzer.simulator import simulate_deck, predict_deck_loss_prob
                with st.spinner(f"Running {n_sims:,} simulations..."):
                    result = simulate_deck(
                        deck_sim, games_df, card_df, seen_played_df, n_sims=n_sims
                    )

                wr = result["win_rate"]
                base = result["baseline_win_rate"]
                delta = wr - base

                st.markdown("### Results")
                c1, c2, c3 = st.columns(3)
                c1.metric("Projected Win Rate", pct(wr), delta=fmt_lift(delta) + " vs your baseline")
                c2.metric("95% Confidence Interval", f"{pct(result['ci_low'])} – {pct(result['ci_high'])}")
                c3.metric("Your Historical Baseline", pct(base))

                # Logistic model second opinion
                if sim_model:
                    to_list = ["OTP", "OTD"] if turn_order_sim == "Both" else [turn_order_sim]
                    lp_rows = []
                    for to in to_list:
                        lp = predict_deck_loss_prob(deck_sim, sim_model, card_df, turn_order=to)
                        lp_rows.append({"Turn Order": to, "Model Loss Prob": pct(lp), "Model Win Prob": pct(1 - lp)})
                    st.markdown("**Logistic Model Estimate** *(trained on your game features)*")
                    st.dataframe(pd.DataFrame(lp_rows), use_container_width=True, hide_index=True)
                    acc = sim_model["train_accuracy"]
                    st.caption(
                        f"Model trained on {sim_model['n_games']} games, training accuracy {pct(acc)}. "
                        "Most important features: opening hand quality, early plays, turn order."
                    )

                st.markdown("---")
                st.subheader("Per-Card Impact on Win Rate")
                st.caption(
                    "**WR Drag** = how much this card (weighted by copies) pulls your projected win rate down. "
                    "Positive drag = hurts you. Negative drag = helps you."
                )
                pcd = result["per_card_df"].copy()
                pcd["loss_lift"] = pcd["loss_lift"].map(fmt_lift)
                pcd["projected_wr_drag"] = pcd["projected_wr_drag"].map(fmt_lift)
                pcd = pcd.rename(columns={
                    "card_name": "Card", "qty": "Copies",
                    "deck_weight": "Deck Weight",
                    "loss_lift": "Loss Lift", "projected_wr_drag": "WR Drag"
                })
                st.dataframe(pcd.reset_index(drop=True), use_container_width=True)

    # ── Tab 2: A vs B Comparison ─────────────────────────────────────────
    with tab2:
        st.subheader("Deck A vs Deck B")
        st.caption(
            "Compare your current deck against a modified version. "
            "Swap a few cards in Deck B to see if the change is projected to help."
        )

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("**Deck A (current build)**")
            text_a = st.text_area("", height=260, key="deck_a_text",
                                  placeholder="4x Card Name\n...")
        with col_b:
            st.markdown("**Deck B (modified build)**")
            text_b = st.text_area("", height=260, key="deck_b_text",
                                  placeholder="4x Card Name\n...")

        n_sims_ab = st.select_slider("Simulations per deck", [500, 1000, 2000], value=1000, key="ab_sims")
        run_ab = st.button("▶ Compare Decks", type="primary", key="run_ab")

        if run_ab:
            deck_a = _parse_sim_deck(text_a)
            deck_b = _parse_sim_deck(text_b)
            if not deck_a or not deck_b:
                st.warning("Please paste both decklists.")
            else:
                from src.lorcana_analyzer.simulator import compare_decklists
                with st.spinner("Simulating both decks..."):
                    summary, res_a, res_b = compare_decklists(
                        deck_a, deck_b, games_df, card_df, seen_played_df, n_sims=n_sims_ab
                    )

                st.markdown("### Summary")
                c1, c2, c3 = st.columns(3)
                c1.metric("Deck A Win Rate", pct(res_a["win_rate"]),
                          delta=fmt_lift(res_a["win_rate"] - res_a["baseline_win_rate"]) + " vs baseline")
                c2.metric("Deck B Win Rate", pct(res_b["win_rate"]),
                          delta=fmt_lift(res_b["win_rate"] - res_b["baseline_win_rate"]) + " vs baseline")
                diff = res_b["win_rate"] - res_a["win_rate"]
                c3.metric("B over A", fmt_lift(diff),
                          delta="↑ B is projected better" if diff > 0 else "↓ B is projected worse",
                          delta_color="normal" if diff > 0 else "inverse")

                st.caption(
                    "If both confidence intervals overlap, the difference is likely noise. "
                    f"A: {pct(res_a['ci_low'])}–{pct(res_a['ci_high'])}  |  "
                    f"B: {pct(res_b['ci_low'])}–{pct(res_b['ci_high'])}"
                )

                # Show diff between decks
                all_cards = set(deck_a) | set(deck_b)
                diff_rows = []
                for c in sorted(all_cards):
                    qa = deck_a.get(c, 0)
                    qb = deck_b.get(c, 0)
                    if qa != qb:
                        diff_rows.append({"Card": c, "Deck A": qa, "Deck B": qb, "Change": qb - qa})
                if diff_rows:
                    st.markdown("**Cards changed between builds**")
                    st.dataframe(pd.DataFrame(diff_rows), use_container_width=True, hide_index=True)

    # ── Tab 3: Opponent Stress Test ───────────────────────────────────────
    with tab3:
        st.subheader("Stress Test vs Your Historical Losses")
        st.caption(
            "Takes the opponent card sequences from every game you lost in your history, "
            "and simulates your deck against each one. "
            "**Low projected win prob** = this opponent's opening is a tough matchup for your deck. "
            "Use this to identify specific archetypes to build against."
        )

        stress_deck_text = st.text_area(
            "Decklist to test (4x Card Name)",
            height=200,
            key="stress_deck_text",
            placeholder="4x Card Name\n...",
        )
        n_sims_stress = st.select_slider("Simulations per matchup", [100, 200, 500], value=200, key="stress_sims")
        max_turn_stress = st.slider("Opponent turns to consider (1–N)", 3, 8, 6, key="stress_turns")
        run_stress = st.button("▶ Run Stress Test", type="primary", key="run_stress")

        if run_stress:
            stress_deck = _parse_sim_deck(stress_deck_text)
            if not stress_deck:
                st.warning("Could not parse decklist.")
            else:
                from src.lorcana_analyzer.simulator import stress_test_deck
                with st.spinner("Simulating against all historical losses... (may take ~30 seconds)"):
                    st_df = stress_test_deck(
                        stress_deck,
                        games_df,
                        events_df if not events_df.empty else pd.DataFrame(),
                        card_df,
                        pair_df,
                        n_sims_per_game=n_sims_stress,
                        max_early_turn=max_turn_stress,
                    )

                if st_df.empty:
                    st.info("No opponent sequence data found. Make sure replays are loaded.")
                else:
                    n_hard = (st_df["projected_win_prob"] < 0.45).sum()
                    n_easy = (st_df["projected_win_prob"] > 0.55).sum()
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Matchups analysed", len(st_df))
                    c2.metric("Tough matchups (< 45%)", int(n_hard))
                    c3.metric("Favourable matchups (> 55%)", int(n_easy))

                    st.markdown("**Hardest matchups (lowest projected win prob)**")
                    hard = st_df.head(10).copy()
                    hard["projected_win_prob"] = hard["projected_win_prob"].map(pct)
                    st.dataframe(
                        hard[["opponent_name", "opponent_colors", "turn_order",
                              "opp_early_plays", "n_opp_cards", "projected_win_prob"]].rename(
                            columns={
                                "opponent_name": "Opponent", "opponent_colors": "Colors",
                                "turn_order": "TO", "opp_early_plays": "Their Early Plays (T1-6)",
                                "n_opp_cards": "Cards Tracked", "projected_win_prob": "Your Win Prob",
                            }
                        ).reset_index(drop=True),
                        use_container_width=True,
                    )

                    with st.expander("All matchups sorted by difficulty"):
                        disp = st_df.copy()
                        disp["projected_win_prob"] = disp["projected_win_prob"].map(pct)
                        st.dataframe(disp.rename(columns={
                            "opponent_name": "Opponent", "opponent_colors": "Colors",
                            "turn_order": "TO", "opp_early_plays": "Their Early Plays",
                            "n_opp_cards": "Cards Tracked", "projected_win_prob": "Your Win Prob",
                        }).reset_index(drop=True), use_container_width=True)

                    # Aggregate by opponent archetype/colors
                    if "opponent_colors" in st_df.columns:
                        st.markdown("**Aggregate by opponent colors**")
                        by_color = (
                            st_df.groupby("opponent_colors")
                            .agg(games=("csv_row_id", "count"),
                                 avg_win_prob=("projected_win_prob", "mean"))
                            .sort_values("avg_win_prob")
                            .reset_index()
                        )
                        by_color["avg_win_prob"] = by_color["avg_win_prob"].map(pct)
                        st.dataframe(by_color.rename(columns={
                            "opponent_colors": "Opponent Colors",
                            "games": "Matchups",
                            "avg_win_prob": "Avg Projected Win Prob"
                        }), use_container_width=True, hide_index=True)

                    st.caption(
                        "Note: the simulation scores hands against opponent sequences using card-level correlations "
                        "and pair synergies from your history. Cards not seen in your games get a neutral score. "
                        "If most matchups cluster near 50%, it means there's limited pair-correlation data "
                        "linking your deck's cards to those opponent sequences — more games = better signal."
                    )


# ════════════════════════════════════════════════════════════════════════════
# PAGE: API Insights
# ════════════════════════════════════════════════════════════════════════════
elif page == "🔌 API Insights":
    st.title("🔌 API Insights")
    st.caption("Extra fields pulled from the API: IDs, queues, MMR, ranked state, and decklist card IDs.")

    if games_df.empty:
        st.info("No game data available. Run analyze.py first.")
        st.stop()

    game_id_col = first_present(games_df, ["game_id", "gameId"])
    replay_id_col = first_present(games_df, ["replay_id", "replayId"])
    gamelog_id_col = first_present(games_df, ["gamelog_id", "gamelogId"])
    queue_col = first_present(games_df, ["queue_name", "Queue", "queue_id"])
    mode_col = first_present(games_df, ["mode", "Source"])
    opp_name_col = first_present(games_df, ["opp_display_name", "opponent_name", "Opponent", "opponent"])
    opp_colors_col = first_present(games_df, ["opp_deck_colors", "opponent_colors", "Opponent Colors"])
    mmr_after_col = first_present(games_df, ["mmr_after", "MMR After"])
    started_col = first_present(games_df, ["started_at", "Started At", "created_at"])

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Games", len(games_df))
    c2.metric("With game_id", int(games_df[game_id_col].notna().sum()) if game_id_col else 0)
    c3.metric("With replay_id", int(games_df[replay_id_col].notna().sum()) if replay_id_col else 0)
    c4.metric("With gamelog_id", int(games_df[gamelog_id_col].notna().sum()) if gamelog_id_col else 0)
    if "ranked" in games_df.columns:
        ranked_rate = games_df["ranked"].fillna(False).astype(bool).mean()
        c5.metric("Ranked Rate", pct(ranked_rate))
    else:
        c5.metric("Ranked Rate", "N/A")

    st.markdown("---")
    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("Queue Performance")
        if queue_col and "is_loss" in games_df.columns:
            queue_perf = (
                games_df.groupby(queue_col, dropna=False)
                .agg(games=("is_loss", "count"), win_rate=("is_loss", lambda s: 1 - s.mean()))
                .reset_index()
                .sort_values("games", ascending=False)
                .head(12)
            )
            queue_perf["win_rate_pct"] = (queue_perf["win_rate"] * 100).round(1)
            st.bar_chart(queue_perf.set_index(queue_col)["win_rate_pct"], height=280)
            queue_table = queue_perf.copy()
            queue_table["win_rate"] = queue_table["win_rate"].map(pct)
            st.dataframe(queue_table.rename(columns={"games": "Games", "win_rate": "Win Rate"}), use_container_width=True)
        else:
            st.info("Queue or result columns are not available.")

    with col_b:
        st.subheader("MMR Trend")
        if mmr_after_col and started_col:
            mmr_view = games_df[[started_col, mmr_after_col]].copy()
            mmr_view[started_col] = pd.to_datetime(mmr_view[started_col], utc=True, errors="coerce")
            mmr_view[mmr_after_col] = pd.to_numeric(mmr_view[mmr_after_col], errors="coerce")
            mmr_view = mmr_view.dropna().sort_values(started_col)
            if not mmr_view.empty:
                st.line_chart(mmr_view.set_index(started_col)[mmr_after_col], height=280)
            else:
                st.info("No numeric MMR data available after filtering.")
        else:
            st.info("MMR fields are not available.")

    st.markdown("---")
    st.subheader("Opponent Deck Colors Breakdown")
    if opp_colors_col and "is_loss" in games_df.columns:
        by_opp_colors = (
            games_df.groupby(opp_colors_col, dropna=False)
            .agg(games=("is_loss", "count"), win_rate=("is_loss", lambda s: 1 - s.mean()))
            .reset_index()
            .sort_values("games", ascending=False)
            .head(12)
        )
        by_opp_colors["win_rate_pct"] = (by_opp_colors["win_rate"] * 100).round(1)
        st.bar_chart(by_opp_colors.set_index(opp_colors_col)["win_rate_pct"], height=260)
    else:
        st.info("Opponent deck color data is not available.")

    st.markdown("---")
    st.subheader("Top Card IDs Seen in Decklists")
    id_col_a, id_col_b = st.columns(2)
    with id_col_a:
        st.markdown("**Opponent deck card IDs**")
        top_opp_ids = top_card_ids(games_df, "opp_decklist", top_n=20)
        if top_opp_ids.empty:
            st.info("No opponent decklist card IDs found.")
        else:
            st.dataframe(top_opp_ids.rename(columns={"card_id": "Card ID", "copies_seen": "Copies Seen"}), use_container_width=True)

    with id_col_b:
        st.markdown("**Your deck card IDs**")
        your_deck_col = first_present(games_df, ["your_decklist"])
        if your_deck_col:
            top_my_ids = top_card_ids(games_df, your_deck_col, top_n=20)
            st.dataframe(top_my_ids.rename(columns={"card_id": "Card ID", "copies_seen": "Copies Seen"}), use_container_width=True)
        else:
            st.info("No API your_decklist column found.")

    st.markdown("---")
    st.subheader("Recent API-Enriched Games")
    recent_cols = [
        col for col in [
            started_col,
            game_id_col,
            replay_id_col,
            gamelog_id_col,
            opp_name_col,
            opp_colors_col,
            queue_col,
            mode_col,
            "match_game_number",
            "is_loss",
        ] if col and col in games_df.columns
    ]
    if recent_cols:
        recent_api = games_df[recent_cols].copy()
        if started_col and started_col in recent_api.columns:
            recent_api[started_col] = pd.to_datetime(recent_api[started_col], utc=True, errors="coerce")
            recent_api = recent_api.sort_values(started_col, ascending=False)
        if "is_loss" in recent_api.columns:
            recent_api["result"] = recent_api["is_loss"].map({True: "Loss", False: "Win"})
            recent_api = recent_api.drop(columns=["is_loss"])
        st.dataframe(recent_api.head(40).reset_index(drop=True), use_container_width=True)
    else:
        st.info("No API-enriched columns are currently available in clean_games.csv.")


# ════════════════════════════════════════════════════════════════════════════
# PAGE: Raw Data
# ════════════════════════════════════════════════════════════════════════════
elif page == "🗂️ Raw Data":
    st.title("🗂️ Raw Data Explorer")

    datasets = {
        "Games (clean_games)": games_df,
        "My Card Loss Correlation": card_df,
        "Seen vs Played Breakdown": seen_played_df,
        "Opponent Card Loss Correlation": opp_card_df,
        "Card Pair Synergy": pair_df,
        "Opening Hand Correlation": opening_df,
        "Archetype Win Rates": arch_df,
        "Matchup Win Rates": matchup_df,
        "Deck Version Win Rates": deck_ver_df,
        "Lore Curves": lore_df,
        "Mulligan Impact": mulligan_df,
        "Tempo Curves": tempo_df,
        "Early Ink Rates": ink_df,
        "First Quest Turn": fq_df,
        "Event Type Correlation": event_type_df,
    }
    selected_ds = st.selectbox("Dataset", list(datasets.keys()))
    df_show = datasets[selected_ds]

    if df_show.empty:
        st.info("This dataset is empty.")
    else:
        st.caption(f"{len(df_show):,} rows × {df_show.shape[1]} columns")

        search_raw = st.text_input("Filter rows containing...", key="raw_search")
        if search_raw:
            mask = df_show.astype(str).apply(lambda col: col.str.contains(search_raw, case=False, na=False)).any(axis=1)
            df_show = df_show[mask]
            st.caption(f"{len(df_show):,} rows after filter")

        st.dataframe(df_show, use_container_width=True)

        csv_bytes = df_show.to_csv(index=False).encode()
        st.download_button(
            label="Download CSV",
            data=csv_bytes,
            file_name=f"{selected_ds.replace(' ', '_').lower()}.csv",
            mime="text/csv",
        )
