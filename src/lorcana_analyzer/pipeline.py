from __future__ import annotations

import gzip
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass
class ReplayPayload:
    game_id: str
    perspective: int
    created_at: pd.Timestamp
    opponent_name: str | None
    my_name: str | None
    winner: int | None
    turn_count: int | None
    victory_reason: str | None
    opening_hand_cards: list[str]
    mulligan_count_my: int | None
    mulligan_count_opp: int | None


def _canonical_col_name(name: str) -> str:
    return "".join(ch for ch in str(name).lower() if ch.isalnum())


def _pick_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    matches: list[str] = []

    direct = set(df.columns)
    for candidate in candidates:
        if candidate in direct and candidate not in matches:
            matches.append(candidate)

    canon_map = {_canonical_col_name(col): col for col in df.columns}
    for candidate in candidates:
        match = canon_map.get(_canonical_col_name(candidate))
        if match and match not in matches:
            matches.append(match)

    if not matches:
        return None

    # Prefer the most populated candidate when multiple schema variants exist.
    scored: list[tuple[int, str]] = []
    for col in matches:
        non_blank = int(
            df[col]
            .fillna("")
            .astype(str)
            .str.strip()
            .ne("")
            .sum()
        )
        scored.append((non_blank, col))
    scored.sort(key=lambda t: t[0], reverse=True)
    return scored[0][1]


def _as_string_series(df: pd.DataFrame, column: str | None, default: str = "") -> pd.Series:
    if not column:
        return pd.Series(default, index=df.index, dtype="object")
    return df[column].fillna(default).astype(str)


def _coerce_bool_series(series: pd.Series) -> pd.Series:
    normalized = series.fillna("").astype(str).str.strip().str.lower()
    return normalized.isin({"1", "true", "t", "yes", "y", "win", "won"})


def parse_decklist(decklist: Any) -> dict[str, int]:
    if isinstance(decklist, list):
        cards: dict[str, int] = {}
        for item in decklist:
            if not isinstance(item, dict):
                continue
            name = str(
                item.get("name")
                or item.get("cardName")
                or item.get("card_id")
                or item.get("cardId")
                or ""
            ).strip()
            qty_raw = item.get("count") or item.get("qty") or item.get("quantity")
            try:
                qty = int(qty_raw)
            except (TypeError, ValueError):
                continue
            if name and qty > 0:
                cards[name] = cards.get(name, 0) + qty
        return cards

    if isinstance(decklist, dict):
        cards: dict[str, int] = {}
        for key, value in decklist.items():
            try:
                qty = int(value)
            except (TypeError, ValueError):
                continue
            name = str(key).strip()
            if name and qty > 0:
                cards[name] = cards.get(name, 0) + qty
        return cards

    if not isinstance(decklist, str) or not decklist.strip():
        return {}

    text = decklist.strip()
    if text.startswith("["):
        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                return parse_decklist(parsed)
        except json.JSONDecodeError:
            pass

    cards: dict[str, int] = {}
    for part in text.split(";"):
        entry = part.strip()
        if not entry:
            continue
        if "x " not in entry:
            continue
        qty_str, name = entry.split("x ", 1)
        qty_str = qty_str.strip()
        name = name.strip()
        if not qty_str.isdigit() or not name:
            continue
        cards[name] = cards.get(name, 0) + int(qty_str)
    return cards


def load_results_csv(results_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(results_csv)

    started_col = _pick_column(df, ["Started At", "started_at", "startedAt", "updated_at", "updatedAt"])
    my_lore_col = _pick_column(df, ["My Lore", "my_lore", "myLore", "myFinalLore", "my_final_lore"])
    opp_lore_col = _pick_column(
        df,
        ["Opponent Lore", "opponent_lore", "opponentLore", "oppFinalLore", "opp_final_lore"],
    )
    turns_col = _pick_column(df, ["Turns", "turns", "turnCount", "turn_count"])
    result_col = _pick_column(df, ["Result", "result", "outcome"])
    is_win_col = _pick_column(df, ["is_win", "isWin", "didWin", "did_win", "won"])
    is_loss_col = _pick_column(df, ["is_loss", "isLoss", "didLose", "did_lose", "lost"])
    turn_order_col = _pick_column(df, ["Turn Order", "turn_order", "turnOrder"])
    opponent_col = _pick_column(df, ["Opponent", "opponent", "opponent_name", "opponentName"])
    my_colors_col = _pick_column(df, ["My Colors", "my_colors", "myColors", "my_ink_colors"])
    opp_colors_col = _pick_column(
        df,
        ["Opponent Colors", "opponent_colors", "opponentColors", "opponent_ink_colors"],
    )
    decklist_col = _pick_column(df, ["Decklist", "decklist", "my_decklist", "myDecklist"])
    game_id_col = _pick_column(df, ["game_id", "gameId", "gameid"])
    replay_id_col = _pick_column(df, ["replay_id", "replayId", "replayid"])

    df["started_at"] = pd.to_datetime(df[started_col], utc=True, errors="coerce") if started_col else pd.NaT
    df["my_lore"] = pd.to_numeric(df[my_lore_col], errors="coerce") if my_lore_col else pd.NA
    df["opponent_lore"] = pd.to_numeric(df[opp_lore_col], errors="coerce") if opp_lore_col else pd.NA
    df["turns"] = pd.to_numeric(df[turns_col], errors="coerce") if turns_col else pd.NA

    if result_col:
        normalized_result = _as_string_series(df, result_col).str.lower().str.strip()
        df["is_loss"] = normalized_result.isin({"loss", "lose", "lost", "l"})
        df["is_win"] = normalized_result.isin({"win", "won", "w"})
    else:
        df["is_win"] = _coerce_bool_series(df[is_win_col]) if is_win_col else False
        if is_loss_col:
            df["is_loss"] = _coerce_bool_series(df[is_loss_col])
        else:
            df["is_loss"] = ~df["is_win"]

    df["turn_order"] = _as_string_series(df, turn_order_col)
    df["opponent"] = _as_string_series(df, opponent_col)
    df["my_colors"] = _as_string_series(df, my_colors_col, default="Unknown")
    df["opponent_colors"] = _as_string_series(df, opp_colors_col, default="Unknown")
    if decklist_col:
        df["decklist_map"] = df[decklist_col].map(parse_decklist)
    else:
        df["decklist_map"] = [{} for _ in range(len(df))]

    if game_id_col:
        df["game_id"] = _as_string_series(df, game_id_col)
    if replay_id_col:
        df["replay_id"] = _as_string_series(df, replay_id_col)

    df["csv_row_id"] = range(1, len(df) + 1)
    return df


def iter_replay_files(replays_root: Path) -> list[Path]:
    return sorted(replays_root.rglob("*.replay.gz"))


def _read_replay_json(replay_file: Path) -> dict[str, Any]:
    with gzip.open(replay_file, "rt", encoding="utf-8") as f:
        return json.load(f)


def _extract_replay_payload(data: dict[str, Any]) -> ReplayPayload:
    perspective = int(data.get("perspective") or 0)
    player_names = data.get("playerNames") or {}

    player_1 = player_names.get("1") or player_names.get(1)
    player_2 = player_names.get("2") or player_names.get(2)
    my_name = player_1 if perspective == 1 else player_2 if perspective == 2 else None
    opponent_name = player_2 if perspective == 1 else player_1 if perspective == 2 else None

    opening_hand_cards: list[str] = []
    mulligan_count_my: int | None = None
    mulligan_count_opp: int | None = None

    for log in data.get("logs") or []:
        log_type = log.get("type")
        log_player = log.get("player")
        if log_type == "INITIAL_HAND" and log_player == perspective:
            opening_hand_cards = [
                ref.get("name", "") for ref in (log.get("cardRefs") or []) if ref.get("name")
            ]
        if log_type == "MULLIGAN":
            count = ((log.get("data") or {}).get("mulliganCount"))
            if isinstance(count, int):
                if log_player == perspective:
                    mulligan_count_my = count
                elif log_player in (1, 2):
                    mulligan_count_opp = count

    return ReplayPayload(
        game_id=str(data.get("gameId") or ""),
        perspective=perspective,
        created_at=pd.to_datetime(data.get("createdAt"), unit="ms", utc=True, errors="coerce"),
        opponent_name=opponent_name,
        my_name=my_name,
        winner=data.get("winner"),
        turn_count=data.get("turnCount"),
        victory_reason=data.get("victoryReason"),
        opening_hand_cards=opening_hand_cards,
        mulligan_count_my=mulligan_count_my,
        mulligan_count_opp=mulligan_count_opp,
    )


def _extract_event_rows(data: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    game_id = data.get("gameId")
    perspective = data.get("perspective")

    for log in data.get("logs") or []:
        card_refs = log.get("cardRefs") or []
        base = {
            "game_id": game_id,
            "perspective": perspective,
            "event_id": log.get("id"),
            "timestamp": pd.to_datetime(log.get("timestamp"), unit="ms", utc=True, errors="coerce"),
            "turn_number": log.get("turnNumber"),
            "player": log.get("player"),
            "is_my_event": log.get("player") == perspective,
            "event_type": log.get("type"),
            "message": log.get("message"),
        }

        if not card_refs:
            rows.append({**base, "card_name": None, "card_id": None})
            continue

        for ref in card_refs:
            rows.append(
                {
                    **base,
                    "card_name": ref.get("name"),
                    "card_id": ref.get("id"),
                }
            )

    return rows


def build_replay_tables(replays_root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    replay_files = iter_replay_files(replays_root)
    game_rows: list[dict[str, Any]] = []
    event_rows: list[dict[str, Any]] = []
    error_rows: list[dict[str, Any]] = []

    for replay_file in replay_files:
        try:
            data = _read_replay_json(replay_file)
            payload = _extract_replay_payload(data)
            game_rows.append(
                {
                    "replay_file": str(replay_file),
                    "game_id": payload.game_id,
                    "perspective": payload.perspective,
                    "created_at": payload.created_at,
                    "opponent_name": payload.opponent_name,
                    "my_name": payload.my_name,
                    "winner": payload.winner,
                    "my_win_replay": payload.winner == payload.perspective if payload.winner in (1, 2) else None,
                    "turn_count_replay": payload.turn_count,
                    "victory_reason": payload.victory_reason,
                    "opening_hand_cards": " | ".join(payload.opening_hand_cards),
                    "mulligan_count_my": payload.mulligan_count_my,
                    "mulligan_count_opp": payload.mulligan_count_opp,
                }
            )
            event_rows.extend(_extract_event_rows(data))
        except Exception as exc:  # noqa: BLE001
            error_rows.append({"replay_file": str(replay_file), "error": str(exc)})

    replay_games_df = pd.DataFrame(game_rows)
    replay_events_df = pd.DataFrame(event_rows)
    replay_errors_df = pd.DataFrame(error_rows)
    return replay_games_df, replay_events_df, replay_errors_df


def link_results_to_replays(results_df: pd.DataFrame, replay_games_df: pd.DataFrame) -> pd.DataFrame:
    if replay_games_df.empty:
        linked = results_df.copy()
        linked["game_id"] = None
        linked["join_score"] = None
        linked["replay_file"] = None
        linked["join_status"] = "unmatched"
        return linked

    candidates = []

    replay_work = replay_games_df.copy()
    replay_work["opponent_name"] = replay_work["opponent_name"].astype(str)
    replay_work["game_id"] = replay_work.get("game_id", pd.Series(dtype="object")).astype(str)

    for _, row in results_df.iterrows():
        row_game_id = str(row.get("game_id") or "").strip()
        if row_game_id:
            exact_subset = replay_work[replay_work["game_id"] == row_game_id]
            for ridx, _ in exact_subset.iterrows():
                candidates.append(
                    {
                        "csv_row_id": row["csv_row_id"],
                        "replay_index": ridx,
                        "join_score": -1_000_000.0,
                    }
                )

        subset = replay_work[replay_work["opponent_name"] == str(row["opponent"])]
        if subset.empty:
            continue

        for ridx, replay_row in subset.iterrows():
            delta_minutes = abs((row["started_at"] - replay_row["created_at"]).total_seconds()) / 60.0
            if delta_minutes > 240:
                continue

            result_match = (
                (bool(row["is_win"]) and replay_row["my_win_replay"] is True)
                or (bool(row["is_loss"]) and replay_row["my_win_replay"] is False)
            )
            turn_delta = abs(float(row["turns"]) - float(replay_row["turn_count_replay"])) if pd.notna(row["turns"]) and pd.notna(replay_row["turn_count_replay"]) else 20.0
            score = delta_minutes + turn_delta * 3.0 + (0.0 if result_match else 40.0)
            candidates.append(
                {
                    "csv_row_id": row["csv_row_id"],
                    "replay_index": ridx,
                    "join_score": score,
                }
            )

    if not candidates:
        linked = results_df.copy()
        linked["game_id"] = None
        linked["join_score"] = None
        linked["replay_file"] = None
        linked["join_status"] = "unmatched"
        return linked

    candidate_df = pd.DataFrame(candidates).sort_values("join_score", ascending=True)

    used_csv: set[int] = set()
    used_replay: set[int] = set()
    accepted_rows: list[dict[str, Any]] = []

    for _, cand in candidate_df.iterrows():
        csv_id = int(cand["csv_row_id"])
        replay_idx = int(cand["replay_index"])
        if csv_id in used_csv or replay_idx in used_replay:
            continue
        used_csv.add(csv_id)
        used_replay.add(replay_idx)
        accepted_rows.append(
            {
                "csv_row_id": csv_id,
                "replay_index": replay_idx,
                "join_score": float(cand["join_score"]),
            }
        )

    accepted_df = pd.DataFrame(accepted_rows)

    linked = results_df.merge(accepted_df, on="csv_row_id", how="left")
    linked = linked.merge(
        replay_work.reset_index().rename(columns={"index": "replay_index"}),
        on="replay_index",
        how="left",
        suffixes=("", "_replay"),
    )
    linked["join_status"] = linked["game_id"].notna().map({True: "matched", False: "unmatched"})
    return linked
