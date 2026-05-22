from __future__ import annotations

import csv
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


MAX_BULK_IDS = 1000


@dataclass
class SyncSummary:
    rows_fetched: int
    history_csv: Path | None
    replay_downloaded: int
    replay_skipped: int
    replay_missing: int
    gamelog_downloaded: int
    gamelog_skipped: int
    gamelog_missing: int


def read_api_token(token_file: Path) -> str:
    if not token_file.exists():
        raise FileNotFoundError(f"Token file not found: {token_file}")

    content = token_file.read_text(encoding="utf-8")
    for raw_line in content.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue

        if "=" in line:
            left, right = line.split("=", 1)
            key = left.strip().upper()
            if key in {"TOKEN", "API_TOKEN", "DUELS_API_TOKEN", "BEARER_TOKEN"}:
                token = right.strip().strip('"').strip("'")
                if token:
                    return token

        return line.strip().strip('"').strip("'")

    raise ValueError(f"No API token found in {token_file}")


def _api_request_json(
    method: str,
    url: str,
    token: str,
    payload: dict[str, Any] | None = None,
    timeout: int = 30,
) -> dict[str, Any]:
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/json",
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/136.0.0.0 Safari/537.36"
        ),
        "Accept-Language": "en-US,en;q=0.9",
    }

    data: bytes | None = None
    if payload is not None:
        headers["Content-Type"] = "application/json"
        data = json.dumps(payload).encode("utf-8")

    request = urllib.request.Request(url=url, data=data, method=method.upper(), headers=headers)

    attempts = 0
    while True:
        attempts += 1
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                body = response.read().decode("utf-8")
                return json.loads(body) if body else {}
        except urllib.error.HTTPError as exc:
            if exc.code == 429 and attempts < 6:
                retry_after = exc.headers.get("Retry-After") if exc.headers else None
                try:
                    wait_seconds = max(1, int(retry_after)) if retry_after else 2**attempts
                except ValueError:
                    wait_seconds = 2**attempts
                time.sleep(wait_seconds)
                continue

            error_body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"API request failed ({exc.code}) for {url}: {error_body}") from exc


def _chunked(values: list[str], size: int) -> list[list[str]]:
    return [values[i : i + size] for i in range(0, len(values), size)]


def _value_from_row(row: dict[str, Any], names: list[str]) -> Any:
    for name in names:
        if name in row and row[name] is not None and row[name] != "":
            return row[name]
    return None


def _normalize_id(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _extract_ids(rows: list[dict[str, Any]]) -> tuple[list[str], list[str]]:
    replay_ids: set[str] = set()
    game_ids: set[str] = set()

    for row in rows:
        replay_value = _value_from_row(
            row,
            ["replay_id", "replayId", "replayid", "id_replay"],
        )
        game_value = _value_from_row(
            row,
            ["game_id", "gameId", "gameid", "id_game", "id"],
        )

        replay_id = _normalize_id(replay_value)
        game_id = _normalize_id(game_value)
        if replay_id:
            replay_ids.add(replay_id)
        if game_id:
            game_ids.add(game_id)

    return sorted(replay_ids), sorted(game_ids)


def _fetch_match_history_rows(
    base_url: str,
    token: str,
    from_ts: str | None,
    to_ts: str | None,
    limit: int = 1000,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    cursor: str | None = None

    while True:
        query: dict[str, str] = {
            "format": "json",
            "limit": str(limit),
        }
        if from_ts:
            query["from"] = from_ts
        if to_ts:
            query["to"] = to_ts
        if cursor:
            query["cursor"] = cursor

        url = f"{base_url.rstrip('/')}/api/me/match-history?{urllib.parse.urlencode(query)}"
        payload = _api_request_json("GET", url, token)

        batch = payload.get("games")
        if not isinstance(batch, list):
            raise RuntimeError("Unexpected response from /api/me/match-history: missing games array")

        rows.extend(item for item in batch if isinstance(item, dict))
        cursor = payload.get("next_cursor")
        if not cursor:
            break

    return rows


def _rows_to_csv(rows: list[dict[str, Any]], output_file: Path) -> None:
    keys: set[str] = set()
    for row in rows:
        keys.update(row.keys())

    fieldnames = sorted(keys)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with output_file.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            normalized: dict[str, Any] = {}
            for key in fieldnames:
                value = row.get(key)
                if isinstance(value, (dict, list)):
                    normalized[key] = json.dumps(value, separators=(",", ":"))
                else:
                    normalized[key] = value
            writer.writerow(normalized)


def _bulk_manifest(base_url: str, token: str, endpoint: str, ids: list[str]) -> tuple[list[dict[str, str]], int]:
    files: list[dict[str, str]] = []
    missing_total = 0

    for chunk in _chunked(ids, MAX_BULK_IDS):
        payload = _api_request_json(
            "POST",
            f"{base_url.rstrip('/')}{endpoint}",
            token,
            payload={"ids": chunk},
        )
        batch_files = payload.get("files") or []
        batch_missing = payload.get("missing") or []

        files.extend(file_entry for file_entry in batch_files if isinstance(file_entry, dict))
        missing_total += len(batch_missing)

    return files, missing_total


def _download_to_path(url: str, destination: Path, timeout: int = 120) -> None:
    request = urllib.request.Request(
        url=url,
        method="GET",
        headers={
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/136.0.0.0 Safari/537.36"
            ),
            "Accept": "*/*",
        },
    )
    attempts = 0
    while True:
        attempts += 1
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                destination.parent.mkdir(parents=True, exist_ok=True)
                with destination.open("wb") as f:
                    while True:
                        chunk = response.read(1024 * 1024)
                        if not chunk:
                            break
                        f.write(chunk)
            return
        except urllib.error.HTTPError as exc:
            if exc.code == 429 and attempts < 6:
                retry_after = exc.headers.get("Retry-After") if exc.headers else None
                try:
                    wait_seconds = max(1, int(retry_after)) if retry_after else 2**attempts
                except ValueError:
                    wait_seconds = 2**attempts
                time.sleep(wait_seconds)
                continue
            raise


def _download_manifest_files(files: list[dict[str, str]], output_root: Path) -> tuple[int, int, int]:
    downloaded = 0
    skipped = 0
    failed = 0
    target_dir = output_root / datetime.now(timezone.utc).strftime("%Y-%m-%d")

    for file_entry in files:
        file_name = file_entry.get("filename") or ""
        url = file_entry.get("url") or ""
        if not file_name or not url:
            continue

        destination = target_dir / Path(file_name).name
        if destination.exists():
            skipped += 1
            continue

        try:
            _download_to_path(url, destination)
            downloaded += 1
        except urllib.error.HTTPError as exc:
            if exc.code in {403, 404, 410}:
                failed += 1
                continue
            raise

    return downloaded, skipped, failed


def sync_from_api(
    *,
    token_file: Path,
    results_dir: Path,
    replays_root: Path,
    gamelogs_root: Path,
    base_url: str = "https://duels.ink",
    from_ts: str | None = None,
    to_ts: str | None = None,
    download_gamelogs: bool = True,
) -> SyncSummary:
    token = read_api_token(token_file)
    rows = _fetch_match_history_rows(
        base_url=base_url,
        token=token,
        from_ts=from_ts,
        to_ts=to_ts,
    )

    history_csv: Path | None = None
    if rows:
        history_csv = results_dir / f"game-history-{datetime.now(timezone.utc):%Y-%m-%d}.csv"
        _rows_to_csv(rows, history_csv)

    replay_ids, game_ids = _extract_ids(rows)

    replay_manifest, replay_missing = _bulk_manifest(
        base_url=base_url,
        token=token,
        endpoint="/api/me/bulk-replays",
        ids=replay_ids,
    ) if replay_ids else ([], 0)

    replay_downloaded, replay_skipped, replay_failed = _download_manifest_files(replay_manifest, replays_root)
    replay_missing += replay_failed

    gamelog_downloaded = 0
    gamelog_skipped = 0
    gamelog_missing = 0
    if download_gamelogs and game_ids:
        gamelog_manifest, gamelog_missing = _bulk_manifest(
            base_url=base_url,
            token=token,
            endpoint="/api/me/bulk-gamelogs",
            ids=game_ids,
        )
        gamelog_downloaded, gamelog_skipped, gamelog_failed = _download_manifest_files(gamelog_manifest, gamelogs_root)
        gamelog_missing += gamelog_failed

    return SyncSummary(
        rows_fetched=len(rows),
        history_csv=history_csv,
        replay_downloaded=replay_downloaded,
        replay_skipped=replay_skipped,
        replay_missing=replay_missing,
        gamelog_downloaded=gamelog_downloaded,
        gamelog_skipped=gamelog_skipped,
        gamelog_missing=gamelog_missing,
    )