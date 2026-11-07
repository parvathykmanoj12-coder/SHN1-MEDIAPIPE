"""Persistent pantry inventory. No photos are stored; uncertain scans are excluded."""
from __future__ import annotations

from datetime import datetime
import json
import os
from pathlib import Path
from threading import RLock

from decision_engine import URGENCY_WEIGHT


class PantryMemory:
    def __init__(self, path: str = "pantry_session.json"):
        self.path = Path(path)
        self._lock = RLock()
        self.items: list[dict] = []
        self._load()

    def _load(self) -> None:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            if isinstance(data, list):
                self.items = [row for row in data if isinstance(row, dict) and row.get("urgency") in URGENCY_WEIGHT]
        except (FileNotFoundError, ValueError, OSError):
            self.items = []

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temp = self.path.with_name(self.path.name + ".tmp")
        temp.write_text(json.dumps(self.items, indent=2), encoding="utf-8")
        os.replace(temp, self.path)

    def add_item(self, scan_result: dict) -> dict | None:
        if scan_result.get("urgency") not in URGENCY_WEIGHT or scan_result.get("review_required"):
            return None
        with self._lock:
            item = {
                "id": max((int(x.get("id", 0)) for x in self.items), default=0) + 1,
                "fruit_name": str(scan_result.get("fruit_name") or "Fruit")[:60],
                "surface_description": str(scan_result.get("surface_description") or "")[:160],
                "urgency": scan_result["urgency"],
                "weight": URGENCY_WEIGHT[scan_result["urgency"]],
                "timestamp": datetime.now().isoformat(timespec="seconds"),
            }
            self.items.append(item)
            self._save()
            return item

    def get_sorted_items(self) -> list[dict]:
        with self._lock:
            return sorted((x.copy() for x in self.items), key=lambda x: (URGENCY_WEIGHT[x["urgency"]], x["id"]))

    def get_spoken_recap(self, max_items: int = 5) -> str:
        ordered = self.get_sorted_items()
        if not ordered:
            return "Your fruit list is empty. Hold a fruit and pinch to scan."
        labels = {"EAT_TODAY": "use first", "USE_SOON": "use soon", "STORE_LONGER": "keep longer"}
        # Short enough for a spoken summary; repeated names carry IDs.
        shown = ordered[:max_items]
        chunks = [f"{x['fruit_name']}, item {x['id']}, {labels[x['urgency']]}" for x in shown]
        more = f" {len(ordered) - len(shown)} more items not read." if len(ordered) > len(shown) else ""
        return "Consumption order: " + "; ".join(chunks) + "." + more

    def remove_last(self) -> bool:
        with self._lock:
            if not self.items:
                return False
            self.items.pop()
            self._save()
            return True

    def clear(self) -> None:
        with self._lock:
            self.items.clear()
            self._save()
