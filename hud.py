"""Optional large-type sighted test HUD. All critical output is also spoken."""
from __future__ import annotations

import cv2
import numpy as np


class HUD:
    @staticmethod
    def render(frame: np.ndarray, state: str, caption: str, items: list[dict],
               bbox=None, progress: float = 0.0, elapsed: float = 0.0,
               turn_degrees=None, muted: bool = False) -> np.ndarray:
        canvas = frame.copy()
        h, w = canvas.shape[:2]
        if bbox:
            cv2.rectangle(canvas, bbox[:2], bbox[2:], (30, 220, 170), 3)
        cv2.rectangle(canvas, (0, 0), (w, 86), (16, 23, 28), -1)
        cv2.putText(canvas, "FRESHPICK - ACCESSIBLE FRUIT SCANNER", (14, 29),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.74, (255, 255, 255), 2)
        line = f"{state} | {elapsed:.1f}s | muted: {muted} | turn estimate: {turn_degrees if turn_degrees is not None else '-'}"
        cv2.putText(canvas, line[:108], (14, 61), cv2.FONT_HERSHEY_SIMPLEX, 0.53,
                    (160, 225, 215), 2)
        if progress:
            cv2.rectangle(canvas, (14, 77), (14 + int((w - 28) * progress), 84), (70, 225, 155), -1)
        if items:
            cv2.rectangle(canvas, (max(0, w - 285), 104), (w, 140 + 30 * min(3, len(items))), (10, 14, 20), -1)
            for index, item in enumerate(items[:3]):
                cv2.putText(canvas, f"{index+1} {item['fruit_name']} [{item['urgency']}]"[:35],
                            (w - 278, 136 + index * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                            (248, 248, 248), 1)
        cv2.rectangle(canvas, (0, h - 100), (w, h), (16, 23, 28), -1)
        words, lines, current = caption.split(), [], ""
        for word in words:
            if len(current) + len(word) < 80:
                current += (" " if current else "") + word
            else:
                lines.append(current)
                current = word
        lines.append(current)
        for index, line in enumerate(lines[:2]):
            cv2.putText(canvas, line, (12, h - 68 + index * 29), cv2.FONT_HERSHEY_SIMPLEX,
                        0.54, (255, 255, 255), 1)
        return canvas
