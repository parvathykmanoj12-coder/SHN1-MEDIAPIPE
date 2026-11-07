"""MediaPipe Hand Landmarker LIVE_STREAM gesture remote and hand-based framing.

Palm-normal changes are approximate orientation hints, NOT proof of 180/360
fruit coverage. The first few webcam frames may have no callback result yet.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import threading
import time
import urllib.request

import cv2
import mediapipe as mp
import numpy as np

MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
MODEL_PATH = Path(__file__).resolve().parent / "assets" / "hand_landmarker.task"
LEGACY_PATH = Path(__file__).resolve().parent / "hand_landmarker.task"


@dataclass
class GestureObservation:
    event: str | None = None
    progress: float = 0.0
    bbox: tuple[int, int, int, int] | None = None
    hint: str | None = None
    hand_visible: bool = False
    estimated_hand_turn_degrees: float | None = None


def _distance(a, b) -> float:
    return math.hypot(a.x - b.x, a.y - b.y)


def _angle_between(a: np.ndarray, b: np.ndarray) -> float:
    cos_value = float(np.clip(np.dot(a, b), -1.0, 1.0))
    return math.degrees(math.acos(cos_value))


class GestureController:
    def __init__(self):
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision as mp_vision

        model_path = MODEL_PATH if MODEL_PATH.exists() or not LEGACY_PATH.exists() else LEGACY_PATH
        if not model_path.exists():
            model_path.parent.mkdir(exist_ok=True, parents=True)
            print("Downloading the official hand_landmarker.task once...")
            try:
                urllib.request.urlretrieve(MODEL_URL, str(model_path))
            except Exception as exc:
                raise RuntimeError("Download the MediaPipe hand_landmarker.task into assets/ manually. " + str(exc)) from exc
        self._lock = threading.Lock()
        self._latest = None
        self._latest_time = 0.0
        self._frame_clock = 0
        self._candidate = None
        self._candidate_since = 0.0
        self._armed = True
        self._last_emit = 0.0
        self._initial_normal = None
        self._fruit_hand_label = None
        options = mp_vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
            running_mode=mp_vision.RunningMode.LIVE_STREAM,
            num_hands=2,
            min_hand_detection_confidence=0.6,
            min_hand_presence_confidence=0.55,
            min_tracking_confidence=0.55,
            result_callback=self._on_result,
        )
        self.landmarker = mp_vision.HandLandmarker.create_from_options(options)

    def _on_result(self, result, output_image, timestamp_ms: int) -> None:
        with self._lock:
            self._latest = result
            self._latest_time = time.monotonic()

    def reset_rotation_reference(self) -> None:
        self._initial_normal = None

    def _palm_normal(self, xyz) -> np.ndarray | None:
        if not xyz:
            return None
        wrist = np.array([xyz[0].x, xyz[0].y, xyz[0].z])
        v1 = np.array([xyz[5].x, xyz[5].y, xyz[5].z]) - wrist
        v2 = np.array([xyz[17].x, xyz[17].y, xyz[17].z]) - wrist
        normal = np.cross(v1, v2)
        size = float(np.linalg.norm(normal))
        return normal / size if size > 1e-6 else None

    def _gesture_kind(self, lms) -> str | None:
        palm_width = max(0.01, _distance(lms[5], lms[17]))
        pinch = _distance(lms[4], lms[8]) / palm_width
        fingers_extended = [
            _distance(lms[tip], lms[0]) > _distance(lms[pip], lms[0]) * 1.14
            for tip, pip in ((8, 6), (12, 10), (16, 14), (20, 18))
        ]
        if pinch < 0.38:
            return "PINCH_SCAN"
        if all(fingers_extended) and pinch > 0.8:
            return "PALM_RECAP"
        if not any(fingers_extended) and _distance(lms[4], lms[0]) > _distance(lms[3], lms[0]) * 1.25:
            return "THUMB_SAFETY"
        return None

    def process_frame(self, frame: np.ndarray, allow_gestures: bool = True) -> GestureObservation:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb = np.ascontiguousarray(rgb)
        now = time.monotonic()
        timestamp = int(now * 1000)
        timestamp = max(timestamp, self._frame_clock + 1)
        self._frame_clock = timestamp
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        self.landmarker.detect_async(image, timestamp)

        with self._lock:
            result = self._latest if now - self._latest_time < 0.6 else None
        if not result or not result.hand_landmarks:
            self._candidate, self._armed = None, True
            return GestureObservation(hint="Place a hand holding one fruit near the middle of the camera.")

        hands = result.hand_landmarks[:2]
        labels = [
            (result.handedness[i][0].category_name if i < len(result.handedness)
             and result.handedness[i] else f"Hand {i}")
            for i in range(len(hands))
        ]
        kinds = [self._gesture_kind(candidate) for candidate in hands]
        controller_index = next((i for i, gesture in enumerate(kinds) if gesture == "PINCH_SCAN"), None)
        if controller_index is None:
            controller_index = next((i for i, gesture in enumerate(kinds) if gesture is not None), 0)
        # With two hands, one may hold the fruit while the other pinches to scan.
        if not allow_gestures and self._fruit_hand_label in labels:
            fruit_index = labels.index(self._fruit_hand_label)
        elif len(hands) == 2 and kinds[controller_index] == "PINCH_SCAN":
            fruit_index = 1 - controller_index
        else:
            fruit_index = 0
        lms = hands[fruit_index]
        h, w = frame.shape[:2]
        # Expand well beyond hand bounds because fruit is usually bigger than hand.
        xs, ys = [p.x for p in lms], [p.y for p in lms]
        cx = (min(xs) + max(xs)) * 0.5
        cy = (min(ys) + max(ys)) * 0.5
        box_w = max(0.66, (max(xs) - min(xs)) + 0.32)
        box_h = max(0.70, (max(ys) - min(ys)) + 0.34)
        x1, x2 = max(0, int((cx - box_w / 2) * w)), min(w, int((cx + box_w / 2) * w))
        y1, y2 = max(0, int((cy - box_h / 2) * h)), min(h, int((cy + box_h / 2) * h))
        box = (x1, y1, x2, y2)
        hint = None
        if cx < 0.29:
            hint = "Move your hand right."
        elif cx > 0.71:
            hint = "Move your hand left."
        elif cy < 0.25:
            hint = "Move your hand down."
        elif cy > 0.75:
            hint = "Move your hand up."
        elif (max(xs) - min(xs)) < 0.11:
            hint = "Move your fruit a little closer."

        normal = self._palm_normal(result.hand_world_landmarks[fruit_index]
                                   if result.hand_world_landmarks
                                   and fruit_index < len(result.hand_world_landmarks) else None)
        degrees = None
        if normal is not None:
            if self._initial_normal is None:
                self._initial_normal = normal
            degrees = _angle_between(self._initial_normal, normal)

        if not allow_gestures:
            self._candidate, self._armed = None, True
            return GestureObservation(bbox=box, hint=hint, hand_visible=True, estimated_hand_turn_degrees=degrees)

        raw = kinds[controller_index]
        if raw != self._candidate:
            self._candidate = raw
            self._candidate_since = now
            if raw is None:
                self._armed = True
        durations = {"PINCH_SCAN": 0.55, "PALM_RECAP": 0.75, "THUMB_SAFETY": 0.75}
        target = durations.get(raw, 0)
        progress = min(1.0, (now - self._candidate_since) / target) if target else 0.0
        event = None
        if raw and self._armed and now - self._candidate_since >= target and now - self._last_emit >= 1.5:
            event = raw
            self._last_emit = now
            self._armed = False
            if event == "PINCH_SCAN":
                self._fruit_hand_label = labels[fruit_index]
        return GestureObservation(event=event, progress=progress, bbox=box, hint=hint,
                                  hand_visible=True, estimated_hand_turn_degrees=degrees)

    def draw_landmarks(self, frame: np.ndarray) -> np.ndarray:
        """Draw visible hand joints and skeleton on a DISPLAY image only.

        Never pass this annotated image into the vision model.
        """
        with self._lock:
            result = self._latest if time.monotonic() - self._latest_time < 0.6 else None
        if result is None or not result.hand_landmarks:
            return frame

        h, w = frame.shape[:2]
        connections = (
            (0, 1), (1, 2), (2, 3), (3, 4),
            (0, 5), (5, 6), (6, 7), (7, 8),
            (5, 9), (9, 10), (10, 11), (11, 12),
            (9, 13), (13, 14), (14, 15), (15, 16),
            (13, 17), (17, 18), (18, 19), (19, 20),
            (0, 17),
        )
        for hand in result.hand_landmarks:
            coords = [
                (max(0, min(w - 1, int(pt.x * w))),
                 max(0, min(h - 1, int(pt.y * h))))
                for pt in hand
            ]
            for a, b in connections:
                cv2.line(frame, coords[a], coords[b], (0, 240, 180), 2,
                         cv2.LINE_AA)
            for x, y in coords:
                cv2.circle(frame, (x, y), 4, (255, 255, 255), -1,
                           cv2.LINE_AA)
                cv2.circle(frame, (x, y), 5, (0, 200, 100), 1,
                           cv2.LINE_AA)
        return frame

    def close(self) -> None:
        self.landmarker.close()
