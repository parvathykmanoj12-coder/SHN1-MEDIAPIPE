"""Multiview fruit inspection using Gemini (existing project's backend) or Moondream2.

A model's answer is not proof of food safety, internal ripeness, or complete
360-degree visual coverage. There is deliberately no HSV->fruit/urgency guess.
"""
from __future__ import annotations

import json
import os
from typing import Literal

import cv2
import numpy as np
from PIL import Image, ImageDraw
from pydantic import BaseModel, Field

from decision_engine import normalize_observation, unknown_result


class FruitObservation(BaseModel):
    fruit_name: str = Field(description="Specific fruit name or 'Unknown fruit'.")
    fruit_visible: bool = Field(description="Is any fruit visible?")
    fruit_count: int = Field(description="Number of distinct fruits visible; zero if uncertain.")
    view_quality: Literal["CLEAR", "PARTIAL", "POOR"]
    ripeness_stage: Literal["OVERRIPE", "RIPE", "UNDERRIPE", "UNKNOWN"]
    possible_mold_or_rot: bool = Field(description="True if suspicious visible growth, holes, leaking or severe decay is seen; conservative flag.")
    visible_features: str = Field(description="At most 12 words about external colors, spots and wrinkles. Avoid claims about internal condition.")


PROMPT = (
    "You are describing a single physical fruit for a person with low or no vision. "
    "Images are sequential views of the same fruit, not separate fruit items. "
    "Name the fruit only if visually recognizable, otherwise say Unknown fruit. "
    "Assess only *visible* peel/skin cues and be conservative. "
    "Classify UNDERRIPE, RIPE, OVERRIPE or UNKNOWN according to this fruit species, "
    "not just universal green/yellow/red rules. An apple can be naturally green when ripe. "
    "Set possible_mold_or_rot true if a visible area may be mold, severe damage or leaking. "
    "Never assert that food is safe, mold-free, edible, or fresh internally. "
    "Set view_quality to PARTIAL or POOR when the view is blocked, blurred or too dark. "
    "If the fruit cannot be reliably recognized or staged, use UNKNOWN. "
    "Give a short visible_features description; no estimates of exact days or expiry dates."
)


class VisionAnalyzer:
    def __init__(self, backend: str | None = None):
        self.backend = (backend or os.getenv("FRESHPICK_BACKEND", "gemini")).strip().lower()
        self.model_name = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
        self.client = None
        self.local_model = None
        self.init_error = None
        if self.backend == "gemini":
            key = os.getenv("GEMINI_API_KEY", "").strip()
            if not key:
                self.init_error = "No GEMINI_API_KEY was found in the root .env file."
            else:
                try:
                    from google import genai
                    self.client = genai.Client(api_key=key)
                except Exception as exc:
                    self.init_error = f"Gemini client could not start: {exc}"
        elif self.backend == "moondream":
            # First use downloads a large model. This path is NOT inherently int4.
            try:
                from transformers import AutoModelForCausalLM
                model_revision = os.getenv("MOONDREAM_REVISION", "2025-06-21")
                self.local_model = AutoModelForCausalLM.from_pretrained(
                    "vikhyatk/moondream2", revision=model_revision,
                    trust_remote_code=True,
                )
                self.local_model.eval()
            except Exception as exc:
                self.init_error = f"Moondream2 could not load: {exc}"
        elif self.backend == "opencv":
            self.init_error = "Offline color-only mode: fruit recognition and ripeness ranking are disabled."
        else:
            self.init_error = f"Unknown backend {self.backend!r}. Use gemini, moondream, or opencv."
        if self.init_error:
            print("[VisionAnalyzer]", self.init_error)

    @staticmethod
    def _prepare_collage(bgr_frames: list[np.ndarray]) -> Image.Image | None:
        good_frames = [f for f in bgr_frames if isinstance(f, np.ndarray) and f.size > 0]
        if not good_frames:
            return None
        # Two views give useful context while keeping inference manageable.
        selected = [good_frames[0], good_frames[-1]] if len(good_frames) > 1 else [good_frames[0]]
        width, height = 440, 370
        collage = Image.new("RGB", (width * len(selected), height + 28), "white")
        drawer = ImageDraw.Draw(collage)
        for index, frame in enumerate(selected):
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb)
            img.thumbnail((width, height))
            x = index * width + (width - img.width) // 2
            y = 28 + (height - img.height) // 2
            collage.paste(img, (x, y))
            drawer.text((index * width + 10, 8), f"View {index + 1}", fill="black")
        return collage

    def analyze_views(self, frames: list[np.ndarray]) -> dict:
        image = self._prepare_collage(frames)
        if image is None:
            return unknown_result("No clear images were captured.")
        if self.init_error:
            outcome = unknown_result("The fruit recognition model is unavailable.")
            colors = self._describe_colors(image)
            if colors:
                outcome["surface_description"] = colors
                outcome["spoken_text"] = (
                    f"I can see {colors}, but cannot identify the fruit or rank its ripeness offline. "
                    "Please scan again with a configured vision model."
                )
            return outcome
        try:
            if self.backend == "gemini":
                observation = self._ask_gemini(image)
            else:
                observation = self._ask_moondream(image)
            return normalize_observation(observation)
        except Exception as exc:
            print("[VisionAnalyzer] analysis failed:", repr(exc))
            # Do not use color heuristics as if they provided reliable fruit identity.
            return unknown_result("The image analysis failed or was inconclusive.")

    @staticmethod
    def _describe_colors(image: Image.Image) -> str:
        """Conservative HSV diagnostic. Color alone cannot identify fruit or ripeness."""
        hsv = cv2.cvtColor(np.asarray(image.convert("RGB")), cv2.COLOR_RGB2HSV)
        masks = {
            "green tones": cv2.inRange(hsv, (36, 55, 55), (85, 255, 255)),
            "yellow tones": cv2.inRange(hsv, (18, 55, 55), (35, 255, 255)),
            "orange tones": cv2.inRange(hsv, (10, 60, 60), (17, 255, 255)),
            "red tones": cv2.bitwise_or(cv2.inRange(hsv, (0, 75, 55), (9, 255, 255)),
                                          cv2.inRange(hsv, (165, 75, 55), (179, 255, 255))),
        }
        count = hsv.shape[0] * hsv.shape[1]
        ranked = sorted(((int(np.count_nonzero(mask)) / count, name)
                         for name, mask in masks.items()), reverse=True)
        return ranked[0][1] if ranked and ranked[0][0] > 0.12 else ""

    def _ask_gemini(self, image: Image.Image) -> dict:
        from google.genai import types
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=[PROMPT, image],
            config=types.GenerateContentConfig(
                response_mime_type="application/json", response_schema=FruitObservation,
                temperature=0.1,
            ),
        )
        if not response.text:
            raise ValueError("Empty model response")
        return FruitObservation.model_validate_json(response.text).model_dump()

    def _ask_moondream(self, image: Image.Image) -> dict:
        # Moondream2 lacks Gemini's enforced JSON schema. Validate its text strictly.
        question = (PROMPT + " Output one JSON object with exactly these keys: "
                    "fruit_name, fruit_visible, fruit_count, view_quality, "
                    "ripeness_stage, possible_mold_or_rot, visible_features. "
                    "No markdown, only JSON.")
        answer = self.local_model.query(image, question)["answer"].strip()
        if answer.startswith("```"):
            answer = answer.strip("`").removeprefix("json").strip()
        start, end = answer.find("{"), answer.rfind("}")
        if start < 0 or end < start:
            raise ValueError("Moondream did not return a JSON object")
        return FruitObservation.model_validate(json.loads(answer[start:end + 1])).model_dump()

    def analyze_single_fruit(self, bgr_crop: np.ndarray) -> dict:
        """Backward-compatible entry point for the previous FreshPick code."""
        return self.analyze_views([bgr_crop])
