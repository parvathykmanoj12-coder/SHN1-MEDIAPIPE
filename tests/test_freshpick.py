import tempfile
from pathlib import Path
import unittest

from decision_engine import normalize_observation, unknown_result
from pantry_memory import PantryMemory
from workflow import Phase, ScanSession


def observation(name, stage):
    return dict(fruit_name=name, visible_features="yellow skin, some spots",
                ripeness_stage=stage, view_quality="CLEAR", fruit_visible=True,
                fruit_count=1, possible_mold_or_rot=False)


class FreshPickTests(unittest.TestCase):
    def test_three_urgency_weights(self):
        self.assertEqual(normalize_observation(observation("Banana", "OVERRIPE"))["weight"], 1)
        self.assertEqual(normalize_observation(observation("Mango", "RIPE"))["weight"], 2)
        self.assertEqual(normalize_observation(observation("Papaya", "UNDERRIPE"))["weight"], 3)

    def test_unknown_is_not_ranked(self):
        self.assertTrue(unknown_result("Blurred")["review_required"])
        self.assertIsNone(unknown_result("Blurred")["weight"])

    def test_suspicious_mold_is_not_recommended_to_eat(self):
        sample = observation("Apple", "OVERRIPE")
        sample["possible_mold_or_rot"] = True
        outcome = normalize_observation(sample)
        self.assertEqual(outcome["urgency"], "REVIEW_REQUIRED")
        self.assertIn("Do not eat based", outcome["spoken_text"])

    def test_pantry_sort_and_storage(self):
        with tempfile.TemporaryDirectory() as folder:
            filename = str(Path(folder) / "pantry.json")
            pantry = PantryMemory(filename)
            for name, stage in (("Mango", "RIPE"), ("Banana", "OVERRIPE"), ("Pear", "UNDERRIPE")):
                pantry.add_item(normalize_observation(observation(name, stage)))
            pantry.add_item(unknown_result())
            self.assertEqual([x["fruit_name"] for x in pantry.get_sorted_items()], ["Banana", "Mango", "Pear"])
            self.assertEqual(len(PantryMemory(filename).items), 3)
            self.assertIn("Banana", pantry.get_spoken_recap(3))
            pantry.remove_last()
            self.assertEqual(len(pantry.items), 2)

    def test_offline_fallback_never_ranks_from_color(self):
        import numpy as np
        from vision_analyzer import VisionAnalyzer
        model = VisionAnalyzer("opencv")
        solid_yellow = np.full((200, 200, 3), (0, 230, 230), dtype=np.uint8)
        result = model.analyze_views([solid_yellow])
        self.assertTrue(result["review_required"])
        self.assertIsNone(result["weight"])
        self.assertIn("yellow", result["surface_description"])

    def test_timeline(self):
        s = ScanSession()
        self.assertTrue(s.begin(10.0))
        self.assertEqual(s.phase, Phase.ALIGN)
        self.assertEqual(s.due_capture_slot(11.5), 0)
        self.assertEqual(s.due_capture_slot(11.6), None)
        self.assertEqual(s.advance(12.1), Phase.ROTATE)
        self.assertEqual(s.due_capture_slot(13.4), 1)
        self.assertEqual(s.due_capture_slot(14.8), 2)
        self.assertEqual(s.advance(15.0), Phase.ANALYZING)
        s.readout(17.0)
        self.assertEqual(s.advance(21.1), Phase.IDLE)


if __name__ == "__main__":
    unittest.main()
