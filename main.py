"""FreshPick multiview, voice-first fruit scanning prototype.

Commands: pinch or SPACE = guided scan; open palm or R = priority summary;
thumbs-up or H = safety reminder; M = mute; X = cancel waiting speech;
U = undo last; Q = quit. No visual UI action is essential to the core scan.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import queue
import threading
import time

import cv2
from dotenv import load_dotenv

from audio_engine import AudioEngine
from gesture_controller import GestureController
from hud import HUD
from pantry_memory import PantryMemory
from vision_analyzer import VisionAnalyzer
from workflow import Phase, ScanSession

ROOT = Path(__file__).resolve().parent
SAFETY_NOTE = (
    "The camera sees only the outside. It cannot establish food safety. "
    "Check for unusual softness, odor, or visible mold and discard suspicious fruit."
)


def crop_around_fruit(frame, bbox):
    h, w = frame.shape[:2]
    if bbox:
        x1, y1, x2, y2 = bbox
    else:
        # Central ROI fallback is for one fruit held in front of the lens.
        x1, y1, x2, y2 = int(w * .16), int(h * .13), int(w * .84), int(h * .90)
    if x2 <= x1 or y2 <= y1:
        return None
    return frame[y1:y2, x1:x2].copy()


def main():
    parser = argparse.ArgumentParser(description="FreshPick voice-first guided fruit scanner")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--backend", choices=("gemini", "moondream", "opencv"), default="gemini")
    parser.add_argument("--tts-volume", type=float, default=0.65)
    args = parser.parse_args()
    load_dotenv(ROOT / ".env", override=False)

    audio = AudioEngine(rate=190, volume=args.tts_volume)
    vision = VisionAnalyzer(backend=args.backend)
    pantry = PantryMemory(path=str(ROOT / "pantry_session.json"))
    gestures = None
    camera = None
    session = ScanSession()
    results = queue.SimpleQueue()
    last_readout = "FreshPick ready."
    caption = "Pinch or press SPACE to scan. Open palm or R for priority list."

    try:
        gestures = GestureController()
        camera = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW) if __import__("sys").platform == "win32" else cv2.VideoCapture(args.camera)
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)
        if not camera.isOpened():
            raise RuntimeError("No working camera found. Try --camera 1 or check permissions.")
        print("Controls: SPACE=scan, R=recap, H=safety, M=mute, X=drop queued speech, U=undo, Q=quit")
        audio.speak("FreshPick ready. Pinch to start a fruit scan. Open your palm to hear the use-first list.")

        def launch_inference(frames):
            def runner():
                try:
                    output = vision.analyze_views(frames)
                except Exception as exc:
                    from decision_engine import unknown_result
                    print("Unexpected inference error:", exc)
                    output = unknown_result("The vision system encountered an error.")
                results.put(output)
            threading.Thread(target=runner, daemon=True).start()

        while True:
            ok, frame = camera.read()
            if not ok:
                print("Camera frame unavailable; exiting.")
                break
            frame = cv2.flip(frame, 1)
            now = time.monotonic()
            # No gesture triggers during active capture, but hand geometry continues.
            hand = gestures.process_frame(frame, allow_gestures=session.phase == Phase.IDLE)
            command = hand.event
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord(" "):
                command = "PINCH_SCAN"
            elif key == ord("r"):
                command = "PALM_RECAP"
            elif key == ord("h"):
                command = "THUMB_SAFETY"
            elif key == ord("m"):
                was_muted = audio.toggle_mute()
                caption = "Audio muted" if was_muted else "Audio unmuted"
                if not was_muted:
                    audio.speak("Sound on.")
            elif key == ord("x"):
                audio.stop_pending()
                audio.stop_current()
                caption = "Speech stopped."
            elif key == ord("u") and session.phase == Phase.IDLE:
                caption = "Last entry removed" if pantry.remove_last() else "Nothing to undo"
                audio.speak(caption, interrupt=True)

            if command == "PINCH_SCAN" and session.begin(now):
                gestures.reset_rotation_reference()
                audio.play_scan_trigger_tone()
                caption = "Hold one fruit steady. Starting guided scan."
                audio.speak("Hold the fruit in the middle.", interrupt=True)
            elif command == "PALM_RECAP" and session.phase == Phase.IDLE:
                last_readout = pantry.get_spoken_recap(max_items=3)
                caption = last_readout
                audio.speak(last_readout, interrupt=True)
            elif command == "THUMB_SAFETY" and session.phase == Phase.IDLE:
                caption = SAFETY_NOTE
                audio.speak(SAFETY_NOTE, interrupt=True)

            # Always capture raw, unannotated frames so HUD graphics do not confuse the VLM.
            if session.phase in (Phase.ALIGN, Phase.ROTATE):
                slot = session.due_capture_slot(now)
                if slot is not None:
                    crop = crop_around_fruit(frame, hand.bbox)
                    if crop is not None:
                        session.frames.append(crop)
                        audio.play_beep(880 + slot * 140, 70)
                if session.elapsed(now) >= 2.0 and not session.turn_announced:
                    session.turn_announced = True
                    caption = "Slowly rotate the fruit to show a different side."
                    audio.speak("Turn fruit slowly.")
                if (hand.estimated_hand_turn_degrees is not None
                        and hand.estimated_hand_turn_degrees > 65
                        and not session.last_rotation_tone):
                    session.last_rotation_tone = True
                    audio.play_beep(1250, 110)  # Only a *hand orientation* hint.

                previous = session.phase
                session.advance(now)
                if previous != Phase.ANALYZING and session.phase == Phase.ANALYZING:
                    caption = "Checking visible fruit features."
                    audio.speak("Checking the fruit.")
                    launch_inference(session.frames.copy())

            elif session.phase == Phase.ANALYZING:
                if now - session.last_tick_at >= 1.5 and not audio.is_speaking:
                    session.last_tick_at = now
                    audio.play_beep(540, 35)
                try:
                    result = results.get_nowait()
                except queue.Empty:
                    result = None
                if result is not None:
                    saved = pantry.add_item(result)
                    last_readout = result["spoken_text"]
                    if saved:
                        last_readout += " Added to your use-first list."
                    caption = last_readout
                    audio.speak(last_readout, interrupt=True)
                    session.readout(now)
            elif session.phase == Phase.READOUT:
                session.advance(now)
                if session.phase == Phase.IDLE:
                    caption = "Ready for another fruit. Pinch or press SPACE."
            elif session.phase == Phase.IDLE and hand.hint:
                # Do not spam if user isn't preparing to scan; only guide when hand is visible.
                if hand.hand_visible:
                    audio.speak_centering_hint(hand.hint)

            img = HUD.render(frame, session.phase.value, caption, pantry.get_sorted_items(),
                             bbox=hand.bbox, progress=hand.progress,
                             elapsed=session.elapsed(now),
                             turn_degrees=(round(hand.estimated_hand_turn_degrees)
                                           if hand.estimated_hand_turn_degrees is not None else None),
                             muted=audio.muted)
            # Visualize MediaPipe joints on the preview, NOT on frames sent to Gemini.
            img = gestures.draw_landmarks(img)
            cv2.imshow("FreshPick | Audio-first prototype", img)

    except (KeyboardInterrupt, Exception) as exc:
        print("FreshPick stopped:", repr(exc))
        raise
    finally:
        if camera is not None:
            camera.release()
        if gestures is not None:
            gestures.close()
        audio.shutdown()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
