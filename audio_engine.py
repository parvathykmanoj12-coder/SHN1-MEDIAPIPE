"""Interruptible speech on Windows (SAPI via isolated PowerShell process).

The audio worker never blocks the webcam. On Windows, cancelling speech or
muting terminates the current speech process. On other systems pyttsx3 is used
but it cannot reliably interrupt a currently speaking voice.
"""
from __future__ import annotations

import base64
import os
import queue
import subprocess
import threading
import time

try:
    import winsound
except ImportError:
    winsound = None
try:
    import pyttsx3
except ImportError:
    pyttsx3 = None


class AudioEngine:
    def __init__(self, rate: int = 190, volume: float = 0.65, muted: bool = False):
        self.rate = rate
        self.volume = max(0.0, min(1.0, volume))
        self.muted = muted
        self.is_speaking = False
        self.last_hint_time = 0.0
        self._messages: queue.Queue[str | None] = queue.Queue()
        self._done = threading.Event()
        self._process_lock = threading.RLock()
        self._current_process = None
        self._worker = threading.Thread(target=self._speak_worker, daemon=True)
        self._worker.start()

    def _windows_sapi(self, message: str) -> bool:
        """Speak using cancellable Windows SAPI, without putting user text in a shell argument."""
        sanitized = message.replace("'", "''")
        voice_rate = max(-10, min(10, round((self.rate - 180) / 18)))
        script = (
            "Add-Type -AssemblyName System.Speech; "
            "$voice=New-Object System.Speech.Synthesis.SpeechSynthesizer; "
            f"$voice.Rate={voice_rate}; $voice.Volume={round(self.volume * 100)}; "
            f"$voice.Speak('{sanitized}');"
        )
        encoded = base64.b64encode(script.encode("utf-16le")).decode("ascii")
        try:
            process = subprocess.Popen(
                ["powershell.exe", "-NoProfile", "-NonInteractive", "-EncodedCommand", encoded],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            with self._process_lock:
                self._current_process = process
            code = process.wait()
            with self._process_lock:
                if self._current_process is process:
                    self._current_process = None
            return code == 0
        except (OSError, subprocess.SubprocessError) as exc:
            print("[FreshPick audio] Windows voice unavailable:", exc)
            return False

    def _speak_worker(self) -> None:
        engine = None
        if os.name != "nt" and pyttsx3 is not None:
            try:
                engine = pyttsx3.init()
                engine.setProperty("rate", self.rate)
                engine.setProperty("volume", self.volume)
            except Exception as exc:
                print("[FreshPick audio warning]", exc)
        while not self._done.is_set():
            try:
                message = self._messages.get(timeout=0.2)
            except queue.Empty:
                continue
            if message is None:
                self._messages.task_done()
                break
            if self.muted:
                self._messages.task_done()
                continue
            print("[FreshPick voice]", message)
            self.is_speaking = True
            try:
                if os.name == "nt":
                    if not self._windows_sapi(message):
                        print("[FreshPick audio] Could not play Windows voice message.")
                elif engine is not None:
                    engine.say(message)
                    engine.runAndWait()
            except Exception as exc:
                print("[FreshPick TTS warning]", exc)
            finally:
                self.is_speaking = False
                self._messages.task_done()

    def stop_pending(self) -> None:
        while True:
            try:
                self._messages.get_nowait()
                self._messages.task_done()
            except queue.Empty:
                break

    def stop_current(self) -> None:
        # On Windows the speech engine lives in a killable subprocess.
        with self._process_lock:
            process = self._current_process
            if process is not None and process.poll() is None:
                process.terminate()

    def speak(self, message: str, interrupt: bool = False) -> None:
        if interrupt:
            self.stop_pending()
            self.stop_current()
        if message and not self.muted:
            self._messages.put(message)

    def speak_centering_hint(self, message: str) -> None:
        now = time.monotonic()
        if message and now - self.last_hint_time >= 4.0 and not self.is_speaking:
            self.last_hint_time = now
            self.speak(message)

    def play_beep(self, freq: int = 850, duration_ms: int = 90) -> None:
        if self.muted or winsound is None:
            return
        def sound():
            try:
                winsound.Beep(freq, duration_ms)
            except RuntimeError:
                pass
        threading.Thread(target=sound, daemon=True).start()

    def play_scan_trigger_tone(self) -> None:
        self.play_beep(1100, 110)

    def toggle_mute(self) -> bool:
        self.muted = not self.muted
        if self.muted:
            self.stop_pending()
            self.stop_current()
        return self.muted

    def shutdown(self) -> None:
        self._done.set()
        self.stop_pending()
        self.stop_current()
        self._messages.put(None)
        self._worker.join(timeout=0.8)
