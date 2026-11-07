"""Testable 10-second-target capture timeline; inference time is not assumed."""
from dataclasses import dataclass, field
from enum import Enum


class Phase(str, Enum):
    IDLE = "IDLE"
    ALIGN = "ALIGN"
    ROTATE = "ROTATE"
    ANALYZING = "ANALYZING"
    READOUT = "READOUT"


@dataclass
class ScanSession:
    phase: Phase = Phase.IDLE
    started_at: float = 0.0
    readout_at: float = 0.0
    frames: list = field(default_factory=list)
    captured_slots: set = field(default_factory=set)
    turn_announced: bool = False
    last_tick_at: float = 0.0
    last_rotation_tone: bool = False

    def begin(self, now: float) -> bool:
        if self.phase != Phase.IDLE:
            return False
        self.phase = Phase.ALIGN
        self.started_at = now
        self.frames.clear()
        self.captured_slots.clear()
        self.turn_announced = False
        self.last_rotation_tone = False
        self.last_tick_at = now
        return True

    def elapsed(self, now: float) -> float:
        return max(0, now - self.started_at) if self.phase != Phase.IDLE else 0.0

    def advance(self, now: float) -> Phase:
        age = self.elapsed(now)
        if self.phase == Phase.ALIGN and age >= 2.0:
            self.phase = Phase.ROTATE
        if self.phase == Phase.ROTATE and age >= 5.0:
            self.phase = Phase.ANALYZING
        if self.phase == Phase.READOUT and now - self.readout_at >= 4.0:
            self.phase = Phase.IDLE
        return self.phase

    def due_capture_slot(self, now: float) -> int | None:
        """Take an initial view and two spaced rotation views, not every video frame."""
        for slot, due in ((0, 1.3), (1, 3.3), (2, 4.7)):
            if self.elapsed(now) >= due and slot not in self.captured_slots and self.phase in {Phase.ALIGN, Phase.ROTATE}:
                self.captured_slots.add(slot)
                return slot
        return None

    def readout(self, now: float) -> None:
        self.phase = Phase.READOUT
        self.readout_at = now
