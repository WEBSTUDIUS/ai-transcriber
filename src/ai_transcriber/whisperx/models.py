from dataclasses import dataclass


@dataclass
class SpeakerSegment:
    speaker: str
    start: float
    end: float
    text: str
