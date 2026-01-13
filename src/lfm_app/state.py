from dataclasses import dataclass


@dataclass
class SessionState:
    transcript: str = ""
    translation: str = ""
