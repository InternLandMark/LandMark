from dataclasses import dataclass


@dataclass
class Camera:
    H: int
    W: int
    focal: int
