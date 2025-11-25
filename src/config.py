from dataclasses import dataclass


@dataclass(frozen=True)
class CourtDimensions:
    SINGLE_LINE_WIDTH_M: float = 8.23
    DOUBLE_LINE_WIDTH_M: float = 10.97
    HALF_COURT_LINE_HEIGHT_M: float = 11.88
    SERVICE_LINE_WIDTH_M: float = 6.40
    DOUBLE_ALLY_DIFFERENCE_M: float = 1.37
    NO_MANS_LAND_HEIGHT_M: float = 5.48


@dataclass(frozen=True)
class PlayerHeights:
    PLAYER_1_HEIGHT_M: float = 1.91
    PLAYER_2_HEIGHT_M: float = 1.88
