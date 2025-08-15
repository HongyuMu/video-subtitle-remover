from enum import Enum, unique

@unique
class SubtitleArea(Enum):
    UPPER_PART = 0
    LOWER_PART = 1


# BGR
BGR_COLOR_WHITE = (255, 255, 255)
BGR_COLOR_BLACK = (0, 0, 0)
BGR_COLOR_RED = (0, 0, 255)
BGR_COLOR_GREEN = (0, 255, 0)
BGR_COLOR_BLUE = (255, 0, 0)
