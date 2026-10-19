"""Left and right: a direction (maplehelper.engine's LEFT and RIGHT, "left" and "right", also the names of
her arrow keys) and its sign along map x, which grows to the right."""

from maplehelper.engine import LEFT, RIGHT


def sign(direction: str) -> int:
    """+1 for RIGHT, -1 for LEFT: which way map x goes that way."""
    return 1 if direction == RIGHT else -1


def toward(dx: float) -> str:
    """The direction of a step of `dx` along map x: RIGHT if it's positive, else LEFT."""
    return RIGHT if dx > 0 else LEFT


def opposite(direction: str) -> str:
    """LEFT for RIGHT, RIGHT for LEFT."""
    return LEFT if direction == RIGHT else RIGHT
