"""Kinds: every layer names monsters and drops by kind, a name from the game in lower case with dashes
("Orange Mushroom" -> "orange-mushroom")."""


def kind_of(name: str) -> str:
    """A monster's or item's kind, from its name in the game: "Orange Mushroom" -> "orange-mushroom"."""
    return name.strip().lower().replace(" ", "-")
