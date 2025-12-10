from typing import Mapping


# A custom Counter for better extensibility
class Counter[T](dict[T, int]):
    """An ordered Counter implementation that counts hashable items."""

    def __missing__(self, key: T) -> int:
        "The count of elements not in the Counter is zero."
        # Needed so that self[missing_item] does not raise KeyError
        return 0

    def __add__(self, other: "Counter[T]") -> "Counter[T]":
        result = Counter[T]()
        for key in set(self) | set(other):
            new_count = self.get(key, 0) + other.get(key, 0)
            if new_count != 0:
                result[key] = new_count
        return result

    def __iadd__(self, other: "Counter[T]") -> "Counter[T]":
        for key in set(self) | set(other):
            new_count = self.get(key, 0) + other.get(key, 0)
            if new_count != 0:
                self[key] = new_count
        return self

    def __mul__(self, other: int) -> "Counter[T]":
        result = Counter[T]()
        for key in self:
            new_count = self[key] * other
            if new_count > 0:
                result[key] = new_count
        return result


def scale(c: Mapping[str, int], k: int) -> Counter:
    """Helper function to multiply all counts in a mapping by k."""
    return Counter({key: value * k for key, value in c.items()})
