from typing import Generic, Iterable, Iterator, MutableSet, TypeVar

T = TypeVar("T")


class OrderedSet(Generic[T], MutableSet[T]):
    def __init__(self, iterable: Iterable[T] | None = None) -> None:
        self._items: list[T] = []
        self._set: set[T] = set()
        if iterable is not None:
            for item in iterable:
                self.add(item)

    def __contains__(self, x: T) -> bool:  # type: ignore[override]
        return x in self._set

    def __iter__(self) -> Iterator[T]:  # type: ignore[override]
        return iter(self._items)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self._items)

    def add(self, value: T) -> None:  # type: ignore[override]
        if value not in self._set:
            self._set.add(value)
            self._items.append(value)

    def discard(self, value: T) -> None:  # type: ignore[override]
        if value in self._set:
            self._set.remove(value)
            try:
                self._items.remove(value)
            except ValueError:
                pass

    def __sub__(self, other: "OrderedSet[T]") -> "OrderedSet[T]":
        return OrderedSet(x for x in self if x not in other)

    def __or__(self, other: "OrderedSet[T]") -> "OrderedSet[T]":
        return OrderedSet([*self, *[x for x in other if x not in self]])

    def update(self, iterable: Iterable[T]) -> None:
        for item in iterable:
            self.add(item)


