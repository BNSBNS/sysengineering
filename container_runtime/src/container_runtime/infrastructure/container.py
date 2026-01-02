"""Dependency injection container."""

from __future__ import annotations

from typing import Any, Callable, TypeVar

T = TypeVar("T")


class Container:
    """Simple dependency injection container."""

    def __init__(self) -> None:
        self._singletons: dict[type, Any] = {}
        self._factories: dict[type, Callable[[Container], Any]] = {}
        self._instances: dict[type, Any] = {}

    def register_singleton(self, interface: type[T], instance: T) -> None:
        """Register a singleton instance."""
        self._singletons[interface] = instance
        self._instances[interface] = instance

    def register_factory(self, interface: type[T], factory: Callable[[Container], T]) -> None:
        """Register a factory function."""
        self._factories[interface] = factory

    def resolve(self, interface: type[T]) -> T:
        """Resolve a dependency."""
        if interface in self._instances:
            return self._instances[interface]
        if interface in self._singletons:
            return self._singletons[interface]
        if interface in self._factories:
            instance = self._factories[interface](self)
            self._instances[interface] = instance
            return instance
        raise KeyError(f"No registration found for {interface}")

    def has(self, interface: type) -> bool:
        """Check if an interface is registered."""
        return interface in self._singletons or interface in self._factories or interface in self._instances

    def clear(self) -> None:
        """Clear all registrations."""
        self._singletons.clear()
        self._factories.clear()
        self._instances.clear()


_container: Container | None = None


def get_container() -> Container:
    """Get the global container instance."""
    global _container
    if _container is None:
        _container = Container()
    return _container


def reset_container() -> None:
    """Reset the global container."""
    global _container
    if _container is not None:
        _container.clear()
    _container = None
