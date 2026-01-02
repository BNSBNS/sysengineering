"""Dependency injection container."""

from __future__ import annotations

from typing import Any, Callable, TypeVar, Generic

T = TypeVar("T")


class Container:
    """
    Simple dependency injection container.

    Supports singleton and factory registrations with lazy initialization.
    """

    def __init__(self) -> None:
        """Initialize the container."""
        self._singletons: dict[type, Any] = {}
        self._factories: dict[type, Callable[[Container], Any]] = {}
        self._instances: dict[type, Any] = {}

    def register_singleton(self, interface: type[T], instance: T) -> None:
        """
        Register a singleton instance.

        Args:
            interface: The interface/type to register
            instance: The singleton instance
        """
        self._singletons[interface] = instance
        self._instances[interface] = instance

    def register_factory(
        self,
        interface: type[T],
        factory: Callable[[Container], T],
    ) -> None:
        """
        Register a factory function for lazy instantiation.

        Args:
            interface: The interface/type to register
            factory: Factory function that takes the container and returns an instance
        """
        self._factories[interface] = factory

    def resolve(self, interface: type[T]) -> T:
        """
        Resolve a dependency.

        Args:
            interface: The interface/type to resolve

        Returns:
            The resolved instance

        Raises:
            KeyError: If no registration exists for the interface
        """
        # Check if already instantiated
        if interface in self._instances:
            return self._instances[interface]

        # Check singletons
        if interface in self._singletons:
            return self._singletons[interface]

        # Check factories
        if interface in self._factories:
            instance = self._factories[interface](self)
            self._instances[interface] = instance
            return instance

        raise KeyError(f"No registration found for {interface}")

    def has(self, interface: type) -> bool:
        """Check if an interface is registered."""
        return (
            interface in self._singletons
            or interface in self._factories
            or interface in self._instances
        )

    def clear(self) -> None:
        """Clear all registrations and instances."""
        self._singletons.clear()
        self._factories.clear()
        self._instances.clear()


class Provider(Generic[T]):
    """
    Lazy dependency provider.

    Allows declaring dependencies that are resolved lazily from the container.
    """

    def __init__(self, container: Container, interface: type[T]) -> None:
        """
        Initialize the provider.

        Args:
            container: The DI container
            interface: The interface to provide
        """
        self._container = container
        self._interface = interface
        self._instance: T | None = None

    def get(self) -> T:
        """Get the provided instance (lazily resolved)."""
        if self._instance is None:
            self._instance = self._container.resolve(self._interface)
        return self._instance

    def __call__(self) -> T:
        """Callable shorthand for get()."""
        return self.get()


# Global container instance
_container: Container | None = None


def get_container() -> Container:
    """Get the global container instance."""
    global _container
    if _container is None:
        _container = Container()
    return _container


def reset_container() -> None:
    """Reset the global container (useful for testing)."""
    global _container
    if _container is not None:
        _container.clear()
    _container = None
