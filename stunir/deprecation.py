"""STUNIR Deprecation Utilities.

Provides decorators and utilities for marking deprecated functions
and classes with proper warnings and documentation.

Example:
    >>> from stunir.deprecation import deprecated
    >>> @deprecated("Use new_function() instead", "1.1.0")
    ... def old_function():
    ...     pass
"""

import functools
import warnings
from typing import Any, Callable, Optional, TypeVar

F = TypeVar("F", bound=Callable[..., Any])


def deprecated(
    reason: str,
    version: str,
    removal_version: Optional[str] = None,
    alternative: Optional[str] = None,
) -> Callable[[F], F]:
    """Mark a function as deprecated.

    Emits a DeprecationWarning when the decorated function is called.
    The warning includes the deprecation reason, version, and migration path.

    Args:
        reason: Explanation of why the function is deprecated.
        version: Version when the function was deprecated.
        removal_version: Version when the function will be removed (optional).
        alternative: Name of the function to use instead (optional).

    Returns:
        A decorator that wraps the function with deprecation warnings.

    Example:
        >>> @deprecated(
        ...     "This function uses an old algorithm",
        ...     "1.1.0",
        ...     removal_version="2.0.0",
        ...     alternative="compute_sha256"
        ... )
        ... def old_hash(data):
        ...     pass
    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            message = f"{func.__qualname__} is deprecated since version {version}. {reason}"
            if alternative:
                message += f" Use {alternative}() instead."
            if removal_version:
                message += f" Will be removed in version {removal_version}."

            warnings.warn(message, DeprecationWarning, stacklevel=2)
            return func(*args, **kwargs)

        # Update docstring
        deprecation_note = f"\n\n.. deprecated:: {version}\n   {reason}"
        if alternative:
            deprecation_note += f"\n   Use :func:`{alternative}` instead."
        if removal_version:
            deprecation_note += f"\n   Will be removed in {removal_version}."

        if wrapper.__doc__:
            wrapper.__doc__ = deprecation_note + "\n\n" + wrapper.__doc__
        else:
            wrapper.__doc__ = deprecation_note

        return wrapper  # type: ignore

    return decorator


def deprecated_parameter(
    param_name: str,
    reason: str,
    version: str,
    alternative: Optional[str] = None,
) -> Callable[[F], F]:
    """Mark a function parameter as deprecated.

    Emits a DeprecationWarning when the decorated function is called
    with the deprecated parameter.

    Args:
        param_name: Name of the deprecated parameter.
        reason: Explanation of why the parameter is deprecated.
        version: Version when the parameter was deprecated.
        alternative: Name of the parameter to use instead (optional).

    Returns:
        A decorator that wraps the function with parameter deprecation warnings.
    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            if param_name in kwargs:
                message = (
                    f"Parameter '{param_name}' of {func.__qualname__} is deprecated "
                    f"since version {version}. {reason}"
                )
                if alternative:
                    message += f" Use '{alternative}' instead."
                warnings.warn(message, DeprecationWarning, stacklevel=2)
            return func(*args, **kwargs)

        return wrapper  # type: ignore

    return decorator


class DeprecatedClass:
    """Base class for deprecated classes.

    Subclass this to create a deprecated class that emits warnings
    when instantiated.

    Example:
        >>> class OldClass(DeprecatedClass):
        ...     _deprecation_version = "1.1.0"
        ...     _deprecation_reason = "Use NewClass instead"
    """

    _deprecation_version: str = "unknown"
    _deprecation_reason: str = "This class is deprecated"
    _alternative: Optional[str] = None

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Emit deprecation warning when subclass is defined."""
        super().__init_subclass__(**kwargs)
        message = (
            f"{cls.__name__} is deprecated since version {cls._deprecation_version}. "
            f"{cls._deprecation_reason}"
        )
        if cls._alternative:
            message += f" Use {cls._alternative} instead."
        warnings.warn(message, DeprecationWarning, stacklevel=2)

    def __new__(cls, *args: Any, **kwargs: Any) -> "DeprecatedClass":
        """Emit deprecation warning when instance is created."""
        message = (
            f"{cls.__name__} is deprecated since version {cls._deprecation_version}. "
            f"{cls._deprecation_reason}"
        )
        if cls._alternative:
            message += f" Use {cls._alternative} instead."
        warnings.warn(message, DeprecationWarning, stacklevel=2)
        return super().__new__(cls)
