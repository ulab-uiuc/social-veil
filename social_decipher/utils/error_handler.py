import math
import time
from functools import wraps
import re

# Attempt to import openai, but allow it to fail gracefully
try:
    import openai
except ImportError:
    openai = None

from beartype.typing import Any, Callable, Optional, TypeVar, cast
from pydantic import BaseModel
from rich import print

INF = float(math.inf)

T = TypeVar("T", bound=Callable[..., Any])


def api_calling_error_exponential_backoff(
    retries: int = 5, base_wait_time: int = 1
) -> Callable[[T], T]:
    """
    Decorator for applying exponential backoff to a function.
    :param retries: Maximum number of retries.
    :param base_wait_time: Base wait time in seconds for the exponential backoff.
    :return: The wrapped function with exponential backoff applied.
    """

    def decorator(func: T) -> T:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            error_handler_mode = kwargs.get("mode", None)
            if error_handler_mode == "TEST":
                modified_retries = 1
                modified_base_wait_time = 1
            else:
                modified_retries = retries
                modified_base_wait_time = base_wait_time

            attempts = 0
            while attempts < modified_retries:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    wait_time = 0.0
                    # Check specifically for OpenAI's RateLimitError
                    if openai and isinstance(e, openai.RateLimitError):
                        error_message = str(e)
                        match = re.search(r"try again in ([\d\.]+)s", error_message)
                        if match:
                            # Use suggested wait time from API + small jitter
                            wait_time = float(match.group(1)) + 0.2
                            print(f"[API calling error] OpenAI rate limit hit. Waiting {wait_time:.2f}s as suggested by the API (attempt {attempts + 1}/{modified_retries})...")
                        else:
                            # Fallback for rate limit errors without a specific time
                            wait_time = modified_base_wait_time * (2**attempts)
                            print(f"[API calling error] OpenAI rate limit hit. Waiting {wait_time:.2f}s (attempt {attempts + 1}/{modified_retries})...")
                    else:
                        wait_time = modified_base_wait_time * (2**attempts)
                        print(
                            f"[API calling error] Attempt {attempts + 1} failed with {type(e).__name__}. Waiting {wait_time} seconds before retrying..."
                        )
                    time.sleep(wait_time)
                    attempts += 1
            print(
                f"Failed to execute '{func.__name__}' after {modified_retries} retries."
            )
            return None

        return cast(T, wrapper)

    return cast(Callable[[T], T], decorator)


TBaseModel = TypeVar("TBaseModel", bound=Callable[..., BaseModel])


def parsing_error_exponential_backoff(
    retries: int = 5, base_wait_time: int = 1
) -> Callable[[TBaseModel], TBaseModel]:
    """
    Decorator for retrying a function that returns a BaseModel with exponential backoff.
    :param retries: Maximum number of retries.
    :param base_wait_time: Base wait time in seconds for the exponential backoff.
    :return: The wrapped function with retry logic applied.
    """

    def decorator(func: TBaseModel) -> TBaseModel:
        @wraps(func)
        def wrapper(self: Any, *args: Any, **kwargs: Any) -> Optional[BaseModel]:
            attempts = 0
            while attempts < retries:
                try:
                    return func(self, *args, **kwargs)
                except Exception as e:
                    wait_time = base_wait_time * (2**attempts)
                    print(f"Attempt {attempts + 1} failed: {e}")
                    print(f"Waiting {wait_time} seconds before retrying...")
                    time.sleep(wait_time)
                    attempts += 1
            print(
                f"Failed to get valid input from {func.__name__} after {retries} retries."
            )
            return None

        return cast(TBaseModel, wrapper)

    return cast(Callable[[TBaseModel], TBaseModel], decorator)
