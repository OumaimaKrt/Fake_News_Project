import time
import logging
import functools
from typing import Callable, Any

logging.basicConfig(
    filename="logs/predictions.log",
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

def timer(func: Callable) -> Callable:

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        start = time.perf_counter()        
        result = func(*args, **kwargs)
        elapsed = round(time.perf_counter() - start, 4)
        logger.info("[TIMER] %s → %.4fs", func.__name__, elapsed)
        return result
    return wrapper

def log_prediction(func: Callable) -> Callable:

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        result = func(*args, **kwargs)
        input_text = str(args[1])[:80] if len(args) > 1 else "N/A"
        label = result.get("label", "UNKNOWN")
        logger.info("[PREDICTION] label=%-4s | text='%s...'", label, input_text)
        return result
    return wrapper

def retry(max_attempts: int = 3, delay: float = 1.5, exceptions: tuple = (Exception,)) -> Callable:
    
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            last_exception = None
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as exc:
                    last_exception = exc
                    logger.warning(
                        "[RETRY] %s | attempt %d/%d | error: %s",
                        func.__name__, attempt, max_attempts, exc
                    )
                    if attempt < max_attempts:
                        time.sleep(delay)
            logger.error("[RETRY] %s failed after %d attempts.", func.__name__, max_attempts)
            raise last_exception
        return wrapper
    return decorator