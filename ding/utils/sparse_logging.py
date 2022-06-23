from ditk import logging
import itertools
import timeit

_log_counter_per_token = {}
_log_timer_per_token = {}


def _get_next_log_count_per_token(token):
    """Wrapper for _log_counter_per_token. Thread-safe.
    Args:
    token: The token for which to look up the count.
    Returns:
    The number of times this function has been called with
    *token* as an argument (starting at 0).
    """
    # Can't use a defaultdict because defaultdict isn't atomic, whereas
    # setdefault is.
    return next(_log_counter_per_token.setdefault(token, itertools.count()))


def _seconds_have_elapsed(token, num_seconds):
    """Tests if 'num_seconds' have passed since 'token' was requested.
    Not strictly thread-safe - may log with the wrong frequency if called
    concurrently from multiple threads. Accuracy depends on resolution of
    'timeit.default_timer()'.
    Always returns True on the first call for a given 'token'.
    Args:
    token: The token for which to look up the count.
    num_seconds: The number of seconds to test for.
    Returns:
    Whether it has been >= 'num_seconds' since 'token' was last requested.
    """
    now = timeit.default_timer()
    then = _log_timer_per_token.get(token, None)
    if then is None or (now - then) >= num_seconds:
        _log_timer_per_token[token] = now
        return True
    else:
        return False


def log_every_n(level, n, msg, *args):
    """Logs 'msg % args' at level 'level' once per 'n' times.
    Logs the 1st call, (N+1)st call, (2N+1)st call,  etc.
    Not threadsafe.
    Args:
    level: int, the absl logging level at which to log.
    msg: str, the message to be logged.
    n: int, the number of times this should be called before it is logged.
    *args: The args to be substituted into the msg.
    """
    count = _get_next_log_count_per_token(logging.getLogger().findCaller())
    if count % n == 0:
        logging.log(level, msg, *args)


def log_every_sec(level, n_seconds, msg, *args):
    """Logs 'msg % args' at level 'level' iff 'n_seconds' elapsed since last call.
    Logs the first call, logs subsequent calls if 'n' seconds have elapsed since
    the last logging call from the same call site (file + line). Not thread-safe.
    Args:
    level: int, the absl logging level at which to log.
    msg: str, the message to be logged.
    n_seconds: float or int, seconds which should elapse before logging again.
    *args: The args to be substituted into the msg.
    """
    should_log = _seconds_have_elapsed(logging.getLogger().findCaller(), n_seconds)
    if should_log:
        logging.log(level, msg, *args)
