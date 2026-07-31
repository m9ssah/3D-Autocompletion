"""
Usage:
    Timer() as t: ...; then read t.elapsed (seconds)
"""

import time


class Timer:
    """
    context manager for timing a block of code
    usage: with Timer() as t: ...; then read t.elapsed (seconds)
    """

    def __enter__(self):
        self._start = time.perf_counter()
        self.elapsed = None
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.elapsed = time.perf_counter() - self._start
        return False
