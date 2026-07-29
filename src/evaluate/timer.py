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


if __name__ == "__main__":
    with Timer() as t:
        total = sum(i * i for i in range(2_000_000))
    print("dummy closed-form-like loop result:", total)
    print("elapsed:", t.elapsed, "seconds")

    with Timer() as t:
        time.sleep(0.05)
    print("dummy iterative-like sleep elapsed:", t.elapsed, "seconds (expect ~0.05)")
