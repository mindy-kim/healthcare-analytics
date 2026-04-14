import time


class Timer:
    start_time: float
    end_time: float
    is_running: bool
    _NANO = 1000000000

    def __init__(self) -> None:
        pass

    def reset(self):
        self.start_time = 0
        self.is_running = False

    def start(self):
        self.start_time = time.time_ns()
        self.is_running = True

    def stop(self):
        if self.is_running:
            self.end_time = time.time_ns()
            self.is_running = False

    def getTime(self):
        if self.is_running:
            return round((time.time_ns() - self.start_time) / self._NANO, 4)
        else:
            return round((self.end_time - self.start_time) / self._NANO, 4)
