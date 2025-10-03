from accelerate import Accelerator
import time

class TrainTimer:

    def __init__(self, step, name="timer", accelerator=None):
        self.name = name
        self.accelerator = accelerator
        assert isinstance(accelerator, Accelerator)
        self.start_time = None
        self.end_time = None
        self.step = step

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.stop()
        self.log()

    def start(self):
        self.start_time = time.perf_counter()

    def stop(self):
        if self.start_time is None:
            raise RuntimeError("Timer must be started before stopping.")
        self.end_time = time.perf_counter()
        elapsed_time = self.end_time - self.start_time
        return elapsed_time

    def reset(self):
        self.start_time = None
        self.end_time = None

    def log(self):
        if self.start_time is None or self.end_time is None:
            raise RuntimeError("Timer must be started and stopped before logging.")
        elapsed_time = self.end_time - self.start_time
        self.accelerator.log({self.name: elapsed_time}, step=self.step)