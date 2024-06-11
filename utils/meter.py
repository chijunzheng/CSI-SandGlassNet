import time
import numpy as np
from functools import wraps

def timeit(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()  # Capture start time
        _ = func(*args, **kwargs)  # Execute the function
        end_time = time.time()  # Capture end time
        # print(f"{func.__name__} executed in {end_time - start_time:.4f} seconds")
        return end_time - start_time
    return wrapper


class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self, meter_name):
        self.name = meter_name
        self.reset()

    def reset(self):
        self.val = 0  # current value
        self.avg = 0  # average value
        self.sum = 0  # sum of all values
        self.count = 0  # number of values

    def update(self, val, n=1):
        """Update the meter with new value and count."""
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0

    def current(self):
        """Prints the current statistics of the meter."""
        print(f"[{self.name}] Average: {self.avg*1000:.4f}, Count: {self.count}")
