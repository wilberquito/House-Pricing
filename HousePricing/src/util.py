import os

def optim_workers():
    return max(1, os.cpu_count() - 1)