import time
import logging

ProjCount = 0


def proj_count_and_time(func):
    def wrapper(*args, **kwargs):
        global ProjCount
        ProjCount += 1
        curTime = time.time()
        result = func(*args, **kwargs)
        logging.debug(f'{ProjCount} time cost {time.time() - curTime} s')
        return result

    return wrapper
