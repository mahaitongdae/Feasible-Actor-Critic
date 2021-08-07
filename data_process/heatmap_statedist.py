import seaborn as sns
import numpy as np
import time

def try_count_state():
    np.random.seed(1)
    a = np.random.random([10000,])
    count = 0
    time_start = time.time()
    # for i in a:
    #     if i < 0.5:
    #         count += 1
    count = (a<0.5).sum() # this is really fast!
    total_time = time.time() - time_start
    print('total count is {}, {:.3e}s used'.format(count, total_time))

if __name__ == '__main__':
    try_count_state()
