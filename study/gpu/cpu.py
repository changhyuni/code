import numpy as np
import timeit
import os

def benchmark(dtype):
    os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
    x = np.random.normal(size=(5000, 5000)).astype(dtype)
    # %%timeit과 동일하게 직접 측정
    t = timeit.timeit(lambda: np.dot(x, x), number=10)
    avg_time = t / 10
    print(f"{dtype}: {avg_time*1000:.3f} ms 평균 실행시간")

benchmark('float32')
benchmark('float64')
benchmark('int32')
benchmark('int64')