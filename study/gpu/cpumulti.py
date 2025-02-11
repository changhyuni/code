import os
import numpy as np
import timeit

def test_dot(threads: int, size=4000, dtype='float32', repeat=3):
    """
    스레드 수(threads) 설정 후, size x size 크기의 행렬 곱(np.dot)을 repeat회 실행 시간 측정
    macOS Accelerate에서는 VECLIB_MAXIMUM_THREADS 환경변수를 통해 
    멀티코어 사용을 제한적 제어할 수 있음 (동작은 버전에 따라 달라질 수 있음)
    """
    os.environ['VECLIB_MAXIMUM_THREADS'] = str(threads)
    
    # 큰 행렬 생성
    A = np.random.normal(size=(size, size)).astype(dtype)
    B = np.random.normal(size=(size, size)).astype(dtype)

    # 명령어에 따라 NumPy가 내부적으로 라이브러리에 접근
    # np.dot(A, B) 수행 시간을 측정
    t = timeit.timeit(lambda: np.dot(A, B), number=repeat)
    avg_ms = (t / repeat) * 1000
    print(f"VECLIB_MAXIMUM_THREADS={threads}, "
          f"{dtype} {size}x{size} matmul: {avg_ms:.2f} ms (avg of {repeat} runs)")

if __name__ == "__main__":
    # 싱글 스레드 vs 멀티 스레드 비교
    print("=== Single-core (VECLIB_MAXIMUM_THREADS=1) ===")
    test_dot(threads=1)
    
    print("\n=== Multi-core (VECLIB_MAXIMUM_THREADS=8) ===")
    test_dot(threads=8)
