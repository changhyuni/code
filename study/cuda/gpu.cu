#include <stdio.h>

// 예: GPU에서 실행될 커널 함수
__global__ void addKernel(int *d_A, int *d_B, int *d_C) {
    int idx = threadIdx.x;  // 간단히 1D 블록이라 가정
    d_C[idx] = d_A[idx] + d_B[idx];
}

int main() {
    int *h_A, *h_B, *h_C;        // 호스트(Host) 메모리 변수
    int *d_A, *d_B, *d_C;        // 디바이스(Device) 메모리 변수
    int size = 100 * sizeof(int);
    
    // 1) 호스트 메모리 할당 및 초기화
    h_A = (int*)malloc(size);
    h_B = (int*)malloc(size);
    h_C = (int*)malloc(size);

    for(int i = 0; i < 100; i++){
        h_A[i] = i;
        h_B[i] = i * 2;
    }

    // 2) 디바이스 메모리 할당
    cudaMalloc((void**)&d_A, size);
    cudaMalloc((void**)&d_B, size);
    cudaMalloc((void**)&d_C, size);

    // 3) 호스트 -> 디바이스 복사
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // 4) 커널 런치 설정 (여기서는 1D로 가정)
    dim3 blockSize(100);     // 블록 하나에 스레드 100개
    dim3 gridSize(1);        // 그리드에는 블록 1개
    addKernel<<<gridSize, blockSize>>>(d_A, d_B, d_C);
    
    // *동기화가 필요한 경우
    cudaDeviceSynchronize(); // 커널 실행이 끝날 때까지 대기

    // 5) 결과를 디바이스 -> 호스트 복사
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    // 6) 결과 확인
    for(int i = 0; i < 5; i++){
        printf("C[%d] = %d\n", i, h_C[i]); // 올바른 덧셈 결과인지 확인
    }

    // 7) 디바이스 메모리 해제
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    // 8) 호스트 메모리 해제
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}
