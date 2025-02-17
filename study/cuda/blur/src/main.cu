#include <cstdio>
#include <cstdlib>
#include <cmath>

// stb 이미지 관련 헤더 (상대 경로 주의)
#include "../lib/stb_image.h"
#include "../lib/stb_image_write.h"

// 블러 커널 선언
#include "../include/blurKernel.cuh"

int main() {
    int width, height, channels;
    const char* input_filename  = "cat.png";
    const char* output_filename = "cat2.jpg";

    // 1) 호스트에 이미지 로드
    unsigned char* h_input = stbi_load(input_filename, &width, &height, &channels, 0);
    if (!h_input) {
        fprintf(stderr, "Failed to load image: %s\n", input_filename);
        return -1;
    }
    printf("Loaded image '%s' (%dx%d, %d channels)\n",
           input_filename, width, height, channels);

    // 2) 호스트 출력 배열 할당
    size_t num_bytes = (size_t)width * height * channels * sizeof(unsigned char);
    unsigned char* h_output = (unsigned char*)malloc(num_bytes);

    // 3) 디바이스 메모리 할당
    unsigned char *d_input, *d_output;
    cudaMalloc((void**)&d_input,  num_bytes);
    cudaMalloc((void**)&d_output, num_bytes);

    // 4) Host -> Device 복사
    cudaMemcpy(d_input, h_input, num_bytes, cudaMemcpyHostToDevice);

    // 5) 커널 실행 (16x16 스레드, 그리드는 올림 처리)
    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x, 
              (height + block.y - 1) / block.y);

    blurKernel<<<grid, block>>>(d_output, d_input, width, height, channels);
    cudaDeviceSynchronize();

    // 6) Device -> Host 복사 (결과 이미지)
    cudaMemcpy(h_output, d_output, num_bytes, cudaMemcpyDeviceToHost);

    // 7) 결과 이미지 저장
    stbi_write_jpg(output_filename, width, height, channels, h_output, 100);
    printf("Output saved to '%s'\n", output_filename);

    // 8) 메모리 해제
    stbi_image_free(h_input);
    free(h_output);
    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}
