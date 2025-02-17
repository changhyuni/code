#include "../include/blurKernel.cuh"

// 간단한 박스 블러(3x3) 예시
__global__
void blurKernel(unsigned char* d_output, 
                const unsigned char* d_input, 
                int width, int height, 
                int channels)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int filterSize = 3;
    int half = filterSize / 2;

    // 픽셀 위치
    int pixelIndex = (y * width + x) * channels;

    // 3채널 예시 (R, G, B)
    float sumR = 0.0f, sumG = 0.0f, sumB = 0.0f;

    for (int fy = -half; fy <= half; fy++) {
        for (int fx = -half; fx <= half; fx++) {
            int nx = x + fx;
            int ny = y + fy;
            // 범위 벗어나면 에지 처리(클램핑)
            nx = max(0, min(nx, width - 1));
            ny = max(0, min(ny, height - 1));

            int nIndex = (ny * width + nx) * channels;
            sumR += d_input[nIndex + 0];
            sumG += d_input[nIndex + 1];
            sumB += d_input[nIndex + 2];
        }
    }

    float size = filterSize * filterSize;
    d_output[pixelIndex + 0] = (unsigned char)(sumR / size);
    d_output[pixelIndex + 1] = (unsigned char)(sumG / size);
    d_output[pixelIndex + 2] = (unsigned char)(sumB / size);
    // channels가 4라면 Alpha 채널도 처리 필요
}
