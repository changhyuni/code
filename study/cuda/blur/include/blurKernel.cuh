#pragma once

// 간단한 박스 블러 커널 선언
__global__ 
void blurKernel(unsigned char* d_output, 
                const unsigned char* d_input, 
                int width, int height, 
                int channels);
