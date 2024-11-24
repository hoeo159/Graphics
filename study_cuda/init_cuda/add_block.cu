#include <iostream>
#include <math.h>
#include <cuda_runtime.h>
#include "../common/common.h"

/*
    index = 2 * 256 + 3 (3번 block의 3번째 thread index)
    gridDim : grid 내 blk 수, girdDim.x = 4096 (1M / 256 = 4096)
    stride  : grid의 총 thread 수
*/
__global__
void add(int n, float *x, float *y)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < n; i += stride)
        y[i] = x[i] + y[i];
}

int main(void)
{
    int N = 1<<30;

    float *h_x = new float[N];
    float *d_x;
    float *h_y = new float[N];
    float *d_y;
    
    CUDA_CHECK(cudaMalloc(&d_x, N*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_y, N*sizeof(float)));

    for (int i = 0; i < N; i++) {
        h_x[i] = 1.0f;
        h_y[i] = 2.0f;
    }

    CUDA_CHECK(cudaMemcpy(d_x, h_x, N*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_y, h_y, N*sizeof(float), cudaMemcpyHostToDevice));

    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;
    add<<<numBlocks, blockSize>>>(N, d_x, d_y);

    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_y, d_y, N*sizeof(float), cudaMemcpyDeviceToHost));

    float maxError = 0.0f;
    for (int i = 0; i < N; i++)
        maxError = fmax(maxError, fabs(h_y[i]-3.0f));
    std::cout << "Max error: " << maxError << std::endl;

    delete[] h_x;
    delete[] h_y;
    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_y));
    return 0;
}