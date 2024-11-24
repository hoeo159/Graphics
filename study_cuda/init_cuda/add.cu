#include <iostream>
#include <math.h>

/*
    nvcc add.cu -o add_gpu
    nvprof ./add_gpu

    device  code : gpu에서 실행되는 코드
    host    code : cpu에서 실행되는 코드

    GPU가 실행할 수 있는 함수, cuda 커널로 바꾸기
    __global__ : cuda c++ 컴파일러에 이 함수가 host 호출 + device 실행 가능하다는 것을 알려줌
    __device__ : cuda device function, device 호출 + device 실행 가능
    __host__   : 일반적인 c 함수와 같고, host 호출 + host 실행 가능
    이 때 __global__ 함수를 커널이라고 부름
*/

__global__
void add(int n, float *x, float *y)
{
    for (int i = 0; i < n; i++)
        y[i] = x[i] + y[i];
}

int main(void)
{
    int N = 1<<20;
    float *x, *y;

/*
    cudaMalloc(&d_data, size) : gpu memory 할당, 명시적인 메모리 전송(cudaMemcpy) 필요
    cudaMallocManaged(&data, size) : unified memory 할당, 자동으로 cpu, gpu 간 데이터 전송을 관리
    따라서 성능 중요할 때는 cudaMalloc이 유효하다
*/
    cudaMallocManaged(&x, N*sizeof(float));
    cudaMallocManaged(&y, N*sizeof(float));

    for (int i = 0; i < N; i++) {
        x[i] = 1.0f;
        y[i] = 2.0f;
    }

/*
    add() kernel 실행
    <<<x, y>>> : gpu 실행에 사용할 병렬 thread 수 x : block 개수, y : thread block의 thread 수
    cpu가 kernel 완료될 때까지 기다렸다가 cpu code에 access -> cudaDeviceSynchronize()
*/
    add<<<1, 1>>>(N, x, y);
    cudaDeviceSynchronize();

    float maxError = 0.0f;
    for (int i = 0; i < N; i++)
    maxError = fmax(maxError, fabs(y[i]-3.0f));
    std::cout << "Max error: " << maxError << std::endl;

    cudaFree(x);
    cudaFree(y);

    return 0;
}