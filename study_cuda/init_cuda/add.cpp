#include <iostream>
#include <math.h>

/*
    g++ add.cpp -o add_cpu
*/

void add(int n, float *x, float *y)
{
    for (int i = 0; i < n; i++)
        y[i] = x[i] + y[i];
}

int main(void)
{
    // 1M array 더하기
    int N = 1<<30;

    float *x = new float[N];
    float *y = new float[N];

    // N loop
    for (int i = 0; i < N; i++) {
        x[i] = 1.0f;
        y[i] = 2.0f;
    }

    add(N, x, y);

    float maxError = 0.0f;
    for (int i = 0; i < N; i++)
    maxError = fmax(maxError, fabs(y[i]-3.0f));
    std::cout << "Max error: " << maxError << std::endl;

    delete [] x;
    delete [] y;

    return 0;
}