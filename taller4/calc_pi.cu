#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>

#define BLOCK_SIZE 256

__global__ void cuda_monte_carlo(int* count, curandState* state, dim3 blockDim, dim3 gridDim, int blockIdx_x) {
    __shared__ int s_counts[BLOCK_SIZE];
    int tid = threadIdx.x;
    int bid = blockIdx_x;
    int idx = bid * blockDim.x + tid;
    int s_idx = tid;

    curand_init(clock() + idx, idx, 0, &state[idx]);

    float x, y, r;
    int s_count = 0;
    for (int i = 0; i < blockDim.x; i++) {
        x = curand_uniform(&state[idx]);
        y = curand_uniform(&state[idx]);
        r = x * x + y * y;
        s_count += (r <= 1.0f);
    }
    s_counts[s_idx] = s_count;

    cudaDeviceSynchronize();

    if (tid == 0) {
        int block_count = 0;
        for (int i = 0; i < blockDim.x; i++) {
            block_count += s_counts[i];
        }
        atomicAdd(count, block_count);
    }
}

double serial_monte_carlo(const int iterations) {
    double x, y, distance_squared, pi_estimate;
    int in_circle = 0;
    for (int i = 0; i < iterations; ++i) {
        x = (double)rand() / RAND_MAX;
        y = (double)rand() / RAND_MAX;
        distance_squared = x * x + y * y;
        if (distance_squared <= 1.0) {
            ++in_circle;
        }
    }
    pi_estimate = 4 * in_circle / ((double)iterations);
    return pi_estimate;
}

int main(int argc, char** argv) {
    int iterations = 100000000;
    int count = 0, * d_count;
    curandState* d_state;

    // Serial Monte Carlo
    clock_t serial_start = clock();
    double pi_serial = serial_monte_carlo(iterations);
    clock_t serial_end = clock();

    // Parallel Monte Carlo
    cudaMalloc(&d_count, sizeof(int));
    cudaMalloc(&d_state, BLOCK_SIZE * gridDim.x * sizeof(curandState));
    cudaMemcpy(d_count, &count, sizeof(int), cudaMemcpyHostToDevice);

    dim3 dimBlock(BLOCK_SIZE);
    dim3 dimGrid(4096);  // 4096 blocks of 256 threads each

    cudaEvent_t parallel_start, parallel_stop;
    cudaEventCreate(&parallel_start);
    cudaEventCreate(&parallel_stop);

    cudaEventRecord(parallel_start);
    for (int i = 0; i < dimGrid.x; i++) {
        cuda_monte_carlo << < dimGrid, dimBlock >> > (d_count, d_state, dimBlock, dimGrid, i);

    }
    cudaMemcpy(&count, d_count, sizeof(int), cudaMemcpyDeviceToHost);
    cudaEventRecord(parallel_stop);
    cudaEventSynchronize(parallel_stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, parallel_start, parallel_stop);

    double pi_parallel = 4 * count / ((double)(iterations * dimGrid.x * BLOCK_SIZE));

    printf("Serial estimate of pi = %f\n", pi_serial);
    printf("Parallel estimate of pi = %f\n", pi_parallel);
    printf("Time taken by serial code = %f seconds\n", ((double)(serial_end - serial_start)) / CLOCKS_PER_SEC);
    printf("Time taken by parallel code = %f seconds\n", milliseconds / 1000);

    cudaFree(d_count);
    cudaFree(d_state);

    return 0;
}
