#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <curand.h>
#include <curand_kernel.h>

#define ITERATIONS 100000000
#define BLOCK_SIZE 1024

__global__ void calculate_pi(float* count, curandState* states)
{
    __shared__ float partial_count[BLOCK_SIZE];
    float x, y;
    int tid = threadIdx.x;
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    curand_init(clock64(), index, 0, &states[index]);
    partial_count[tid] = 0.0f;
    for (int i = threadIdx.x; i < ITERATIONS; i += blockDim.x * gridDim.x)
    {
        x = curand_uniform(&states[index]);
        y = curand_uniform(&states[index]);
        if (x * x + y * y <= 1.0f)
        {
            partial_count[tid]++;
        }
    }
    __syncthreads();
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (tid < s)
        {
            partial_count[tid] += partial_count[tid + s];
        }
        __syncthreads();
    }
    if (tid == 0)
    {
        atomicAdd(count, partial_count[0]);
    }
}

__global__ void setup_curand(curandState* state, unsigned long seed)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    curand_init(seed, id, 0, &state[id]);
}

void pi_serial() {
    float x, y, count = 0;
    clock_t start, end;
    double executionTime;

    start = clock();

    for (int i = 0; i < ITERATIONS; i++)
    {
        x = (float)rand() / RAND_MAX;
        y = (float)rand() / RAND_MAX;
        if (x * x + y * y <= 1.0f)
        {
            count++;
        }
    }
    
    end = clock();
    printf("Estimated value of pi with serial processing: %f\n", 4.0f * count / ITERATIONS);
    executionTime = ((double)(end - start)) *1000 / CLOCKS_PER_SEC;
    printf("Time elapsed: %.2f ms \n", executionTime);
}


int main()
{
    float* count_device;
    float* count_host;
    curandState* states_device;
    int blocks_per_grid = (ITERATIONS + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int seed = time(NULL);

    //serial execution
    pi_serial();

    //Cuda execution
    cudaMalloc((void**)&count_device, sizeof(float));
    cudaMalloc((void**)&states_device, blocks_per_grid * BLOCK_SIZE * sizeof(curandState));
    setup_curand << <blocks_per_grid, BLOCK_SIZE >> > (states_device, seed);
    count_host = (float*)malloc(sizeof(float));
    *count_host = 0.0f;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    calculate_pi <<< blocks_per_grid, BLOCK_SIZE >>> (count_device, states_device);
    cudaMemcpy(count_host, count_device, sizeof(float), cudaMemcpyDeviceToHost);

    printf("Estimated value of pi using CUDA processing: %f\n", 4.0f * (*count_host) / ITERATIONS);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Time elapsed: %f ms\n", milliseconds);

    cudaFree(count_device);
    cudaFree(states_device);
    free(count_host);

    return 0;
}