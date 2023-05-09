#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>


#define N 20000

void init_matrix(int* matrix, int size) {
    srand((unsigned int)time(NULL));
    for (int i = 0; i < size; i++) {
        matrix[i] = rand() % 2;
    }
}

// Function to add values in a serial way
int sum_matrix_serial(int* matrix, int size) {
    int sum = 0;
    for (int i = 0; i < size; i++) {
        sum += matrix[i];
    }
    return sum;
}

// Kernel for adding values in parallel 
__global__ void sum_matrix_parallel(int* matrix, int* sum) {
    __shared__ int partial_sum[1024];

    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    partial_sum[tid] = (i < N * N) ? matrix[i] : 0;

    // Threads syncronization
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        __syncthreads();
        if (tid < s) {
            partial_sum[tid] += partial_sum[tid + s];
        }
    }

    // Just the 1st thread of the blocks will add
    if (tid == 0) {
        atomicAdd(sum, partial_sum[0]);
    }
}

int main(int argc, char** argv) {
    int* matrix;       // Host data
    int* d_matrix;     // Device data
    int* d_sum;        // Device sum

    int size = N * N * sizeof(int);

    // Allocate memory on host
    matrix = (int*)malloc(size);

    // Initialize matrix with random integer values
    init_matrix(matrix, N * N);

    // Compute sum of matrix serially
    clock_t serial_start = clock();
    int sum_serial = sum_matrix_serial(matrix, N * N);
    clock_t serial_end = clock();

    // Allocate memory on device
    cudaMalloc(&d_matrix, size);
    cudaMalloc(&d_sum, sizeof(int));

    // Copy matrix from host to device
    cudaMemcpy(d_matrix, matrix, size, cudaMemcpyHostToDevice);

    // Initialize sum on device
    cudaMemset(d_sum, 0, sizeof(int));


    // Set grid and block sizes
    int threads_per_block = 1024;
    int blocks_per_grid = (N * N + threads_per_block - 1) / threads_per_block;

    // Launch kernel to compute sum of matrix in parallel
    clock_t cuda_start = clock();
    sum_matrix_parallel <<<blocks_per_grid, threads_per_block>>> (d_matrix, d_sum);

    // Wait for kernel to finish
    cudaDeviceSynchronize();
    clock_t cuda_end = clock();

    // Copy sum from device to host
    int sum_parallel; // Cambiado a int
    cudaMemcpy(&sum_parallel, d_sum, sizeof(int), cudaMemcpyDeviceToHost);

    //Time computing
    double serial_time = (double)(serial_end - serial_start) / CLOCKS_PER_SEC;
    double cuda_time = (double)(cuda_end - cuda_start) / CLOCKS_PER_SEC;

    // Print results
    printf("N: %d\n", N);
    printf("Suma serial: %d, CPU time = %f\n", sum_serial, serial_time); 
    printf("Suma paralela: %d, GPU time = %f\n", sum_parallel, cuda_time); 

    return 0;
}