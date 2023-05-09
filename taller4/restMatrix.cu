#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define N 10000

// Función para inicializar la matriz con valores enteros aleatorios entre 0 y 10
void init_matrix(int* matrix, int size) {
    srand((unsigned int)time(NULL));
    for (int i = 0; i < size; i++) {
        matrix[i] = rand() % 11;
    }
}

// Función para sumar los valores de la matriz de forma serial
int sum_matrix_serial(int* matrix, int size) {
    int sum = 0;
    for (int i = 0; i < size; i++) {
        sum += matrix[i];
    }
    return sum;
}

// Kernel para sumar los valores de la matriz de forma paralela
__global__ void sum_matrix_parallel(int* matrix, int* sum) {
    __shared__ int partial_sum[1024];

    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Cada hilo sumará un valor de la matriz
    partial_sum[tid] = (i < N * N) ? matrix[i] : 0;

    // Sincronización de hilos para evitar condiciones de carrera
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        __syncthreads();
        if (tid < s) {
            partial_sum[tid] += partial_sum[tid + s];
        }
    }

    // Solo el primer hilo de cada bloque sumará los valores parciales
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
    cudaMemcpy(&sum_parallel, d_sum, sizeof(int), cudaMemcpyDeviceToHost); // Cambiado a sizeof(int)

    //Time computing
    double serial_time = (double)(serial_end - serial_start) / CLOCKS_PER_SEC;
    double cuda_time = (double)(cuda_end - cuda_start) / CLOCKS_PER_SEC;

    // Print results
    printf("Suma serial: %d\nCPU time = %f\n", sum_serial, serial_time); // Cambiado a %d
    printf("Suma paralela: %d\nGPU time = %f\n", sum_parallel, cuda_time); // Cambiado a %d

    return 0;
}