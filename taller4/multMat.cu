/*
#include <stdio.h>
#include <cuda.h>

#define N 4

__global__ 
void matrix_mult(int* a, int* b, int* c) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < N && j < N) {
        int sum = 0;
        for (int k = 0; k < N; k++) {
            sum += a[i * N + k] * b[k * N + j];
        }
        c[i * N + j] = sum;
    }
}

int main(int argc, char** argv) {
    int* a, * b, * c;         // host data
    int* d_a, * d_b, * d_c;   // device data

    int size = N * N * sizeof(int);

    // Allocate memory on host
    a = (int*)malloc(size);
    b = (int*)malloc(size);
    c = (int*)malloc(size);

    // Matrix values
    int a_values[N][N] = {{10, 20, 30, 40},
                          {55, 65, 75, 85},
                          {96, 106, 116, 126},
                          {137, 147, 158, 168}};

    int b_values[N][N] = {{12, 23, 32, 43},
                          {52, 62, 72, 82},
                          {94, 104, 115, 125},
                          {139, 149, 150, 160}};

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            a[i * N + j] = a_values[i][j];
            b[i * N + j] = b_values[i][j];
        }
    }

    // Allocate memory on device
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);

    // Copy data from host to device
    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);

    // Set grid and block sizes
    dim3 dimBlock(2, 2);
    dim3 dimGrid((N + dimBlock.x - 1) / dimBlock.x, (N + dimBlock.y - 1) / dimBlock.y);

    // Launch kernel
    matrix_mult << <dimGrid, dimBlock >> > (d_a, d_b, d_c);

    // Wait for kernel to finish
    cudaDeviceSynchronize();

    // Copy data from device to host
    cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);

    // Print result
    printf(" Matriz Resultante:\n");
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            printf(" %d ", c[i * N + j]);
        }
        printf("\n");
    }

    // Free memory
    free(a);
    free(b);
    free(c);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}*/