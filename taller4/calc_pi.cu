#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <curand.h>
#include <curand_kernel.h>
#include <opencv2>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

using namespace cv;
using namespace std;

__global__ void sobelFilter(unsigned char* inputImage, unsigned char* outputImage, int width, int height)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < height && col < width)
    {
        int index = row * width + col;

        int gx = 0, gy = 0;

        if (row > 0 && row < height - 1 && col > 0 && col < width - 1)
        {
            gx = inputImage[(row - 1) * width + col - 1] - inputImage[(row - 1) * width + col + 1] +
                2 * (inputImage[row * width + col - 1] - inputImage[row * width + col + 1]) +
                inputImage[(row + 1) * width + col - 1] - inputImage[(row + 1) * width + col + 1];

            gy = inputImage[(row - 1) * width + col - 1] + 2 * inputImage[(row - 1) * width + col] +
                inputImage[(row - 1) * width + col + 1] - inputImage[(row + 1) * width + col - 1] -
                2 * inputImage[(row + 1) * width + col] - inputImage[(row + 1) * width + col + 1];
        }

        int magnitude = sqrt(gx * gx + gy * gy);

        if (magnitude > 255)
        {
            magnitude = 255;
        }

        outputImage[index] = (unsigned char)magnitude;
    }
}

int main(int argc, char** argv)
{
    if (argc < 2)
    {
        cout << "No image specified!" << endl;
        return -1;
    }

    string filename = argv[1];
    Mat inputImage = imread(filename, IMREAD_GRAYSCALE);
    if (inputImage.empty())
    {
        cout << "Unable to open image file: " << filename << endl;
        return -1;
    }

    int width = inputImage.cols;
    int height = inputImage.rows;

    Size imageSize = inputImage.size();
    Mat outputImage(imageSize, CV_8UC1);

    int imageSizeBytes = width * height * sizeof(unsigned char);

    unsigned char* d_inputImage = NULL;
    unsigned char* d_outputImage = NULL;

    cudaMalloc((void**)&d_inputImage, imageSizeBytes);
    cudaMalloc((void**)&d_outputImage, imageSizeBytes);

    cudaMemcpy(d_inputImage, inputImage.data, imageSizeBytes, cudaMemcpyHostToDevice);

    dim3 blockSize(32, 32);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    sobelFilter << <gridSize, blockSize >> > (d_inputImage, d_outputImage, width, height);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    printf("Time elapsed: %.2f ms \n", milliseconds);

    // Save output image
    imwrite("output.jpg", output_image);

    // Free memory
    cudaFree(d_input_image);
    cudaFree(d_output_image);
    free(h_input_image.data);
    free(h_output_image.data);

    return 0;
}