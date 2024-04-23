#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>

#define TILE_mat_width 16  

__global__ void matrixMulTiled(float *A, float *B, float *C, int mat_width) {
    __shared__ float tileA[TILE_mat_width][TILE_mat_width];
    __shared__ float tileB[TILE_mat_width][TILE_mat_width];

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0;

    for (int m = 0; m < mat_width / TILE_mat_width; ++m) {
        tileA[threadIdx.y][threadIdx.x] = A[row * mat_width + (m * TILE_mat_width + threadIdx.x)];
        tileB[threadIdx.y][threadIdx.x] = B[(m * TILE_mat_width + threadIdx.y) * mat_width + col];
        __syncthreads();

        for (int k = 0; k < TILE_mat_width; ++k) {
            sum += tileA[threadIdx.y][k] * tileB[k][threadIdx.x];
        }
        __syncthreads();
    }
    if (row < mat_width && col < mat_width) {
        C[row * mat_width + col] = sum;
    }
}

int main() {
    int mat_width = 1024;  // Define the size of the matrix
    size_t size = mat_width * mat_width * sizeof(float);

    float *h_A, *h_B, *h_C;  // Host copies of A, B, C
    float *d_A, *d_B, *d_C;  // Device copies of A, B, C

    h_A = (float *)malloc(size);
    h_B = (float *)malloc(size);
    h_C = (float *)malloc(size);

    for (int i = 0; i < mat_width * mat_width; i++) {
        h_A[i] = rand() / (float)RAND_MAX;
        h_B[i] = rand() / (float)RAND_MAX;
    }

    cudaMalloc((void **)&d_A, size);
    cudaMalloc((void **)&d_B, size);
    cudaMalloc((void **)&d_C, size);

    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    dim3 dimBlock(TILE_mat_width, TILE_mat_width);
    dim3 dimGrid((mat_width + TILE_mat_width - 1) / TILE_mat_width, (mat_width + TILE_mat_width - 1) / TILE_mat_width);
    clock_t t;
    t = clock();
    matrixMulTiled<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, mat_width);
    t = clock() - t;
    double time_taken = ((double)t)/CLOCKS_PER_SEC; 
    printf("%lf",time_taken);
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(h_A); free(h_B); free(h_C);

    return 0;
}


