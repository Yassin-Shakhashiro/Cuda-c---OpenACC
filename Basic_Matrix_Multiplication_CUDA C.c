#include <cuda.h>
#include <stdio.h>

__global__ void matrixMulBasic(float *A, float *B, float *C, int mat_width) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    float sum = 0;
    if(row < mat_width && col < mat_width) {
        for (int k = 0; k < mat_width; k++) {
            sum += A[row * mat_width + k] * B[k * mat_width + col];
        }
        C[row * mat_width + col] = sum;
    }
}

int main() {
    int mat_width = 1024;  
    size_t size = mat_width * mat_width * sizeof(float);

    float *h_A, *h_B, *h_C;  
    float *d_A, *d_B, *d_C;  

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

    matrixMulTiled<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, mat_width);

    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(h_A); free(h_B); free(h_C);

    return 0;
}
