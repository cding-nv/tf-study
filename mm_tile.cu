#include <stdio.h>

#define TILE_WIDTH 32  //block size ,each thread to calucate each block

__global__ void matrixMultiplyShared(float *A, float *B, float *C, 
									int M, int K, int N) {
    __shared__ float sharedM[TILE_WIDTH][TILE_WIDTH];
    __shared__ float sharedN[TILE_WIDTH][TILE_WIDTH];
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = by * TILE_WIDTH + ty;
    int col = bx * TILE_WIDTH + tx;
    float v = 0.0;

	// 在K的维度上优化
	// 每一个block负责完成A的某一行 和B的某一列的乘加
	// v 先做tile内的累加，再做K/Tile 的累加
    for (int i = 0; i < (int)(ceil((float)K / TILE_WIDTH)); i++) {
        if (i * TILE_WIDTH + tx < K && row < M)
            sharedM[ty][tx] = A[row * K + i * TILE_WIDTH + tx];
        else
            sharedM[ty][tx] = 0.0;

        if (i * TILE_WIDTH + ty < K && col < N)
            sharedN[ty][tx] = B[(i * TILE_WIDTH + ty) * N + col];
        else
            sharedN[ty][tx] = 0.0;
        __syncthreads();

        for(int j = 0; j < TILE_WIDTH; j++)
            v += sharedM[ty][j] * sharedN[j][tx];
        __syncthreads();
    }

    if (row < M && col < N)
        C[row * N + col] = v;
}

int main(int argc, char **argv) {
 	float A[] = {2, 3, -1, 6, 1, -2}; // M x K = 2 x 3
	    //  2 3 -1
		//  6 1 -2
	float B[] = {4, -5, -3, 0, 1, 2}; // K x N = 3 x 2
        //  4 -5
		// -3  0
		//  1  2 
	float C[4] = {0};

	int M = 2;
	int K = 3;
	int N = 2;
	float *d_a;
	float *d_b;
	float *d_c;
	cudaMalloc((void**)&d_a,M * K * sizeof(float));
	cudaMalloc((void**)&d_b,K * N * sizeof(float));
	cudaMalloc((void**)&d_c,M * N * sizeof(float));
 
	cudaMemcpy(d_a, A, sizeof(float) * M * K, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, B, sizeof(float) * K * N, cudaMemcpyHostToDevice);
	cudaMemcpy(d_c, C, sizeof(float) * M * N, cudaMemcpyHostToDevice);

	dim3 dim_grid((N - 1) / 16 + 1, (M - 1) / 16 + 1, 1);
    dim3 dim_block(16, 16, 1);
	matrixMultiplyShared<<<dim_grid, dim_block>>>(d_a, d_b, d_c, M, K, N);
	cudaMemcpy(C, d_c, sizeof(float) * M * N, cudaMemcpyDeviceToHost);
	
	for (int i = 0; i < M * N; i++) {
		printf("%f ", C[i]);
	}
	printf("\n");
	
	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);
	
  return 0;
}
