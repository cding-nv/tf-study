#include <stdio.h>

// 每个thread负责一个 C(i, j), 每个线程for循环次数是K
// C(0, 0) A的第0行 乘 B的第0列
__global__ void matrixMultiply(float *A, float *B, float *C,
						int M, int K, int N) {
    float sum = 0.0f;

	// thread(row, col) is for C(i,j)
    int row = blockIdx.y * blockDim.y + threadIdx.y; // y is for row
    int col = blockIdx.x * blockDim.x + threadIdx.x; // x is for cloumn

    if(row < M && col < N) {
        for (int i = 0; i < K; ++i) {    // K loop for a thread
            sum += A[row * K + i] * B[i * N + col];
        }
        C[row * N + col] = sum;
    }
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
	matrixMultiply<<<dim_grid, dim_block>>>(d_a, d_b, d_c, M, K, N);
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
