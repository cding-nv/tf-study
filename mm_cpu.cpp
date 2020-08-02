#include <stdio.h>

#define OPT_MOVE_K 1

#if OPT_MOVE_K
void matrixMulCPU(float* C, const float* A, const float* B, int K, int M, int N) {
	printf("OPT_MOVE_K\n");
	for (int k = 0; k < K; k++) {
	    for (int i = 0; i < M; i++) {
		    for (int j = 0; j < N; j++) {
				C[i * N + j] += A[i * K + k] * B[k * N + j];
			}
		}
	}
}
#else
void matrixMulCPU(float* C, const float* A, const float* B, int K, int M, int N) {
	for (int i = 0; i < M; i++) {
		for (int j = 0; j < N; j++) {
			for (int k = 0; k < K; k++) {
				C[i * N + j] += A[i * K + k] * B[k * N + j];
			}
		}
	}
}
#endif

int main() {
	float A[] = {2, 3, -1, 6, 1, -2}; // M x K = 2 x 3
	    //  2 3 -1
		//  6 1 -2
	float B[] = {4, -5, -3, 0, 1, 2}; // K x N = 3 x 2
				//  4 -5
		        // -3  0
		        //  1  2 
	float C[4] = {0};
	
	matrixMulCPU(C, A, B, 3, 2, 2);
	printf("%f,  %f\n", C[0], C[1]);
	printf("%f,  %f\n", C[2], C[3]);
	
	return 0;
}
