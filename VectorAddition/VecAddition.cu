#include<cuda_runtime.h>
#include<iostream>
#include <chrono>
#include<math.h>

using namespace::std;
using namespace chrono;

__global__ void VecAdd(float* a, float* b, float *c, int N)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i < N)
		c[i] = a[i] + b[i];
}

int main()
{
	long int N = pow(2, 50);
	size_t size = N * sizeof(float);

	// allocate host memory arrays
	float* A = (float*)malloc(size);
	float* B = (float*)malloc(size);
	float* C = (float*)malloc(size);

	// allocate device memory arrays
	float* d_A, * d_B, *d_C;
	cudaMalloc(&d_A, size);
	cudaMalloc(&d_B, size);
	cudaMalloc(&d_C, size);
	
	for (int i = 0;i < N;i++)
	{
		A[i] = i;
		B[i] = 2*i;
	}

	// copy data from host to device for parallel execution
	cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);

	// Kernel invocation
	int threadsPerBlock = 2;
	int blocksPerGrid = (N - 1) / threadsPerBlock + 1;

	auto start = high_resolution_clock::now();

	VecAdd << < blocksPerGrid, threadsPerBlock>> > (d_A, d_B, d_C, N);

	auto stop = high_resolution_clock::now();
	auto duration = duration_cast<milliseconds>(stop - start);

	// copy result from device to host
	cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);

	// Print details
	cout << "\nNumber of elements: " << N << endl;
	cout << "Threads per block: " << threadsPerBlock << endl;
	cout << "Blocks per grid: " << blocksPerGrid << endl;
	cout << "Grid size: " << threadsPerBlock * blocksPerGrid << endl;
	cout << "Parallel - Total execution time: " << duration.count() << " milliseconds" << endl;

	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);

	start = high_resolution_clock::now();

	for (int i = 0;i < N;i++)
		C[i] = A[i] + B[i];
	stop = high_resolution_clock::now();
	duration = duration_cast<milliseconds>(stop - start);
	cout << "Sequential - Total execution time: " << duration.count() << " milliseconds" << endl;

	free(A);
	free(B);
	free(C);

	return 0;
}