#include<iostream>
#include<cuda_runtime.h>
#define BLOCK_SIZE 4

using namespace std;

typedef struct //matrix structure
{
	int width;
	int height;
	int stride;
	float* ele;
}Matrix;

__device__ Matrix GetSubMatrix(const Matrix A, int row, int col)
{
	// Device method to extract single tile from the entire matrix based on row and col indices of the block (~tile)
	Matrix sub;
	sub.width = BLOCK_SIZE;
	sub.height = BLOCK_SIZE;
	sub.stride = A.stride;
	sub.ele = &A.ele[row * BLOCK_SIZE * A.stride + col * BLOCK_SIZE];
	return sub;
}

__device__ float GetElement(const Matrix A, int row, int col)
{
	//getter method
	return A.ele[row * A.stride + col];
}

__device__ void SetElement(Matrix A, int row, int col, float val)
{
	//setter method
	A.ele[row * A.stride + col] = val;
}

__global__ void Tile_Shared_MatMul(Matrix A, Matrix B, Matrix C)
{
	// Tiled Matrix Multiplication approach using shared memory
	int blockRow = blockIdx.y;
	int blockCol = blockIdx.x;
	int row = threadIdx.y;
	int col = threadIdx.x;

	float c_val = 0.0;	//place holder to accumulate sum

	Matrix C_sub = GetSubMatrix(C, blockRow, blockCol);	//get the submatrix (tile of C in which the result to be stored)

	for (int tiles = 0;tiles < BLOCK_SIZE / A.width;tiles++)	// loop through all the tiles of entire matrix
	{
		Matrix A_sub = GetSubMatrix(A, blockRow, tiles);	//get tile of A
		Matrix B_sub = GetSubMatrix(B, tiles, blockCol);	//get tile of B

		__shared__ float As[BLOCK_SIZE][BLOCK_SIZE];	//create shared memory for each element of tile of matrix A
		__shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];	//create shared memory for each element of tile of matrix B

		As[row][col] = GetElement(A_sub, row, col);	//move element of A from global memory to shared memory
		Bs[row][col] = GetElement(B_sub, row, col);	//move element of B from global memory to shared memory

		__syncthreads();	//synchronize the threads to finish copying data from global to shared memory

		for (int e = 0;e < BLOCK_SIZE;e++)
			c_val += As[row][e] * Bs[e][col];

		__syncthreads();	//synchronize the threads to confirm that the tiled calculation is done
	}
	SetElement(C_sub, row, col, c_val);	//set the calculated element to the tile of C matrix
}

int main()
{
	Matrix A, B, C, d_A, d_B, d_C;	//host and device matrix declaration
	size_t size;

	A.width = A.height = A.stride = BLOCK_SIZE;
	d_A.width = d_A.height = d_A.stride = BLOCK_SIZE;
	size = A.height * A.width * sizeof(float);
	A.ele = (float*)malloc(size);	//allocate host memory for A matrix
	for(int i=0;i<A.height*A.width;i++)	//initialize A matrix
		A.ele[i] = (i + 1);
	cudaMalloc(&d_A.ele, size);	//allocate device memory for matrix A
	cudaMemcpy(d_A.ele, A.ele, size, cudaMemcpyHostToDevice);	//copy matrix A from host to device
	
	B.width = B.height = B.stride = BLOCK_SIZE;
	d_B.width = d_B.height = d_B.stride = BLOCK_SIZE;
	size = B.height * B.width * sizeof(float);
	B.ele = (float*)malloc(size);	//allocate host memory for B matrix
	for (int i = 0;i < B.height * B.width;i++)	//initialize B matrix
		B.ele[i] = (i + 1)*2;
	cudaMalloc(&d_B.ele, size);	//allocate device memory for matrix B
	cudaMemcpy(d_B.ele, B.ele, size, cudaMemcpyHostToDevice);	//copy matrix B from host to device
	
	C.width = C.height = C.stride = BLOCK_SIZE;
	d_C.width = d_C.height = d_C.stride = BLOCK_SIZE;
	size = C.height * C.width * sizeof(float);	
	C.ele = (float*)malloc(size);	//allocate host memory for C matrix
	cudaMalloc(&d_C.ele, size);	//allocate device memory for matrix C

	// MxN * NxO = MxO
	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
	dim3 dimGrid(A.width / BLOCK_SIZE, B.height / BLOCK_SIZE);

	Tile_Shared_MatMul << <dimGrid, dimBlock >> > (d_A, d_B, d_C);	//invoke the kernel for mat mul with tiled approach using shared memory

	cudaMemcpy(C.ele, d_C.ele, size, cudaMemcpyDeviceToHost);	//copy back result matrix from device to host

	// display matrices
	for (int i = 0;i < A.height;i++)
	{
		for (int j = 0;j < A.width;j++)
			cout << A.ele[i * A.width + j] << "\t";
		cout << "\n";
	}
	cout << "\n";

	for (int i = 0;i < B.height;i++)
	{
		for (int j = 0;j < B.width;j++)
			cout << B.ele[i * B.width + j] << "\t";
		cout << "\n";
	}
	cout << "\n";

	for (int i = 0;i < C.height;i++)
	{
		for (int j = 0;j < C.width;j++)
			cout << C.ele[i * C.width + j] << "\t";
		cout << "\n";
	}

	cudaFree(d_A.ele); cudaFree(d_B.ele); cudaFree(d_C.ele);
	free(A.ele); free(B.ele); free(C.ele);

	return 0;
}