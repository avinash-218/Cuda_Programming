#include<iostream>
#include<cuda_runtime.h>

using namespace std;

typedef struct {
	int width;	//col
	int height;	//row
	float* ele;	//pointer to the first element of the matrix (linear memory)
}Matrix;

__global__ void MatMul(const Matrix A, const Matrix B, Matrix C, const int c_width, const int c_height)
{
	// MxN * NxO = MxO
	int row = threadIdx.y + blockIdx.y * blockDim.y;
	int col = threadIdx.x + blockIdx.x * blockDim.x;
	float val = 0;

	if (row < c_height && col < c_width)	//only valid indices
	{
		for (int i = 0;i < A.width;i++)
			val += A.ele[row * A.width + i] * B.ele[col + i * B.width];

		C.ele[row * C.width + col] = val;
	}
}

int main()
{
	Matrix A, B, C, d_A, d_B, d_C;	//declare host and device data

	//specify dimension of the matrices
	A.height = 4; A.width = 3;	//3x2
	B.height = 3; B.width = 2;	//2x3
	C.height = A.height; C.width = B.width;	// MxN * NxO = MxO

	d_A.height = A.height; d_A.width = A.width;
	d_B.height = B.height; d_B.width = B.width;
	d_C.height = C.height; d_C.width = C.width;

	// dynamic allocation of host data of size of the float matrix
	A.ele = (float*)malloc(A.width * A.height * sizeof(float));
	B.ele = (float*)malloc(B.width * B.height * sizeof(float));
	C.ele = (float*)malloc(C.width * C.height * sizeof(float));

	//initialization of host data
	for (int i = 0;i < A.width * A.height;i++)
		A.ele[i] = float(i + 1);

	for (int i = 0;i < B.width * B.height;i++)
		B.ele[i] = float((i + 1) * 2);

	//dynamic allocation of device data of corresponding sizes
	cudaMalloc(&d_A.ele, A.width * A.height * sizeof(float));
	cudaMalloc(&d_B.ele, B.width * B.height * sizeof(float));
	cudaMalloc(&d_C.ele, C.width * C.height * sizeof(float));

	cudaMemcpy(d_A.ele, A.ele, A.width * A.height * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_B.ele, B.ele, B.width * B.height * sizeof(float), cudaMemcpyHostToDevice);

	// MxN * NxO = MxO
	int num_threads = 16;
	dim3 dimBlock(num_threads, num_threads);	//num_threads x num_threads
	dim3 dimGrid((C.width - 1) / dimBlock.x + 1, (C.height - 1) / dimBlock.y + 1); //calculate grid size

	MatMul << <dimGrid, dimBlock >> > (d_A, d_B, d_C, C.width, C.height);

	cudaMemcpy(C.ele, d_C.ele, C.width * C.height * sizeof(float), cudaMemcpyDeviceToHost);	//copy data from device to host

	//display the matrices
	for (int r = 0;r < A.height;r++)
	{
		for (int c = 0;c < A.width;c++)
			cout << A.ele[r * A.width + c] << "\t";
		cout << endl;
	}
	cout << endl;

	for (int r = 0;r < B.height;r++)
	{
		for (int c = 0;c < B.width;c++)
			cout << B.ele[r * B.width + c] << "\t";
		cout << endl;
	}
	cout << endl;

	for (int r = 0;r < C.height;r++)
	{
		for (int c = 0;c < C.width;c++)
			cout << C.ele[r * C.width + c] << "\t";
		cout << endl;
	}

	//free device and host memory
	cudaFree(d_A.ele);cudaFree(d_B.ele);cudaFree(d_C.ele);
	free(A.ele);free(B.ele);free(C.ele);

	return 0;
}