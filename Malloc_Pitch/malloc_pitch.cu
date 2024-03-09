#include<iostream>
#include<cuda_runtime.h>

using namespace std;

__global__ void MemAccess(float* d_data, size_t pitch, int COL, int ROW)
{
	for (int r = 0;r < ROW;r++)
	{
		float* row = (float*)((char*)d_data + r * pitch);	//get the index in which rth row starts (linear memory with padding denoted by pitch)
		for (int c = 0;c < COL;c++)
			float ele = row[c]; //row[c] => row + c th memory location (pointer arithmetic)
	}
}

int main()
{
	int ROW = 64, COL = 64;
	float* d_data;
	size_t pitch;
	int threadsPerBlock = 512, blocksPerGrid = 100;

	cudaMallocPitch(&d_data, &pitch, COL * sizeof(float), ROW);

	MemAccess << < blocksPerGrid, threadsPerBlock >> > (d_data, pitch, COL, ROW);
	return 0;
}