#include<iostream>
#include<cuda_runtime.h>

using namespace std;

__global__ void ExtentView(cudaPitchedPtr pitchPtr, int COL, int ROW, int DEPTH)
{
	size_t pitch = pitchPtr.pitch;
	size_t slicePatch = pitch * ROW;
	char* data = (char*)pitchPtr.ptr;
	for (int z = 0;z < DEPTH;z++)
	{
		char* slice = data + z * slicePatch;
		for (int y = 0;y < ROW;y++)
		{
			float* row = (float*)(slice + y * pitch);
			for (int x = 0;x < COL;x++)
			{
				float val = row[x];
			}
		}
	}
}

int main()
{
	int ROW = 64, COL = 64, DEPTH = 64;
	cudaExtent extent_size = make_cudaExtent(COL * sizeof(float), ROW, DEPTH);	// structure representing the size of the extent
	cudaPitchedPtr data;
	cudaMalloc3D(&data, extent_size);

	ExtentView << <100, 512 >> > (data, COL, ROW, DEPTH);

	return 0;
}