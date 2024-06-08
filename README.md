# Cuda Programs

###

### Vector Addition
```
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
```
<hr>

### RGB To Grayscale
```
#include<opencv2/opencv.hpp>
#include<cuda_runtime.h>
#include<iostream>

using namespace std;

cv::Mat load_img(const string& file_path)	//return image loaded from file_path
{
	cv::Mat img = cv::imread(file_path, cv::IMREAD_COLOR);	//read image
	if (img.empty())	//if image is empty, throw error
	{
		cerr << "Image invalid";
		exit(1);
	}
	return img;	//if image is not invalid, return the image
}

__global__
void rgb_to_gray(const unsigned char* d_inp, unsigned char* d_out, int width, int height)
{
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int row = blockIdx.y * blockDim.y + threadIdx.y;

	if (col < width && row < height)
	{
		int idx = row * width + col;
		idx *= 3;	//since three channels

		float r = d_inp[idx];
		float g = d_inp[idx + 1];
		float b = d_inp[idx + 2];

		float gray = 0.2989f * r + 0.5870f * g + 0.1140f * b;
		d_out[row * width + col] = (unsigned char)gray;	//store grayscale value
	}
}

int main()
{
	string file_path = "Image.png";	//input image file path
	cv::Mat img = load_img(file_path);

	unsigned char* h_inp = img.data;	//extract data from the cv's Mat structure
	int height = img.rows;	//extract height
	int width = img.cols;	//extract width
	size_t inp_size = img.step * height * sizeof(unsigned char);	//calculate size of the image (considering pads)

	unsigned char* d_inp, *d_out;
	cudaMalloc(&d_inp, inp_size);	//dynamic allocation of input image in device
	cudaMalloc(&d_out, width * height * sizeof(unsigned char));	//dynamic allocation of output image in device

	cudaMemcpy(d_inp, h_inp, inp_size, cudaMemcpyHostToDevice);	//copy image data from host to device

	dim3 blockDim(32, 32);
	dim3 gridDim((width - 1) / blockDim.x + 1, (height - 1) / blockDim.y + 1);

	rgb_to_gray<<<gridDim, blockDim >>>(d_inp, d_out, width, height);	//convert rgb to grayscale

	size_t out_size = height * width * sizeof(unsigned char);
	unsigned char* h_out = (unsigned char*) malloc(out_size);
	cudaMemcpy(h_out, d_out, out_size, cudaMemcpyDeviceToHost);	//copy output image from device to host

	cv::Mat out_img(height, width, CV_8UC1, h_out);	//create cv matrix from result
	cv::imwrite("out.png", out_img);	//save output image

	free(h_out);	//free host memory
	cudaFree(d_inp); cudaFree(d_out);	//free device memory


	return 0;
}
```

<hr>

### Malloc Pitch
```
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
```

<hr>

### Malloc 3D
```
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
```

<hr>

### Image Bluring - Grayscale

```
#include<iostream>
#include<cuda_runtime.h>
#include<opencv2/opencv.hpp>
#define BLUR_SIZE 7

using namespace std;

cv::Mat load_img(const string file_path)    //load image
{
    cv::Mat img = cv::imread(file_path, cv::IMREAD_GRAYSCALE);    //read color image
    if (img.empty())
    {
        cerr << "Image can not be read";
        exit(1);
    }
    return img;
}

__global__ void image_blur(const unsigned char* d_inp, unsigned char* d_out, int width, int height)
{
    int col = threadIdx.x + blockIdx.x * blockDim.x;
    int row = threadIdx.y + blockIdx.y * blockDim.y;

    if (col < width && row < height)
    {
        int pixval = 0;
        int pixels = 0;

        for (int blur_row = -BLUR_SIZE; blur_row <= BLUR_SIZE; blur_row++)
        {
            for (int blur_col = -BLUR_SIZE; blur_col <= BLUR_SIZE; blur_col++)
            {
                int cur_row = row + blur_row;
                int cur_col = col + blur_col;

                if (cur_row >= 0 && cur_row < height && cur_col >= 0 && cur_col < width)
                {
                    pixval += d_inp[cur_row * width + cur_col];
                    pixels++;
                }
            }
        }
        d_out[row * width + col] = (unsigned char)(pixval / pixels);
    }
}

int main()
{
    string file_path = "Image.png";    //input image file path
    cv::Mat img = load_img(file_path);

    int width = img.cols;    //number of cols
    int height = img.rows;    //number of rows
    unsigned char* h_inp = img.data;    //image data
    size_t size = img.step * height * sizeof(unsigned char);    //calculate input image size

    unsigned char* d_inp, * d_out;
    cudaMalloc(&d_inp, size);
    cudaMemcpy(d_inp, h_inp, size, cudaMemcpyHostToDevice);

    cudaMalloc(&d_out, width * height * sizeof(unsigned char));

    dim3 blockDim(BLUR_SIZE+1, BLUR_SIZE+1);
    dim3 gridDim((width - 1) / blockDim.x + 1, (height - 1) / blockDim.y + 1);

    image_blur << < gridDim, blockDim >> > (d_inp, d_out, width, height);

    size = width * height * sizeof(unsigned char);
    unsigned char* h_out = (unsigned char*)malloc(size);

    cudaMemcpy(h_out, d_out, size, cudaMemcpyDeviceToHost);

    cv::Mat out_img(height, width, CV_8UC1, h_out);
    cv::imwrite("out.png", out_img);

    cudaFree(d_inp); cudaFree(d_out);
    free(h_out);

    return 0;
}
```

### Image Bluring - RGB
```
#include<iostream>
#include<cuda_runtime.h>
#include<opencv2/opencv.hpp>
#define BLUR_SIZE 7

using namespace std;

cv::Mat load_img(const string file_path)    //load image
{
    cv::Mat img = cv::imread(file_path, cv::IMREAD_COLOR);    //read color image
    if (img.empty())
    {
        cerr << "Image can not be read";
        exit(1);
    }
    return img;
}

__global__ void image_blur(const unsigned char* d_inp, unsigned char* d_out, int width, int height)
{
    int col = threadIdx.x + blockIdx.x * blockDim.x;
    int row = threadIdx.y + blockIdx.y * blockDim.y;

    if (col < width && row < height)
    {
        int pixval_r = 0, pixval_g = 0, pixval_b = 0;
        int pixels = 0;

        for (int blur_row = -BLUR_SIZE; blur_row <= BLUR_SIZE; blur_row++)
        {
            for (int blur_col = -BLUR_SIZE; blur_col <= BLUR_SIZE; blur_col++)
            {
                int cur_row = row + blur_row;
                int cur_col = col + blur_col;

                if (cur_row >= 0 && cur_row < height && cur_col >= 0 && cur_col < width)
                {
                    pixval_r += d_inp[(cur_row * width + cur_col) * 3];
                    pixval_g += d_inp[(cur_row * width + cur_col) * 3 + 1];
                    pixval_b += d_inp[(cur_row * width + cur_col) * 3 + 2];
                    pixels++;
                }
            }
        }
        d_out[(row * width + col) * 3] = (unsigned char)(pixval_r / pixels);
        d_out[(row * width + col) * 3 + 1] = (unsigned char)(pixval_g / pixels);
        d_out[(row * width + col) * 3 + 2] = (unsigned char)(pixval_b / pixels);
    }
}

int main()
{
    string file_path = "Image.png";    //input image file path
    cv::Mat img = load_img(file_path);

    int width = img.cols;    //number of cols
    int height = img.rows;    //number of rows
    unsigned char* h_inp = img.data;    //image data
    size_t size = img.step * height * sizeof(unsigned char);    //calculate input image size

    unsigned char* d_inp, * d_out;
    cudaMalloc(&d_inp, size);
    cudaMemcpy(d_inp, h_inp, size, cudaMemcpyHostToDevice);

    cudaMalloc(&d_out, width * height * sizeof(unsigned char) * 3);

    dim3 blockDim(BLUR_SIZE+1, BLUR_SIZE+1);
    dim3 gridDim((width - 1) / blockDim.x + 1, (height - 1) / blockDim.y + 1);

    image_blur << < gridDim, blockDim >> > (d_inp, d_out, width, height);

    size = width * height * sizeof(unsigned char) * 3;
    unsigned char* h_out = (unsigned char*)malloc(size);

    cudaMemcpy(h_out, d_out, size, cudaMemcpyDeviceToHost);

    cv::Mat out_img(height, width, CV_8UC3, h_out);
    cv::imwrite("out.png", out_img);

    cudaFree(d_inp); cudaFree(d_out);
    free(h_out);

    return 0;
}
```

<hr>

### Matrix Multiplication - Version 1

```
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
```

<hr>

### Matrix Multiplication - Tiled Approach With Shared  Memory
```
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
```

<hr>

### Vector Addition
Problem Statement : Assume that we want to use each thread to calculate two (adjacent) elects of a vector addiction
```
#include<iostream>
#include<cuda_runtime.h>

using namespace std;

__global__
void VecAdd(const float* A, const float* B, float* C, int n)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x*2;
	int j = i / 2;
	if (i < n - 1)
		C[j] = A[i] + B[i] + A[i + 1] + B[i + 1];
}

int main()
{
	int n = 16;	//n is even always
	size_t size = n * sizeof(float);

	float *A = (float*)malloc(size);
	float *B = (float*)malloc(size);
	float* C = (float*)malloc(size/2);

	for (int i = 0;i < n;i++)
	{
		A[i] = i;
		B[i] = 2 * i;
	}

	cout << "A" << endl;
	for (int i = 0;i < n;i++)
		cout << A[i] << "\t";

	cout << "\n\nB" << endl;
	for (int i = 0;i < n;i++)
		cout << B[i] << "\t";

	float* d_A, * d_B, *d_C;
	cudaMalloc(&d_A, size);
	cudaMalloc(&d_B, size);
	cudaMalloc(&d_C, size);

	cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);

	dim3 blockDim(4);
	dim3 gridDim((n - 1) / blockDim.x + 1);

	VecAdd << <gridDim, blockDim >> > (d_A, d_B, d_C, n);

	cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);

	cout << "\n\nC" << endl;
	for (int i = 0;i < n/2;i++)
		cout << C[i] << "\t";

	return 1;
}
```

<hr>

