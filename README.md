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

	return 1;
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


	return 1;
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
	return 1;
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

	return 1;
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

    return 1;
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

    return 1;
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

	return 1;
}
```

<hr>

### Matrix Multiplication - Tiled Approach With Shared  Memory
```
#include<iostream>
#include<cuda_runtime.h>
#define TILE_WIDTH 2

using namespace std;

typedef struct {
	int width;	//col
	int height;	//row
	float* ele;	//pointer to the first element of the matrix (linear memory)
}Matrix;

__global__ void MatMul(const Matrix A, const Matrix B, Matrix C)	//MxN * NxO = MxO
{
	__shared__ float A_SM[TILE_WIDTH][TILE_WIDTH];
	__shared__ float B_SM[TILE_WIDTH][TILE_WIDTH];

	int by = blockIdx.y;
	int bx = blockIdx.x;
	int ty = threadIdx.y;
	int tx = threadIdx.x;

	int row = by * TILE_WIDTH + ty;
	int col = bx * TILE_WIDTH + tx;

	float val = 0;

	for (int t = 0; t < (A.width-1) / TILE_WIDTH+1; ++t)
	{
		// Load tiles into shared memory
		if (row < A.height && (t * TILE_WIDTH + tx) < A.width)
			A_SM[ty][tx] = A.ele[row * A.width + t * TILE_WIDTH + tx];
		else
			A_SM[ty][tx] = 0.0f;
		if ((t * TILE_WIDTH + ty) < B.height && col < B.width)
			B_SM[ty][tx] = B.ele[(t * TILE_WIDTH + ty) * B.width + col];
		else
			B_SM[ty][tx] = 0.0f;

		__syncthreads();

		// Compute dot product within the tile
		for (int k = 0; k < TILE_WIDTH; ++k)
		{
			val += A_SM[ty][k] * B_SM[k][tx];
		}

		__syncthreads();
	}

	// Write the computed value back to matrix C
	if (row < C.height && col < C.width)
	{
		C.ele[row * C.width + col] = val;
	}
}

int main()
{
	Matrix A, B, C, d_A, d_B, d_C;	//declare host and device data

	//specify dimension of the matrices
	A.height = 4; A.width = 4;
	B.height = 4; B.width = 4;
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
	dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);	//TILE_WIDTH x TILE_WIDTH
	dim3 dimGrid((C.width - 1) / dimBlock.x + 1, (C.height - 1) / dimBlock.y + 1); //calculate grid size

	MatMul << <dimGrid, dimBlock >> > (d_A, d_B, d_C);

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

	return 1;
}
```

<hr>

### Matrix Multiplication - Tiled - Shared Memory - Corner Turning
```
#include<iostream>
#include<cuda_runtime.h >
#define TILE_WIDTH 2

using namespace std;

typedef struct
{
	float* ele;
	int width;
	int height;
}Matrix;

__global__
void MatMul(const Matrix A, const Matrix B, Matrix C)
{
	int by = blockIdx.y;
	int bx = blockIdx.x;
	int ty = threadIdx.y;
	int tx = threadIdx.x;

	// declare shared memory
	__shared__ float A_SM[TILE_WIDTH][TILE_WIDTH];
	__shared__ float B_SM[TILE_WIDTH][TILE_WIDTH];

	int row = by * TILE_WIDTH + ty;
	int col = bx * TILE_WIDTH + tx;

	float val = 0.0;

	for (int p = 0; p < (A.width - 1) / TILE_WIDTH + 1; p++)
	{
		// load data from global memory to shared memory
		if (row < A.height && p * TILE_WIDTH + tx < A.width)
			A_SM[ty][tx] = A.ele[row * A.width + p * TILE_WIDTH + tx];
		else
			A_SM[ty][tx] = 0.0f;

		if (p * TILE_WIDTH + ty < B.height && col < B.width)
			B_SM[ty][tx] = B.ele[col * B.height + p * TILE_WIDTH + ty]; // Access B in column-major layout
		else
			B_SM[ty][tx] = 0.0f;

		__syncthreads();

		// Perform the multiplication
		for (int k = 0; k < TILE_WIDTH; k++)
			val += A_SM[ty][k] * B_SM[k][tx];

		__syncthreads();
	}

	if (row < C.height && col < C.width)
		C.ele[row * C.width + col] = val;
}

int main()
{
	Matrix A, B, C, d_A, d_B, d_C;	//declare host and device data

	//specify dimension of the matrices
	A.height = 4; A.width = 4;
	B.height = 4; B.width = 4;
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
	dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);	//TILE_WIDTH x TILE_WIDTH
	dim3 dimGrid((C.width - 1) / dimBlock.x + 1, (C.height - 1) / dimBlock.y + 1); //calculate grid size

	MatMul << <dimGrid, dimBlock >> > (d_A, d_B, d_C);

	cudaMemcpy(C.ele, d_C.ele, C.width * C.height * sizeof(float), cudaMemcpyDeviceToHost);	//copy data from device to host

	//display the matrices
	for (int r = 0;r < A.height;r++)
	{
		for (int c = 0;c < A.width;c++)
			cout << A.ele[r * A.width + c] << "\t";
		cout << endl;
	}
	cout << endl;

	for (int r = 0;r < B.width;r++)
	{
		for (int c = 0;c < B.height;c++)
			cout << B.ele[c * B.height + r] << "\t";
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

	return 1;
}
```
<hr>

### Matrix Multiplication - Tiled - Shared Memory - Coarsening
```
#include<iostream>
#include<cuda_runtime.h>
#define TILE_WIDTH 2
#define COARSE_FACTOR 4

using namespace std;

typedef struct {
	int width;	//col
	int height;	//row
	float* ele;	//pointer to the first element of the matrix (linear memory)
}Matrix;

__global__ void MatMul(const Matrix A, const Matrix B, Matrix C)	//MxN * NxO = MxO
{
	__shared__ float A_SM[TILE_WIDTH][TILE_WIDTH];
	__shared__ float B_SM[TILE_WIDTH][TILE_WIDTH];

	int by = blockIdx.y;
	int bx = blockIdx.x;
	int ty = threadIdx.y;
	int tx = threadIdx.x;

	int row = by * TILE_WIDTH + ty;
	int colstart = bx * TILE_WIDTH * COARSE_FACTOR + tx;

	float val[COARSE_FACTOR];
	for (int c = 0;c < COARSE_FACTOR;c++)
		val[c] = 0.0f;

	for (int t = 0; t < (A.width - 1) / TILE_WIDTH + 1; ++t)
	{
		A_SM[ty][tx] = A.ele[row * A.width + t * TILE_WIDTH + tx];

		for (int c = 0;c < COARSE_FACTOR;c++)
		{
			int col = colstart + c * TILE_WIDTH;
			B_SM[ty][tx] = B.ele[(t * TILE_WIDTH + ty) * B.width + col];

			__syncthreads();

			for (int k = 0; k < TILE_WIDTH; ++k)
				val[c] += A_SM[ty][k] * B_SM[k][tx];
		}
		__syncthreads();
	}

	for (int c = 0;c < COARSE_FACTOR;c++)
	{
		int col = colstart + c * TILE_WIDTH;
		if (row < C.height && col < C.width)
		{
			C.ele[row * C.width + col] = val[c];
		}
	}
}

int main()
{
	Matrix A, B, C, d_A, d_B, d_C;	//declare host and device data

	//specify dimension of the matrices
	A.height = 4; A.width = 4;
	B.height = 4; B.width = 4;
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
	dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);	//TILE_WIDTH x TILE_WIDTH
	dim3 dimGrid((C.width - 1) / dimBlock.x * COARSE_FACTOR + 1, (C.height - 1) / dimBlock.y * COARSE_FACTOR + 1); //calculate grid size

	MatMul << <dimGrid, dimBlock >> > (d_A, d_B, d_C);

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

	return 1;
}
```
<hr>

### Vector Addition
Problem Statement : Consider two vectors. Add two consecutive elements of both the array.
##### For example:
###### Vector A
0--1--2--3--4--5--6--7
###### Vector B
9--10--11--12--13--14--15--16
###### Output:
20--28--36--44
###### Explanation
- 0+1+9+10 = 20
- 2+3+11+12 = 28
- 4+5+13+14 = 36
- 6+7+15+16 = 44

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

### Vector Addition
Problem Statement : Consider two vectors. Add n consecutive elements of both the array.
##### For example:
- Consecutive elements = 2
###### Vector A
0--1--2--3--4--5--6--7
###### Vector B
9--10--11--12--13--14--15--16
###### Output:
20--28--36--44
###### Explanation
- 0+1+9+10 = 20
- 2+3+11+12 = 28
- 4+5+13+14 = 36
- 6+7+15+16 = 44
```
#include<iostream>
#include<cuda_runtime.h>

using namespace std;

__global__
void VecAdd(const float* A, const float* B, float* C, const int n,const int consec_ele)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x * consec_ele;
	if (i < n - consec_ele +1)
		for(int j=0;j< consec_ele;j++)
			C[i] = A[i+j] + B[i+j] + C[i];
}

int main()
{
	int n = 18;	//total number of elements
	int consec_ele = 4;	//number of consecutive elements to sum
	size_t size = n * sizeof(float);

	float *A = (float*)malloc(size);
	float *B = (float*)malloc(size);
	float* C = (float*)malloc(size);

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

	dim3 blockDim(n/consec_ele);
	dim3 gridDim((n - 1) / blockDim.x + 1);

	VecAdd << <gridDim, blockDim >> > (d_A, d_B, d_C, n, consec_ele);

	cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);

	cout << "\n\nC" << endl;
	for (int i = 0;i < n;i++)
		cout << C[i] << "\t";

	return 1;
}
```

<hr>

### Vector Addition
Problem Statement : Given two vectors. The vector is split in x blocks in which each block contains n sections and each section is of length l. The problem is to map thread so that ith thread should add ith elements in each section of a block.
##### For example:
- Section Length = 2
- Number of section per block = 2
###### Vector A
0       1       2       3       4       5       6       7       8       9       10      11
###### Vector B
0       2       4       6       8       10      12      14      16      18      20      22
###### Output:
6       12      0       0       30      36      0       0       54      60      0       0
###### Explanation
- 0_A + 2_A + 0_B + 4_B = 6
- 1_A + 3_A + 2_B + 6_B = 12

- X_A : means X belongs to the A vector
```
#include<iostream>
#include<cuda_runtime.h>

using namespace std;

__global__
void VecAdd(const float* A, const float* B, float* C, const int n, const int number_of_section_per_block, const int section_len)
{
	int block_start = blockIdx.x * blockDim.x * number_of_section_per_block;
	for (int j = 0;j < number_of_section_per_block;j++)
	{
		int sec_start = block_start + j * section_len;
		C[block_start + threadIdx.x] += A[sec_start + threadIdx.x] + B[sec_start + threadIdx.x];
	}
}

int main()
{
	int n = 12;	//total number of elements
	int section_len = 2;	//number of elements in a section
	int number_of_section_per_block = 2;
	size_t size = n * sizeof(float);

	float *A = (float*)malloc(size);
	float *B = (float*)malloc(size);
	float* C = (float*)malloc(size);

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

	dim3 blockDim(section_len);
	dim3 gridDim(n / (number_of_section_per_block*section_len));

	VecAdd << <gridDim, blockDim >> > (d_A, d_B, d_C, n, number_of_section_per_block, section_len);

	cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);

	cout << "\n\nC" << endl;
	for (int i = 0;i < n;i++)
		cout << C[i] << "\t";

	return 1;
}
```

<hr>

### Matrix Addition

A matrix addition takes two input matrices A and B and produces one output matrix C. Each element of the output matrix C is the sum of the corresponding elements of the input matrices A and B,
i.e., C[ i ][ j ] = A[ i ][ j ] + B[ i ][ j ]. For simplicity, we will only handle square matrices whose elements are single-precision floating-point
numbers. Write a matrix addition kernel and the host stub function that can be called with four parameters:
pointer-to-the-output matrix, pointer-to-the-first-input matrix, pointer-to-the-second-input matrix, and the
number of elements in each dimension. Follow the instructions below:
- Write the host stub function by allocating memory for the input and output matrices, transferring input
data to device; launch the kernel, transferring the output data to host and freeing the device memory for
the input and output data. Leave the execution configuration parameters open for this step.
- Write a kernel that has each thread to produce one output matrix element. Fill in the execution
configuration parameters for this design.
- Write a kernel that has each thread to produce one output matrix row. Fill in the execution configuration
parameters for the design.
- Write a kernel that has each thread to produce one output matrix column. Fill in the execution
configuration parameters for the design.
```
#include<iostream>
#include<cuda_runtime.h>
# define N 4

using namespace std;

typedef struct
{
	int rows;
	int cols;
	float* ele;
}Matrix;

__global__
void Ele_Mat_Add(const Matrix d_A, const Matrix d_B, Matrix d_C, const int n)
{
	int row = blockDim.y * blockIdx.y + threadIdx.y;
	int col = blockDim.x * blockIdx.x + threadIdx.x;

	if (row < d_C.rows && col < d_C.cols)
	{
		int i = row * n + col;
		d_C.ele[i] = d_A.ele[i] + d_B.ele[i];
	}
}

__global__
void Row_Mat_Add(const Matrix d_A, const Matrix d_B, Matrix d_C, const int n)
{
	int row = blockDim.x * blockIdx.x + threadIdx.x;
	for (int col = 0;col < n;col++)
	{
		int ind = row * n + col;
		d_C.ele[ind] = d_A.ele[ind] + d_B.ele[ind];
	}
}

__global__
void Col_Mat_Add(const Matrix d_A, const Matrix d_B, Matrix d_C, const int n)
{
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	for (int row = 0;row < n;row++)
	{
		int ind = row * n + col;
		d_C.ele[ind] = d_A.ele[ind] + d_B.ele[ind];
	}
}

void matrixAdd(Matrix *A, Matrix* B, Matrix* C, int n)	//host stab
{
	Matrix d_A, d_B, d_C;
	size_t size = n * n * sizeof(float);
	d_A.rows = n; d_A.cols = n;
	d_B.rows = n; d_B.cols = n;
	d_C.rows = n; d_C.cols = n;

	// allocate memory for input and output matrix
	cudaMalloc(&d_A.ele, size);
	cudaMalloc(&d_B.ele, size);
	cudaMalloc(&d_C.ele, size);

	// transfer input data from host to memory
	cudaMemcpy(d_A.ele, A->ele, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_B.ele, B->ele, size, cudaMemcpyHostToDevice);

	//launch the kernels

	//kernel 1 - each thread - each element
	//dim3 blockDim(N, N);
	//dim3 gridDim((n - 1) / blockDim.x + 1, (n - 1) / blockDim.y + 1);
	//Ele_Mat_Add << <gridDim, blockDim >> > (d_A, d_B, d_C, n);

	//kernel 2 - each thread - one row
	//int blockDim = N;
	//int gridDim = (N - 1) / blockDim + 1;
	//Row_Mat_Add << <gridDim, blockDim >> > (d_A, d_B, d_C, N);

	// kernel 2 - each thread - one row
	int blockDim = N;
	int gridDim = (N - 1) / blockDim + 1;
	Col_Mat_Add << <gridDim, blockDim >> > (d_A, d_B, d_C, N);

	//transfer output data from device to host
	cudaMemcpy(C->ele, d_C.ele, size, cudaMemcpyDeviceToHost);

	// free device memory
	cudaFree(d_A.ele);
	cudaFree(d_B.ele);
	cudaFree(d_C.ele);
}

int main()
{
	Matrix A, B, C;
	int n;

	// all are  matrices
	A.rows = A.cols = N;
	B.rows = B.cols = N;
	C.rows = C.cols = N;

	n = N * N;

	size_t size = n * sizeof(float);

	// allocate host data memory for input and output
	A.ele = (float*)malloc(size);
	B.ele = (float*)malloc(size);
	C.ele = (float*)malloc(size);

	// initialize host data
	for (int i = 0;i < n;i++)
	{
		A.ele[i] = (i + 1);
		B.ele[i] = (i + 1) * 2;
	}

	//display the matrices
	cout << "Matrix A"<<endl;
	for (int r = 0;r < A.rows;r++)
	{
		for (int c = 0;c < A.cols;c++)
			cout << A.ele[r * A.cols + c] << "\t";
		cout << endl;
	}
	cout << endl;

	cout << "Matrix B" << endl;
	for (int r = 0;r < B.rows;r++)
	{
		for (int c = 0;c < B.cols;c++)
			cout << B.ele[r * B.cols + c] << "\t";
		cout << endl;
	}
	cout << endl;

	matrixAdd(&A, &B, &C, N);

	cout << "Matrix C" << endl;
	for (int r = 0;r < C.rows;r++)
	{
		for (int c = 0;c < C.cols;c++)
			cout << C.ele[r * C.cols + c] << "\t";
		cout << endl;
	}
	cout << endl;

	free(A.ele); free(B.ele); free(C.ele);
	

	return 1;
}
```

### Matrix Multiplication - Row Wise
```
#include<iostream>
#include<cuda_runtime.h>

using namespace std;

typedef struct
{
	int height;
	int width;
	float* ele;
}Matrix;

__global__
void MatMulRowWise(const Matrix d_A, const Matrix d_B, Matrix d_C)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;

	if (i < d_C.height)
	{
		for (int c_col = 0; c_col < d_C.width; c_col++)	//loop d_c's column times to account for one entire row
		{
			float val = 0;
			for (int k = 0; k < d_A.width; k++)
				val += d_A.ele[i * d_A.width + k] * d_B.ele[k * d_B.width + c_col];
			d_C.ele[i * d_C.width + c_col] = val;
		}
	}
}

int main()
{
	Matrix A, B, C, d_A, d_B, d_C;

	//specify dimension of the matrices
	A.height = 4; A.width = 3;	//4x3
	B.height = 3; B.width = 2;	//3x2
	C.height = A.height; C.width = B.width;	// MxN * NxO = MxO

	d_A.height = A.height; d_A.width = A.width;
	d_B.height = B.height; d_B.width = B.width;
	d_C.height = C.height; d_C.width = C.width;

	// dynamic allocate host matrices
	A.ele = (float*)malloc(A.height * A.width * sizeof(float));
	B.ele = (float*)malloc(B.height * B.width * sizeof(float));
	C.ele = (float*)malloc(C.height * C.width * sizeof(float));

	//initialization of host data
	for (int i = 0;i < A.width * A.height;i++)
		A.ele[i] = float(i + 1);

	for (int i = 0;i < B.width * B.height;i++)
		B.ele[i] = float((i + 1) * 2);

	// dynamic allocate device matrices
	cudaMalloc(&d_A.ele, d_A.height * d_A.width * sizeof(float));
	cudaMalloc(&d_B.ele, d_B.height * d_B.width * sizeof(float));
	cudaMalloc(&d_C.ele, d_C.height * d_C.width * sizeof(float));

	// transfer data from host to device
	cudaMemcpy(d_A.ele, A.ele, d_A.height * d_A.width * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_B.ele, B.ele, d_B.height * d_B.width * sizeof(float), cudaMemcpyHostToDevice);

	// kernel config
	// MxN * NxO = MxO
	dim3 blockDim(C.height);	//number of threads is number of blocks
	dim3 gridDim(1);	//only one block

	MatMulRowWise << <gridDim, blockDim >> > (d_A, d_B, d_C);

	cudaMemcpy(C.ele, d_C.ele, d_C.height * d_C.width * sizeof(float), cudaMemcpyDeviceToHost);

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

	cudaFree(d_A.ele);	cudaFree(d_B.ele); cudaFree(d_C.ele);
	free(A.ele), free(B.ele), free(C.ele);

	return 1;
}
```

<hr>

### Matrix Multiplication - Column wise
```
#include<iostream>
#include<cuda_runtime.h>

using namespace std;

typedef struct
{
	int height;
	int width;
	float* ele;
}Matrix;

__global__
void MatMulColWise(const Matrix d_A, const Matrix d_B, Matrix d_C)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;

	if (i < d_C.width)
	{
		for (int row = 0;row < d_C.height;row++)
		{
			float val = 0.0;
			for (int k = 0;k < d_A.width;k++)
				val += d_A.ele[row * d_A.width + k] * d_B.ele[k * d_B.width + i];

			d_C.ele[row * d_B.width + i] = val;
		}
	}
}

int main()
{
	Matrix A, B, C, d_A, d_B, d_C;

	//specify dimension of the matrices
	A.height = 4; A.width = 3;	//4x3
	B.height = 3; B.width = 2;	//3x2
	C.height = A.height; C.width = B.width;	// MxN * NxO = MxO

	d_A.height = A.height; d_A.width = A.width;
	d_B.height = B.height; d_B.width = B.width;
	d_C.height = C.height; d_C.width = C.width;

	// dynamic allocate host matrices
	A.ele = (float*)malloc(A.height * A.width * sizeof(float));
	B.ele = (float*)malloc(B.height * B.width * sizeof(float));
	C.ele = (float*)malloc(C.height * C.width * sizeof(float));

	//initialization of host data
	for (int i = 0;i < A.width * A.height;i++)
		A.ele[i] = float(i + 1);

	for (int i = 0;i < B.width * B.height;i++)
		B.ele[i] = float((i + 1) * 2);

	// dynamic allocate device matrices
	cudaMalloc(&d_A.ele, d_A.height * d_A.width * sizeof(float));
	cudaMalloc(&d_B.ele, d_B.height * d_B.width * sizeof(float));
	cudaMalloc(&d_C.ele, d_C.height * d_C.width * sizeof(float));

	// transfer data from host to device
	cudaMemcpy(d_A.ele, A.ele, d_A.height * d_A.width * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_B.ele, B.ele, d_B.height * d_B.width * sizeof(float), cudaMemcpyHostToDevice);

	// kernel config
	// MxN * NxO = MxO
	dim3 blockDim(C.width);	//number of threads is number of blocks
	dim3 gridDim(1);	//only one block

	MatMulColWise << <gridDim, blockDim >> > (d_A, d_B, d_C);

	cudaMemcpy(C.ele, d_C.ele, d_C.height * d_C.width * sizeof(float), cudaMemcpyDeviceToHost);

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

	cudaFree(d_A.ele);	cudaFree(d_B.ele); cudaFree(d_C.ele);
	free(A.ele), free(B.ele), free(C.ele);

	return 1;
}
```
<hr>

### Matrix Vector Multiplication
```
#include<iostream>
#include<cuda_runtime.h>

using namespace std;

typedef struct
{
	int rows;
	int cols;
	float* ele;
}Matrix;

__global__
void MatVecMul(const Matrix d_A, const Matrix d_B, Matrix d_C)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	float val = 0.0;
	for (int k = 0;k < d_A.cols;k++)
		val += d_A.ele[i * d_A.cols + k] * d_B.ele[k * d_B.cols];
	d_C.ele[i * d_C.cols]=val;
}

int main()
{
	Matrix A, B, C, d_A, d_B, d_C;

	// 3x3 * 3x1 = 3x1
	A.rows = A.cols = B.rows = C.rows = 3;
	B.cols = C.cols = 1;
	d_A.rows = A.rows; d_A.cols = A.cols;
	d_B.rows = B.rows; d_B.cols = B.cols;
	d_C.rows = C.rows; d_C.cols = C.cols;

	// host memory allocation
	A.ele = (float*)malloc(A.rows * A.cols * sizeof(float));
	B.ele = (float*)malloc(B.rows * B.cols * sizeof(float));
	C.ele = (float*)malloc(C.rows * C.cols * sizeof(float));

	// device memory allocation
	cudaMalloc(&d_A.ele, d_A.rows * d_A.cols * sizeof(float));
	cudaMalloc(&d_B.ele, d_B.rows * d_B.cols * sizeof(float));
	cudaMalloc(&d_C.ele, d_C.rows * d_C.cols * sizeof(float));

	// host data initialization
	for (int row = 0;row < A.rows; row++)
		for (int col = 0; col < A.cols; col++)
			A.ele[row * A.cols + col] = row * A.cols + col + 1;
	
	for (int row = 0;row < B.rows; row++)
		for (int col = 0; col < B.cols; col++)
			B.ele[row * B.cols + col] = (row * B.cols + col + 1) * 2;

	cudaMemcpy(d_A.ele, A.ele, d_A.rows * d_A.cols * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_B.ele, B.ele, d_B.rows * d_B.cols * sizeof(float), cudaMemcpyHostToDevice);

	dim3 blockDim(C.rows);
	dim3 gridDim(1);

	MatVecMul << <gridDim, blockDim >> > (d_A, d_B, d_C);

	cudaMemcpy(C.ele, d_C.ele, d_C.rows * d_C.cols * sizeof(float), cudaMemcpyDeviceToHost);

	//display the matrices
	for (int r = 0;r < A.rows;r++)
	{
		for (int c = 0;c < A.cols;c++)
			cout << A.ele[r * A.cols + c] << "\t";
		cout << endl;
	}
	cout << endl;

	for (int r = 0;r < B.rows;r++)
	{
		for (int c = 0;c < B.cols;c++)
			cout << B.ele[r * B.cols + c] << "\t";
		cout << endl;
	}
	cout << endl;

	for (int r = 0;r < C.rows;r++)
	{
		for (int c = 0;c < C.cols;c++)
			cout << C.ele[r * C.cols + c] << "\t";
		cout << endl;
	}

	//free device and host memory
	cudaFree(d_A.ele);cudaFree(d_B.ele);cudaFree(d_C.ele);
	free(A.ele);free(B.ele);free(C.ele);

	return 1;
}
```
<hr>

### Cuda Device Property Query
```
#include<iostream>
#include<cuda_runtime.h>

using namespace std;

int main()
{
	int devcount;
	cudaGetDeviceCount(&devcount);
	cout << "Number of Devices : " << devcount << endl << endl;
	
	cudaDeviceProp devProp;	//device properties
	for (unsigned int d = 0;d < devcount;d++)	//loop through each device
	{
		cudaGetDeviceProperties(&devProp, d);	//get dth device properties
		cout << "Device : " << d<<endl;
		cout << "Total Global Memory (Bytes): " << devProp.totalGlobalMem<<endl;
		cout << "Shared Memory Per Block (Bytes): " << devProp.sharedMemPerBlock<<endl;
		cout << "Number of Registers Per Block: " << devProp.regsPerBlock<<endl;
		cout << "Warp Size: " << devProp.warpSize<< endl;
		cout << "Max Threads Per Block: " << devProp.maxThreadsPerBlock<< endl;
		cout << "Total Constant Memory (Bytes): " << devProp.totalConstMem<< endl;
		cout << "Number of streaming multi-processors: " << devProp.multiProcessorCount << endl;
		cout << "Total Constant Memory (Bytes): " << devProp.integrated << endl;
		cout << "L2 Cache Size (Bytes): " << devProp.l2CacheSize<< endl;
		cout << "Maximum number of threads per SM: " << devProp.maxThreadsPerMultiProcessor << endl;
		cout << "Number of Registers per multiprocessor: " << devProp.regsPerMultiprocessor << endl;
	}

	return 1;
}
```

<hr>

### 1D Convolution - Zero Padding - Stride 1
```
#include<iostream>
#include<cuda_runtime.h>

using namespace std;

__global__
void Convolution_1D(const float* d_inp, float* d_out, const float* d_filter, const int array_len, const int r)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	float val = 0.0;

	if (i < array_len)
	{
		for (int c = 0; c < 2 * r + 1; c++)
		{
			int final_i = i - r + c;
			if (final_i >= 0 && final_i < array_len)
				val += d_inp[final_i] * d_filter[c];
		}
		d_out[i] = val;
	}
}

int main()
{
	float* inp, * out, * filter, * d_inp, * d_out, * d_filter;

	int array_len = 20;
	int filter_radius = 3;	//number of filter elements is 2r+1

	size_t array_size = array_len * sizeof(float);
	size_t filter_size = (2 * filter_radius + 1) * sizeof(float);

	inp = (float*)malloc(array_size);	// input array memory allocation
	out = (float*)malloc(array_size);	// output array memory allocation
	filter = (float*)malloc(filter_size);	// kernel array memory allocation

	for (int i = 0;i < 2 * filter_radius + 1; i++)	// initialize filter
		filter[i] = (i + 1);

	for (int i = 0;i < array_len;i++)	//initialize input array
		inp[i] = (i + 1) * 2;

	// cuda memory allocation
	cudaMalloc(&d_inp, array_size);
	cudaMalloc(&d_out, array_size);
	cudaMalloc(&d_filter, filter_size);

	// copy data from host to device
	cudaMemcpy(d_inp, inp, array_size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_filter, filter, filter_size, cudaMemcpyHostToDevice);

	// kernel configuration
	dim3 blockDim(2 * filter_radius + 1);
	dim3 gridDim((array_len - 1) / blockDim.x + 1);

	Convolution_1D << < gridDim, blockDim >> > (d_inp, d_out, d_filter, array_len, filter_radius);	//invoke the kernel

	// copy output data from device to host
	cudaMemcpy(out, d_out, array_size, cudaMemcpyDeviceToHost);

	// display data
	cout << "Convolution Kernel" << endl;
	for (int i = 0;i < 2 * filter_radius + 1; i++)	// initialize filter
		cout << filter[i] << '\t';

	cout << "\n\nInput Array" << endl;
	for (int i = 0;i < array_len;i++)	//initialize input array
		cout << inp[i] << '\t';

	cout << "\n\nOutputArray" << endl;
	for (int i = 0;i < array_len;i++)	//initialize input array
		cout << out[i] << '\t';

	free(inp); free(out); free(filter);
	cudaFree(d_inp); cudaFree(d_out); cudaFree(d_filter);

	return 1;
}
```

### 1D Convolution - Zero Padding - Stride 1 - Constant Memory For Kernel
```
#include<iostream>
#include<cuda_runtime.h>
#define FILTER_RADIUS 3	// filter radius (known at compile time)
__constant__ float F[2 * FILTER_RADIUS + 1];	//declare constant memory during compilation time

using namespace std;

__global__
void Convolution_1D(const float* d_inp, float* d_out, const int array_len, const int r)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	float val = 0.0;

	if (i < array_len)  // Ensure within bounds
	{
		for (int c = 0; c < 2 * r + 1; c++)
		{
			int final_i = i - r + c;
			if (final_i >= 0 && final_i < array_len)
				val += d_inp[final_i] * F[c];
		}
		d_out[i] = val;
	}
}

int main()
{
	float* inp, * out, * filter, * d_inp, * d_out;

	int array_len = 20;

	size_t array_size = array_len * sizeof(float);
	size_t filter_size = (2 * FILTER_RADIUS + 1) * sizeof(float);

	inp = (float*)malloc(array_size);	// input array memory allocation
	out = (float*)malloc(array_size);	// output array memory allocation
	filter = (float*)malloc(filter_size);	// kernel array memory allocation

	for (int i = 0;i < 2 * FILTER_RADIUS + 1; i++)	// initialize filter in host memory
		filter[i] = (i + 1);

	cudaMemcpyToSymbol(F, filter, filter_size);	//copy filter data from host memory to constant memory

	for (int i = 0;i < array_len;i++)	//initialize input array
		inp[i] = (i + 1) * 2;

	// cuda memory allocation
	cudaMalloc(&d_inp, array_size);
	cudaMalloc(&d_out, array_size);

	// copy data from host to device
	cudaMemcpy(d_inp, inp, array_size, cudaMemcpyHostToDevice);

	// kernel configuration
	dim3 blockDim(2 * FILTER_RADIUS + 1);
	dim3 gridDim((array_len - 1) / blockDim.x + 1);

	Convolution_1D << < gridDim, blockDim >> > (d_inp, d_out, array_len, FILTER_RADIUS);	//invoke the kernel

	// copy output data from device to host
	cudaMemcpy(out, d_out, array_size, cudaMemcpyDeviceToHost);

	// display data
	cout << "Convolution Kernel" << endl;
	for (int i = 0;i < 2 * FILTER_RADIUS + 1; i++)	// initialize filter
		cout << filter[i] << '\t';

	cout << "\n\nInput Array" << endl;
	for (int i = 0;i < array_len;i++)	//initialize input array
		cout << inp[i] << '\t';

	cout << "\n\nOutputArray" << endl;
	for (int i = 0;i < array_len;i++)	//initialize input array
		cout << out[i] << '\t';

	free(inp); free(out); free(filter);
	cudaFree(d_inp); cudaFree(d_out);

	return 1;
}
```

### 2D Convolution - Zero Padding - Stride 1
```
#include<iostream>
#include<cuda_runtime.h>

using namespace std;

typedef struct
{
    float* ele;
    int rows;
    int cols;
} Matrix;

typedef struct
{
    int* ele;
    int rows;
    int cols;
    int r;
} Kernel;

__global__
void Convolution_2D(const Matrix d_inp, Matrix d_out, const Kernel d_k)
{
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    float val = 0.0f;

    if (row < d_out.rows && col < d_out.cols) {  // Ensure within bounds
        for (int k_r = 0; k_r < d_k.rows; k_r++)
        {
            for (int k_c = 0; k_c < d_k.cols; k_c++)
            {
                int f_row = row - d_k.r + k_r;
                int f_col = col - d_k.r + k_c;

                if (f_row >= 0 && f_row < d_inp.rows && f_col >= 0 && f_col < d_inp.cols)
                    val += d_inp.ele[f_row * d_inp.cols + f_col] * d_k.ele[k_r * d_k.cols + k_c];
            }
        }
        d_out.ele[row * d_out.cols + col] = val;
    }
}

int main()
{
    // Input and Output Matrices
    Matrix inp, d_inp, out, d_out;
    inp.rows = d_inp.rows = out.rows = d_out.rows = 4;
    inp.cols = d_inp.cols = out.cols = d_out.cols = 4;

    size_t mat_size = inp.rows * inp.cols * sizeof(float);

    inp.ele = (float*)malloc(mat_size);
    out.ele = (float*)malloc(mat_size);
    memset(out.ele, 0, mat_size);  // Initialize output matrix to 0

    cudaMalloc(&d_inp.ele, mat_size);
    cudaMalloc(&d_out.ele, mat_size);

    // Initialization of input matrix
    for (int i = 0; i < inp.cols * inp.rows; i++)
        inp.ele[i] = float(i + 1);

    cudaMemcpy(d_inp.ele, inp.ele, mat_size, cudaMemcpyHostToDevice);  // Copy input matrix from host to device

    Kernel k, d_k;
    k.r = d_k.r = 1;
    k.cols = d_k.cols = k.rows = d_k.rows = 2 * k.r + 1;
    size_t kernel_size = k.rows * k.cols * sizeof(int);
    k.ele = (int*)malloc(kernel_size);
    cudaMalloc(&d_k.ele, kernel_size);

    // Initialization of kernel
    for (int i = 0; i < k.rows * k.cols; i++)
        k.ele[i] = (i + 1) * 2;
    cudaMemcpy(d_k.ele, k.ele, kernel_size, cudaMemcpyHostToDevice);

    dim3 blockDim(16, 16);
    dim3 gridDim((inp.cols + blockDim.x - 1) / blockDim.x, (inp.rows + blockDim.y - 1) / blockDim.y);

    Convolution_2D << <gridDim, blockDim >> > (d_inp, d_out, d_k);

    // Check for any errors launching the kernel
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        cout << "Error: " << cudaGetErrorString(err) << endl;
        return -1;
    }

    cudaMemcpy(out.ele, d_out.ele, mat_size, cudaMemcpyDeviceToHost);  // Copy output matrix from device to host

    // Display the matrices
    cout << "Input" << endl;
    for (int r = 0; r < inp.rows; r++)
    {
        for (int c = 0; c < inp.cols; c++)
            cout << inp.ele[r * inp.cols + c] << "\t";
        cout << endl;
    }
    cout << endl << "Kernel" << endl;

    for (int r = 0; r < k.rows; r++)
    {
        for (int c = 0; c < k.cols; c++)
            cout << k.ele[r * k.cols + c] << "\t";
        cout << endl;
    }
    cout << endl << "Output" << endl;

    for (int r = 0; r < out.rows; r++)
    {
        for (int c = 0; c < out.cols; c++)
            cout << out.ele[r * out.cols + c] << "\t";
        cout << endl;
    }
    cout << endl;

    free(inp.ele); free(out.ele); free(k.ele);
    cudaFree(d_inp.ele); cudaFree(d_out.ele); cudaFree(d_k.ele);

    return 1;
}
```
<hr>

### 3D Convolution - Zero Padding - Stride 1
```
#include <iostream>
#include <cuda_runtime.h>

using namespace std;

typedef struct
{
    float* ele;
    int rows;
    int cols;
    int depth;
} Matrix;

typedef struct
{
    int* ele;
    int rows;
    int cols;
    int depth;
    int r;
} Kernel;

__global__
void Convolution_3D(const Matrix d_inp, Matrix d_out, const Kernel d_k)
{
    int z = blockDim.z * blockIdx.z + threadIdx.z;
    int y = blockDim.y * blockIdx.y + threadIdx.y;
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    float val = 0.0f;

    if (z < d_out.depth && y < d_out.rows && x < d_out.cols) {  // Ensure within bounds
        for (int k_z = 0; k_z < d_k.depth; k_z++)
        {
            for (int k_y = 0; k_y < d_k.rows; k_y++)
            {
                for (int k_x = 0; k_x < d_k.cols; k_x++)
                {
                    int f_z = z - d_k.r + k_z;
                    int f_y = y - d_k.r + k_y;
                    int f_x = x - d_k.r + k_x;

                    if (f_z >= 0 && f_z < d_inp.depth && f_y >= 0 && f_y < d_inp.rows && f_x >= 0 && f_x < d_inp.cols)
                        val += d_inp.ele[f_z * d_inp.rows * d_inp.cols + f_y * d_inp.cols + f_x] * d_k.ele[k_z * d_k.rows * d_k.cols + k_y * d_k.cols + k_x];
                }
            }
        }
        d_out.ele[z * d_out.rows * d_out.cols + y * d_out.cols + x] = val;
    }
}

int main()
{
    // Input and Output Matrices
    Matrix inp, d_inp, out, d_out;
    inp.rows = d_inp.rows = out.rows = d_out.rows = 4;
    inp.cols = d_inp.cols = out.cols = d_out.cols = 4;
    inp.depth = d_inp.depth = out.depth = d_out.depth = 4;

    size_t mat_size = inp.rows * inp.cols * inp.depth * sizeof(float);

    inp.ele = (float*)malloc(mat_size);
    out.ele = (float*)malloc(mat_size);
    memset(out.ele, 0, mat_size);  // Initialize output matrix to 0

    cudaMalloc(&d_inp.ele, mat_size);
    cudaMalloc(&d_out.ele, mat_size);

    // Initialization of input matrix
    for (int i = 0; i < inp.cols * inp.rows * inp.depth; i++)
        inp.ele[i] = float(i + 1);

    cudaMemcpy(d_inp.ele, inp.ele, mat_size, cudaMemcpyHostToDevice);  // Copy input matrix from host to device

    Kernel k, d_k;
    k.r = d_k.r = 1;
    k.cols = d_k.cols = k.rows = d_k.rows = 2 * k.r + 1;
    k.depth = d_k.depth = 2 * k.r + 1;
    size_t kernel_size = k.rows * k.cols * k.depth * sizeof(int);
    k.ele = (int*)malloc(kernel_size);
    cudaMalloc(&d_k.ele, kernel_size);

    // Initialization of kernel
    for (int i = 0; i < k.depth * k.rows * k.cols; i++)
        k.ele[i] = (i + 1) * 2;
    cudaMemcpy(d_k.ele, k.ele, kernel_size, cudaMemcpyHostToDevice);

    dim3 blockDim(4, 4, 4);  // Adjust block dimensions as needed
    dim3 gridDim((inp.cols + blockDim.x - 1) / blockDim.x, (inp.rows + blockDim.y - 1) / blockDim.y, (inp.depth + blockDim.z - 1) / blockDim.z);

    Convolution_3D << <gridDim, blockDim >> > (d_inp, d_out, d_k);

    // Check for any errors launching the kernel
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        cout << "Error: " << cudaGetErrorString(err) << endl;
        return -1;
    }

    cudaMemcpy(out.ele, d_out.ele, mat_size, cudaMemcpyDeviceToHost);  // Copy output matrix from device to host

    // Display the matrices
    cout << "Input" << endl;
    for (int z = 0; z < inp.depth; z++)
    {
        cout << "Depth " << z << endl;
        for (int y = 0; y < inp.rows; y++)
        {
            for (int x = 0; x < inp.cols; x++)
                cout << inp.ele[z * inp.rows * inp.cols + y * inp.cols + x] << "\t";
            cout << endl;
        }
        cout << endl;
    }

    cout << "Kernel" << endl;
    for (int z = 0; z < k.depth; z++)
    {
        cout << "Depth " << z << endl;
        for (int y = 0; y < k.rows; y++)
        {
            for (int x = 0; x < k.cols; x++)
                cout << k.ele[z * k.rows * k.cols + y * k.cols + x] << "\t";
            cout << endl;
        }
        cout << endl;
    }

    cout << "Output" << endl;
    for (int z = 0; z < out.depth; z++)
    {
        cout << "Depth " << z << endl;
        for (int y = 0; y < out.rows; y++)
        {
            for (int x = 0; x < out.cols; x++)
                cout << out.ele[z * out.rows * out.cols + y * out.cols + x] << "\t";
            cout << endl;
        }
        cout << endl;
    }

    free(inp.ele); free(out.ele); free(k.ele);
    cudaFree(d_inp.ele); cudaFree(d_out.ele); cudaFree(d_k.ele);

    return 1;
}
```
<hr>

### 2D Convolution - Zero Padding - Stride 1 - Constant Memory
```
#include<iostream>
#include<cuda_runtime.h>
#define FILTER_RADIUS 1	// filter radius (known at compile time)
__constant__ int F[2 * FILTER_RADIUS + 1][2 * FILTER_RADIUS + 1];	//declare constant memory during compilation time

using namespace std;

typedef struct
{
	float* ele;
	int rows;
	int cols;
}Matrix;

typedef struct
{
	int* ele;
	int rows;
	int cols;
	int r;
}Kernel;

__global__
void Convolution_2D(const Matrix d_inp, Matrix d_out)
{
	int row = blockDim.y * blockIdx.y + threadIdx.y;
	int col = blockDim.x * blockIdx.x + threadIdx.x;
	float val = 0.0f;

	if (row < d_inp.rows && col < d_inp.cols) // Ensure within bounds
	{
		for (int k_r = 0;k_r < 2 * FILTER_RADIUS + 1;k_r++)
		{
			for (int k_c = 0;k_c < 2 * FILTER_RADIUS + 1;k_c++)
			{
				int f_row = row - FILTER_RADIUS + k_r;
				int f_col = col - FILTER_RADIUS + k_c;

				if (f_row >= 0 && f_row < d_inp.rows && f_col >= 0 && f_col < d_inp.cols)
					val += d_inp.ele[f_row * d_inp.cols + f_col] * F[k_r][k_c];
			}
		}
		d_out.ele[row * d_out.cols + col] = val;
	}
}

int main()
{
	// Input and Output Matrices
	Matrix inp, d_inp, out, d_out;
	inp.rows = d_inp.rows = out.rows = d_out.rows = 4;
	inp.cols = d_inp.cols = out.cols = d_out.cols = 4;

	size_t mat_size = inp.rows * inp.cols * sizeof(float);

	inp.ele = (float*)malloc(mat_size);
	out.ele = (float*)malloc(mat_size);

	cudaMalloc(&d_inp.ele, mat_size);
	cudaMalloc(&d_out.ele, mat_size);

	//initialization of input matrix
	for (int i = 0;i < inp.cols * inp.rows;i++)
		inp.ele[i] = float(i + 1);

	cudaMemcpy(d_inp.ele, inp.ele, mat_size, cudaMemcpyHostToDevice);	// copy input matrix from host to device

	Kernel k;
	k.r = FILTER_RADIUS;
	k.cols = k.rows = 2 * k.r + 1;
	size_t kernel_size = k.rows * k.cols * sizeof(int);
	k.ele = (int*)malloc(kernel_size);

	// initialization of kernel
	for (int i = 0;i < k.rows * k.cols;i++)
		k.ele[i] = (i + 1) * 2;
	cudaMemcpyToSymbol(F, k.ele, kernel_size);	//copy filter data from host memory to constant memory

	dim3 blockDim(k.cols, k.rows);
	dim3 gridDim((inp.cols - 1) / blockDim.x + 1, (inp.rows - 1) / blockDim.y + 1);

	Convolution_2D << <gridDim, blockDim >> > (d_inp, d_out);

	cudaMemcpy(out.ele, d_out.ele, mat_size, cudaMemcpyDeviceToHost);	// copy outut matrix from device to host

	//display the matrices
	cout << "Input" << endl;
	for (int r = 0;r < inp.rows;r++)
	{
		for (int c = 0;c < inp.cols;c++)
			cout << inp.ele[r * inp.cols + c] << "\t";
		cout << endl;
	}
	cout << endl << "Kernel" << endl;

	for (int r = 0;r < k.rows;r++)
	{
		for (int c = 0;c < k.cols;c++)
			cout << k.ele[r * k.cols + c] << "\t";
		cout << endl;
	}
	cout << endl << "Output" << endl;

	for (int r = 0;r < out.rows;r++)
	{
		for (int c = 0;c < out.cols;c++)
			cout << out.ele[r * out.cols + c] << "\t";
		cout << endl;
	}
	cout << endl;

	free(inp.ele); free(out.ele); free(k.ele);
	cudaFree(d_inp.ele);cudaFree(d_out.ele);

	return 1;
}
```
<hr>

### 3D Convolution - Zero Padding - Stride 1 - Constant Memory
```
#include <iostream>
#include <cuda_runtime.h>
#define FILTER_RADIUS 1  // Filter radius (known at compile time)

__constant__ int F[2 * FILTER_RADIUS + 1][2 * FILTER_RADIUS + 1][2 * FILTER_RADIUS + 1];  // Declare constant memory during compilation time

using namespace std;

typedef struct
{
    float* ele;
    int rows;
    int cols;
    int depth;
} Matrix;

typedef struct
{
    int* ele;
    int rows;
    int cols;
    int depth;
    int r;
} Kernel;

__global__
void Convolution_3D(const Matrix d_inp, Matrix d_out)
{
    int z = blockDim.z * blockIdx.z + threadIdx.z;
    int y = blockDim.y * blockIdx.y + threadIdx.y;
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    float val = 0.0f;

    if (z < d_out.depth && y < d_out.rows && x < d_out.cols)  // Ensure within bounds
    {
        for (int k_z = 0; k_z < 2 * FILTER_RADIUS + 1; k_z++)
        {
            for (int k_y = 0; k_y < 2 * FILTER_RADIUS + 1; k_y++)
            {
                for (int k_x = 0; k_x < 2 * FILTER_RADIUS + 1; k_x++)
                {
                    int f_z = z - FILTER_RADIUS + k_z;
                    int f_y = y - FILTER_RADIUS + k_y;
                    int f_x = x - FILTER_RADIUS + k_x;

                    if (f_z >= 0 && f_z < d_inp.depth && f_y >= 0 && f_y < d_inp.rows && f_x >= 0 && f_x < d_inp.cols)
                        val += d_inp.ele[f_z * d_inp.rows * d_inp.cols + f_y * d_inp.cols + f_x] * F[k_z][k_y][k_x];
                }
            }
        }
        d_out.ele[z * d_out.rows * d_out.cols + y * d_out.cols + x] = val;
    }
}

int main()
{
    // Input and Output Matrices
    Matrix inp, d_inp, out, d_out;
    inp.rows = d_inp.rows = out.rows = d_out.rows = 4;
    inp.cols = d_inp.cols = out.cols = d_out.cols = 4;
    inp.depth = d_inp.depth = out.depth = d_out.depth = 4;

    size_t mat_size = inp.rows * inp.cols * inp.depth * sizeof(float);

    inp.ele = (float*)malloc(mat_size);
    out.ele = (float*)malloc(mat_size);

    cudaMalloc(&d_inp.ele, mat_size);
    cudaMalloc(&d_out.ele, mat_size);

    // Initialization of input matrix
    for (int i = 0; i < inp.cols * inp.rows * inp.depth; i++)
        inp.ele[i] = float(i + 1);

    cudaMemcpy(d_inp.ele, inp.ele, mat_size, cudaMemcpyHostToDevice);  // Copy input matrix from host to device

    Kernel k;
    k.r = FILTER_RADIUS;
    k.cols = k.rows = k.depth = 2 * k.r + 1;
    size_t kernel_size = k.rows * k.cols * k.depth * sizeof(int);
    k.ele = (int*)malloc(kernel_size);

    // Initialization of kernel
    for (int i = 0; i < k.depth * k.rows * k.cols; i++)
        k.ele[i] = (i + 1) * 2;
    cudaMemcpyToSymbol(F, k.ele, kernel_size);  // Copy filter data from host memory to constant memory

    dim3 blockDim(4, 4, 4);  // Adjust block dimensions as needed
    dim3 gridDim((inp.cols + blockDim.x - 1) / blockDim.x, (inp.rows + blockDim.y - 1) / blockDim.y, (inp.depth + blockDim.z - 1) / blockDim.z);

    Convolution_3D << <gridDim, blockDim >> > (d_inp, d_out);

    // Check for any errors launching the kernel
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        cout << "Error: " << cudaGetErrorString(err) << endl;
        return -1;
    }

    cudaMemcpy(out.ele, d_out.ele, mat_size, cudaMemcpyDeviceToHost);  // Copy output matrix from device to host

    // Display the matrices
    cout << "Input" << endl;
    for (int z = 0; z < inp.depth; z++)
    {
        cout << "Depth " << z << endl;
        for (int y = 0; y < inp.rows; y++)
        {
            for (int x = 0; x < inp.cols; x++)
                cout << inp.ele[z * inp.rows * inp.cols + y * inp.cols + x] << "\t";
            cout << endl;
        }
        cout << endl;
    }

    cout << "Kernel" << endl;
    for (int z = 0; z < k.depth; z++)
    {
        cout << "Depth " << z << endl;
        for (int y = 0; y < k.rows; y++)
        {
            for (int x = 0; x < k.cols; x++)
                cout << k.ele[z * k.rows * k.cols + y * k.cols + x] << "\t";
            cout << endl;
        }
        cout << endl;
    }

    cout << "Output" << endl;
    for (int z = 0; z < out.depth; z++)
    {
        cout << "Depth " << z << endl;
        for (int y = 0; y < out.rows; y++)
        {
            for (int x = 0; x < out.cols; x++)
                cout << out.ele[z * out.rows * out.cols + y * out.cols + x] << "\t";
            cout << endl;
        }
        cout << endl;
    }

    free(inp.ele); free(out.ele); free(k.ele);
    cudaFree(d_inp.ele); cudaFree(d_out.ele);

    return 1;
}
```
<hr>

### 2D Convolution - Zero Padding - Stride 1 - Constant Memory - Shared Memory
```
#include <iostream>
#include <cuda_runtime.h>

#define KERNEL_RADIUS 1 // filter radius (known at compile time)
#define IN_TILE_WIDTH 4
#define OUT_TILE_WIDTH ((IN_TILE_WIDTH) - 2*(KERNEL_RADIUS))    //since this is macro, it will be placed as it is in the code
// so considering that, add brackets as much as needed considering BODMAS rule

__constant__ int d_K[2 * KERNEL_RADIUS + 1][2 * KERNEL_RADIUS + 1]; // constant memory for kernel

using namespace std;

typedef struct
{
    int cols;
    int rows;
    float* ele;
} Matrix;

typedef struct
{
    int cols;
    int rows;
    int* ele;
    int r; // radius
} Kernel;

__global__
void Convolution(const Matrix inp, Matrix out)
{
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int col = blockDim.x * blockIdx.x + threadIdx.x;

    __shared__ float SM[IN_TILE_WIDTH][IN_TILE_WIDTH];

    if (row < inp.rows && col < inp.cols)
        SM[threadIdx.y][threadIdx.x] = inp.ele[row * inp.cols + col];
    else
        SM[threadIdx.y][threadIdx.x] = 0.0f;

    __syncthreads();

    int sm_row = threadIdx.y - KERNEL_RADIUS;
    int sm_col = threadIdx.x - KERNEL_RADIUS;

    if (row < out.rows && col < out.cols)
    {
        float val = 0.0f;
        for (int r = 0; r < 2 * KERNEL_RADIUS + 1; ++r)
        {
            for (int c = 0; c < 2 * KERNEL_RADIUS + 1; ++c)
            {
                int sm_r = sm_row + r;
                int sm_c = sm_col + c;
                if (sm_r >= 0 && sm_r < IN_TILE_WIDTH && sm_c >= 0 && sm_c < IN_TILE_WIDTH)
                    val += SM[sm_r][sm_c] * d_K[r][c];
            }
        }
        out.ele[row * out.cols + col] = val;
    }
}

int main()
{
    // matrix
    Matrix inp, d_inp, out, d_out;
    inp.rows = d_inp.rows = out.rows = d_out.rows = 4;
    inp.cols = d_inp.cols = out.cols = d_out.cols = 4;

    // allocate memory for host data
    size_t mat_size = inp.rows * inp.cols * sizeof(float);
    inp.ele = (float*)malloc(mat_size);
    out.ele = (float*)malloc(mat_size);

    // allocate memory for device data
    cudaMalloc(&d_inp.ele, mat_size);
    cudaMalloc(&d_out.ele, mat_size);

    // initialization of input matrix
    for (int i = 0; i < inp.cols * inp.rows; i++)
        inp.ele[i] = float(i + 1);

    // copy input matrix from host to device
    cudaMemcpy(d_inp.ele, inp.ele, mat_size, cudaMemcpyHostToDevice);

    // kernel
    Kernel K;
    K.r = KERNEL_RADIUS;
    K.rows = K.cols = 2 * K.r + 1;
    size_t kernel_size = K.rows * K.cols * sizeof(int);

    // allocate kernel memory
    K.ele = (int*)malloc(kernel_size);

    // initialization of kernel
    for (int i = 0; i < K.rows * K.cols; i++)
        K.ele[i] = (int)((i + 1));

    cudaMemcpyToSymbol(d_K, K.ele, kernel_size); // copy kernel data from host to constant memory

    dim3 blockDim(IN_TILE_WIDTH, IN_TILE_WIDTH);
    dim3 gridDim((inp.cols - 1) / blockDim.x + 1, (inp.rows - 1) / blockDim.y + 1);

    Convolution << <gridDim, blockDim >> > (d_inp, d_out);

    cudaMemcpy(out.ele, d_out.ele, mat_size, cudaMemcpyDeviceToHost);

    // display the matrices
    cout << "Input" << endl;
    for (int r = 0; r < inp.rows; r++)
    {
        for (int c = 0; c < inp.cols; c++)
            cout << inp.ele[r * inp.cols + c] << "\t";
        cout << endl;
    }
    cout << endl
        << "Kernel" << endl;

    for (int r = 0; r < K.rows; r++)
    {
        for (int c = 0; c < K.cols; c++)
            cout << K.ele[r * K.cols + c] << "\t";
        cout << endl;
    }
    cout << endl
        << "Output" << endl;

    for (int r = 0; r < out.rows; r++)
    {
        for (int c = 0; c < out.cols; c++)
            cout << out.ele[r * out.cols + c] << "\t";
        cout << endl;
    }
    cout << endl;

    free(inp.ele);free(out.ele);free(K.ele);
    cudaFree(d_inp.ele);cudaFree(d_out.ele);

    return 1;
}
```
<hr>

### 3D Convolution - Zero Padding - Stride 1 - Constant Memory - Shared Memory
```
#include <iostream>
#include <cuda_runtime.h>

#define KERNEL_RADIUS 1 // filter radius (known at compile time)
#define IN_TILE_WIDTH 4
#define OUT_TILE_WIDTH ((IN_TILE_WIDTH) - 2*(KERNEL_RADIUS))    // Adjust according to BODMAS rule

__constant__ int d_K[2 * KERNEL_RADIUS + 1][2 * KERNEL_RADIUS + 1][2 * KERNEL_RADIUS + 1]; // constant memory for kernel

using namespace std;

typedef struct
{
    int cols;
    int rows;
    int depth;
    float* ele;
} Matrix;

typedef struct
{
    int cols;
    int rows;
    int depth;
    int* ele;
    int r; // radius
} Kernel;

__global__
void Convolution3D(const Matrix inp, Matrix out)
{
    int z = blockDim.z * blockIdx.z + threadIdx.z;
    int y = blockDim.y * blockIdx.y + threadIdx.y;
    int x = blockDim.x * blockIdx.x + threadIdx.x;

    __shared__ float SM[IN_TILE_WIDTH][IN_TILE_WIDTH][IN_TILE_WIDTH];

    // Load data into shared memory
    if (z < inp.depth && y < inp.rows && x < inp.cols)
        SM[threadIdx.z][threadIdx.y][threadIdx.x] = inp.ele[z * inp.rows * inp.cols + y * inp.cols + x];
    else
        SM[threadIdx.z][threadIdx.y][threadIdx.x] = 0.0f;

    __syncthreads();

    int sm_z = threadIdx.z - KERNEL_RADIUS;
    int sm_y = threadIdx.y - KERNEL_RADIUS;
    int sm_x = threadIdx.x - KERNEL_RADIUS;

    if (z < out.depth && y < out.rows && x < out.cols)
    {
        float val = 0.0f;
        for (int dz = 0; dz < 2 * KERNEL_RADIUS + 1; ++dz)
        {
            for (int dy = 0; dy < 2 * KERNEL_RADIUS + 1; ++dy)
            {
                for (int dx = 0; dx < 2 * KERNEL_RADIUS + 1; ++dx)
                {
                    int sm_r = sm_z + dz;
                    int sm_c = sm_y + dy;
                    int sm_d = sm_x + dx;
                    if (sm_r >= 0 && sm_r < IN_TILE_WIDTH && sm_c >= 0 && sm_c < IN_TILE_WIDTH && sm_d >= 0 && sm_d < IN_TILE_WIDTH)
                        val += SM[sm_r][sm_c][sm_d] * d_K[dz][dy][dx];
                }
            }
        }
        out.ele[z * out.rows * out.cols + y * out.cols + x] = val;
    }
}

int main()
{
    // Matrix dimensions
    Matrix inp, d_inp, out, d_out;
    inp.rows = d_inp.rows = out.rows = d_out.rows = 4;
    inp.cols = d_inp.cols = out.cols = d_out.cols = 4;
    inp.depth = d_inp.depth = out.depth = d_out.depth = 4;

    size_t mat_size = inp.rows * inp.cols * inp.depth * sizeof(float);

    // Allocate memory for host data
    inp.ele = (float*)malloc(mat_size);
    out.ele = (float*)malloc(mat_size);

    // Allocate memory for device data
    cudaMalloc(&d_inp.ele, mat_size);
    cudaMalloc(&d_out.ele, mat_size);

    // Initialization of input matrix
    for (int i = 0; i < inp.cols * inp.rows * inp.depth; i++)
        inp.ele[i] = float(i + 1);

    cudaMemcpy(d_inp.ele, inp.ele, mat_size, cudaMemcpyHostToDevice); // Copy input matrix from host to device

    // Kernel
    Kernel K;
    K.r = KERNEL_RADIUS;
    K.cols = K.rows = K.depth = 2 * K.r + 1;
    size_t kernel_size = K.cols * K.rows * K.depth * sizeof(int);

    // Allocate kernel memory
    K.ele = (int*)malloc(kernel_size);

    // Initialization of kernel
    for (int i = 0; i < K.cols * K.rows * K.depth; i++)
        K.ele[i] = (i + 1);

    cudaMemcpyToSymbol(d_K, K.ele, kernel_size); // Copy kernel data from host memory to constant memory

    dim3 blockDim(IN_TILE_WIDTH, IN_TILE_WIDTH, IN_TILE_WIDTH);
    dim3 gridDim((inp.cols - 1) / blockDim.x + 1, (inp.rows - 1) / blockDim.y + 1, (inp.depth - 1) / blockDim.z + 1);

    Convolution3D << <gridDim, blockDim >> > (d_inp, d_out);

    cudaMemcpy(out.ele, d_out.ele, mat_size, cudaMemcpyDeviceToHost);

    // Display the matrices
    cout << "Input" << endl;
    for (int z = 0; z < inp.depth; z++)
    {
        cout << "Depth " << z << endl;
        for (int y = 0; y < inp.rows; y++)
        {
            for (int x = 0; x < inp.cols; x++)
                cout << inp.ele[z * inp.rows * inp.cols + y * inp.cols + x] << "\t";
            cout << endl;
        }
        cout << endl;
    }

    cout << "Kernel" << endl;
    for (int z = 0; z < K.depth; z++)
    {
        cout << "Depth " << z << endl;
        for (int y = 0; y < K.rows; y++)
        {
            for (int x = 0; x < K.cols; x++)
                cout << K.ele[z * K.rows * K.cols + y * K.cols + x] << "\t";
            cout << endl;
        }
        cout << endl;
    }

    cout << "Output" << endl;
    for (int z = 0; z < out.depth; z++)
    {
        cout << "Depth " << z << endl;
        for (int y = 0; y < out.rows; y++)
        {
            for (int x = 0; x < out.cols; x++)
                cout << out.ele[z * out.rows * out.cols + y * out.cols + x] << "\t";
            cout << endl;
        }
        cout << endl;
    }

    free(inp.ele); free(out.ele); free(K.ele);
    cudaFree(d_inp.ele); cudaFree(d_out.ele);

    return 1;
}
```
<hr>

### 2D Convolution - Zero Padding - Stride 1 - Constant Memory - Shared Memory - Caching
```
#include <iostream>
#include <cuda_runtime.h>
#define KERNEL_RADIUS 1
#define TILE_WIDTH 4

using namespace std;

__constant__ int d_K[2 * KERNEL_RADIUS + 1][2 * KERNEL_RADIUS + 1];

typedef struct {
    float* ele;
    int rows;
    int cols;
} Matrix;

typedef struct {
    int r;
    int rows;
    int cols;
    int* ele;
} Kernel;

__global__
void Convolution(const Matrix inp, Matrix out) {
    int row = TILE_WIDTH * blockIdx.y + threadIdx.y;
    int col = TILE_WIDTH * blockIdx.x + threadIdx.x;

    __shared__ float SM[TILE_WIDTH][TILE_WIDTH];

    if (row < inp.rows && col < inp.cols)
        SM[threadIdx.y][threadIdx.x] = inp.ele[row * inp.cols + col];
    else
        SM[threadIdx.y][threadIdx.x] = 0.0f;

    __syncthreads();

    if (row < inp.rows && col < inp.cols)   //within matrix
    {
        float val = 0.0f;
        for (int r = -KERNEL_RADIUS;r <=KERNEL_RADIUS;r++)
        {
            for (int c = -KERNEL_RADIUS;c <= KERNEL_RADIUS;c++)
            {
                if (threadIdx.y + r > 0 && threadIdx.y + r < TILE_WIDTH &&
                    threadIdx.x + c > 0 && threadIdx.x + c < TILE_WIDTH)    //within tile
                {
                    val += d_K[r + KERNEL_RADIUS][c + KERNEL_RADIUS] * SM[threadIdx.y + r][threadIdx.x + c];
                }
                else//outside tile
                {
                    if (row + r >= 0 && row + r < inp.rows && col + c >= 0 && col + c < inp.cols)
                    {
                        val += d_K[r + KERNEL_RADIUS][c + KERNEL_RADIUS] * inp.ele[(row + r) * inp.cols + (col + c)];
                    }
                }
            }
        }
        out.ele[row * out.cols + col] = val;
    }
}

int main() {
    Matrix inp, d_inp, out, d_out;
    inp.rows = d_inp.rows = out.rows = d_out.rows = 4;
    inp.cols = d_inp.cols = out.cols = d_out.cols = 4;

    size_t mat_size = inp.rows * inp.cols * sizeof(float);

    inp.ele = (float*)malloc(mat_size);
    out.ele = (float*)malloc(mat_size);
    cudaMalloc(&d_inp.ele, mat_size);
    cudaMalloc(&d_out.ele, mat_size);

    // Initialization of input matrix
    for (int i = 0; i < inp.cols * inp.rows; i++)
        inp.ele[i] = float(i + 1);

    cudaMemcpy(d_inp.ele, inp.ele, mat_size, cudaMemcpyHostToDevice);

    Kernel K;
    K.r = KERNEL_RADIUS;
    K.rows = K.cols = 2 * K.r + 1;

    size_t kernel_size = K.rows * K.cols * sizeof(int);
    K.ele = (int*)malloc(kernel_size);

    // Initialization of kernel
    for (int i = 0; i < K.rows * K.cols; i++)
        K.ele[i] = (int)((i + 1));

    cudaMemcpyToSymbol(d_K, K.ele, kernel_size);

    dim3 blockDim(TILE_WIDTH, TILE_WIDTH);
    dim3 gridDim((inp.cols - 1) / TILE_WIDTH + 1, (inp.rows - 1) / TILE_WIDTH + 1);

    Convolution << <gridDim, blockDim >> > (d_inp, d_out);

    cudaMemcpy(out.ele, d_out.ele, mat_size, cudaMemcpyDeviceToHost);

    // Display the matrices
    cout << "Input" << endl;
    for (int r = 0; r < inp.rows; r++) {
        for (int c = 0; c < inp.cols; c++)
            cout << inp.ele[r * inp.cols + c] << "\t";
        cout << endl;
    }
    cout << endl << "Kernel" << endl;

    for (int r = 0; r < K.rows; r++) {
        for (int c = 0; c < K.cols; c++)
            cout << K.ele[r * K.cols + c] << "\t";
        cout << endl;
    }
    cout << endl << "Output" << endl;

    for (int r = 0; r < out.rows; r++) {
        for (int c = 0; c < out.cols; c++)
            cout << out.ele[r * out.cols + c] << "\t";
        cout << endl;
    }
    cout << endl;

    free(inp.ele); free(out.ele); free(K.ele);
    cudaFree(d_inp.ele); cudaFree(d_out.ele);

    return 1;
}
```

<hr>

### Stencil - Single Order Derivative - Three Variables - Constant Memory
```
#include<iostream>
#include<cuda_runtime.h>
#define BLOCK_DIM 4
using namespace std;

__constant__ float coef[7];

__global__
void Stencil(const float* d_in, float* d_out, const int N)
{
	int i = blockDim.z * blockIdx.z + threadIdx.z;
	int j = blockDim.y * blockIdx.y + threadIdx.y;
	int k = blockDim.x * blockIdx.x + threadIdx.x;

	if (i>=1 && i< N-1 &&j >= 1 && j < N - 1 && k >= 1 && k < N - 1)
	{
		d_out[i * N * N + j * N + k] = coef[0] * d_in[i * N * N + j * N + k] +
			coef[1] * d_in[i * N * N + j * N + (k - 1)] +
			coef[2] * d_in[i * N * N + j * N + (k + 1)] +
			coef[3] * d_in[i * N * N + (j - 1) * N + k] +
			coef[4] * d_in[i * N * N + (j + 1) * N + k] +
			coef[5] * d_in[(i - 1) * N * N + j * N + k] +
			coef[6] * d_in[(i + 1) * N * N + j * N + k];
	}
}

int main()
{
	float* h_in, * d_in, * h_out, * d_out;
	int N = 32;	// grid points in each dimension
	size_t size = N * N * N * sizeof(float);

	// allocate
	h_in = (float*)malloc(size);
	h_out = (float*)malloc(size);
	cudaMalloc(&d_in, size);
	cudaMalloc(&d_out, size);

	//initialize
	for (int i = 0;i < N * N * N;i++)
		h_in[i] = (float)i + 1;

	// copy data from host to device
	cudaMemcpy(d_in, h_in, size, cudaMemcpyHostToDevice);

	float h_cooef[7] = {1,1,1,1,1,1,1};

	//copy constant data to constant memory
	cudaMemcpyToSymbol(coef, h_cooef, 7 * sizeof(float));

	dim3 blockDim(BLOCK_DIM, BLOCK_DIM, BLOCK_DIM);	//3D stencil
	dim3 gridDim((N - 1) / blockDim.x + 1, (N - 1) / blockDim.y + 1, (N - 1) / blockDim.z + 1);

	Stencil << <gridDim, blockDim >> > (d_in, d_out, N);
	cudaMemcpy(h_out, d_out, size, cudaMemcpyDeviceToHost);

	cout << "Testing :" << endl;
	int i = 1, j = 1, k = 1;
	cout << "Inp[" << i << "][" << j << "][" << k << "] : " << h_in[i * N * N + j * N + k] << endl;
	cout << "Inp[" << i << "][" << j << "][" << k - 1 << "] : " << h_in[i * N * N + j * N + (k - 1)] << endl;
	cout << "Inp[" << i << "][" << j << "][" << k + 1 << "] : " << h_in[i * N * N + j * N + (k + 1)] << endl;
	cout << "h_in[" << i << "][" << j - 1 << "][" << k << "] : " << h_in[i * N * N + (j - 1) * N + k] << endl;
	cout << "h_in[" << i << "][" << j + 1 << "][" << k << "] : " << h_in[i * N * N + (j + 1) * N + k] << endl;
	cout << "h_in[" << i - 1 << "][" << j << "][" << k << "] : " << h_in[(i - 1) * N * N + j * N + k] << endl;
	cout << "h_in[" << i + 1 << "][" << j << "][" << k << "] : " << h_in[(i + 1) * N * N + j * N + k] << endl;
	cout << "\nOut[" << i << "][" << j << "][" << k << "] : " << h_out[i * N * N + j * N + k] << endl;

	cudaFree(d_in);cudaFree(d_out);
	free(h_in); free(h_out);

	return 1;
}
```

<hr>

### Stencil - Single Order Derivative - Three Variables - Shared Memory - Constant Memory
```
#include <iostream>
#include <cuda_runtime.h>

#define IN_TILE_WIDTH 4
#define OUT_TILE_WIDTH (IN_TILE_WIDTH - 2)
#define N 32

using namespace std;

__constant__ float coef[7];

typedef struct {
    int rows;
    int cols;
    int depth;
    float* ele;
} Matrix;

__global__ void Stencil(const Matrix d_in, const Matrix d_out) {
    int i = blockIdx.z * OUT_TILE_WIDTH + threadIdx.z - 1;  //to include immediate boundary cells to load in SM
    int j = blockIdx.y * OUT_TILE_WIDTH + threadIdx.y - 1;
    int k = blockIdx.x * OUT_TILE_WIDTH + threadIdx.x - 1;

    __shared__ float s_data[IN_TILE_WIDTH][IN_TILE_WIDTH][IN_TILE_WIDTH];

    if (i >= 0 && i < N && j >= 0 && j < N && k >= 0 && k < N)  //within matrix
        s_data[threadIdx.z][threadIdx.y][threadIdx.x] = d_in.ele[i * N * N + j * N + k];
    else
        s_data[threadIdx.z][threadIdx.y][threadIdx.x] = 0.0f;

    __syncthreads();

    if (i >= 1 && i < N-1 && j >= 1 && j < N-1 && k >= 1 && k < N-1)   //turn off boundary ignore assumption made - boundary are ignored in the calc
    {
        if (threadIdx.x >= 1 && threadIdx.x < IN_TILE_WIDTH - 1 &&  // only inner boundary is considered (remember : when loading already -1 is done)
            threadIdx.y >= 1 && threadIdx.y < IN_TILE_WIDTH - 1 &&
            threadIdx.z >= 1 && threadIdx.z < IN_TILE_WIDTH - 1)
        {
            d_out.ele[i * N * N + j * N + k] =
                coef[0] * s_data[threadIdx.z][threadIdx.y][threadIdx.x]
                + coef[1] * s_data[threadIdx.z][threadIdx.y][threadIdx.x - 1]
                + coef[2] * s_data[threadIdx.z][threadIdx.y][threadIdx.x + 1]
                + coef[3] * s_data[threadIdx.z][threadIdx.y - 1][threadIdx.x]
                + coef[4] * s_data[threadIdx.z][threadIdx.y + 1][threadIdx.x]
                + coef[5] * s_data[threadIdx.z - 1][threadIdx.y][threadIdx.x]
                + coef[6] * s_data[threadIdx.z + 1][threadIdx.y][threadIdx.x];
        }
    }
}

int main() {
    Matrix inp, d_inp, out, d_out;
    inp.depth = d_inp.depth = out.depth = d_out.depth = N;
    inp.rows = d_inp.rows = out.rows = d_out.rows = N;
    inp.cols = d_inp.cols = out.cols = d_out.cols = N;

    size_t mat_size = inp.depth * inp.rows * inp.cols * sizeof(float);

    inp.ele = (float*)malloc(mat_size);
    out.ele = (float*)malloc(mat_size);
    cudaMalloc(&d_inp.ele, mat_size);
    cudaMalloc(&d_out.ele, mat_size);

    // Initialization of input matrix
    for (int i = 0; i < inp.cols * inp.rows * inp.depth; i++)
        inp.ele[i] = float(i + 1);

    cudaMemcpy(d_inp.ele, inp.ele, mat_size, cudaMemcpyHostToDevice);

    float h_coef[7] = {1,1,1,1,1,1,1};

    // Copy constant data to constant memory
    cudaMemcpyToSymbol(coef, h_coef, 7 * sizeof(float));

    dim3 blockDim(IN_TILE_WIDTH, IN_TILE_WIDTH, IN_TILE_WIDTH);
    dim3 gridDim((N + OUT_TILE_WIDTH - 1) / OUT_TILE_WIDTH,
        (N + OUT_TILE_WIDTH - 1) / OUT_TILE_WIDTH,
        (N + OUT_TILE_WIDTH - 1) / OUT_TILE_WIDTH);

    Stencil << <gridDim, blockDim >> > (d_inp, d_out);
    cudaMemcpy(out.ele, d_out.ele, mat_size, cudaMemcpyDeviceToHost);

    cout << "Testing :" << endl;
    int i = 1, j = 1, k = 1;
    cout << "Inp[" << i << "][" << j << "][" << k << "] : " << inp.ele[i * N * N + j * N + k] << endl;
    cout << "Inp[" << i << "][" << j << "][" << k - 1 << "] : " << inp.ele[i * N * N + j * N + (k - 1)] << endl;
    cout << "Inp[" << i << "][" << j << "][" << k + 1 << "] : " << inp.ele[i * N * N + j * N + (k + 1)] << endl;
    cout << "Inp[" << i << "][" << j - 1 << "][" << k << "] : " << inp.ele[i * N * N + (j - 1) * N + k] << endl;
    cout << "Inp[" << i << "][" << j + 1 << "][" << k << "] : " << inp.ele[i * N * N + (j + 1) * N + k] << endl;
    cout << "Inp[" << i - 1 << "][" << j << "][" << k << "] : " << inp.ele[(i - 1) * N * N + j * N + k] << endl;
    cout << "Inp[" << i + 1 << "][" << j << "][" << k << "] : " << inp.ele[(i + 1) * N * N + j * N + k] << endl;
    cout << "\nOut[" << i << "][" << j << "][" << k << "] : " << out.ele[i * N * N + j * N + k] << endl;

    cudaFree(d_inp.ele);
    cudaFree(d_out.ele);
    free(inp.ele);
    free(out.ele);

    return 1;
}
```

<hr>

### Stencil - Single Order Derivative - Three Variables - Shared Memory - Constant Memory - Thread Coarsening

```
#include <iostream>
#include <cuda_runtime.h>

#define IN_TILE_WIDTH 4
#define OUT_TILE_WIDTH (IN_TILE_WIDTH - 2)
#define N 32

using namespace std;

__constant__ float coef[7];

typedef struct {
    int rows;
    int cols;
    int depth;
    float* ele;
} Matrix;

__global__ void Stencil(const Matrix inp, Matrix out)
{
    int i_start = blockIdx.z * OUT_TILE_WIDTH;
    int j = blockIdx.y * OUT_TILE_WIDTH + threadIdx.y - 1;
    int k = blockIdx.x * OUT_TILE_WIDTH + threadIdx.x - 1;

    __shared__ float prev[IN_TILE_WIDTH][IN_TILE_WIDTH];
    __shared__ float cur[IN_TILE_WIDTH][IN_TILE_WIDTH];
    __shared__ float next[IN_TILE_WIDTH][IN_TILE_WIDTH];

    if (i_start-1 >= 0 && i_start -1< N  && j >= 0 && j < N && k >= 0 && k < N) //load prev layer
        prev[threadIdx.y][threadIdx.x] = inp.ele[(i_start - 1) * N * N + j * N + k];

    if (i_start >= 0 && i_start < N && j >= 0 && j < N && k >= 0 && k < N)  //load current layer
        cur[threadIdx.y][threadIdx.x] = inp.ele[i_start * N * N + j * N + k];

    for (int i = i_start;i < i_start + OUT_TILE_WIDTH;i++)  // loop through each layer in z axis
    {
        if (i + 1 >= 0 && i + 1 < N && j >= 0 && j < N && k >= 0 && k < N)  //loda next layer
            next[threadIdx.y][threadIdx.x] = inp.ele[(i + 1) * N * N + j * N + k];

        __syncthreads();

        if (i >= 1 && i < N - 1 && j >= 1 && j < N - 1 && k >= 1 && k < N)  
        {   //boundary layer of input matrix is ignored (as per assumption) - to access correct locations of output matrix
            if (threadIdx.y >= 1 && threadIdx.y < IN_TILE_WIDTH - 1 && threadIdx.x >= 1 && threadIdx.x < IN_TILE_WIDTH - 1)
            {       //if thread is inner part of tile to access correct location of SM(exclude the border)
                out.ele[i * N * N + j * N + k] = coef[0] * cur[threadIdx.y][threadIdx.x] +
                                                 coef[1] * cur[threadIdx.y][threadIdx.x-1] +
                                                 coef[2] * cur[threadIdx.y][threadIdx.x+1] +
                                                 coef[3] * cur[threadIdx.y-1][threadIdx.x] +
                                                 coef[4] * cur[threadIdx.y+1][threadIdx.x] +
                                                 coef[5] * prev[threadIdx.y][threadIdx.x] +
                                                 coef[6] * next[threadIdx.y][threadIdx.x];
            }
        }
        __syncthreads();
        prev[threadIdx.y][threadIdx.x] = cur[threadIdx.y][threadIdx.x];
        cur[threadIdx.y][threadIdx.x] = next[threadIdx.y][threadIdx.x];
    }
}

int main() {
    Matrix inp, d_inp, out, d_out;
    inp.depth = d_inp.depth = out.depth = d_out.depth = N;
    inp.rows = d_inp.rows = out.rows = d_out.rows = N;
    inp.cols = d_inp.cols = out.cols = d_out.cols = N;

    size_t mat_size = inp.depth * inp.rows * inp.cols * sizeof(float);

    inp.ele = (float*)malloc(mat_size);
    out.ele = (float*)malloc(mat_size);
    cudaMalloc(&d_inp.ele, mat_size);
    cudaMalloc(&d_out.ele, mat_size);

    // Initialization of input matrix
    for (int i = 0; i < inp.cols * inp.rows * inp.depth; i++)
        inp.ele[i] = float(i + 1);

    cudaMemcpy(d_inp.ele, inp.ele, mat_size, cudaMemcpyHostToDevice);

    float h_coef[7] = { 1,1,1,1,1,1,1 };

    // Copy constant data to constant memory
    cudaMemcpyToSymbol(coef, h_coef, 7 * sizeof(float));

    dim3 blockDim(IN_TILE_WIDTH, IN_TILE_WIDTH, 1);
    dim3 gridDim((N + OUT_TILE_WIDTH - 1) / OUT_TILE_WIDTH,
        (N + OUT_TILE_WIDTH - 1) / OUT_TILE_WIDTH,
        (N + OUT_TILE_WIDTH - 1) / OUT_TILE_WIDTH);

    Stencil << <gridDim, blockDim >> > (d_inp, d_out);
    cudaMemcpy(out.ele, d_out.ele, mat_size, cudaMemcpyDeviceToHost);

    cout << "Testing :" << endl;
    int i = 2, j = 1, k = 5;
    cout << "Inp[" << i << "][" << j << "][" << k << "] : " << inp.ele[i * N * N + j * N + k] << endl;
    cout << "Inp[" << i << "][" << j << "][" << k - 1 << "] : " << inp.ele[i * N * N + j * N + (k - 1)] << endl;
    cout << "Inp[" << i << "][" << j << "][" << k + 1 << "] : " << inp.ele[i * N * N + j * N + (k + 1)] << endl;
    cout << "Inp[" << i << "][" << j - 1 << "][" << k << "] : " << inp.ele[i * N * N + (j - 1) * N + k] << endl;
    cout << "Inp[" << i << "][" << j + 1 << "][" << k << "] : " << inp.ele[i * N * N + (j + 1) * N + k] << endl;
    cout << "Inp[" << i - 1 << "][" << j << "][" << k << "] : " << inp.ele[(i - 1) * N * N + j * N + k] << endl;
    cout << "Inp[" << i + 1 << "][" << j << "][" << k << "] : " << inp.ele[(i + 1) * N * N + j * N + k] << endl;
    cout << "\nOut[" << i << "][" << j << "][" << k << "] : " << out.ele[i * N * N + j * N + k] << endl;

    cudaFree(d_inp.ele);
    cudaFree(d_out.ele);
    free(inp.ele);
    free(out.ele);

    return 1;
}
```

<hr>

### Stencil - Single Order Derivative - Three Variables - Shared Memory - Constant Memory - Thread Coarsening - Register Tiling
```
#include <iostream>
#include <cuda_runtime.h>

#define IN_TILE_WIDTH 4
#define OUT_TILE_WIDTH (IN_TILE_WIDTH - 2)
#define N 32

using namespace std;

__constant__ float coef[7];

typedef struct {
    int rows;
    int cols;
    int depth;
    float* ele;
} Matrix;

__global__ void Stencil(const Matrix inp, Matrix out)
{
    int i_start = blockIdx.z * OUT_TILE_WIDTH;
    int j = blockIdx.y * OUT_TILE_WIDTH + threadIdx.y - 1;
    int k = blockIdx.x * OUT_TILE_WIDTH + threadIdx.x - 1;

    __shared__ float cur[IN_TILE_WIDTH][IN_TILE_WIDTH];
    float prev, next;

    if (i_start - 1 >= 0 && i_start - 1 < N && j >= 0 && j < N && k >= 0 && k < N) //load only prev layer element
        prev = inp.ele[(i_start - 1) * N * N + j * N + k];

    if (i_start >= 0 && i_start < N && j >= 0 && j < N && k >= 0 && k < N)  //load current layer
        cur[threadIdx.y][threadIdx.x] = inp.ele[i_start * N * N + j * N + k];

    for (int i = i_start;i < i_start + OUT_TILE_WIDTH;i++)  // loop through each layer in z axis
    {
        if (i + 1 >= 0 && i + 1 < N && j >= 0 && j < N && k >= 0 && k < N)  //load only next layer element
            next = inp.ele[(i + 1) * N * N + j * N + k];

        __syncthreads();

        if (i >= 1 && i < N - 1 && j >= 1 && j < N - 1 && k >= 1 && k < N-1)
        {   //boundary layer of input matrix is ignored (as per assumption) - to access correct locations of output matrix
            if (threadIdx.y >= 1 && threadIdx.y < IN_TILE_WIDTH - 1 && threadIdx.x >= 1 && threadIdx.x < IN_TILE_WIDTH - 1)
            {       //if thread is inner part of tile to access correct location of SM(exclude the border)
                out.ele[i * N * N + j * N + k] = coef[0] * cur[threadIdx.y][threadIdx.x] +
                    coef[1] * cur[threadIdx.y][threadIdx.x - 1] +
                    coef[2] * cur[threadIdx.y][threadIdx.x + 1] +
                    coef[3] * cur[threadIdx.y - 1][threadIdx.x] +
                    coef[4] * cur[threadIdx.y + 1][threadIdx.x] +
                    coef[5] * prev +
                    coef[6] * next;
            }
        }
        __syncthreads();
        prev = cur[threadIdx.y][threadIdx.x];
        cur[threadIdx.y][threadIdx.x] = next;
    }
}

int main() {
    Matrix inp, d_inp, out, d_out;
    inp.depth = d_inp.depth = out.depth = d_out.depth = N;
    inp.rows = d_inp.rows = out.rows = d_out.rows = N;
    inp.cols = d_inp.cols = out.cols = d_out.cols = N;

    size_t mat_size = inp.depth * inp.rows * inp.cols * sizeof(float);

    inp.ele = (float*)malloc(mat_size);
    out.ele = (float*)malloc(mat_size);
    cudaMalloc(&d_inp.ele, mat_size);
    cudaMalloc(&d_out.ele, mat_size);

    // Initialization of input matrix
    for (int i = 0; i < inp.cols * inp.rows * inp.depth; i++)
        inp.ele[i] = float(i + 1);

    cudaMemcpy(d_inp.ele, inp.ele, mat_size, cudaMemcpyHostToDevice);

    float h_coef[7] = { 1,1,1,1,1,1,1 };

    // Copy constant data to constant memory
    cudaMemcpyToSymbol(coef, h_coef, 7 * sizeof(float));

    dim3 blockDim(IN_TILE_WIDTH, IN_TILE_WIDTH, 1);
    dim3 gridDim((N + OUT_TILE_WIDTH - 1) / OUT_TILE_WIDTH,
        (N + OUT_TILE_WIDTH - 1) / OUT_TILE_WIDTH,
        (N + OUT_TILE_WIDTH - 1) / OUT_TILE_WIDTH);

    Stencil << <gridDim, blockDim >> > (d_inp, d_out);
    cudaMemcpy(out.ele, d_out.ele, mat_size, cudaMemcpyDeviceToHost);

    cout << "Testing :" << endl;
    int i = 2, j = 1, k = 5;
    cout << "Inp[" << i << "][" << j << "][" << k << "] : " << inp.ele[i * N * N + j * N + k] << endl;
    cout << "Inp[" << i << "][" << j << "][" << k - 1 << "] : " << inp.ele[i * N * N + j * N + (k - 1)] << endl;
    cout << "Inp[" << i << "][" << j << "][" << k + 1 << "] : " << inp.ele[i * N * N + j * N + (k + 1)] << endl;
    cout << "Inp[" << i << "][" << j - 1 << "][" << k << "] : " << inp.ele[i * N * N + (j - 1) * N + k] << endl;
    cout << "Inp[" << i << "][" << j + 1 << "][" << k << "] : " << inp.ele[i * N * N + (j + 1) * N + k] << endl;
    cout << "Inp[" << i - 1 << "][" << j << "][" << k << "] : " << inp.ele[(i - 1) * N * N + j * N + k] << endl;
    cout << "Inp[" << i + 1 << "][" << j << "][" << k << "] : " << inp.ele[(i + 1) * N * N + j * N + k] << endl;
    cout << "\nOut[" << i << "][" << j << "][" << k << "] : " << out.ele[i * N * N + j * N + k] << endl;

    cudaFree(d_inp.ele);
    cudaFree(d_out.ele);
    free(inp.ele);
    free(out.ele);

    return 1;
}
```
<hr>

### Basic Histogram Calculation - Atomic Operation

```
#include<iostream>
#include<cuda_runtime.h>
#define ALPHABET 26

using namespace std;

__global__
void HistCalc(const char* data, unsigned int* hist, size_t len)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i < len)
	{
		int val = (int)(data[i] - 'a');	//hash index value
		if (val >= 0 && val < ALPHABET)
			atomicAdd(&hist[val], 1);	// atomic addition
	}
}

int main()
{
	string input;
	cout << "Enter input string:"<<endl;
	cin >> input;
	size_t size = input.size();

	char* data = (char*)input.c_str();	//convert to c char aray from string
	unsigned int hist[ALPHABET] = { 0 };

	char* d_data;
	unsigned int* d_hist;
	
	cudaMalloc(&d_data, size);	//allocate device data for string
	cudaMalloc(&d_hist, ALPHABET * sizeof(unsigned int));	//allocate device data for histogram

	//copy data from host to device
	cudaMemcpy(d_data, data, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_hist, hist, ALPHABET * sizeof(unsigned int), cudaMemcpyHostToDevice);

	dim3 blockDim(256, 1);
	dim3 gridDim((size - 1) / blockDim.x + 1, 1);

	HistCalc << <gridDim, blockDim >> > (d_data, d_hist, size);

	cudaMemcpy(hist, d_hist, ALPHABET * sizeof(unsigned int), cudaMemcpyDeviceToHost);

	for (int i = 0;i < ALPHABET;i++)
		if (hist[i] != 0)
			cout << (char)('a' + i) << "-" << hist[i]<<endl;

	cudaFree(d_data);cudaFree(d_hist);

	return 1;
}
```

<hr>

### Histogram Calculation - Privatisation
```
#include<iostream>
#include<string.h>
#include<cuda_runtime.h>

#define ALPHABET 26
#define BIN_WIDTH 5	//width of one bin
#define NUM_BINS (((ALPHABET-1)/BIN_WIDTH)+1)	//number of bins
#define BLOCK_SIZE 3

using namespace std;

__global__
void HistCalc_Privatization(const char* data, unsigned int* hist, const size_t len)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;

	if (i < len)
	{
		int pos = data[i] - 'a';	//get the index
		if(pos>=0 && pos<ALPHABET)	//if present in valid range
			atomicAdd(&hist[blockIdx.x * NUM_BINS + pos / BIN_WIDTH], 1);	//atomic add on the private loc of block in hist
	}
	__syncthreads();

	if (blockIdx.x > 0)	//skip block 0 since accumulation result is block 0's bins only
	{
		for (unsigned int j = threadIdx.x;j < NUM_BINS;j += blockDim.x)	//one thread takes care of multiple bins
		{
			unsigned int binValue = hist[blockIdx.x * NUM_BINS + j];	//get bin value
			if (binValue > 0) {
				atomicAdd(&(hist[j]), binValue);	//add the private hist value to block 0's bin val
			}
		}
	}
}

int main()
{
	string input;
	cout << "Enter the string input:\n";
	cin >> input;
	size_t inp_size = input.size();	//size of input string

	char* data = (char*)input.c_str();	//string to C char array
	char* d_data;
	unsigned int* d_hist;
	unsigned int hist[NUM_BINS];

	int grid_size = (inp_size - 1) / BLOCK_SIZE + 1;

	// allocate device memory
	cudaMalloc(&d_data, inp_size);	
	cudaMalloc(&d_hist, sizeof(unsigned int) * NUM_BINS * grid_size);
	cudaMemset(d_hist, 0, sizeof(unsigned int) * NUM_BINS * grid_size);	//initialize value

	// copy data from host to device
	cudaMemcpy(d_data, data, inp_size, cudaMemcpyHostToDevice);

	dim3 blockDim(BLOCK_SIZE, 1);
	dim3 gridDim(grid_size, 1);

	HistCalc_Privatization << <gridDim, blockDim >> > (d_data, d_hist, inp_size);

	// copy data from device to host
	cudaMemcpy(hist, d_hist, NUM_BINS * sizeof(unsigned int), cudaMemcpyDeviceToHost);

	// display data
	for (unsigned int i = 0;i < NUM_BINS;i++)
	{
		char start = 'a' + i * BIN_WIDTH;
		char end = start + BIN_WIDTH - 1;
		if (end > 'z')
			end = 'z';
		if (hist[i] > 0)
			cout << start << '-' << end <<':'<<hist[i] << endl;
	}
	cudaFree(d_data);cudaFree(d_hist);
	return 1;
}
```
<hr>

### Histogram Calculation - Privatization - Shared Memory
```
#include<iostream>
#include<cuda_runtime.h>
#include<string.h>

#define ALPHABET 26
#define BIN_WIDTH 4	//width of one bin
#define NUM_BINS (((ALPHABET-1)/BIN_WIDTH)+1)	//number of bins
#define BLOCK_WIDTH 3

using namespace std;

__global__
void HistCalcSM(const char* data, unsigned int* hist, const int len)
{
	//initialize histogram
	__shared__ unsigned int SM_hist[NUM_BINS];	
	for (int i = threadIdx.x;i < NUM_BINS;i+=blockDim.x)
		SM_hist[i] = 0;

	__syncthreads();

	unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i < len)
	{
		int pos = data[i] - 'a';//get the index
		if (pos >= 0 && pos < ALPHABET)//if present in valid range
			atomicAdd(&SM_hist[pos/BIN_WIDTH], 1);//atomic add on SM
	}

	__syncthreads();

	for (int bin = threadIdx.x;bin < NUM_BINS;bin += blockDim.x)
	{
		int val = SM_hist[bin];
		if(val >0)
			atomicAdd(&hist[bin], val);	//add to global memory
	}

}

int main()
{
	string input;
	cout << "Enter a string:\n";
	cin >> input;
	size_t size = input.size();
	int grid_dim = (size - 1) / BLOCK_WIDTH + 1;

	char* data = (char*)input.c_str();
	char* d_data;
	unsigned int hist[NUM_BINS] = { 0 };
	unsigned int* d_hist;

	cudaMalloc(&d_data, size);
	cudaMemcpy(d_data, data, size, cudaMemcpyHostToDevice);

	cudaMalloc(&d_hist, NUM_BINS * sizeof(unsigned int));
	cudaMemset(d_hist, 0, NUM_BINS * sizeof(unsigned int));

	dim3 blockDim(BLOCK_WIDTH, 1);
	dim3 gridDim(grid_dim, 1);

	HistCalcSM << <gridDim, blockDim >> > (d_data, d_hist, size);

	cudaMemcpy(hist, d_hist, NUM_BINS * sizeof(unsigned int), cudaMemcpyDeviceToHost);

	for (int i = 0;i < NUM_BINS;i++)
	{
		char start = (char)('a' + i*BIN_WIDTH);
		char end = start + BIN_WIDTH - 1;
		if (end > 'z')end = 'z';
		if (hist[i] > 0)
			cout << start << "-" << end << ":" << hist[i] << endl;
	}

	cudaFree(d_data); cudaFree(d_hist);
	
	return 1;
}
```
<hr>

### Histogram Calculation - Privatization - Shared Memory - Coarsen - Contiguous Partition
```
#include<iostream>
#include<cuda_runtime.h>
#include<string.h>

#define BLOCK_SIZE 3
#define ALPHABET 26
#define BIN_WIDTH 5
#define NUM_BINS (((ALPHABET-1)/BIN_WIDTH)+1)
#define COARSE_FACTOR 3
using namespace std;

__global__
void HistCalcContiguousPartition(const char* data, unsigned int* hist, const int len)
{
	__shared__ unsigned int SM[NUM_BINS];
	for (int i = threadIdx.x; i < NUM_BINS;i += blockDim.x)
		SM[i] = 0;

	__syncthreads();

	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	for (int i = tid * COARSE_FACTOR;i < min(len, (tid + 1) * COARSE_FACTOR);i++)
	{
		int pos = data[i] - 'a';
		if (pos >= 0 && pos < ALPHABET)
		{
			atomicAdd(&SM[pos / BIN_WIDTH], 1);
		}
	}

	__syncthreads();
	for (int j = threadIdx.x; j < NUM_BINS;j += blockDim.x)
	{
		int val = SM[j];
		if (val > 0)
			atomicAdd(&hist[j], val);
	}
}

int main()
{
	string input;
	cout << "Enter the string:\n";
	cin >> input;

	size_t size = input.size();
	unsigned int hist[NUM_BINS] = { 0 };
	unsigned int* d_hist;
	int grid_size = (size - 1) / (BLOCK_SIZE*COARSE_FACTOR)+1;

	char* data = (char*)input.c_str();
	char* d_data;
	cudaMalloc(&d_data, size);
	cudaMemcpy(d_data, data, size, cudaMemcpyHostToDevice);
	cudaMalloc(&d_hist, NUM_BINS * sizeof(unsigned int));
	cudaMemset(d_hist, 0, NUM_BINS * sizeof(unsigned int));

	dim3 blockDim(BLOCK_SIZE);
	dim3 gridDim(grid_size);

	HistCalcContiguousPartition << <gridDim, blockDim >> > (d_data, d_hist, size);

	cudaMemcpy(hist, d_hist, NUM_BINS * sizeof(unsigned int), cudaMemcpyDeviceToHost);

	for (int i = 0;i < NUM_BINS;i++)
	{
		char start = (char)('a' + i * BIN_WIDTH);
		char end = start + BIN_WIDTH - 1;
		if (end > 'z') end = 'z';
			cout << start << "-" << end << ":" << hist[i]<<endl;
	}

	cudaFree(d_data); cudaFree(d_hist);

	return 1;
}
```

<hr>

### Histogram Calculation - Privatization - Shared Memory - Coarsen - Interleaved Partition

```
#include<iostream>
#include<cuda_runtime.h>
#include<string.h>

#define BLOCK_SIZE 5
#define ALPHABET 26
#define BIN_WIDTH 4
#define NUM_BINS (((ALPHABET-1)/BIN_WIDTH)+1)
#define COARSE_FACTOR 3

using namespace std;

__global__
void HistCalcContiguousPartition (const char* data, unsigned int* hist, const int len)
{
	__shared__ unsigned int SM[NUM_BINS];
	for (int i = threadIdx.x; i < NUM_BINS;i += blockDim.x)
		SM[i] = 0;

	__syncthreads();

	int tid = threadIdx.x + blockDim.x * blockIdx.x;
	for (int i = tid ; i <len ;i+=blockDim.x*gridDim.x)
	{
		int pos = data[i] - 'a';
		if (pos >= 0 && pos < ALPHABET)
			atomicAdd(&SM[pos / BIN_WIDTH], 1);
	}

	__syncthreads();

	for (int i = threadIdx.x; i < NUM_BINS;i += blockDim.x)
	{
		int val = SM[i];
		if (val > 0)
			atomicAdd(&hist[i], val);
	}
}

int main()
{
	string input;
	cout << "Enter the string:\n";
	cin >> input;
	size_t size = input.size();
	int grid_size = (size - 1) / (BLOCK_SIZE * COARSE_FACTOR) + 1;

	char* data = (char*)input.c_str();
	char* d_data;
	cudaMalloc(&d_data, size);
	cudaMemcpy(d_data, data, size, cudaMemcpyHostToDevice);

	unsigned int hist[NUM_BINS] = { 0 };
	unsigned int* d_hist;
	cudaMalloc(&d_hist, NUM_BINS * sizeof(unsigned int));
	cudaMemset(d_hist, 0, NUM_BINS * sizeof(unsigned int));

	dim3 blockDim(BLOCK_SIZE, 1);
	dim3 gridDim(grid_size, 1);

	HistCalcContiguousPartition << <gridDim, blockDim >> > (d_data, d_hist, size);

	cudaMemcpy(hist, d_hist, NUM_BINS * sizeof(unsigned int), cudaMemcpyDeviceToHost);

	for (int i = 0;i < NUM_BINS;i++)
	{
		char start = 'a' + i * BIN_WIDTH;
		char end = start + BIN_WIDTH-1;
		if (end > 'z') end = 'z';
		cout << start << "-" << end << ":" << hist[i] << endl;
	}

	cudaFree(d_data); cudaFree(d_hist);
	return 1;
}
```

<hr>

### Histogram Calculation - Privatization - Shared Memory - Coarsen - Interleaved Partition - Accumulator
```
#include<iostream>
#include<cuda_runtime.h>
#include<string.h>

#define ALPHABET 26
#define BIN_WIDTH 4
#define NUM_BINS (((ALPHABET-1)/BIN_WIDTH)+1)
#define BLOCK_WIDTH 3

using namespace std;

__global__
void HistCalc(const char* data, unsigned int* hist, int len)
{
	__shared__ unsigned int SM[NUM_BINS];	// initialize shared memory
	for (int i = threadIdx.x;i < NUM_BINS;i += blockDim.x)
		SM[i] = 0;

	__syncthreads();

	unsigned int accumulator = 0;	//accumulator to accumulate consecutive identical bins
	int prev_bin = -1;	// track variable to track previous bin
	int tid = blockDim.x * blockIdx.x + threadIdx.x;

	for (int i = tid; i < len;i += gridDim.x * blockDim.x)
	{
		int pos = data[i] - 'a';
		if (pos >= 0 && pos < ALPHABET)
		{
			int bin_pos = pos / BIN_WIDTH;
			if (bin_pos == prev_bin)	// if current bin match prev bin
				accumulator += 1;	//increment the accumulator
			else  // if not matched
			{
				if (accumulator > 0)
				{
					atomicAdd(&SM[prev_bin], accumulator);	//flush the existing accumulator to SM
				}
				accumulator = 1;	//set 1 for curent bin
				prev_bin = bin_pos;	//update the prev bin
			}
		}
	}

	if (accumulator > 0)	//flush out remaining accumulator (for last character)
		atomicAdd(&SM[prev_bin], accumulator);

	__syncthreads();

	for (int i = threadIdx.x; i < NUM_BINS; i += blockDim.x)
	{
		int val = SM[i];
		if (val > 0)
		{
			atomicAdd(&hist[i], val);
		}
	}

}

int main()
{
	string input;
	cout << "Enter a string:\n";
	cin >> input;

	size_t size = input.size();
	int grid_size = (size - 1) / BLOCK_WIDTH + 1;

	char* data = (char*)input.c_str();
	char* d_data;
	cudaMalloc(&d_data, size);
	cudaMemcpy(d_data, data, size, cudaMemcpyHostToDevice);

	unsigned int hist[NUM_BINS] = { 0 };
	unsigned int* d_hist;

	cudaMalloc(&d_hist, NUM_BINS * sizeof(unsigned int));
	cudaMemset(d_hist, 0, NUM_BINS * sizeof(unsigned int));

	dim3 blockDim(BLOCK_WIDTH, 1);
	dim3 gridDim(grid_size, 1);

	HistCalc << <gridDim, blockDim >> > (d_data, d_hist, size);

	cudaMemcpy(hist, d_hist, NUM_BINS * sizeof(unsigned int), cudaMemcpyDeviceToHost);

	for (int i = 0;i < NUM_BINS; i += 1)
	{
		char start = 'a' + i * BIN_WIDTH;
		char end = start + BIN_WIDTH-1;
		if (end > 'z') end = 'z';
		cout << start << "-" << end << ":" << hist[i] << endl;
	}

	cudaFree(d_data);cudaFree(d_hist);
	return 1;
}
```
<hr>

### Sum Reduction
```
#include <iostream>
#include <cuda_runtime.h>

using namespace std;

// Kernel function for reduction
__global__ void Reduction(float* data, float* out)
{
    int i = 2 * threadIdx.x;
    for (int stride = 1; stride <= blockDim.x; stride *= 2)
    {
        if (threadIdx.x % stride == 0)
        {
            data[i] += data[i + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0)
        *out = data[0];
}

// Function to determine the optimal block size
int getOptimalBlockSize(int size)
{
    int blockSize = 1;
    while (blockSize < size)
    {
        blockSize *= 2;
    }
    return blockSize / 2; // Return the largest power of 2 <= size
}

int main()
{
    int l = 46; // Size of the data
    float* data, *d_data, *out, *d_out;
    size_t size = l * sizeof(float);

    // Allocate and initialize host memory
    data = (float*)malloc(size);
    out = (float*)malloc(sizeof(float)); // Allocate memory for output

    for (int i = 0; i < l; i++)
        data[i] = i + 1;

    // Allocate device memory
    cudaMalloc(&d_data, size);
    cudaMalloc(&d_out, sizeof(float));

    // Copy data from host to device
    cudaMemcpy(d_data, data, size, cudaMemcpyHostToDevice);

    // Determine optimal block size based on l
    int blockSize = getOptimalBlockSize(l);
    int numBlocks = 1;

    cout << blockSize;

    // Launch kernel
    Reduction << <numBlocks, blockSize >> > (d_data, d_out);

    // Copy result from device to host
    cudaMemcpy(out, d_out, sizeof(float), cudaMemcpyDeviceToHost);

    // Print input data
    cout << "Inputs:\n";
    for (int i = 0; i < l; i++)
        cout << data[i] << "   ";
    cout << "\nOutput: " << *out << endl;

    // Free allocated memory
    cudaFree(d_out);
    cudaFree(d_data);
    free(data);
    free(out);

    return 1;
}
```
<hr>

### Sum Reduction - Minimized Control Divergence - Improved Memory Coalescing
```
#include <iostream>
#include <cuda_runtime.h>

using namespace std;

// Kernel function for reduction
__global__ void Reduction(float* data, float* out)
{
    int i = threadIdx.x;
    for (int stride = blockDim.x; stride >= 1; stride /= 2)
    {
        if (threadIdx.x < stride)
            data[i] += data[i + stride];
        __syncthreads();
    }
    if(threadIdx.x==0)
        *out = data[0];
}

// Function to determine the optimal block size
int getOptimalBlockSize(int size)
{
    int blockSize = 1;
    while (blockSize < size)
    {
        blockSize *= 2;
    }
    return blockSize / 2; // Return the largest power of 2 <= size
}

int main()
{
    int l = 47; // Size of the data
    float* data, * d_data, * out, * d_out;
    size_t size = l * sizeof(float);

    // Allocate and initialize host memory
    data = (float*)malloc(size);
    out = (float*)malloc(sizeof(float)); // Allocate memory for output

    for (int i = 0; i < l; i++)
        data[i] = i + 1;

    // Allocate device memory
    cudaMalloc(&d_data, size);
    cudaMalloc(&d_out, sizeof(float));

    // Copy data from host to device
    cudaMemcpy(d_data, data, size, cudaMemcpyHostToDevice);

    // Determine optimal block size based on l
    int blockSize = getOptimalBlockSize(l);
    int numBlocks = 1;

    // Launch kernel
    Reduction << <numBlocks, blockSize >> > (d_data, d_out);

    // Copy result from device to host
    cudaMemcpy(out, d_out, sizeof(float), cudaMemcpyDeviceToHost);

    // Print input data
    cout << "Inputs:\n";
    for (int i = 0; i < l; i++)
        cout << data[i] << "   ";
    cout << "\nOutput: " << *out << endl;

    // Free allocated memory
    cudaFree(d_out);
    cudaFree(d_data);
    free(data);
    free(out);

    return 1;
}
```
<hr>

### Sum Reduction - Minimized Control Divergence - Improved Memory Coalescing - Rightside Aggregation
```
#include <iostream>
#include <cuda_runtime.h>

using namespace std;

__global__ void Reduction(float* data, float* out)
{
    int i = threadIdx.x + blockDim.x;
    int prev_stride = 0;
    for (int stride = blockDim.x; stride >= 1; stride /= 2)
    {
        if (i >= stride + prev_stride)
            data[i] += data[i - stride];
        __syncthreads();
        prev_stride += stride;
    }
    if (i == 2*blockDim.x-1)
        *out = data[i];
}

int getOptimalBlockSize(int size)
{
    int blockSize = 1;
    while (blockSize < size)
    {
        blockSize *= 2;
    }
    return blockSize / 2; // Return the largest power of 2 <= size
}

int main()
{
    int l = 16; // Size of the data
    float* data, * d_data, * out, * d_out;
    size_t size = l * sizeof(float);

    // Allocate and initialize host memory
    data = (float*)malloc(size);
    out = (float*)malloc(sizeof(float)); // Allocate memory for output

    for (int i = 0; i < l; i++)
        data[i] = i;

    // Allocate device memory
    cudaMalloc(&d_data, size);
    cudaMalloc(&d_out, sizeof(float));

    // Copy data from host to device
    cudaMemcpy(d_data, data, size, cudaMemcpyHostToDevice);

    // Determine optimal block size based on l
    int blockSize = getOptimalBlockSize(l);
    int numBlocks = 1;

    // Launch kernel
    Reduction << <numBlocks, blockSize >> > (d_data, d_out);

    // Copy result from device to host
    cudaMemcpy(out, d_out, sizeof(float), cudaMemcpyDeviceToHost);

    // Print input data
    cout << "Inputs:\n";
    for (int i = 0; i < l; i++)
        cout << data[i] << "   ";
    cout << "\nOutput: " << *out << endl;

    // Free allocated memory
    cudaFree(d_out);
    cudaFree(d_data);
    free(data);
    free(out);

    return 1;
}
```
<hr>

### Sum Reduction - Minimized Control Divergence - Improved Memory Coalescing - Minimized Global Memory Access
```
#include <iostream>
#include <cuda_runtime.h>
#define BLOCK_SIZE 1024

using namespace std;

// Kernel function for reduction
__global__ void Reduction(float* data, float* out, int n)
{
    __shared__ float SM[BLOCK_SIZE];
    int i = threadIdx.x;

    SM[i] = data[i] + data[i+BLOCK_SIZE];

    // Perform reduction in shared memory
    for (int stride = blockDim.x / 2; stride >= 1; stride /= 2)
    {
        __syncthreads();

        if (threadIdx.x < stride)
            SM[i] += SM[i + stride];
    }

    if (threadIdx.x == 0) {
        *out = SM[0];
    }
}

int main()
{
    int l = 100; // Size of the data
    float* data, * d_data, * out, * d_out;
    size_t size = l * sizeof(float);

    // Allocate and initialize host memory
    data = (float*)malloc(size);
    out = (float*)malloc(sizeof(float)); // Allocate memory for output

    for (int i = 0; i < l; i++)
        data[i] = i + 1;

    // Allocate device memory
    cudaMalloc(&d_data, size);
    cudaMalloc(&d_out, sizeof(float));

    // Copy data from host to device
    cudaMemcpy(d_data, data, size, cudaMemcpyHostToDevice);

    // Launch kernel with one block
    Reduction << <1, BLOCK_SIZE >> > (d_data, d_out, l);

    // Copy result from device to host
    cudaMemcpy(out, d_out, sizeof(float), cudaMemcpyDeviceToHost);

    // Print input data
    cout << "Inputs:\n";
    for (int i = 0; i < l; i++)
        cout << data[i] << "   ";
    cout << "\nOutput: " << *out << endl;

    // Free allocated memory
    cudaFree(d_out);
    cudaFree(d_data);
    free(data);
    free(out);

    return 1;
}
```

<hr>

### Sum Reduction - Segmented to Multiblock
```
#include <iostream>
#include <cuda_runtime.h>
#define BLOCK_SIZE 32
#define COARSENING_FACTOR 2

using namespace std;

// Kernel function for reduction
__global__ void Reduction(float* data, float* out, int n)
{
    __shared__ float SM[BLOCK_SIZE];
    int segment = COARSENING_FACTOR * 2 * blockDim.x * blockIdx.x;
    int i = segment + threadIdx.x;
    int t = threadIdx.x;

    float sum = data[i];
    for (int j = 1; j < COARSENING_FACTOR * 2; j++)
        sum += data[i + j * BLOCK_SIZE];

    SM[t] = sum;
    for (int stride = blockDim.x / 2; stride >= 1; stride /= 2)
    {
        __syncthreads();
        if (t < stride)
            SM[t] += SM[t + stride];
    }
    if (t == 0)
        atomicAdd(out, SM[0]);
}

int main()
{
    int l = 1021; // Size of the data
    float* data, * d_data, * out, * d_out;
    size_t size = l * sizeof(float);

    // Allocate and initialize host memory
    data = (float*)malloc(size);
    out = (float*)malloc(sizeof(float)); // Allocate memory for output

    for (int i = 0; i < l; i++)
        data[i] = i + 1;

    // Allocate device memory
    cudaMalloc(&d_data, size);
    cudaMalloc(&d_out, sizeof(float));

    // Copy data from host to device
    cudaMemcpy(d_data, data, size, cudaMemcpyHostToDevice);

    // Compute grid and block dimensions
    dim3 blockDim(BLOCK_SIZE);
    dim3 gridDim((l - 1) / (blockDim.x * COARSENING_FACTOR) + 1);

    // Launch kernel
    Reduction << <gridDim, blockDim >> > (d_data, d_out, l);

    // Copy result from device to host
    cudaMemcpy(out, d_out, sizeof(float), cudaMemcpyDeviceToHost);

    // Print input data
    cout << "Inputs:\n";
    for (int i = 0; i < l; i++)
        cout << data[i] << "   ";
    cout << "\nOutput: " << *out << endl;

    // Free allocated memory
    cudaFree(d_out);
    cudaFree(d_data);
    free(data);
    free(out);

    return 1;
}
```
<hr>

### Sum Reduction - Thread Coarsening
```
#include <iostream>
#include <cuda_runtime.h>
#define BLOCK_SIZE 32
#define COARSENING_FACTOR 2

using namespace std;

// Kernel function for reduction
__global__ void Reduction(float* data, float* out, int n)
{
    __shared__ float SM[BLOCK_SIZE];
    int segment = COARSENING_FACTOR * 2 * blockDim.x * blockIdx.x;
    int i = segment + threadIdx.x;
    int t = threadIdx.x;

    float sum = 0.0f;
    // Sum values within the segment
    for (int j = 0; j < COARSENING_FACTOR * 2 && i + j * BLOCK_SIZE < n; j++)
        sum += data[i + j * BLOCK_SIZE];

    SM[t] = sum;
    __syncthreads();

    // Perform reduction in shared memory
    for (int stride = blockDim.x / 2; stride >= 1; stride /= 2)
    {
        if (t < stride)
            SM[t] += SM[t + stride];
        __syncthreads();
    }

    // Write the result for this block to global memory
    if (t == 0)
        atomicAdd(out, SM[0]);
}

int main()
{
    int l = 1021; // Size of the data
    float* data, * d_data, * out, * d_out;
    size_t size = l * sizeof(float);

    // Allocate and initialize host memory
    data = (float*)malloc(size);
    out = (float*)malloc(sizeof(float)); // Allocate memory for output

    for (int i = 0; i < l; i++)
        data[i] = i + 1;

    // Allocate device memory
    cudaMalloc(&d_data, size);
    cudaMalloc(&d_out, sizeof(float));
    cudaMemset(d_out, 0, sizeof(float));
    cudaMemcpy(d_data, data, size, cudaMemcpyHostToDevice);

    // Compute grid and block dimensions
    dim3 blockDim(BLOCK_SIZE);
    dim3 gridDim((l - 1) / (BLOCK_SIZE * COARSENING_FACTOR) + 1);

    // Launch kernel
    Reduction << <gridDim, blockDim >> > (d_data, d_out, l);

    cudaMemcpy(out, d_out, sizeof(float), cudaMemcpyDeviceToHost);

    // Print input data
    cout << "Inputs:\n";
    for (int i = 0; i < l; i++)
        cout << data[i] << "   ";
    cout << "\nOutput: " << *out << endl;

    // Free allocated memory
    cudaFree(d_out);
    cudaFree(d_data);
    free(data);
    free(out);

    return 1;
}
```
<hr>

### Max Reduction - Thread Coarsening
```
#include <iostream>
#include <cuda_runtime.h>
#define BLOCK_SIZE 32
#define COARSENING_FACTOR 2

using namespace std;

// Kernel function for reduction (max)
__global__ void Reduction(float* data, float* out, int n)
{
    __shared__ float SM[BLOCK_SIZE];
    int segment = COARSENING_FACTOR * 2 * blockDim.x * blockIdx.x;
    int i = segment + threadIdx.x;
    int t = threadIdx.x;

    float max_val = -FLT_MAX; // Use a very small value initially
    // Find the maximum value within the segment
    for (int j = 0; j < COARSENING_FACTOR * 2 && i + j * BLOCK_SIZE < n; j++)
        max_val = max(max_val, data[i + j * BLOCK_SIZE]);

    SM[t] = max_val;
    __syncthreads();

    // Perform reduction in shared memory
    for (int stride = blockDim.x / 2; stride >= 1; stride /= 2)
    {
        if (t < stride)
            SM[t] = max(SM[t], SM[t + stride]);
        __syncthreads();
    }

    // Write the result for this block to global memory
    if (t == 0)
        atomicMax(reinterpret_cast<int*>(out), __float_as_int(SM[0]));
}

int main()
{
    int l = 648; // Size of the data
    float* data, * d_data, * out, * d_out;
    size_t size = l * sizeof(float);

    // Allocate and initialize host memory
    data = (float*)malloc(size);
    out = (float*)malloc(sizeof(float)); // Allocate memory for output

    for (int i = 0; i < l; i++)
        data[i] = i + 1;

    // Allocate device memory
    cudaMalloc(&d_data, size);
    cudaMalloc(&d_out, sizeof(float));
    cudaMemset(d_out, 0, sizeof(float)); // Initialize d_out with 0
    cudaMemcpy(d_data, data, size, cudaMemcpyHostToDevice);

    // Compute grid and block dimensions
    dim3 blockDim(BLOCK_SIZE);
    dim3 gridDim((l - 1) / (BLOCK_SIZE * COARSENING_FACTOR) + 1);

    // Launch kernel
    Reduction << <gridDim, blockDim >> > (d_data, d_out, l);

    cudaMemcpy(out, d_out, sizeof(float), cudaMemcpyDeviceToHost);

    // Print input data
    cout << "Inputs:\n";
    for (int i = 0; i < l; i++)
        cout << data[i] << "   ";
    cout << "\nMax Output: " << *out << endl;

    // Free allocated memory
    cudaFree(d_out);
    cudaFree(d_data);
    free(data);
    free(out);

    return 1;
}
```

### Koggne Stone Prefix Sum - Inclusive Scan Algorithm
```
#include<iostream>
#include<cuda_runtime.h>
#define SECTION_SIZE 10

using namespace std;

__global__ void Kogge_Stone_Inclusive_Prefix_Sum(int* data, int* out, const int L)
{
	__shared__ int SM[SECTION_SIZE];	//declare shared memory
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i < L)
		SM[threadIdx.x] = data[i];
	else
		SM[threadIdx.x] = 0;

	for (int stride = 1;stride < blockDim.x;stride *= 2)
	{
		__syncthreads();
		int temp;
		if (threadIdx.x >= stride)
			temp = SM[threadIdx.x] + SM[threadIdx.x - stride];

		__syncthreads();

		if (threadIdx.x >= stride)
			SM[threadIdx.x] = temp;
	}

	if (i < L)
		out[i] = SM[threadIdx.x];
}

int main()
{
	int* data, * d_data, *out, *d_out;	// input data
	int l = SECTION_SIZE;	//assuming number of elements is same as section size
	size_t size = l * sizeof(int);

	data = (int*)malloc(size);	// allocate host memory for input data
	out = (int*)malloc(size);	// allocate host memory for output data

	cudaMalloc(&d_data, size);	//allocate device memory for input data
	cudaMalloc(&d_out, size);	//allocate device memory for output data

	for (int i = 0;i < l;i++)	// initialize data
		data[i] = (i + 1);

	cudaMemcpy(d_data, data, size, cudaMemcpyHostToDevice);	// copy data from host to device

	dim3 blockDim(SECTION_SIZE);	//block size is section size
	dim3 gridDim((l - 1) / blockDim.x + 1);
	Kogge_Stone_Inclusive_Prefix_Sum<<<gridDim, blockDim>>>(d_data, d_out, l);	//invoke the kernel

	cudaMemcpy(out, d_out, size, cudaMemcpyDeviceToHost);	//copy output from device to host

	cout << "\nInput :" << endl;
	for (int i = 0;i < l;i++)
		cout << data[i] << "\t";

	cout << "\nOutput :" << endl;
	for (int i = 0;i < l;i++)
		cout<<out[i] << "\t";

	cudaFree(d_data);
	free(data);

	return 1;
}
```

### Koggne Stone Prefix Sum - Inclusive Scan Algorithm - Double Buffering
```
#include<iostream>
#include<cuda_runtime.h>
#define SECTION_SIZE 10

using namespace std;

__global__ void Kogge_Stone_Inclusive_Prefix_Sum(int* data, int* out, const int L)
{
    __shared__ int SM1[SECTION_SIZE];  // First shared memory buffer
    __shared__ int SM2[SECTION_SIZE];  // Second shared memory buffer

    int i = blockDim.x * blockIdx.x + threadIdx.x;

    // Initialize shared memory buffers from global memory
    if (i < L) {
        SM1[threadIdx.x] = data[i];
    }
    else {
        SM1[threadIdx.x] = 0;
    }

    // Perform the Kogge-Stone inclusive prefix sum using double buffering
    int* inputBuffer = SM1;
    int* outputBuffer = SM2;

    for (int stride = 1; stride < blockDim.x; stride *= 2) {
        __syncthreads();  // Ensure all threads have loaded data into shared memory

        if (threadIdx.x >= stride) {
            outputBuffer[threadIdx.x] = inputBuffer[threadIdx.x] + inputBuffer[threadIdx.x - stride];
        }
        else {
            outputBuffer[threadIdx.x] = inputBuffer[threadIdx.x];
        }

        // Swap buffers for the next iteration
        int* temp = inputBuffer;
        inputBuffer = outputBuffer;
        outputBuffer = temp;
    }

    __syncthreads();  // Ensure all threads have completed writing to shared memory

    // Write the final result back to global memory
    if (i < L) {
        out[i] = inputBuffer[threadIdx.x];
    }
}

int main()
{
	int* data, * d_data, *out, *d_out;	// input data
	int l = SECTION_SIZE;	//assuming number of elements is same as section size
	size_t size = l * sizeof(int);

	data = (int*)malloc(size);	// allocate host memory for input data
	out = (int*)malloc(size);	// allocate host memory for output data

	cudaMalloc(&d_data, size);	//allocate device memory for input data
	cudaMalloc(&d_out, size);	//allocate device memory for output data

	for (int i = 0;i < l;i++)	// initialize data
		data[i] = (i + 1);

	cudaMemcpy(d_data, data, size, cudaMemcpyHostToDevice);	// copy data from host to device

	dim3 blockDim(SECTION_SIZE);	//block size is section size
	dim3 gridDim((l - 1) / blockDim.x + 1);
	Kogge_Stone_Inclusive_Prefix_Sum<<<gridDim, blockDim>>>(d_data, d_out, l);	//invoke the kernel

	cudaMemcpy(out, d_out, size, cudaMemcpyDeviceToHost);	//copy output from device to host

	cout << "\nInput :" << endl;
	for (int i = 0;i < l;i++)
		cout << data[i] << "\t";

	cout << "\nOutput :" << endl;
	for (int i = 0;i < l;i++)
		cout<<out[i] << "\t";

	cudaFree(d_data);
	free(data);

	return 1;
}
```

### Koggne Stone Prefix Sum - Exclusive Scan Algorithm
```
#include<iostream>
#include<cuda_runtime.h>
#define SECTION_SIZE 10

using namespace std;

__global__ void Kogge_Stone_Inclusive_Prefix_Sum(int* data, int* out, const int L)
{
	__shared__ int SM[SECTION_SIZE];	//declare shared memory
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i < L && threadIdx.x != 0)
		SM[threadIdx.x] = data[i-1];
	else
		SM[threadIdx.x] = 0;

	for (int stride = 1;stride < blockDim.x;stride *= 2)
	{
		__syncthreads();
		int temp;
		if (threadIdx.x >= stride)
			temp = SM[threadIdx.x] + SM[threadIdx.x - stride];

		__syncthreads();

		if (threadIdx.x >= stride)
			SM[threadIdx.x] = temp;
	}

	if (i < L)
		out[i] = SM[threadIdx.x];
}

int main()
{
	int* data, * d_data, *out, *d_out;	// input data
	int l = SECTION_SIZE;	//assuming number of elements is same as section size
	size_t size = l * sizeof(int);

	data = (int*)malloc(size);	// allocate host memory for input data
	out = (int*)malloc(size);	// allocate host memory for output data

	cudaMalloc(&d_data, size);	//allocate device memory for input data
	cudaMalloc(&d_out, size);	//allocate device memory for output data

	for (int i = 0;i < l;i++)	// initialize data
		data[i] = (i + 1);

	cudaMemcpy(d_data, data, size, cudaMemcpyHostToDevice);	// copy data from host to device

	dim3 blockDim(SECTION_SIZE);	//block size is section size
	dim3 gridDim((l - 1) / blockDim.x + 1);
	Kogge_Stone_Inclusive_Prefix_Sum<<<gridDim, blockDim>>>(d_data, d_out, l);	//invoke the kernel

	cudaMemcpy(out, d_out, size, cudaMemcpyDeviceToHost);	//copy output from device to host

	cout << "\nInput :" << endl;
	for (int i = 0;i < l;i++)
		cout << data[i] << "\t";

	cout << "\nOutput :" << endl;
	for (int i = 0;i < l;i++)
		cout<<out[i] << "\t";

	cudaFree(d_data);
	free(data);

	return 1;
}
```

### Koggne Stone Prefix Sum - Exclusive Scan Algorithm - Double Buffering
```
#include<iostream>
#include<cuda_runtime.h>
#define SECTION_SIZE 10

using namespace std;

__global__ void Kogge_Stone_Inclusive_Prefix_Sum(int* data, int* out, const int L)
{
    __shared__ int SM1[SECTION_SIZE];  // First shared memory buffer
    __shared__ int SM2[SECTION_SIZE];  // Second shared memory buffer

    int i = blockDim.x * blockIdx.x + threadIdx.x;

    // Initialize shared memory buffers from global memory
    if (i < L && threadIdx.x != 0)
        SM1[threadIdx.x] = data[i - 1];
    else {
        SM1[threadIdx.x] = 0;
    }

    // Perform the Kogge-Stone inclusive prefix sum using double buffering
    int* inputBuffer = SM1;
    int* outputBuffer = SM2;

    for (int stride = 1; stride < blockDim.x; stride *= 2) {
        __syncthreads();  // Ensure all threads have loaded data into shared memory

        if (threadIdx.x >= stride) {
            outputBuffer[threadIdx.x] = inputBuffer[threadIdx.x] + inputBuffer[threadIdx.x - stride];
        }
        else {
            outputBuffer[threadIdx.x] = inputBuffer[threadIdx.x];
        }

        // Swap buffers for the next iteration
        int* temp = inputBuffer;
        inputBuffer = outputBuffer;
        outputBuffer = temp;
    }

    __syncthreads();  // Ensure all threads have completed writing to shared memory

    // Write the final result back to global memory
    if (i < L) {
        out[i] = inputBuffer[threadIdx.x];
    }
}

int main()
{
    int* data, * d_data, * out, * d_out;	// input data
    int l = SECTION_SIZE;	//assuming number of elements is same as section size
    size_t size = l * sizeof(int);

    data = (int*)malloc(size);	// allocate host memory for input data
    out = (int*)malloc(size);	// allocate host memory for output data

    cudaMalloc(&d_data, size);	//allocate device memory for input data
    cudaMalloc(&d_out, size);	//allocate device memory for output data

    for (int i = 0;i < l;i++)	// initialize data
        data[i] = (i + 1);

    cudaMemcpy(d_data, data, size, cudaMemcpyHostToDevice);	// copy data from host to device

    dim3 blockDim(SECTION_SIZE);	//block size is section size
    dim3 gridDim((l - 1) / blockDim.x + 1);
    Kogge_Stone_Inclusive_Prefix_Sum << <gridDim, blockDim >> > (d_data, d_out, l);	//invoke the kernel

    cudaMemcpy(out, d_out, size, cudaMemcpyDeviceToHost);	//copy output from device to host

    cout << "\nInput :" << endl;
    for (int i = 0;i < l;i++)
        cout << data[i] << "\t";

    cout << "\nOutput :" << endl;
    for (int i = 0;i < l;i++)
        cout << out[i] << "\t";

    cudaFree(d_data);
    free(data);

    return 1;
}
```