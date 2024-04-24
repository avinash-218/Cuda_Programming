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