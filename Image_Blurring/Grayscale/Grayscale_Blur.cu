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
