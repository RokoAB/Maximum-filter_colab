##5x5 max filter kernel
#using opencv


%%writefile test.cu
//#include <stdio.h>
//#include <stdlib.h>
//#include <opencv2/opencv.hpp>
using namespace cv;
__global__ void max_filter_5x5(const uchar* input, uchar* output, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int halfSize = 2;
    int maxValR = 0;
    int maxValG = 0;
    int maxValB = 0;

    for (int dy = -halfSize; dy <= halfSize; ++dy) {
        for (int dx = -halfSize; dx <= halfSize; ++dx) {
            int nx = min(max(x + dx, 0), width - 1);
            int ny = min(max(y + dy, 0), height - 1);
            int idx = (ny * width + nx) * 3; // assuming 3 channels (RGB)
            maxValR = max(maxValR, input[idx]);
            maxValG = max(maxValG, input[idx + 1]);
            maxValB = max(maxValB, input[idx + 2]);
        }
    }

    int idx = (y * width + x) * 3;
    output[idx] = maxValR;
    output[idx + 1] = maxValG;
    output[idx + 2] = maxValB;
}

int main(int argc, char** argv) {
    if (argc != 2) {
        printf("usage: DisplayImage.out <Image_Path>\n");
        return -1;
    }

    Mat image = imread(argv[1], IMREAD_COLOR);
    if (image.empty()) {
        printf("Could not open or find the image\n");
        return -1;
    }

    int height = image.rows;
    int width = image.cols;
    size_t sizeInBytes = image.total() * image.elemSize();
    int sizeMat = static_cast<int>(sizeInBytes);

    printf("Image size: %i bytes\n", sizeMat);
    printf("Image dimensions: %i x %i\n", height, width);

    uchar *a = image.ptr<uchar>(0);
    uchar *d_a = NULL; // device copy of input image
    uchar *d_b = NULL; // device copy of output image
    uchar *b = (uchar*)malloc(sizeMat); // host copy of output image

    // Alloc space for device copies of a and b
    cudaMalloc((void**)&d_a, sizeMat);
    cudaMalloc((void**)&d_b, sizeMat);

    // Copy input image to device
    cudaMemcpy(d_a, a, sizeMat, cudaMemcpyHostToDevice);

    // Define block and grid sizes
    dim3 blockDim(16, 16);
    dim3 gridDim((width + blockDim.x - 1) / blockDim.x, (height + blockDim.y - 1) / blockDim.y);

    // Launch max_filter_5x5 kernel
    max_filter_5x5<<<gridDim, blockDim>>>(d_a, d_b, width, height);

    // Copy result back to host
    cudaMemcpy(b, d_b, sizeMat, cudaMemcpyDeviceToHost);

    printf("Data copied back from device\n");

    // Save the result image
    Mat image_out = Mat(height, width, CV_8UC3, b);
    imwrite("new.jpg", image_out);

    // Clean up
    cudaFree(d_a);
    cudaFree(d_b);
    free(b);

    return 0;
}
