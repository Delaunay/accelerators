#include <cstdio>
#include <cuda_runtime_api.h>
#include <cudnn.h>
#include <sstream>

#if __has_include("check.h")
#   include "check.h"
#else
#   define CHK(X)
#endif

#define cudnnConvAlgoPerf_t cudnnConvolutionFwdAlgoPerf_t
#define cudnnFloat CUDNN_DATA_FLOAT
#define cudnnConvolution CUDNN_CONVOLUTION

#define cudaMemcpyDtoH(x, y, c) cudaMemcpy(x, y, c, cudaMemcpyDeviceToHost)
#define cudaMemcpyHtoD(x, y, c) cudaMemcpy(x, y, c, cudaMemcpyHostToDevice)
#define cudaHostFree cudaFreeHost
#define cudaHostMalloc cudaMallocHost
#define cudaDeviceMalloc cudaMalloc
#define cudaDeviceFree cudaFree

#define cudnnGetConvolutionForwardOutputDim(conv, idesc, fdesc, n, c, h, w)\
      cudnnGetConvolution2dForwardOutputDim(conv, idesc, fdesc, n, c, h, w)

// CUDNN_CROSS_CORRELATION
#define CONV_MODE CUDNN_CONVOLUTION

#define cudnnSet4dTensorDescriptor(desc,                    dtype, n, c, h, w)\
        cudnnSetTensor4dDescriptor(desc, CUDNN_TENSOR_NCHW, dtype, n, c, h, w)

#define cudnnInitConvolutionDescriptor(conv, mode, padh, padw, sh, sw, dh, dw)\
       cudnnSetConvolution2dDescriptor(conv, padh, padw, sh, sw, dh, dw, mode, cudnnFloat)

#define cudnnConvolutionForwardGetWorkSpaceSize(h, w, x, conv, y, ws)\
        cudnnGetConvolutionForwardWorkspaceSize(h, x, w, conv, y, CUDNN_CONVOLUTION_FWD_ALGO_GEMM, ws)

#define cudnnFindConvolutionForwardAlgorithm(h, xd, x, wd, w, cd, yd, y, c, rc, perf, ws, wsize, ex)\
        cudnnFindConvolutionForwardAlgorithm(h, xd, wd, cd, yd, c, rc, perf)

#define cudnnConvolutionForward(h, a, xd, x, wd, w, cd, algo, b, yd, y, ws, wsize)\
        cudnnConvolutionForward(h, a, xd, x, wd, w, cd, algo, ws, wsize, b, yd, y)

#define cudnnGet4dTensorDescriptor cudnnGetTensor4dDescriptor


void print_tensor(float *tensor, cudnnTensorDescriptor_t desc, int batch_id);

struct Arguments {
    int w = 8;
    int h = 8;
    int in_c = 3;
    int kernel_w = 3;
    int out_channel = 32;
    int batch_size = 32;
};

Arguments parse_arguments(int argc, const char *argv[]) {
    Arguments args;

    for (int i = 0; i < argc; ++i) {
        auto str = std::string(argv[i]);
        std::stringstream ss;
        ss << argv[i + 1];

        if (str == "-w") {
            ss >> args.w;
            i += 1;
        } else if (str == "-h") {
            ss >> args.h;
            i += 1;
        } else if (str == "-c") {
            ss >> args.in_c;
            i += 1;
        } else if (str == "-k") {
            ss >> args.kernel_w;
            i += 1;
        } else if (str == "-b") {
            ss >> args.batch_size;
            i += 1;
        } else if (str == "-o") {
            ss >> args.out_channel;
            i += 1;
        }
    }

    return args;
}

int main(int argc, const char *argv[]) {
    Arguments args = parse_arguments(argc, argv);

    printf("w=%d h=%d\n", args.w, args.h);

    // Runtime
    int device_count = 0;
    CHK(cudaGetDeviceCount(&device_count));

    printf("Device Count: %d\n", device_count);
    CHK(cudaSetDevice(0));

    // Allocate memory for our tensors
    float *tensor_hx = nullptr;
    float *tensor_dx = nullptr;
    size_t size_x = args.batch_size * args.in_c * args.w * args.h * sizeof(float);

    CHK(cudaHostMalloc(&tensor_hx, size_x));
    CHK(cudaDeviceMalloc(&tensor_dx, size_x));

    float *kernel_hk = nullptr;
    float *kernel_dk = nullptr;
    size_t size_k = args.in_c * args.kernel_w * args.kernel_w * sizeof(float);

    CHK(cudaHostMalloc(&kernel_hk, size_k));
    CHK(cudaDeviceMalloc(&kernel_dk, size_k));

    // Create a MIOpen Handle
    cudnnHandle_t handle;
    CHK(cudnnCreate(&handle));

    cudnnTensorDescriptor_t desc_x;
    CHK(cudnnCreateTensorDescriptor(&desc_x));
    CHK(cudnnSet4dTensorDescriptor(desc_x, cudnnFloat,
                                    args.batch_size, // N
                                    args.in_c,       // C
                                    args.w,          // W
                                    args.h));        // H

    //cudnnTensorDescriptor_t desc_k;
    cudnnFilterDescriptor_t desc_k;
    CHK(cudnnCreateFilterDescriptor(&desc_k));
    // Output channel becomes the batch_size
    CHK(cudnnSetFilter4dDescriptor(desc_k, cudnnFloat,
                                   CUDNN_TENSOR_NCHW,
                                    args.out_channel, // OC
                                    args.in_c,        // IC
                                    args.kernel_w,    //  W
                                    args.kernel_w));  //  H

    // Copy Host to Device
    CHK(cudaMemcpyHtoD(tensor_dx, tensor_hx, size_x));

    // >> Play Ground
    // >>>>>>>>>>>>>>

    // Set tensor value to 1
    float value = 1;
    CHK(cudnnSetTensor(handle, desc_x, tensor_dx, &value));
    // CHK(cudnnSetFilter(handle, desc_k, kernel_dk, &value));

    // Make a convolution
    cudnnConvolutionDescriptor_t convDesc;
    CHK(cudnnCreateConvolutionDescriptor(&convDesc));
    CHK(cudnnInitConvolutionDescriptor(convDesc,
                                        cudnnConvolution, // mode
                                        0,                // pad_h
                                        0,                // pad_w
                                        1,                // stride_h
                                        1,                // stride_w
                                        1,                // dilatation_h
                                        1                 // dilatation_w
                                        ));

    int on = 0, oc = 0, oh = 0, ow = 0;
    // Make Output Tensor
    CHK(cudnnGetConvolutionForwardOutputDim(
            convDesc,
            desc_x,
            desc_k,
            &on, &oc, &oh, &ow));

    printf("Output: (n: %d, c: %d, h: %d, w: %d)\n", on, oc, oh, ow);

    float *output_ho = nullptr;
    float *output_do = nullptr;
    size_t size_o = on * oc * oh * ow * sizeof(float);
    CHK(cudaHostMalloc(&output_ho, size_o));
    CHK(cudaDeviceMalloc(&output_do, size_o));

    cudnnTensorDescriptor_t desc_o;
    CHK(cudnnCreateTensorDescriptor(&desc_o));
    CHK(cudnnSet4dTensorDescriptor(desc_o, cudnnFloat, on, oc, oh, ow));

    // Get Workspace size
    size_t workspace_size = 0;
    // cudnnGetConvolutionForwardWorkspaceSize
    CHK(cudnnConvolutionForwardGetWorkSpaceSize(
        handle,
        desc_k,
        desc_x,
        convDesc,
        desc_o,
        &workspace_size));

    printf("Workspace Size %zu\n", workspace_size);

    float *workspace = nullptr;
    CHK(cudaDeviceMalloc(&workspace, workspace_size));

    int algo_count = 1;
    cudnnConvAlgoPerf_t perf;
    CHK(cudnnFindConvolutionForwardAlgorithm(
        handle,
        desc_x, tensor_dx,
        desc_k, kernel_dk,
        convDesc,
        desc_o, output_do,
        1,
        &algo_count,
        &perf,
        &workspace, workspace_size, true));

    // Given the best convolution algo allocate workspace
    workspace_size = perf.memory;

    printf("Workspace Size %zu\n", workspace_size);
    CHK(cudaDeviceFree(workspace));
    CHK(cudaDeviceMalloc(&workspace, workspace_size));

    printf("Picked Algo %d\n", perf.algo);

    // Execute the convolution at last
    float alpha = 1;
    float beta = 0;

    CHK(cudnnConvolutionForward(
        handle, &alpha,
        desc_x, tensor_dx,
        desc_k, kernel_dk,
        convDesc,
        perf.algo, &beta,
        desc_o, output_do,
        workspace, workspace_size));

    // <<<<<<<<<<<<<
    // Copy Device back to host
    CHK(cudaMemcpyDtoH(tensor_hx, tensor_dx, size_x));
    CHK(cudaMemcpyDtoH(output_ho, output_do, size_o));

    print_tensor(tensor_hx, desc_x, 0);
    print_tensor(output_ho, desc_o, 0);

    // --------------------------
    CHK(cudnnDestroyTensorDescriptor(desc_x));
    CHK(cudnnDestroyTensorDescriptor(desc_o));
    CHK(cudnnDestroyFilterDescriptor(desc_k));

    CHK(cudnnDestroy(handle));

    CHK(cudaHostFree(tensor_hx));
    CHK(cudaHostFree(kernel_hk));
    CHK(cudaHostFree(output_ho));

    CHK(cudaDeviceFree(tensor_dx));
    CHK(cudaDeviceFree(kernel_dk));
    CHK(cudaDeviceFree(output_do));

    return 0;
}

void print_tensor(float *tensor, cudnnTensorDescriptor_t desc, int batch_id) {
    cudnnDataType_t dtype;
    int n, c, h, w, ns, cs, hs, ws;
    CHK(cudnnGet4dTensorDescriptor(desc, &dtype, &n, &c, &h, &w, &ns, &cs, &hs,
                                    &ws));

    printf("Tensor: Size(%d, %d, %d, %d) Strides(%d, %d, %d, %d)\n", n, c, h, w,
           ns, cs, hs, ws);

    size_t color_size = w * h;
    size_t img_size = color_size * c;

    for (int row = 0; row < h; row += 1) {
        for (int col = 0; col < w; col += 1) {
            printf("[");
            for (int channel = 0; channel < c; channel += 1) {
                size_t offset =
                    col * ws + row * hs + channel * cs + batch_id * ns;
                printf("%.f ", tensor[offset]);
            }
            printf("] ");
        }
        printf("\n");
    }
}
