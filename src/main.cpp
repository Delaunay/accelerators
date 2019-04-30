#include <miopen/miopen.h>
#include <cstdio>
#include <sstream>

#define hipDeviceMalloc hipMalloc
#define hipDeviceFree hipFree

template<typename EC>
void check(EC error, const char* file, const char* fun, int line, const char* call_str){

}

template<>
void check(hipError_t error, const char* file, const char* fun, int line, const char* call_str){
        if (error != hipSuccess){
                printf("[!] %s/%s:%d %s %s %s\n",
                        file,
                        fun,
                        line,
                        hipGetErrorName(error),
                        hipGetErrorString(error),
			call_str
                );
        } else {
		printf("[s] %s/%s:%d %s\n", file, fun, line, call_str);
	};
}

template<>
void check(miopenStatus_t error, const char* file, const char* fun, int line, const char* call_str){
	if (error != miopenStatusSuccess){
		printf("[!] %s/%s:%d %d %s\n",
			file,
			fun,
			line,
			error,
			call_str
		);
	} else{
		printf("[s] %s/%s:%d %s\n", file, fun, line, call_str);
	}
}

#define CHK(X) check(X, __FILE__, __func__, __LINE__, #X)

void print_tensor(float* tensor, miopenTensorDescriptor_t desc, int batch_id);

int main(int argc, const char* argv[]){
	int w = 8;
	int h = 8;
	int in_c = 3;

	for(int i = 0; i < argc; ++i){
		auto str = std::string(argv[i]);
		std::stringstream ss;
		ss << argv[i + 1];

		if (str == "-w"){
			ss >> w;
			i += 1;
		} else if (str == "-h"){
			ss >> h;
			i += 1;
		} else if (str == "-c"){
			ss >> in_c;
			i += 1;
		} 
	}

	printf("w=%d h=%d\n", w, h);

	// Runtime
	int device_count = 0;
	CHK(hipGetDeviceCount(&device_count));
	
	printf("Device Count: %d\n", device_count);
	CHK(hipSetDevice(0));	

	// Allocate memory for our tensors
	float* tensor_hx = nullptr;
	float* tensor_dx = nullptr;
	size_t size_x = 32 * in_c * w * h * sizeof(float);
	CHK(hipHostMalloc  (&tensor_hx, size_x));
	CHK(hipDeviceMalloc(&tensor_dx, size_x));	

	float* kernel_hk = nullptr;
	float* kernel_dk = nullptr;
	size_t size_k = in_c * 3 * 3 * sizeof(float);
	CHK(hipHostMalloc  (&kernel_hk, size_k));
	CHK(hipDeviceMalloc(&kernel_dk, size_k));

	float* output_ho = nullptr;
	float* output_do = nullptr;
	size_t size_o = 0;
	//hip_status = hipMallocDevice(&output_o, size_o);

	// Create a MIOpen Handle
	miopenHandle_t handle;
	CHK(miopenCreate(&handle));

	miopenTensorDescriptor_t desc_x;
	CHK(miopenCreateTensorDescriptor(&desc_x));
	CHK(miopenSet4dTensorDescriptor(desc_x, miopenFloat, 32, in_c, w, h));

	miopenTensorDescriptor_t desc_k;
	CHK(miopenCreateTensorDescriptor(&desc_k));
	// Output channel becomes the batch_size        
	int out_channel = 4;
	CHK(miopenSet4dTensorDescriptor(desc_k, miopenFloat, out_channel, in_c, 3, 3));

	// Copy Host to Device
	CHK(hipMemcpyHtoD(tensor_dx, tensor_hx, size_x));
	
	// >> Play Ground
	// >>>>>>>>>>>>>>
	
	// Set tensor value to 1
	float value = 1;
	CHK(miopenSetTensor(handle, desc_x, tensor_dx, &value));	
	CHK(miopenSetTensor(handle, desc_k, kernel_dk, &value));

	// Make a convolution
	miopenConvolutionDescriptor_t convDesc;
	CHK(miopenCreateConvolutionDescriptor(&convDesc));
	CHK(miopenInitConvolutionDescriptor(
		convDesc,
		miopenConvolution, // mode
		0, // pad_h
		0, // pad_w
		1, // stride_h
		1, // stride_w
		1, // dilatation_h
		1  // dilatation_w
	));

        int on = 0, oc = 0, oh = 0, ow = 0;
	// Make Output Tensor
	CHK(miopenGetConvolutionForwardOutputDim(
		convDesc,
		desc_x,
		desc_k,
		&on, &oc, &oh, &ow
	)); 

	printf("Output: (n: %d, c: %d, h: %d, w: %d)\n", on, oc, oh, ow); 		
	
	size_o = on * oc * oh * ow * sizeof(float);
	CHK(hipHostMalloc  (&output_ho, size_o));
        CHK(hipDeviceMalloc(&output_do, size_o));

	miopenTensorDescriptor_t desc_o;
	CHK(miopenCreateTensorDescriptor(&desc_o));
	CHK(miopenSet4dTensorDescriptor(desc_o, miopenFloat, on, oc, oh, ow));

	// Get Workspace size
	size_t workspace_size = 0;
	CHK(miopenConvolutionForwardGetWorkSpaceSize(
		handle,
		desc_k,
		desc_x,
		convDesc,
		desc_o,
		&workspace_size
	));
	printf("Workspace Size %zu\n", workspace_size);

	float* workspace = nullptr;
	CHK(hipDeviceMalloc(&workspace, workspace_size));

	int algo_count = 0;
	miopenConvAlgoPerf_t perf[5];
	CHK(miopenFindConvolutionForwardAlgorithm(
		handle,
		desc_x, tensor_dx,
		desc_k, kernel_dk,
		convDesc,
		desc_o, output_do,
		1,
		&algo_count,
		perf,
		&workspace,
		workspace_size,
		true
	));
	// Given the best convolution algo allocate workspace
	workspace_size = perf[0].memory;

	printf("Workspace Size %zu\n", workspace_size);
	CHK(hipDeviceFree(workspace));
        CHK(hipDeviceMalloc(&workspace, workspace_size));

	printf("Picked Algo %d\n", perf[0].fwd_algo);
	
	// Execute the convolution at last
	float alpha = 1;
	float beta = 0;
	CHK(miopenConvolutionForward(
		handle,
		&alpha,
		desc_x, tensor_dx,
		desc_k, kernel_dk,
		convDesc,
		perf[0].fwd_algo,
		&beta,
		desc_o, output_do,
		workspace, workspace_size
	));	


	// <<<<<<<<<<<<<
	// Copy Device back to host
	CHK(hipMemcpyDtoH(tensor_hx, tensor_dx, size_x));
	
	int batch = 0;
	int img_size = w * h;

	for (int i = 0; i < w; ++i){
		for(int j = 0; j < h; ++j){
			printf("(%.2f, %.2f, %.2f) ",
				tensor_hx[i * w + j + 0 * img_size],
				tensor_hx[i * w + j + 1 * img_size],
				tensor_hx[i * w + j + 2 * img_size]
			);
		}
		printf("\n");
	}

	CHK(hipMemcpyDtoH(output_ho, output_do, size_o));

	print_tensor(output_ho, desc_o, 0);
	
	CHK(miopenDestroyTensorDescriptor(desc_x));
	CHK(miopenDestroyTensorDescriptor(desc_o));
	CHK(miopenDestroyTensorDescriptor(desc_k));

	CHK(miopenDestroy(handle));
	
	CHK(hipHostFree(tensor_hx));
	CHK(hipHostFree(kernel_hk));
	CHK(hipHostFree(output_ho));
	
	CHK(hipDeviceFree(tensor_dx));
	CHK(hipDeviceFree(kernel_dk));
	CHK(hipDeviceFree(output_do));

	return 0;
}


void print_tensor(float* tensor, miopenTensorDescriptor_t desc, int batch_id){
	miopenDataType_t dtype;
	int n, c, h, w, ns, cs, hs, ws;
	CHK(miopenGet4dTensorDescriptor(
		desc,
		&dtype,
		&n , &c , &h , &w,
		&ns, &cs, &hs, &ws
	));

	printf("Tensor: Size(%d, %d, %d, %d) Strides(%d, %d, %d, %d)\n", 
	       n, c, h, w, ns, cs, hs, ws);

	size_t color_size = w * h;
	size_t img_size = color_size * c;
	
	for (int row = 0; row < h; row += 1){
		for (int col = 0; col < w; col += 1){
			printf("[");
			for (int channel = 0; channel < c; channel += 1){
				size_t offset = col * ws + row * hs + channel * cs + batch_id * ns;
				printf("%.f ", tensor[offset]);
			}
			printf("] ");
		}
		printf("\n");
	}
}


