#include <miopen/miopen.h>
#include <cstdio>

#define hipDeviceMalloc hipMalloc
#define hipDeviceFree hipFree

template<typename EC>
void check(EC error, const char* file, const char* fun, int line){

}

template<>
void check(hipError_t error, const char* file, const char* fun, int line){
        if (error != hipSuccess){
                printf("[!] %s/%s:%d %s %s\n",
                        file,
                        fun,
                        line,
                        hipGetErrorName(error),
                        hipGetErrorString(error)
                );
        } else {
		printf("[s] %s/%s:%d\n", file, fun, line);
	};
}

template<>
void check(miopenStatus_t error, const char* file, const char* fun, int line){
	if (error != miopenStatusSuccess){
		printf("[!] %s/%s:%d %d\n",
			file,
			fun,
			line,
			error
		);
	} else{
		printf("[s] %s/%s:%d\n", file, fun, line);
	}
}

//template<>
//void check(int i){
//}



#define CHK(X) check(X, __FILE__, __func__, __LINE__)

int main(){
	// Runtime
	int device_count = 0;
	CHK(hipGetDeviceCount(&device_count));
	
	printf("Device Count: %d\n", device_count);
	auto hip_status = hipSetDevice(0);
	
	int w = 8;
	int h = 8;

	// Allocate memory for our tensors
	float* tensor_hx = nullptr;
	float* tensor_dx = nullptr;
	size_t size_x = 32 * 3 * w * h * sizeof(float);
	CHK(hipHostMalloc  (&tensor_hx, size_x));
	CHK(hipDeviceMalloc(&tensor_dx, size_x));	

	float* kernel_hk = nullptr;
	float* kernel_dk = nullptr;
	size_t size_k = 3 * 3 * 3 * sizeof(float);
	CHK(hipHostMalloc  (&kernel_hk, size_k));
	CHK(hipDeviceMalloc(&kernel_dk, size_k));

	float* output_ho = nullptr;
	float* output_do = nullptr;
	size_t size_o = 0;
	//hip_status = hipMallocDevice(&output_o, size_o);

	// Create a MIOpen Handle
	miopenHandle_t handle;
	auto mi_status = miopenCreate(&handle);

	miopenTensorDescriptor_t desc_x;
	mi_status = miopenCreateTensorDescriptor(&desc_x);
	mi_status = miopenSet4dTensorDescriptor(desc_x, miopenFloat, 32, 3, w, h);

	miopenTensorDescriptor_t desc_k;
	mi_status = miopenCreateTensorDescriptor(&desc_k);
	// Output channel becomes the batch_size        
	int out_channel = 32;
	mi_status = miopenSet4dTensorDescriptor(desc_k, miopenFloat, out_channel, 3, 3, 3);

	// Copy Host to Device
	hip_status = hipMemcpyHtoD(tensor_dx, tensor_hx, size_x);
	
	// >> Play Ground
	// >>>>>>>>>>>>>>
	
	// Set tensor value to 1
	float value = 1;
	CHK(miopenSetTensor(handle, desc_x, tensor_dx, &value));	

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
	miopenConvAlgoPerf_t perf;
	CHK(miopenFindConvolutionForwardAlgorithm(
		handle,
		desc_x, tensor_dx,
		desc_k, kernel_dk,
		convDesc,
		desc_o, output_do,
		1,
		&algo_count,
		&perf,
		&workspace,
		workspace_size,
		false
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
	
	mi_status = miopenDestroyTensorDescriptor(desc_x);
	mi_status = miopenDestroyTensorDescriptor(desc_o);
	mi_status = miopenDestroyTensorDescriptor(desc_k);

	mi_status = miopenDestroy(handle);
	
	hip_status = hipHostFree(tensor_hx);
	hip_status = hipHostFree(kernel_hk);
	hip_status = hipHostFree(output_ho);
	
	hip_status = hipDeviceFree(tensor_dx);
	hip_status = hipDeviceFree(kernel_dk);
	hip_status = hipDeviceFree(output_do);
	return 0;
}
