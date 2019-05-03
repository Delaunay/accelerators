#define HIP_ENABLE_PRINTF
#include <hip/hip_runtime.h>

__global__ void device_print(float* tensor, int n, int c, int width, int height)

{
    int size = n * c * width * height;
    for(int i = 0; i < size; ++i){
        printf("%.f ", tensor[i]);
    }
}


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

#define hipDeviceMalloc hipMalloc
#define hipDeviceFree hipFree
#define CHK(X) check(X, __FILE__, __func__, __LINE__, #X)

int main(){


	int size_x = 32 * 3 * 224 * 224;
	float* tensor_dx = nullptr;
	CHK(hipDeviceMalloc(&tensor_dx, size_x));	

	CHK(hipLaunchKernelGGL(
		device_print, dim3(1), dim3(1), 0, 0, 
		tensor_dx, 32, 3, 224, 224
	));

	hipDeviceSynchronize();	

	CHK(hipDeviceFree(tensor_dx));
	return 0;
}
