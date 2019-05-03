#ifndef ACC_ERROR_HEADER_H_
#define ACC_ERROR_HEADER_H_

#include <cstdio>

template <typename EC>
void check(EC error, const char *file, const char *fun, int line,
           const char *call_str) {}

#if defined(__HCC__)
template <>
void check(hipError_t error, const char *file, const char *fun, int line,
           const char *call_str) {
    if (error != hipSuccess) {
        printf("[!] %s/%s:%d %s %s %s\n", file, fun, line,
               hipGetErrorName(error), hipGetErrorString(error), call_str);
    } else {
        printf("[s] %s/%s:%d %s\n", file, fun, line, call_str);
    };
}

template <>
void check(miopenStatus_t error, const char *file, const char *fun, int line,
           const char *call_str) {
    if (error != miopenStatusSuccess) {
        printf("[!] %s/%s:%d %d %s\n", file, fun, line, error, call_str);
    } else {
        printf("[s] %s/%s:%d %s\n", file, fun, line, call_str);
    }
}
#elif defined(__CUDACC__)
template <>
void check(cudaError_t error, const char *file, const char *fun, int line,
           const char *call_str) {
    if (error != cudaSuccess) {
        printf("[!] %s/%s:%d %s %s %s\n", file, fun, line,
               cudaGetErrorName(error), cudaGetErrorString(error), call_str);
    } else {
        printf("[s] %s/%s:%d %s\n", file, fun, line, call_str);
    };
}

template <>
void check(cudnnStatus_t error, const char *file, const char *fun, int line,
           const char *call_str) {
    if (error != CUDNN_STATUS_SUCCESS) {
        printf("[!] %s/%s:%d %d %s\n", file, fun, line, error, call_str);
    } else {
        printf("[s] %s/%s:%d %s\n", file, fun, line, call_str);
    }
}
#endif

#define CHK(X) check(X, __FILE__, __func__, __LINE__, #X)
#endif

