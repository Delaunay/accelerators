#if 0
__global__ void device_print(float *tensor, int n, int c, int width, int height)
{
    int size = n * c * width * height;
    for (int i = 0; i < size; ++i) {
        printf("%.f ", tensor[i]);
    }
}
#endif
