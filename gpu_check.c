// code used in configure.ac to test for presence of a GPU
//
// Note:  configure.ac does not include this file, rather the code
// here is duplicated within configure.ac.  This copy allows us to
// compile and test the code separately...
#include <stdio.h>
#include <cuda_runtime.h>
int main() {
    int deviceCount, device;
    int gpuDeviceCount = 0;
    struct cudaDeviceProp properties;
    cudaError_t cudaResultCode = cudaGetDeviceCount(&deviceCount);
    if (cudaResultCode != cudaSuccess)
	deviceCount = 0;
    /* machines with no GPUs can still report one emulation device */
    for (device = 0; device < deviceCount; ++device) {
	cudaGetDeviceProperties(&properties, device);
	if (properties.major != 9999){ /* 9999 means emulation only */
	    ++gpuDeviceCount;
	    printf("%d.%d\n",
		properties.major,properties.minor);
	    return 0;	// just print out for first device
	}
    }
    printf("0.0\n");

    /* dont just return the number of gpus, because other runtime cuda
       errors can also yield non-zero return values */
    if (gpuDeviceCount > 0)
	return 0; /* success */
    else
	return 1; /* failure */
}

