/* This program is incorporated into configure.ac to determine
 * the compute capability of the cuda device when configure
 * is run.  This file is provided to test and debug the program
 * and the cuda installation.
 */

/* Compile with:
gcc -I /usr/local/cuda/include -L/usr/local/cuda/lib comp_cap.c -lcudart
OR
gcc -I /usr/local/cuda/include -L/usr/local/cuda/lib64 comp_cap.c -lcudart
*/

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
//fprintf(stderr,"deviceCount = %d\n",deviceCount);
    for (device = 0; device < deviceCount; ++device) {
	cudaGetDeviceProperties(&properties, device);
	if (properties.major != 9999){ /* 9999 means emulation only */
	    ++gpuDeviceCount;
	    printf("%d.%d\n",
		properties.major,properties.minor);
	    return 0;	// just print out for first device
	}
    }
    printf("0.01\n");

    /* dont just return the number of gpus, because other runtime cuda
       errors can also yield non-zero return values */
    if (gpuDeviceCount > 0)
	return 0; /* success */
    else
	return 1; /* failure */
}

