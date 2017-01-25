
/*
 * Copyright 1993-2009 NVIDIA Corporation.  All rights reserved.
 *
 * NOTICE TO USER:
 *
 * This source code is subject to NVIDIA ownership rights under U.S. and
 * international Copyright laws.  Users and possessors of this source code
 * are hereby granted a nonexclusive, royalty-free license to use this code
 * in individual and commercial software.
 *
 * NVIDIA MAKES NO REPRESENTATION ABOUT THE SUITABILITY OF THIS SOURCE
 * CODE FOR ANY PURPOSE.  IT IS PROVIDED "AS IS" WITHOUT EXPRESS OR
 * IMPLIED WARRANTY OF ANY KIND.  NVIDIA DISCLAIMS ALL WARRANTIES WITH
 * REGARD TO THIS SOURCE CODE, INCLUDING ALL IMPLIED WARRANTIES OF
 * MERCHANTABILITY, NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
 * IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY SPECIAL, INDIRECT, INCIDENTAL,
 * OR CONSEQUENTIAL DAMAGES, OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS
 * OF USE, DATA OR PROFITS,  WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE
 * OR OTHER TORTIOUS ACTION,  ARISING OUT OF OR IN CONNECTION WITH THE USE
 * OR PERFORMANCE OF THIS SOURCE CODE.
 *
 * U.S. Government End Users.   This source code is a "commercial item" as
 * that term is defined at  48 C.F.R. 2.101 (OCT 1995), consisting  of
 * "commercial computer  software"  and "commercial computer software
 * documentation" as such terms are  used in 48 C.F.R. 12.212 (SEPT 1995)
 * and is provided to the U.S. Government only as a commercial end item.
 * Consistent with 48 C.F.R.12.212 and 48 C.F.R. 227.7202-1 through
 * 227.7202-4 (JUNE 1995), all U.S. Government End Users acquire the
 * source code with only those rights set forth herein.
 *
 * Any use of this source code in individual and commercial software must
 * include, in the user documentation and internal comments to the code,
 * the above Disclaimer and U.S. Government End Users Notice.
 */
/* This sample queries the properties of the CUDA devices present in the system via CUDA Runtime API. */

#include "quip_config.h"

// includes, system
#include <stdio.h>

#ifdef HAVE_STDLIB_H
#include <stdlib.h>
#endif

#ifdef HAVE_STRING_H
#include <string.h>
#endif

#ifdef HAVE_CUDA
#define BUILD_FOR_CUDA
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <curand.h>
#endif // HAVE_CUDA

#include "quip_prot.h"
#include "my_cuda.h"

#ifdef HAVE_CUDA


#ifdef USE_DLL_LINKING
void mapCudaFunctionPointers()
{
	cudaError_t cuda_detected_sts = cudaRuntimeDynload();

	if (cuda_detected_sts == cudaErrorUnknown) {
		printf("CUDA Runtime not available, exiting\n");
		exit(0);
	}
}
#endif // USE_DLL_LINKING

void print_cudev_properties(QSP_ARG_DECL  int dev, cudaDeviceProp *propp)
{
	sprintf(msg_str,"\nDevice %d: \"%s\"", dev, propp->name);
	prt_msg(msg_str);
	sprintf(msg_str,"  CUDA Capability Major revision number:		%d", propp->major);
	prt_msg(msg_str);
	sprintf(msg_str,"  CUDA Capability Minor revision number:		%d", propp->minor);
	prt_msg(msg_str);
	sprintf(msg_str,"  Total amount of global memory:\t\t\t%lu bytes", propp->totalGlobalMem);
	prt_msg(msg_str);
#if CUDART_VERSION >= 2000
	sprintf(msg_str,"  Number of multiprocessors:\t\t\t\t%d", propp->multiProcessorCount);
	prt_msg(msg_str);
	sprintf(msg_str,"  Number of cores:\t\t\t\t\t%d", 8 * propp->multiProcessorCount);
	prt_msg(msg_str);
#endif
	sprintf(msg_str,"  Total amount of constant memory:\t\t\t%lu bytes", propp->totalConstMem); 
	prt_msg(msg_str);
	sprintf(msg_str,"  Total amount of shared memory per block:\t\t%lu bytes", propp->sharedMemPerBlock);
	prt_msg(msg_str);
	sprintf(msg_str,"  Total number of registers available per block:\t%d", propp->regsPerBlock);
	prt_msg(msg_str);
	sprintf(msg_str,"  Warp size:\t\t\t\t\t\t%d", propp->warpSize);
	prt_msg(msg_str);
	sprintf(msg_str,"  Maximum number of threads per block:\t\t\t%d", propp->maxThreadsPerBlock);
	prt_msg(msg_str);
	sprintf(msg_str,"  Maximum sizes of each dimension of a block:\t\t%d x %d x %d",
		propp->maxThreadsDim[0],
		propp->maxThreadsDim[1],
		propp->maxThreadsDim[2]);
	prt_msg(msg_str);
	sprintf(msg_str,"  Maximum sizes of each dimension of a grid:\t\t%d x %d x %d",
		propp->maxGridSize[0],
		propp->maxGridSize[1],
		propp->maxGridSize[2]);
	prt_msg(msg_str);
	sprintf(msg_str,"  Maximum memory pitch:\t\t\t\t\t%lu bytes", propp->memPitch);
	prt_msg(msg_str);
	sprintf(msg_str,"  Texture alignment:\t\t\t\t\t%lu bytes", propp->textureAlignment);
	prt_msg(msg_str);
	sprintf(msg_str,"  Clock rate:\t\t\t\t\t\t%.2f GHz", propp->clockRate * 1e-6f);
	prt_msg(msg_str);
#if CUDART_VERSION >= 2000
	sprintf(msg_str,"  Concurrent copy and execution:\t\t\t%s", propp->deviceOverlap ? "Yes" : "No");
	prt_msg(msg_str);
#endif

	sprintf(msg_str,"  Run time limit on kernels:\t\t\t\t%s", propp->kernelExecTimeoutEnabled ? "Yes" : "No");
	prt_msg(msg_str);
	sprintf(msg_str,"  Integrated:\t\t\t\t\t\t%s", propp->integrated ? "Yes" : "No");
	prt_msg(msg_str);
	sprintf(msg_str,"  Support host page-locked memory mapping:\t\t%s", propp->canMapHostMemory ? "Yes" : "No");
	prt_msg(msg_str);
	sprintf(msg_str,"  Compute mode:\t\t\t\t\t\t%s",
		propp->computeMode == cudaComputeModeDefault ?
	"Default (multiple host threads can use this device simultaneously)" :
		propp->computeMode == cudaComputeModeExclusive ?
	"Exclusive (only one host thread at a time can use this device)" :
		propp->computeMode == cudaComputeModeProhibited ?
			"Prohibited (no host thread can use this device)" : "Unknown");
	prt_msg(msg_str);

	prt_msg("");		// a blank line at the end
}

extern "C" {

void query_cuda_device(QSP_ARG_DECL  int dev)
{
	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, dev);

	// This function call returns 9999 for both major & minor fields, if no CUDA capable devices are present
	if (deviceProp.major == 9999 && deviceProp.minor == 9999)
		printf("There is no CUDA device with dev = %d!?.\n",dev);

	print_cudev_properties(QSP_ARG  dev,&deviceProp);
}

}

COMMAND_FUNC( query_cuda_devices ){

#if defined(USE_DLL_LINKING)
	printf("CUDA Device Query (Runtime API) version (CUDART dynamic linking)\n");

	mapCudaFunctionPointers();
#else
	printf("CUDA Device Query (Runtime API) version (CUDART static linking)\n");
#endif

	int deviceCount;

	cudaGetDeviceCount(&deviceCount);

	// This function call returns 0 if there are no CUDA capable devices.
	if (deviceCount == 0)
		printf("There is no device supporting CUDA\n");

	else if (deviceCount == 1)
		printf("There is 1 device supporting CUDA\n");
	else
		printf("There are %d devices supporting CUDA\n", deviceCount);

	int dev;
	for (dev = 0; dev < deviceCount; ++dev) {
		if (dev == 0) {
		}
		query_cuda_device(QSP_ARG  dev);
	}
}

#else	// ! HAVE_CUDA

COMMAND_FUNC( query_cuda_devices )
{}

#endif	// ! HAVE_CUDA

