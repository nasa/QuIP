#include "quip_config.h"
#ifdef HAVE_CUDA
#define BUILD_FOR_CUDA
#endif // HAVE_CUDA

#include "cuda_supp.h"
#include "my_cu2.h"
#include "quip_prot.h"
#include "veclib/cu2_veclib_prot.h"
#include "platform.h"


#ifdef HAVE_CUDA
//CUFFT
//static const char* getCUFFTError(cufftResult_t status)
const char* getCUFFTError(cufftResult status)
{
	switch (status) {
		case CUFFT_SUCCESS:
			return "Success";
		case CUFFT_NOT_SUPPORTED:
			return "CuFFT not supported";
		case CUFFT_INVALID_PLAN:
			return "Invalid Plan";
		case CUFFT_ALLOC_FAILED:
			return "Allocation Failed";
		case CUFFT_INVALID_TYPE:
			return "Invalid Type";
		case CUFFT_INVALID_VALUE:
			return "Invalid Error";
		case CUFFT_INTERNAL_ERROR:
			return "Internal Error";
		case CUFFT_EXEC_FAILED:
			return "Execution Failed";
		case CUFFT_SETUP_FAILED:
			return "Setup Failed";
		case CUFFT_INVALID_SIZE:
			return "Invalid Size";
		case CUFFT_UNALIGNED_DATA:
			return "Unaligned data";
		// these were added later on iMac - present in older versions?
		// BUG - find correct version number...
#if CUDA_VERSION >= 5050
		case CUFFT_INCOMPLETE_PARAMETER_LIST:
			return "Incomplete parameter list";
		case CUFFT_INVALID_DEVICE:
			return "Invalid device";
		case CUFFT_PARSE_ERROR:
			return "Parse error";
		case CUFFT_NO_WORKSPACE:
			return "No workspace";
#endif
#if CUDA_VERSION >= 6050
		case CUFFT_NOT_IMPLEMENTED:
			return "Not implemented";
		case CUFFT_LICENSE_ERROR:
			return "License error";
#endif
	}
	sprintf(DEFAULT_ERROR_STRING,"Unexpected CUFFT return value:  %d",status);
	return(DEFAULT_ERROR_STRING);
}
#endif // HAVE_CUDA
// What precision is this?  assume single preciscion?
// This doesn't seem to do anything yet!?

void g_cu2_vfft(QSP_ARG_DECL  Data_Obj *dst_dp, Data_Obj *src1_dp)
{
#ifdef HAVE_CUDA
	//Variable declarations
	int NX = 256;
	//int BATCH = 10;
	int BATCH = 1;
	enum cufftResult_t status;

	//Declare plan for FFT
	cufftHandle plan;
	//cufftComplex *data;
	//cufftComplex *result;
	void *data;
	void *result;
	cudaError_t drv_err;

	//Allocate RAM
	//cutilSafeCall(cudaMalloc(&data, sizeof(cufftComplex)*NX*BATCH));	
	//cutilSafeCall(cudaMalloc(&result, sizeof(cufftComplex)*NX*BATCH));
	drv_err = cudaMalloc(&data, sizeof(cufftComplex)*NX*BATCH);
	if( drv_err != cudaSuccess ){
		WARN("error allocating cuda data buffer for fft!?");
		return;
	}
	drv_err = cudaMalloc(&result, sizeof(cufftComplex)*NX*BATCH);
	if( drv_err != cudaSuccess ){
		WARN("error allocating cuda result buffer for fft!?");
		// BUG clean up previous malloc...
		return;
	}

	//Create plan for FFT
	status = cufftPlan1d(&plan, NX, CUFFT_C2C, BATCH);
	if (status != CUFFT_SUCCESS) {
		sprintf(ERROR_STRING, "Error in cufftPlan1d: %s\n", getCUFFTError(status));
		NWARN(ERROR_STRING);
	}

	//Run forward fft on data
	status = cufftExecC2C(plan, (cufftComplex *)data,
			(cufftComplex *)result, CUFFT_FORWARD);
	if (status != CUFFT_SUCCESS) {
		sprintf(ERROR_STRING, "Error in cufftExecC2C: %s\n", getCUFFTError(status));
		NWARN(ERROR_STRING);
	}

	//Run inverse fft on data
	/*status = cufftExecC2C(plan, data, result, CUFFT_INVERSE);
	if (status != CUFFT_SUCCESS)
	{
		sprintf(ERROR_STRING, "Error in cufftExecC2C: %s\n", getCUFFTError(status));
		NWARN(ERROR_STRING);
	}*/

	//Free resources
	cufftDestroy(plan);
	cudaFree(data);
#else // ! HAVE_CUDA
	NO_CUDA_MSG(g_fwdfft)
#endif // ! HAVE_CUDA
}

#ifdef FOOBAR
void h_cu2_fft2d( FFT_FUNC_ARG_DECLS ) { NERROR1("Sorry, h_cu2_fft2d not implemented!?"); }
void h_cu2_ift2d( FFT_FUNC_ARG_DECLS ) { NERROR1("Sorry, h_cu2_ift2d not implemented!?"); }
void h_cu2_fftrows( FFT_FUNC_ARG_DECLS ) { NERROR1("Sorry, h_cu2_fftrows not implemented!?"); }
void h_cu2_iftrows( FFT_FUNC_ARG_DECLS ) { NERROR1("Sorry, h_cu2_iftrows not implemented!?"); }
#endif // FOOBAR



