
#include "quip_config.h"

char VersionId_cuda_cuda_error[] = QUIP_VERSION_STRING;

#ifdef HAVE_CUDA

#include "my_cuda.h"
#include "cuda_supp.h"

#define CUDA_RUNTIME_ERROR( e , s )					\
									\
case e:									\
	sprintf(DEFAULT_ERROR_STRING, "%s:  CUDA runtime error:  %s",whence,s);	\
	NWARN(DEFAULT_ERROR_STRING);						\
	break;

#define CUDA_DRIVER_ERROR( e, s ) \
case cudaErrorApiFailureBase+e:						\
	sprintf(DEFAULT_ERROR_STRING, "%s:  CUDA driver error:  %s",whence,s);	\
	NWARN(DEFAULT_ERROR_STRING);						\
	break;



void describe_cuda_error2(const char *whence, const char *msg,
							cudaError_t e)
{
	char str[LLEN];

	sprintf(str,"%s:  %s",whence,msg);
	describe_cuda_error(str,e);
}

void describe_cuda_error(const char *whence, cudaError_t e)
{
	cudaError_t e2;

	switch(e){
		case cudaSuccess:
			sprintf(DEFAULT_ERROR_STRING,"%s:  No errors.",whence);
			advise(DEFAULT_ERROR_STRING);
			break;
		CUDA_RUNTIME_ERROR( cudaErrorMissingConfiguration ,
					"Missing configuration error." )
		CUDA_RUNTIME_ERROR( cudaErrorMemoryAllocation,
					"Memory allocation error." )
		CUDA_RUNTIME_ERROR( cudaErrorInitializationError ,
					"Initialization error." )
		CUDA_RUNTIME_ERROR( cudaErrorLaunchFailure ,
					"Launch failure." )
		CUDA_RUNTIME_ERROR( cudaErrorPriorLaunchFailure ,
					"Prior launch failure." )
		CUDA_RUNTIME_ERROR( cudaErrorLaunchTimeout ,
					"Launch timeout error." )
		CUDA_RUNTIME_ERROR( cudaErrorLaunchOutOfResources ,
					"Launch out of resources error." )
		CUDA_RUNTIME_ERROR( cudaErrorInvalidDeviceFunction ,
					"Invalid device function." )
		CUDA_RUNTIME_ERROR( cudaErrorInvalidConfiguration ,
					"Invalid configuration." )
		CUDA_RUNTIME_ERROR( cudaErrorInvalidDevice ,
					"Invalid device." )
		CUDA_RUNTIME_ERROR( cudaErrorInvalidValue ,
					"Invalid value." )
		CUDA_RUNTIME_ERROR( cudaErrorInvalidPitchValue ,
					"Invalid pitch value." )
		CUDA_RUNTIME_ERROR( cudaErrorInvalidSymbol ,
					"Invalid symbol." )
		CUDA_RUNTIME_ERROR( cudaErrorMapBufferObjectFailed ,
					"Map buffer object failed." )
		CUDA_RUNTIME_ERROR( cudaErrorUnmapBufferObjectFailed ,
					"Unmap buffer object failed." )
		CUDA_RUNTIME_ERROR( cudaErrorInvalidHostPointer ,
					"Invalid host pointer." )
		CUDA_RUNTIME_ERROR( cudaErrorInvalidDevicePointer ,
					"Invalid device pointer." )
		CUDA_RUNTIME_ERROR( cudaErrorInvalidTexture ,
					"Invalid texture." )
		CUDA_RUNTIME_ERROR( cudaErrorInvalidTextureBinding ,
					"Invalid texture binding." )
		CUDA_RUNTIME_ERROR( cudaErrorInvalidChannelDescriptor ,
					"Invalid channel descriptor." )
		CUDA_RUNTIME_ERROR( cudaErrorInvalidMemcpyDirection ,
					"Invalid memcpy direction." )
		CUDA_RUNTIME_ERROR( cudaErrorAddressOfConstant ,
					"Address of constant error." )
		CUDA_RUNTIME_ERROR( cudaErrorTextureFetchFailed ,
					"Texture fetch failed." )
		CUDA_RUNTIME_ERROR( cudaErrorTextureNotBound ,
					"Texture not bound error." )
		CUDA_RUNTIME_ERROR( cudaErrorSynchronizationError ,
					"Synchronization error." )
		CUDA_RUNTIME_ERROR( cudaErrorInvalidFilterSetting ,
					"Invalid filter setting." )
		CUDA_RUNTIME_ERROR( cudaErrorInvalidNormSetting ,
					"Invalid norm setting." )
		CUDA_RUNTIME_ERROR( cudaErrorMixedDeviceExecution ,
					"Mixed device execution." )
		CUDA_RUNTIME_ERROR( cudaErrorCudartUnloading ,
					"CUDA runtime unloading." )
		CUDA_RUNTIME_ERROR( cudaErrorUnknown ,
					"Unknown error condition." )
		CUDA_RUNTIME_ERROR( cudaErrorNotYetImplemented ,
					"Function not yet implemented." )
		CUDA_RUNTIME_ERROR( cudaErrorMemoryValueTooLarge ,
					"Memory value too large." )
		CUDA_RUNTIME_ERROR( cudaErrorInvalidResourceHandle ,
					"Invalid resource handle." )
		CUDA_RUNTIME_ERROR( cudaErrorNotReady ,
					"Not ready error." )
		CUDA_RUNTIME_ERROR( cudaErrorInsufficientDriver ,
					"CUDA runtime is newer than driver." )
		CUDA_RUNTIME_ERROR( cudaErrorSetOnActiveProcess ,
					"Set on active process error." )
		CUDA_RUNTIME_ERROR( cudaErrorNoDevice ,
					"No available CUDA device." )
		//CUDA_RUNTIME_ERROR( cudaErrorECCUncorrectable , "Uncorrectable ECC error detected." )
		CUDA_RUNTIME_ERROR( cudaErrorStartupFailure ,
					"Startup failure." )
		CUDA_DRIVER_ERROR( CUDA_ERROR_INVALID_VALUE ,
				"Invalid parameter value." )
		CUDA_DRIVER_ERROR( CUDA_ERROR_OUT_OF_MEMORY ,
				"out of memory." )
		CUDA_DRIVER_ERROR( CUDA_ERROR_NOT_INITIALIZED ,
				"not initialized." )
		CUDA_DRIVER_ERROR( CUDA_ERROR_DEINITIALIZED ,
				"de-initialized." )
		CUDA_DRIVER_ERROR( CUDA_ERROR_PROFILER_DISABLED ,
				"profiler is disabled." )
		CUDA_DRIVER_ERROR( CUDA_ERROR_PROFILER_NOT_INITIALIZED ,
				"profiler not initialized." )
		CUDA_DRIVER_ERROR( CUDA_ERROR_PROFILER_ALREADY_STARTED ,
				"profiler already started." )
		CUDA_DRIVER_ERROR( CUDA_ERROR_PROFILER_ALREADY_STOPPED ,
				"profiler already stopped." )
		CUDA_DRIVER_ERROR( CUDA_ERROR_NO_DEVICE ,
				"no cuda-capable devices." )
		CUDA_DRIVER_ERROR( CUDA_ERROR_INVALID_DEVICE ,
				"invalid device." )
		CUDA_DRIVER_ERROR( CUDA_ERROR_INVALID_IMAGE ,
				"invalid kernal image/module." )
		CUDA_DRIVER_ERROR( CUDA_ERROR_INVALID_CONTEXT ,
				"invalid context." )
		CUDA_DRIVER_ERROR( CUDA_ERROR_CONTEXT_ALREADY_CURRENT ,
				"context already current." )
		CUDA_DRIVER_ERROR( CUDA_ERROR_MAP_FAILED ,
				"mapping failure." )
		CUDA_DRIVER_ERROR( CUDA_ERROR_UNMAP_FAILED ,
				"unmapping failure." )
		CUDA_DRIVER_ERROR( CUDA_ERROR_ARRAY_IS_MAPPED ,
				"array is mapped and cannot be destroyed." )
		CUDA_DRIVER_ERROR( CUDA_ERROR_ALREADY_MAPPED ,
				"already mapped." )
		CUDA_DRIVER_ERROR( CUDA_ERROR_NO_BINARY_FOR_GPU ,
				"no binary for GPU." )
		CUDA_DRIVER_ERROR( CUDA_ERROR_ALREADY_ACQUIRED ,
				"resource already acquired." )
		CUDA_DRIVER_ERROR( CUDA_ERROR_NOT_MAPPED ,
				"resource not mapped." )
		CUDA_DRIVER_ERROR( CUDA_ERROR_NOT_MAPPED_AS_ARRAY ,
				"not mapped as array." )
		CUDA_DRIVER_ERROR( CUDA_ERROR_NOT_MAPPED_AS_POINTER ,
				"not mapped as pointer." )
		CUDA_DRIVER_ERROR( CUDA_ERROR_ECC_UNCORRECTABLE ,
				"uncorrectable ECC error." )
		CUDA_DRIVER_ERROR( CUDA_ERROR_UNSUPPORTED_LIMIT ,
				"unsupported limit." )
		CUDA_DRIVER_ERROR( CUDA_ERROR_CONTEXT_ALREADY_IN_USE ,
				"context already in use." )
		CUDA_DRIVER_ERROR( CUDA_ERROR_INVALID_SOURCE ,
				"invalide device kernel source." )
		CUDA_DRIVER_ERROR( CUDA_ERROR_FILE_NOT_FOUND ,
				"file not found." )
		CUDA_DRIVER_ERROR( CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND ,
				"shared object symbol not found." )
		CUDA_DRIVER_ERROR( CUDA_ERROR_SHARED_OBJECT_INIT_FAILED ,
				"shared object init failed." )
		CUDA_DRIVER_ERROR( CUDA_ERROR_OPERATING_SYSTEM ,
				"OS call failed." )
		CUDA_DRIVER_ERROR( CUDA_ERROR_INVALID_HANDLE ,
				"invalid handle." )
		CUDA_DRIVER_ERROR( CUDA_ERROR_NOT_FOUND ,
				"named symbol not found." )
		CUDA_DRIVER_ERROR( CUDA_ERROR_NOT_READY ,
				"async operation not completed (not an error)." )
		CUDA_DRIVER_ERROR( CUDA_ERROR_LAUNCH_FAILED ,
				"launch failed)." )
		CUDA_DRIVER_ERROR( CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES ,
				"launch out of resources)." )
		CUDA_DRIVER_ERROR( CUDA_ERROR_LAUNCH_TIMEOUT ,
				"launch timeout)." )
		CUDA_DRIVER_ERROR( CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING ,
				"incompatible texturing)." )
		CUDA_DRIVER_ERROR( CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED ,
				"peer access already enabled)." )
		CUDA_DRIVER_ERROR( CUDA_ERROR_PEER_ACCESS_NOT_ENABLED ,
				"peer access not enabled)." )
		CUDA_DRIVER_ERROR( CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE ,
				"primary context already initialized)." )
		CUDA_DRIVER_ERROR( CUDA_ERROR_CONTEXT_IS_DESTROYED ,
				"current context has been destroyed)." )
		CUDA_DRIVER_ERROR( CUDA_ERROR_UNKNOWN ,
				"unknown error)." )

		default:
			sprintf(DEFAULT_ERROR_STRING,
		"%s:  unrecognized cuda error code %d",whence,e);
			NWARN(DEFAULT_ERROR_STRING);
			break;
	}
	e2 = cudaGetLastError();		// clear error
#ifdef CAUTIOUS
	if( e2 != e ){
		NERROR1("CAUTIOUS:  describe_cuda_error:  errors do not match!?");
	}
#endif /* CAUTIOUS */
}

void xfer_cuda_flag(Data_Obj *dpto, Data_Obj *dpfr, uint32_t flagbit)
{
	if( dpfr->dt_flags & flagbit ){
		dpto->dt_flags |= flagbit;
	} else {
		dpto->dt_flags &= ~flagbit;
	}
}

#endif /* HAVE_CUDA */

