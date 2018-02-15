
#include "quip_config.h"

#ifdef HAVE_CUDA

#define BUILD_FOR_CUDA
#include "my_cuda.h"
#include "cuda_supp.h"
#include "quip_prot.h"

#define CUDA_RUNTIME_ERROR( e , s )					\
									\
case e:									\
	sprintf(DEFAULT_ERROR_STRING, "%s:  CUDA runtime error:  %s",whence,s);	\
	NWARN(DEFAULT_ERROR_STRING);						\
	break;

//#if CUDA_VERSION<6000

#define CUDA_DRIVER_ERROR( e, s ) \
case cudaErrorApiFailureBase+e:						\
	sprintf(DEFAULT_ERROR_STRING, "%s:  CUDA driver error:  %s",whence,s);	\
	NWARN(DEFAULT_ERROR_STRING);						\
	break;

/*#else // CUDA_VERSION >= 6000
//
//#define CUDA_DRIVER_ERROR( e, s ) \
//case e:						\
//	sprintf(DEFAULT_ERROR_STRING, "%s:  CUDA driver error:  %s",whence,s);	\
//	NWARN(DEFAULT_ERROR_STRING);						\
//	break;
//
//#endif // CUDA_VERSION >= 6000
*/


// cudaError_t not good in 6.0?
void describe_cuda_error2(const char *whence, const char *msg,
							/*cudaError_t*/
							CUresult e)
{
	char str[LLEN];

	sprintf(str,"%s:  %s",whence,msg);
	describe_cuda_error(str,e);
}

void describe_cuda_driver_error2(const char *whence, const char *msg,
							cudaError_t e)
{
	char str[LLEN];

	sprintf(str,"%s:  %s",whence,msg);
	describe_cuda_driver_error(str,e);
}

#define RUNTIME_ERROR_CASE(code,msg)					\
									\
	case code:							\
		sprintf(DEFAULT_ERROR_STRING,"%s:  %s.",whence,msg);	\
		NWARN(DEFAULT_ERROR_STRING);				\
		break;

void describe_cuda_error(const char *whence, CUresult e)
{
//	CUresult e2;

	switch(e){
		case CUDA_SUCCESS:
			sprintf(DEFAULT_ERROR_STRING,"%s:  No errors.",whence);
			NADVISE(DEFAULT_ERROR_STRING);
			break;
#if CUDA_VERSION >= 6050
RUNTIME_ERROR_CASE(CUDA_ERROR_INVALID_GRAPHICS_CONTEXT,"Invalid graphics context")
#endif

#if CUDA_VERSION > 4000
RUNTIME_ERROR_CASE(CUDA_ERROR_PEER_ACCESS_UNSUPPORTED,"Peer access unsupported")
RUNTIME_ERROR_CASE(CUDA_ERROR_INVALID_PTX,"Invalid PTX")
RUNTIME_ERROR_CASE(CUDA_ERROR_ILLEGAL_ADDRESS,"Illegal address")
RUNTIME_ERROR_CASE(CUDA_ERROR_ASSERT,"Assertion error")
RUNTIME_ERROR_CASE(CUDA_ERROR_TOO_MANY_PEERS,"Too many peers")
RUNTIME_ERROR_CASE(CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED,"Host mem already registered")
RUNTIME_ERROR_CASE(CUDA_ERROR_HOST_MEMORY_NOT_REGISTERED,"Host mem not registered")
RUNTIME_ERROR_CASE(CUDA_ERROR_HARDWARE_STACK_ERROR,"H/W stack error")

RUNTIME_ERROR_CASE(CUDA_ERROR_ILLEGAL_INSTRUCTION,"Illegal instruction");
RUNTIME_ERROR_CASE(CUDA_ERROR_MISALIGNED_ADDRESS,"Misaligned address")
RUNTIME_ERROR_CASE(CUDA_ERROR_INVALID_ADDRESS_SPACE,"Invalid address space")
RUNTIME_ERROR_CASE(CUDA_ERROR_INVALID_PC,"Invalid PC")
RUNTIME_ERROR_CASE(CUDA_ERROR_NOT_PERMITTED,"Not permitted")
RUNTIME_ERROR_CASE(CUDA_ERROR_NOT_SUPPORTED,"Not supported")
#endif // CUDA_VERSION > 4000
RUNTIME_ERROR_CASE(CUDA_ERROR_LAUNCH_FAILED,"Launch failed")
RUNTIME_ERROR_CASE(CUDA_ERROR_UNKNOWN,"Unknown error")
RUNTIME_ERROR_CASE( CUDA_ERROR_INVALID_DEVICE , "Invalid device." )
RUNTIME_ERROR_CASE( CUDA_ERROR_NO_DEVICE , "No device" )
RUNTIME_ERROR_CASE( CUDA_ERROR_INVALID_VALUE , "Invalid value." )
RUNTIME_ERROR_CASE(CUDA_ERROR_INVALID_IMAGE,"Invalid Image")
RUNTIME_ERROR_CASE(CUDA_ERROR_INVALID_CONTEXT,"Invalid context")
#ifdef CUDA_ERROR_NVLINK_UNCORRECTABLE
RUNTIME_ERROR_CASE(CUDA_ERROR_NVLINK_UNCORRECTABLE,"uncorrectable NVLink error")
#endif // CUDA_ERROR_NVLINK_UNCORRECTABLE
//RUNTIME_ERROR_CASE( CUDA_ERROR_INVALID_PITCH_VALUE , "Invalid pitch value." )
//RUNTIME_ERROR_CASE( CUDA_ERROR_INVALID_SYMBOL , "Invalid symbol." )
//RUNTIME_ERROR_CASE( CUDA_ERROR_MAP_OBJECT_FAILED , "Map buffer object failed." )
//RUNTIME_ERROR_CASE( CUDA_ERROR_UNMAP_OBJECT_FAILED , "Unmap buffer object failed." )
//RUNTIME_ERROR_CASE( CUDA_ERROR_INVALID_HOST_POINTER , "Invalid host pointer." )
//RUNTIME_ERROR_CASE( CUDA_ERROR_INVALID_DEVICE_POINTER , "Invalid device pointer." )
//RUNTIME_ERROR_CASE( CUDA_ERROR_INVALID_TEXTURE , "Invalid texture." )
//RUNTIME_ERROR_CASE( CUDA_ERROR_INVALID_TEXTURE_BINDING , "Invalid texture binding." )
RUNTIME_ERROR_CASE( CUDA_ERROR_OUT_OF_MEMORY , "out of memory." )
RUNTIME_ERROR_CASE( CUDA_ERROR_NOT_INITIALIZED , "not initialized." )
RUNTIME_ERROR_CASE( CUDA_ERROR_DEINITIALIZED , "de-initialized." )
RUNTIME_ERROR_CASE( CUDA_ERROR_PROFILER_DISABLED , "profiler is disabled." )
RUNTIME_ERROR_CASE( CUDA_ERROR_PROFILER_NOT_INITIALIZED , "profiler not initialized." )
RUNTIME_ERROR_CASE( CUDA_ERROR_PROFILER_ALREADY_STARTED , "profiler already started." )
RUNTIME_ERROR_CASE( CUDA_ERROR_PROFILER_ALREADY_STOPPED , "profiler already stopped." )
RUNTIME_ERROR_CASE( CUDA_ERROR_CONTEXT_ALREADY_CURRENT , "context already current." )
RUNTIME_ERROR_CASE( CUDA_ERROR_MAP_FAILED , "mapping failure." )
RUNTIME_ERROR_CASE( CUDA_ERROR_UNMAP_FAILED , "unmapping failure." )
RUNTIME_ERROR_CASE( CUDA_ERROR_ARRAY_IS_MAPPED , "array is mapped and cannot be destroyed." )
RUNTIME_ERROR_CASE( CUDA_ERROR_ALREADY_MAPPED , "already mapped." )
RUNTIME_ERROR_CASE( CUDA_ERROR_NO_BINARY_FOR_GPU , "no binary for GPU." )
RUNTIME_ERROR_CASE( CUDA_ERROR_ALREADY_ACQUIRED , "resource already acquired." )
RUNTIME_ERROR_CASE( CUDA_ERROR_NOT_MAPPED , "resource not mapped." )
RUNTIME_ERROR_CASE( CUDA_ERROR_NOT_MAPPED_AS_ARRAY , "not mapped as array." )
RUNTIME_ERROR_CASE( CUDA_ERROR_NOT_MAPPED_AS_POINTER , "not mapped as pointer." )
RUNTIME_ERROR_CASE( CUDA_ERROR_ECC_UNCORRECTABLE , "uncorrectable ECC error." )
RUNTIME_ERROR_CASE( CUDA_ERROR_UNSUPPORTED_LIMIT , "unsupported limit." )
RUNTIME_ERROR_CASE( CUDA_ERROR_CONTEXT_ALREADY_IN_USE , "context already in use." )
RUNTIME_ERROR_CASE( CUDA_ERROR_INVALID_SOURCE , "invalide device kernel source." )
RUNTIME_ERROR_CASE( CUDA_ERROR_FILE_NOT_FOUND , "file not found." )
RUNTIME_ERROR_CASE( CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND , "shared object symbol not found." )
RUNTIME_ERROR_CASE( CUDA_ERROR_SHARED_OBJECT_INIT_FAILED , "shared object init failed." )
RUNTIME_ERROR_CASE( CUDA_ERROR_OPERATING_SYSTEM , "OS call failed." )
RUNTIME_ERROR_CASE( CUDA_ERROR_INVALID_HANDLE , "invalid handle." )
RUNTIME_ERROR_CASE( CUDA_ERROR_NOT_FOUND , "named symbol not found." )
RUNTIME_ERROR_CASE( CUDA_ERROR_NOT_READY , "async operation not completed (not an error)." )
RUNTIME_ERROR_CASE( CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES , "launch out of resources)." )
RUNTIME_ERROR_CASE( CUDA_ERROR_LAUNCH_TIMEOUT , "launch timeout)." )
RUNTIME_ERROR_CASE( CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING , "incompatible texturing)." )
RUNTIME_ERROR_CASE( CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED , "peer access already enabled)." )
RUNTIME_ERROR_CASE( CUDA_ERROR_PEER_ACCESS_NOT_ENABLED , "peer access not enabled)." )
RUNTIME_ERROR_CASE( CUDA_ERROR_CONTEXT_IS_DESTROYED , "current context has been destroyed)." )
RUNTIME_ERROR_CASE( CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE , "primary context already initialized)." )

#ifdef FOOBAR
		//CUDA_RUNTIME_ERROR( cudaErrorECCUncorrectable , "Uncorrectable ECC error detected." )
		CUDA_RUNTIME_ERROR( cudaErrorStartupFailure ,
					"Startup failure." )
#endif // FOOBAR
		default:
			sprintf(DEFAULT_ERROR_STRING,
		"%s:  unrecognized cuda error code %d",whence,e);
			NWARN(DEFAULT_ERROR_STRING);
			break;
	}
#ifdef FOOBAR
	e2 = cudaGetLastError();		// clear error
#ifdef CAUTIOUS
	if( e2 != e ){
		NERROR1("CAUTIOUS:  describe_cuda_error:  errors do not match!?");
	}
#endif /* CAUTIOUS */
#endif // FOOBAR
}

#define DRIVER_ERROR_CASE(code,msg)					\
									\
	case code:							\
		sprintf(DEFAULT_ERROR_STRING,"%s:  %s.",whence,msg);	\
		NADVISE(DEFAULT_ERROR_STRING);				\
		break;

void describe_cuda_driver_error(const char *whence, cudaError_t e)
{
	cudaError_t e2;

	switch(e){
DRIVER_ERROR_CASE(cudaSuccess,"No driver errors")
#if CUDA_VERSION >= 6050
DRIVER_ERROR_CASE(cudaErrorInvalidGraphicsContext,"Invalid graphics context")
DRIVER_ERROR_CASE( cudaErrorInvalidPtx , "Invalid PTX." )
#endif
DRIVER_ERROR_CASE( cudaErrorInvalidDevice , "Invalid device." )
DRIVER_ERROR_CASE( cudaErrorInvalidValue , "Invalid value." )
DRIVER_ERROR_CASE( cudaErrorInvalidPitchValue , "Invalid pitch value." )
DRIVER_ERROR_CASE( cudaErrorInvalidSymbol , "Invalid symbol." )
DRIVER_ERROR_CASE( cudaErrorMapBufferObjectFailed , "Map buffer object failed." )
DRIVER_ERROR_CASE( cudaErrorUnmapBufferObjectFailed , "Unmap buffer object failed." )
DRIVER_ERROR_CASE( cudaErrorInvalidHostPointer , "Invalid host pointer." )
DRIVER_ERROR_CASE( cudaErrorInvalidDevicePointer , "Invalid device pointer." )
DRIVER_ERROR_CASE( cudaErrorInvalidTexture , "Invalid texture." )
DRIVER_ERROR_CASE( cudaErrorInvalidTextureBinding , "Invalid texture binding." )
DRIVER_ERROR_CASE( cudaErrorInvalidChannelDescriptor , "Invalid channel descriptor." )
DRIVER_ERROR_CASE( cudaErrorInvalidMemcpyDirection , "Invalid memcpy direction." )
DRIVER_ERROR_CASE( cudaErrorAddressOfConstant , "Address of constant error." )
DRIVER_ERROR_CASE( cudaErrorTextureFetchFailed , "Texture fetch failed." )
DRIVER_ERROR_CASE( cudaErrorTextureNotBound , "Texture not bound error." )
DRIVER_ERROR_CASE( cudaErrorSynchronizationError , "Synchronization error." )
DRIVER_ERROR_CASE( cudaErrorInvalidResourceHandle , "Invalid resource handle." )
DRIVER_ERROR_CASE( cudaErrorNotReady , "Not ready error." )
DRIVER_ERROR_CASE( cudaErrorInsufficientDriver , "CUDA runtime is newer than driver." )
DRIVER_ERROR_CASE( cudaErrorSetOnActiveProcess , "Set on active process error." )
DRIVER_ERROR_CASE( cudaErrorNoDevice , "No available CUDA device." )
DRIVER_ERROR_CASE( cudaErrorMissingConfiguration , "Missing configuration error." )
DRIVER_ERROR_CASE( cudaErrorMemoryAllocation, "Memory allocation error." )
DRIVER_ERROR_CASE( cudaErrorInitializationError , "Initialization error." )
DRIVER_ERROR_CASE( cudaErrorLaunchFailure , "Launch failure." )
DRIVER_ERROR_CASE( cudaErrorPriorLaunchFailure , "Prior launch failure." )
DRIVER_ERROR_CASE( cudaErrorLaunchTimeout , "Launch timeout error." )
DRIVER_ERROR_CASE( cudaErrorLaunchOutOfResources , "Launch out of resources error." )
DRIVER_ERROR_CASE( cudaErrorInvalidDeviceFunction , "Invalid device function." )
DRIVER_ERROR_CASE( cudaErrorInvalidConfiguration , "Invalid configuration." )
DRIVER_ERROR_CASE( cudaErrorInvalidFilterSetting , "Invalid filter setting." )
DRIVER_ERROR_CASE( cudaErrorInvalidNormSetting , "Invalid norm setting." )
DRIVER_ERROR_CASE(cudaErrorMixedDeviceExecution,"Mixed device execution")
DRIVER_ERROR_CASE(cudaErrorCudartUnloading,"CUDA runtime unloading")
DRIVER_ERROR_CASE(cudaErrorUnknown,"Unknown error condition")
DRIVER_ERROR_CASE(cudaErrorNotYetImplemented,"Function not yet implemented")
DRIVER_ERROR_CASE(cudaErrorMemoryValueTooLarge,"Memory value too large")
DRIVER_ERROR_CASE(cudaErrorInvalidSurface,"Invalid surface")
DRIVER_ERROR_CASE(cudaErrorECCUncorrectable,"ECC uncorrectable")
DRIVER_ERROR_CASE(cudaErrorSharedObjectSymbolNotFound,"Shared object symbol not found")
DRIVER_ERROR_CASE(cudaErrorSharedObjectInitFailed,"Shared object init failed")

DRIVER_ERROR_CASE(cudaErrorUnsupportedLimit,"Unsupported limit")
DRIVER_ERROR_CASE(cudaErrorDuplicateVariableName,"Duplicate variable name")
DRIVER_ERROR_CASE(cudaErrorDuplicateTextureName,"Duplicate texture name")
DRIVER_ERROR_CASE(cudaErrorDuplicateSurfaceName,"Duplicate surface name")
DRIVER_ERROR_CASE(cudaErrorDevicesUnavailable,"Devices unavailable")
DRIVER_ERROR_CASE(cudaErrorInvalidKernelImage,"Invalid kernel image")
DRIVER_ERROR_CASE(cudaErrorNoKernelImageForDevice,"No kernel image for device")
DRIVER_ERROR_CASE(cudaErrorIncompatibleDriverContext,"Incompatible driver context")
DRIVER_ERROR_CASE(cudaErrorPeerAccessAlreadyEnabled,"Peer access already enabled")

DRIVER_ERROR_CASE(cudaErrorPeerAccessNotEnabled,"Peer access not enabled")
DRIVER_ERROR_CASE(cudaErrorDeviceAlreadyInUse,"Device already in use")
DRIVER_ERROR_CASE(cudaErrorProfilerDisabled,"Profiler disabled")
DRIVER_ERROR_CASE(cudaErrorProfilerNotInitialized,"Profiler not intialized")
DRIVER_ERROR_CASE(cudaErrorProfilerAlreadyStarted,"Profiler already started")
DRIVER_ERROR_CASE(cudaErrorProfilerAlreadyStopped,"Profiler already stopped")
#if CUDA_VERSION > 4000
DRIVER_ERROR_CASE(cudaErrorAssert,"Assertion error")
DRIVER_ERROR_CASE(cudaErrorTooManyPeers,"Too many peers")
DRIVER_ERROR_CASE(cudaErrorHostMemoryAlreadyRegistered,"Host mem already registered")

DRIVER_ERROR_CASE(cudaErrorHostMemoryNotRegistered,"Host memory not registered")
DRIVER_ERROR_CASE(cudaErrorOperatingSystem,"OS error")
DRIVER_ERROR_CASE(cudaErrorPeerAccessUnsupported,"Peer access unsupported")
DRIVER_ERROR_CASE(cudaErrorLaunchMaxDepthExceeded,"Launch max depth exceeded")
DRIVER_ERROR_CASE(cudaErrorLaunchFileScopedTex,"Launch file scoped tex")
DRIVER_ERROR_CASE(cudaErrorLaunchFileScopedSurf,"Launch file scoped surf")
DRIVER_ERROR_CASE(cudaErrorSyncDepthExceeded,"Sync depth exceeded")
DRIVER_ERROR_CASE(cudaErrorLaunchPendingCountExceeded,"Launch pending count exceeded")
DRIVER_ERROR_CASE(cudaErrorNotPermitted,"Not permitted")
DRIVER_ERROR_CASE(cudaErrorNotSupported,"Not supported")
DRIVER_ERROR_CASE(cudaErrorHardwareStackError,"H/W Stack Error")
DRIVER_ERROR_CASE(cudaErrorIllegalInstruction,"Illegal instruction")
DRIVER_ERROR_CASE(cudaErrorMisalignedAddress,"Mis-aligned address")
DRIVER_ERROR_CASE(cudaErrorInvalidAddressSpace,"Invalid address space")
DRIVER_ERROR_CASE(cudaErrorInvalidPc,"Invalid PC")
DRIVER_ERROR_CASE(cudaErrorIllegalAddress,"Illegal address")
#endif // CUDA_VERSION > 4000
DRIVER_ERROR_CASE(cudaErrorStartupFailure,"Startup failure")
DRIVER_ERROR_CASE(cudaErrorApiFailureBase,"Unexpected driver error")

#ifdef WHAT_CUDA_VERSION
		// need to fix for cuda 6?
		// not in cuda 6?
		CUDA_DRIVER_ERROR( CUDA_ERROR_LAUNCH_FAILED ,
				"launch failed)." )
		// not in cuda 6?
		CUDA_DRIVER_ERROR( CUDA_ERROR_UNKNOWN ,
				"unknown error)." )
#endif // WHAT_CUDA_VERSION

		default:
			sprintf(DEFAULT_ERROR_STRING,
		"%s:  unrecognized cuda error code %d",whence,e);
			NWARN(DEFAULT_ERROR_STRING);
			break;
	}
	e2 = cudaGetLastError();		// clear error
#ifdef CAUTIOUS
	if( e2 != e ){
		sprintf(DEFAULT_ERROR_STRING,
	"e = %d (0x%x), cudaGetLastError() = %d (0x%x)",e,e,e2,e2);
		NADVISE(DEFAULT_ERROR_STRING);
		NERROR1("CAUTIOUS:  describe_cuda_driver_error:  errors do not match!?");
	}
#endif /* CAUTIOUS */
}

#ifdef FOOBAR
// There is nothing about this that is cuda-specific - xfer to libdobj...
void xfer_cuda_flag(Data_Obj *dpto, Data_Obj *dpfr, uint32_t flagbit)
{
	if( OBJ_FLAGS(dpfr) & flagbit ){
		SET_OBJ_FLAG_BITS(dpto, flagbit);
	} else {
		CLEAR_OBJ_FLAG_BITS(dpto, flagbit);
	}
}
#endif // FOOBAR

#endif /* HAVE_CUDA */

