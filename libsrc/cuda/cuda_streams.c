
#include "quip_config.h"

char VersionId_cuda_cuda_streams[] = QUIP_VERSION_STRING;

#ifdef HAVE_CUDA

#include "my_cuda.h"
#include "cuda_supp.h"
#include "items.h"

typedef struct my_cuda_stream {
	Item		cs_item;
	cudaStream_t	cs_stream;
} My_Cuda_Stream;

#define NO_CUDA_STREAM ((My_Cuda_Stream *)NULL)
#define PICK_CUDA_STREAM(s)	pick_cuda_stream(QSP_ARG s)


ITEM_INTERFACE_DECLARATIONS( My_Cuda_Stream , cuda_stream )

static int cuda_streams_inited=0;

void init_cuda_streams(SINGLE_QSP_ARG_DECL)
{
	My_Cuda_Stream *csp;

	csp=new_cuda_stream(QSP_ARG  "default");

#ifdef CAUTIOUS
	if( csp == NO_CUDA_STREAM ) ERROR1("CAUTIOUS:  failed to create default stream object");
#endif /* CAUTIOUS */

	csp->cs_stream = (cudaStream_t) 0;
	cuda_streams_inited=1;
}

COMMAND_FUNC( do_new_stream )
{
	const char *s;
	My_Cuda_Stream *csp;
	cudaError_t e;

	if( ! cuda_streams_inited ) init_cuda_streams(SINGLE_QSP_ARG);

	s=NAMEOF("name for stream");
	csp = new_cuda_stream(QSP_ARG  s);
	if( csp == NO_CUDA_STREAM ) return;

	e = cudaStreamCreate(&csp->cs_stream);
	if( e != cudaSuccess ){
		describe_cuda_error2("do_new_stream","cudaStreamCreate",e);
		WARN("Error creating cuda stream");
	}
}

COMMAND_FUNC( do_sync_stream )
{
	My_Cuda_Stream *csp;
	cudaError_t e;

	if( ! cuda_streams_inited ) init_cuda_streams(SINGLE_QSP_ARG);

	csp = PICK_CUDA_STREAM("stream");
	if( csp == NO_CUDA_STREAM ) return;

	e = cudaStreamSynchronize(csp->cs_stream);
	if( e != cudaSuccess ){
		describe_cuda_error2("do_sync_stream","cudaStreamSynchronize",e);
		WARN("Synchronization failure.");
	}
}






#endif /* HAVE_CUDA */

