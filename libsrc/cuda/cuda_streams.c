
#include "quip_config.h"


#ifdef HAVE_CUDA
#define BUILD_FOR_CUDA
#endif // HAVE_CUDA

#include "quip_prot.h"
#include "my_cuda.h"
#include "cuda_supp.h"

typedef struct my_cuda_stream {
	Item		cs_item;
#ifdef HAVE_CUDA
	cudaStream_t	cs_stream;
#endif // HAVE_CUDA
} My_Cuda_Stream;

#define NO_CUDA_STREAM ((My_Cuda_Stream *)NULL)
#define PICK_CUDA_STREAM(s)	pick_cuda_stream(QSP_ARG s)


//ITEM_INTERFACE_DECLARATIONS_STATIC( My_Cuda_Stream , cuda_stream )
static Item_Type *cuda_stream_itp=NULL;
static ITEM_INIT_FUNC(My_Cuda_Stream,cuda_stream,0)
//static ITEM_CHECK_FUNC(My_Cuda_Stream,cuda_stream)
static ITEM_NEW_FUNC(My_Cuda_Stream,cuda_stream)
static ITEM_PICK_FUNC(My_Cuda_Stream,cuda_stream)
static ITEM_LIST_FUNC(My_Cuda_Stream,cuda_stream)
static ITEM_GET_FUNC(My_Cuda_Stream,cuda_stream)

#define GET_CUDA_STREAM(p)	get_cuda_stream(QSP_ARG  p)

static int have_first_cuda_stream=0;

static void init_first_cuda_stream(SINGLE_QSP_ARG_DECL)
{
	My_Cuda_Stream *csp;

	csp=new_cuda_stream(QSP_ARG  "default");

#ifdef CAUTIOUS
	if( csp == NO_CUDA_STREAM ) ERROR1("CAUTIOUS:  failed to create default stream object");
#endif /* CAUTIOUS */

#ifdef HAVE_CUDA
	csp->cs_stream = (cudaStream_t) 0;
#endif // HAVE_CUDA
	have_first_cuda_stream=1;
}

COMMAND_FUNC( do_new_stream )
{
	const char *s;
	My_Cuda_Stream *csp;
#ifdef HAVE_CUDA
	cudaError_t e;
#endif // HAVE_CUDA

	if( ! have_first_cuda_stream ) init_first_cuda_stream(SINGLE_QSP_ARG);

	s=NAMEOF("name for stream");
	csp = new_cuda_stream(QSP_ARG  s);
	if( csp == NO_CUDA_STREAM ) return;

#ifdef HAVE_CUDA
	e = cudaStreamCreate(&csp->cs_stream);
	if( e != cudaSuccess ){
		describe_cuda_driver_error2("do_new_stream","cudaStreamCreate",e);
		WARN("Error creating cuda stream");
	}
#else // ! HAVE_CUDA
	NO_CUDA_MSG(do_new_stream)
#endif // ! HAVE_CUDA
}

COMMAND_FUNC( do_list_cuda_streams )
{
	list_cuda_streams(QSP_ARG  tell_msgfile(SINGLE_QSP_ARG));
}

COMMAND_FUNC( do_cuda_stream_info )
{
	My_Cuda_Stream *csp;
	
	csp = GET_CUDA_STREAM("");
	if( csp == NO_CUDA_STREAM ) return;

	prt_msg("No information...");
}

COMMAND_FUNC( do_sync_stream )
{
	My_Cuda_Stream *csp;
#ifdef HAVE_CUDA
	cudaError_t e;
#endif // HAVE_CUDA

	if( ! have_first_cuda_stream ) init_first_cuda_stream(SINGLE_QSP_ARG);

	csp = PICK_CUDA_STREAM("stream");
	if( csp == NO_CUDA_STREAM ) return;

#ifdef HAVE_CUDA
	e = cudaStreamSynchronize(csp->cs_stream);
	if( e != cudaSuccess ){
		describe_cuda_driver_error2("do_sync_stream","cudaStreamSynchronize",e);
		WARN("Synchronization failure.");
	}
#else // ! HAVE_CUDA
	NO_CUDA_MSG(do_sync_stream)
#endif  // ! HAVE_CUDA
}






//#endif /* HAVE_CUDA */

