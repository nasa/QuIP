
#include "quip_config.h"

#include "my_cu2.h"
#include "quip_prot.h"

typedef struct my_cu2_stream {
	Item		os_item;
//	cudaStream_t	os_stream;
} My_Cuda_Stream;

#define NO_CUDA_STREAM ((My_Cuda_Stream *)NULL)
#define PICK_CUDA_STREAM(s)	pick_cu2_stream(QSP_ARG s)


//ITEM_INTERFACE_DECLARATIONS_STATIC( My_Cuda_Stream , cu2_stream )
static Item_Type *cu2_stream_itp=NULL;
static ITEM_INIT_FUNC(My_Cuda_Stream,cu2_stream)
//static ITEM_CHECK_FUNC(My_Cuda_Stream,cu2_stream)
static ITEM_NEW_FUNC(My_Cuda_Stream,cu2_stream)
static ITEM_PICK_FUNC(My_Cuda_Stream,cu2_stream)
static ITEM_LIST_FUNC(My_Cuda_Stream,cu2_stream)
static ITEM_GET_FUNC(My_Cuda_Stream,cu2_stream)

#define GET_CUDA_STREAM(p)	get_cu2_stream(QSP_ARG  p)

static int have_first_cu2_stream=0;

static void init_first_cu2_stream(SINGLE_QSP_ARG_DECL)
{
	My_Cuda_Stream *csp;

	csp=new_cu2_stream(QSP_ARG  "default");

#ifdef CAUTIOUS
	if( csp == NO_CUDA_STREAM ) ERROR1("CAUTIOUS:  failed to create default stream object");
#endif /* CAUTIOUS */

	//csp->cs_stream = (cudaStream_t) 0;
	have_first_cu2_stream=1;
}

static PF_COMMAND_FUNC( new_stream )
{
	const char *s;
	My_Cuda_Stream *csp;
	//cudaError_t e;

	if( ! have_first_cu2_stream ) init_first_cu2_stream(SINGLE_QSP_ARG);

	s=NAMEOF("name for stream");
	csp = new_cu2_stream(QSP_ARG  s);
	if( csp == NO_CUDA_STREAM ) return;

	/*
	e = cudaStreamCreate(&csp->cs_stream);
	if( e != cudaSuccess ){
		describe_cu2_driver_error2("new_stream","cudaStreamCreate",e);
		WARN("Error creating cuda stream");
	}
	*/
}

static PF_COMMAND_FUNC( list_streams )
{
	list_cu2_streams(SINGLE_QSP_ARG);
}

static PF_COMMAND_FUNC( stream_info )
{
	My_Cuda_Stream *csp;
	
	csp = GET_CUDA_STREAM("");
	if( csp == NO_CUDA_STREAM ) return;

	prt_msg("No information...");
}

static PF_COMMAND_FUNC( sync_stream )
{
	My_Cuda_Stream *csp;
	//cudaError_t e;

	if( ! have_first_cu2_stream ) init_first_cu2_stream(SINGLE_QSP_ARG);

	csp = PICK_CUDA_STREAM("stream");
	if( csp == NO_CUDA_STREAM ) return;

/*
	e = cudaStreamSynchronize(csp->cs_stream);
	if( e != cudaSuccess ){
		describe_cu2_driver_error2("sync_stream","cudaStreamSynchronize",e);
		WARN("Synchronization failure.");
	}
*/
}


