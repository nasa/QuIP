
#include "quip_config.h"

#ifdef HAVE_OPENCL

#include "quip_prot.h"
#include "my_ocl.h"
#include "platform.h"
#include "veclib/ocl_port.h"


ITEM_INTERFACE_DECLARATIONS( Platform_Stream , stream, 0 )

#ifdef NOT_USED
static int have_first_ocl_stream=0;

static Platform_Stream *PF_FUNC_NAME(new_stream)(QSP_ARG_DECL  const char *s)
{
	WARN("ocl_new_stream:  not implemented!?");
	return NULL;
}

static void init_first_ocl_stream(SINGLE_QSP_ARG_DECL)
{
	Platform_Stream *psp;

	psp=PF_FUNC_NAME(new_stream)(QSP_ARG  "default");

#ifdef CAUTIOUS
	if( psp == NO_STREAM ) ERROR1("CAUTIOUS:  failed to create default stream object");
#endif /* CAUTIOUS */

	//psp->cs_stream = (cudaStream_t) 0;
	have_first_ocl_stream=1;
}

static PF_COMMAND_FUNC( new_stream )
{
	const char *s;
	Platform_Stream *psp;
	//cudaError_t e;

	if( ! have_first_ocl_stream ) init_first_ocl_stream(SINGLE_QSP_ARG);

	s=NAMEOF("name for stream");
	psp = PF_FUNC_NAME(new_stream)(QSP_ARG  s);
	if( psp == NO_STREAM ) return;

	/*
	e = cudaStreamCreate(&psp->cs_stream);
	if( e != cudaSuccess ){
		describe_ocl_driver_error2("do_new_stream","cudaStreamCreate",e);
		WARN("Error creating cuda stream");
	}
	*/
}

static void PF_FUNC_NAME(list_streams)(SINGLE_QSP_ARG_DECL)
{
	WARN("ocl_list_streams not implemented!?");
}

static PF_COMMAND_FUNC( list_streams )
{
	PF_FUNC_NAME(list_streams)(SINGLE_QSP_ARG);
}

static PF_COMMAND_FUNC( stream_info )
{
	Platform_Stream *psp;
	
	psp = GET_STREAM("");
	if( psp == NO_STREAM ) return;

	prt_msg("No information...");
}

static PF_COMMAND_FUNC( sync_stream )
{
	Platform_Stream *psp;
	//cudaError_t e;

	if( ! have_first_ocl_stream ) init_first_ocl_stream(SINGLE_QSP_ARG);

	psp = PICK_STREAM("stream");
	if( psp == NO_STREAM ) return;

/*
	e = cudaStreamSynchronize(psp->cs_stream);
	if( e != cudaSuccess ){
		describe_ocl_driver_error2("do_sync_stream","cudaStreamSynchronize",e);
		WARN("Synchronization failure.");
	}
*/
}
#endif // NOT_USED


#endif // HAVE_OPENCL

