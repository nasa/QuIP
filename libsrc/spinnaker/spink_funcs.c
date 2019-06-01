#include "quip_config.h"
#include "spink.h"
#include "quip_prot.h"

#ifdef HAVE_LIBSPINNAKER
// This file contains wrappers for all of the library functions

// define this to empty to quiet debugging...
#ifdef MAX_DEBUG
#define WRAPPER_REPORT(my_name,spin_name)							\
fprintf(stderr,"SPINNAKER API CALL:  %s calling spin%s\n",#my_name,#spin_name);
#else // ! MAX_DEBUG
#define WRAPPER_REPORT(my_name,spin_name)
#endif // ! MAX_DEBUG

#define SPINK_WRAPPER_ONE_ARG(my_name,spin_name,decl1,name1)					\
												\
int _##my_name(QSP_ARG_DECL  decl1 name1)							\
{												\
	spinError err;										\
	WRAPPER_REPORT(my_name,spin_name)							\
	err = spin##spin_name(name1);								\
	if( err != SPINNAKER_ERR_SUCCESS ){							\
		report_spink_error(err,#spin_name);						\
		return -1;									\
	}											\
	return 0;										\
}

#define SPINK_WRAPPER_TWO_ARG(my_name,spin_name,decl1,name1,decl2,name2)			\
												\
int _##my_name(QSP_ARG_DECL  decl1 name1, decl2 name2)						\
{												\
	spinError err;										\
	WRAPPER_REPORT(my_name,spin_name)							\
	err = spin##spin_name(name1,name2);							\
	if( err != SPINNAKER_ERR_SUCCESS ){							\
		report_spink_error(err,#spin_name);					\
		return -1;									\
	}											\
	return 0;										\
}

#define SPINK_WRAPPER_THREE_ARG(my_name,spin_name,decl1,name1,decl2,name2,decl3,name3)		\
												\
int _##my_name(QSP_ARG_DECL  decl1 name1, decl2 name2, decl3 name3)				\
{												\
	spinError err;										\
	WRAPPER_REPORT(my_name,spin_name)							\
	err = spin##spin_name(name1,name2,name3);						\
	if( err != SPINNAKER_ERR_SUCCESS ){							\
		report_spink_error(err,#spin_name);					\
		return -1;									\
	}											\
	return 0;										\
}

void _report_spink_error(QSP_ARG_DECL  spinError error, const char *whence )
{
	const char *msg;

	switch(error){
		case SPINNAKER_ERR_SUCCESS:
			msg = "Success"; break;
		case SPINNAKER_ERR_ERROR:
			msg = "Error"; break;
		case SPINNAKER_ERR_NOT_INITIALIZED:
			msg = "Not initialized"; break;
		case SPINNAKER_ERR_NOT_IMPLEMENTED:
			msg = "Not implemented"; break;
		case SPINNAKER_ERR_RESOURCE_IN_USE:
			msg = "Resource in use"; break;
		case SPINNAKER_ERR_ACCESS_DENIED:
			msg = "Access denied"; break;
		case SPINNAKER_ERR_INVALID_HANDLE:
			msg = "Invalid handle"; break;
		case SPINNAKER_ERR_INVALID_ID:
			msg = "Invalid ID"; break;
		case SPINNAKER_ERR_NO_DATA:
			msg = "No data"; break;
		case SPINNAKER_ERR_INVALID_PARAMETER:
			msg = "Invalid parameter"; break;
		case SPINNAKER_ERR_IO:
			msg = "I/O error"; break;
		case SPINNAKER_ERR_TIMEOUT:
			msg = "Timeout"; break;
		case SPINNAKER_ERR_ABORT:
			msg = "Abort"; break;
		case SPINNAKER_ERR_INVALID_BUFFER:
			msg = "Invalid buffer"; break;
		case SPINNAKER_ERR_NOT_AVAILABLE:
			msg = "Not available"; break;
		case SPINNAKER_ERR_INVALID_ADDRESS:
			msg = "Invalid address"; break;
		case SPINNAKER_ERR_BUFFER_TOO_SMALL:
			msg = "Buffer too small"; break;
		case SPINNAKER_ERR_INVALID_INDEX:
			msg = "Invalid index"; break;
		case SPINNAKER_ERR_PARSING_CHUNK_DATA:
			msg = "Chunk data parsing error"; break;
		case SPINNAKER_ERR_INVALID_VALUE:
			msg = "Invalid value"; break;
		case SPINNAKER_ERR_RESOURCE_EXHAUSTED:
			msg = "Resource exhausted"; break;
		case SPINNAKER_ERR_OUT_OF_MEMORY:
			msg = "Out of memory"; break;
		case SPINNAKER_ERR_BUSY:
			msg = "Busy"; break;

		case GENICAM_ERR_INVALID_ARGUMENT:
			msg = "genicam invalid argument"; break;
		case GENICAM_ERR_OUT_OF_RANGE:
			msg = "genicam range error"; break;
		case GENICAM_ERR_PROPERTY:
			msg = "genicam property error"; break;
		case GENICAM_ERR_RUN_TIME:
			msg = "genicam run time error"; break;
		case GENICAM_ERR_LOGICAL:
			msg = "genicam logical error"; break;
		case GENICAM_ERR_ACCESS:
			msg = "genicam access error"; break;
		case GENICAM_ERR_TIMEOUT:
			msg = "genicam timeout error"; break;
		case GENICAM_ERR_DYNAMIC_CAST:
			msg = "genicam dynamic cast error"; break;
		case GENICAM_ERR_GENERIC:
			msg = "genicam generic error"; break;
		case GENICAM_ERR_BAD_ALLOCATION:
			msg = "genicam bad allocation"; break;

		case SPINNAKER_ERR_IM_CONVERT:
			msg = "image conversion error"; break;
		case SPINNAKER_ERR_IM_COPY:
			msg = "image copy error"; break;
		case SPINNAKER_ERR_IM_MALLOC:
			msg = "image malloc error"; break;
		case SPINNAKER_ERR_IM_NOT_SUPPORTED:
			msg = "image operation not supported"; break;
		case SPINNAKER_ERR_IM_HISTOGRAM_RANGE:
			msg = "image histogram range error"; break;
		case SPINNAKER_ERR_IM_HISTOGRAM_MEAN:
			msg = "image histogram mean error"; break;
		case SPINNAKER_ERR_IM_MIN_MAX:
			msg = "image min/max error"; break;
		case SPINNAKER_ERR_IM_COLOR_CONVERSION:
			msg = "image color conversion error"; break;

//		case SPINNAKER_ERR_CUSTOM_ID = -10000

		default:
			sprintf(ERROR_STRING,
		"report_spink_error (%s):  unhandled error code %d!?\n",
				whence,error);
			warn(ERROR_STRING);
			msg = "unhandled error code";
			break;
	}
	sprintf(ERROR_STRING,"spin%s:  %s",whence,msg);
	//warn(ERROR_STRING);
	error1(ERROR_STRING);
}

#include "spink_wrappers.c"

#endif // HAVE_LIBSPINNAKER

