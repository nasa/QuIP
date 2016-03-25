#include "quip_config.h"

#ifdef HAVE_STRING_H
#include <string.h>
#endif

#include "quip_prot.h"
#include "veclib/vecgen.h"
#include "nvf.h"
#include "warn.h"	// advise

const char *type_strings[N_ARGSET_PRECISIONS]={
"byte",		"short",	"int32",	"int64",	"float",	"double",
"u_byte",	"u_short",	"uint32",	"uint64",
"u_byte/short",	"short/byte",	"u_short/long",	"float/double", "bitmap"
};


/* This version takes a buffer, so it can be used by different threads simultaneously */

void private_show_obj_args(QSP_ARG_DECL  char *buf, const Vec_Obj_Args *oap, void (*report_func)(QSP_ARG_DECL  const char *))
{
	int i;

	if( OA_DEST(oap) != NO_OBJ ){
		sprintf(buf,"Destination object: %s",OBJ_NAME( OA_DEST(oap) ) );
		(*report_func)(DEFAULT_QSP_ARG  buf);
longlist(QSP_ARG  OA_DEST(oap) );
	}
	for(i=0;i<MAX_N_ARGS; i++){
		if( OA_SRC_OBJ(oap,i) != NO_OBJ ){
			sprintf(buf,"Source object %d:  %s",i+1,OBJ_NAME( OA_SRC_OBJ(oap,i) ) );
			(*report_func)(DEFAULT_QSP_ARG  buf);
longlist(QSP_ARG  OA_SRC_OBJ(oap,i) );
		}
	}
	for(i=0;i<MAX_RETSCAL_ARGS; i++){
		if( OA_SCLR_OBJ(oap,i) != NO_OBJ ){
			sprintf(buf,"Scalar object %d:  %s",i+1,OBJ_NAME( OA_SCLR_OBJ(oap,i) ) );
			(*report_func)(DEFAULT_QSP_ARG  buf);
		}
	}
	for(i=0;i<MAX_SRCSCAL_ARGS; i++){
		if( OA_SVAL(oap,i) != NULL ){
			prec_t prec;
			char msgbuf[LLEN];
			prec = PREC_FOR_ARGSET( OA_ARGSPREC(oap) );
//fprintf(stderr,"formatting as float:  %g\n",*((float *)OA_SVAL(oap,i)) );
			if( prec != PREC_NONE ){
				format_scalar_value(QSP_ARG  msgbuf,(void *)OA_SVAL(oap,i),prec_for_code(prec));
			}
			else	strcpy(msgbuf,"(invalid precision)");
				
			sprintf(buf,"Scalar value at addr 0x%lx = %s",
				(int_for_addr)OA_SVAL(oap,i),msgbuf);
			(*report_func)(DEFAULT_QSP_ARG  buf);
			/* argset precision is not the same as regular precision? */
		}
	}
	if(  /* OA_ARGSPREC(oap)  >= 0 && */  OA_ARGSPREC(oap)  < N_ARGSET_PRECISIONS ){
		sprintf(buf,"\targsprec:  %s (%d)",type_strings[ OA_ARGSPREC(oap) ], OA_ARGSPREC(oap) );
		(*report_func)(DEFAULT_QSP_ARG  buf);
	} else if(  OA_ARGSPREC(oap)  == INVALID_ARGSET_PREC ){
		(*report_func)(DEFAULT_QSP_ARG  "\targsprec not set");
	} else {
		sprintf(buf,"\targsprec:  garbage value (%d)", OA_ARGSPREC(oap) );
		(*report_func)(DEFAULT_QSP_ARG  buf);
	}

	if( /* OA_ARGSTYPE(oap) >= 0 && */ OA_ARGSTYPE(oap) < N_ARGSET_TYPES ){
		sprintf(buf,"\targstype:  %s (%d)",argset_type_name[OA_ARGSTYPE(oap)],OA_ARGSTYPE(oap));
		(*report_func)(DEFAULT_QSP_ARG  buf);
	} else if( OA_ARGSTYPE(oap) == INVALID_ARGSET_TYPE ){
		(*report_func)(DEFAULT_QSP_ARG  "\targstype not set");
	} else {
		sprintf(buf,"\targstype:  garbage value (%d)",OA_ARGSTYPE(oap));
		(*report_func)(DEFAULT_QSP_ARG  buf);
	}

	/* BUG uninitialized functype generates garbage values */
	if( OA_FUNCTYPE(oap) == -1 ){
		(*report_func)(DEFAULT_QSP_ARG  "\tfunctype not set");
	} else {
		argset_prec ap;
		//prec_t p;
		int dt;
		const char *ap_string;

		/* these are reported, but not used!?!? */
		ap = (argset_prec) (OA_FUNCTYPE(oap) % N_ARGSET_PRECISIONS);
		//ap = ARGSET_PREC(p);
		dt = 1 + OA_FUNCTYPE(oap) / N_ARGSET_PRECISIONS;

		/* BUG (just ugly and easy to break) this code relies on the argset precisions
		 * being in the same order as the data object machine precisions,
		 * but without the null precision at 0.
		 *
		 * Worse than that, this is only good for the simple machine precisions.
		 * The mixed types and bit precision don't correspond to prec codes...
		 */
		if( ap < (N_MACHINE_PRECS-1) ){
			ap_string=NAME_FOR_PREC_CODE(ap+1);
		} else {
			switch(ap){
				case BYIN_ARGS: ap_string="byte/int16"; break;
				case INBY_ARGS: ap_string="int16/byte"; break;
				case INDI_ARGS: ap_string="int16/int32"; break;
				case SPDP_ARGS: ap_string="float/double"; break;
				case BIT_ARGS:  ap_string="bit"; break;
#ifdef CAUTIOUS
				default:
					NWARN("CAUTIOUS:  private_show_obj_args:  bad argset precision!?");
					ap_string="(invalid)";
					break;
#endif /* CAUTIOUS */
			}
		}
		sprintf(buf,"\tfunctype:  %d (0x%x), argset_prec = %s (%d), argset type = %d",
			OA_FUNCTYPE(oap),OA_FUNCTYPE(oap),
			ap_string, ap,dt);
		(*report_func)(DEFAULT_QSP_ARG  buf);
	}
} // end private_show_obj_args

void show_obj_args(QSP_ARG_DECL  const Vec_Obj_Args *oap)
{
	private_show_obj_args(QSP_ARG  ERROR_STRING,oap,_advise);
}

void set_obj_arg_flags(Vec_Obj_Args *oap)
{
	 SET_OA_ARGSPREC(oap, ARGSET_PREC( OBJ_PREC( OA_DEST(oap) ) ) );

	if( IS_COMPLEX(OA_DEST(oap)) )
		OA_ARGSTYPE(oap) = COMPLEX_ARGS;
	else
		OA_ARGSTYPE(oap) = REAL_ARGS;

	SET_OA_FUNCTYPE(oap , FUNCTYPE_FOR( OA_ARGSPREC(oap),OA_ARGSTYPE(oap)) );
	//TELL_FUNCTYPE( OA_ARGSPREC(oap),OA_ARGSTYPE(oap))
}

void clear_obj_args(Vec_Obj_Args *oap)
{
#ifdef HAVE_MEMSET

	memset(oap,0,sizeof(Vec_Obj_Args));

#else // ! HAVE_MEMSET

	unsigned char *p;
	int n;

	p = (unsigned char *) oap;
	n = sizeof(Vec_Obj_Args);
	while(n--)
		*p++ = 0;

#endif // ! HAVE_MEMSET

	// These fields have non-null initial values...

	SET_OA_ARGSPREC(oap, INVALID_ARGSET_PREC );
	SET_OA_ARGSTYPE(oap, INVALID_ARGSET_TYPE);

	SET_OA_FUNCTYPE(oap, -1);
}

const char *name_for_argsprec(argset_prec t)
{
	switch(t){
		case BY_ARGS:	return "byte"; break;
		case IN_ARGS:	return "short"; break;
		case DI_ARGS:	return "int32"; break;
		case LI_ARGS:	return "int64"; break;
		case SP_ARGS:	return "float"; break;
		case DP_ARGS:	return "double"; break;
		case UBY_ARGS:	return "u_byte"; break;
		case UIN_ARGS:	return "u_short"; break;
		case UDI_ARGS:	return "uint32"; break;
		case ULI_ARGS:	return "uint64"; break;
		case BYIN_ARGS:	return "byte/short"; break;
		case INBY_ARGS:	return "short/byte"; break;
		case INDI_ARGS:	return "short/int32"; break;
		case SPDP_ARGS:	return "float/double"; break;
		case BIT_ARGS:	return "bit"; break;
		default: return "invalid"; break;
	}
}

const char *name_for_argtype(argset_type t)
{
	switch(t){
		case UNKNOWN_ARGS: return "unknown"; break;
		case REAL_ARGS: return "real"; break;
		case COMPLEX_ARGS: return "complex"; break;
		case MIXED_ARGS: return "real/complex"; break;
		case QUATERNION_ARGS: return "quaternion"; break;
		case QMIXED_ARGS: return "real/quaternion"; break;
		default: return "invalid"; break;
	}
}

