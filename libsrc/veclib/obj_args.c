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

void _private_show_obj_args(QSP_ARG_DECL  char *buf, const Vec_Obj_Args *oap, void (*report_func)(QSP_ARG_DECL  const char *))
{
	int i;

	if( OA_DEST(oap) != NULL ){
		sprintf(buf,"Destination object: %s",OBJ_NAME( OA_DEST(oap) ) );
		(*report_func)(QSP_ARG  buf);
longlist(OA_DEST(oap) );
	}
	for(i=0;i<MAX_N_ARGS; i++){
		if( OA_SRC_OBJ(oap,i) != NULL ){
			sprintf(buf,"Source object %d:  %s",i+1,OBJ_NAME( OA_SRC_OBJ(oap,i) ) );
			(*report_func)(QSP_ARG  buf);
longlist(OA_SRC_OBJ(oap,i) );
		}
	}
	for(i=0;i<MAX_RETSCAL_ARGS; i++){
		if( OA_SCLR_OBJ(oap,i) != NULL ){
			sprintf(buf,"Scalar object %d:  %s",i+1,OBJ_NAME( OA_SCLR_OBJ(oap,i) ) );
			(*report_func)(QSP_ARG  buf);
		}
	}
	for(i=0;i<MAX_SRCSCAL_ARGS; i++){
		if( OA_SVAL(oap,i) != NULL ){
			prec_t prec;
#define MSG_LEN	80
			char msgbuf[MSG_LEN];
			prec = PREC_FOR_ARGSET( OA_ARGSPREC_CODE(oap) );
//fprintf(stderr,"formatting as float:  %g\n",*((float *)OA_SVAL(oap,i)) );
			if( prec != PREC_NONE ){
				format_scalar_value(msgbuf,MSG_LEN,(void *)OA_SVAL(oap,i),prec_for_code(prec),NO_PADDING);
			}
			else	strcpy(msgbuf,"(invalid precision)");
				
			sprintf(buf,"Scalar value at addr 0x%"PRIxPTR" = %s",
				(uintptr_t)OA_SVAL(oap,i),msgbuf);
			(*report_func)(QSP_ARG  buf);
			/* argset precision is not the same as regular precision? */
		}
	}
	if(  /* OA_ARGSPREC_CODE(oap)  >= 0 && */  OA_ARGSPREC_CODE(oap)  < N_ARGSET_PRECISIONS ){
		sprintf(buf,"\targsprec:  %s (%d)",type_strings[ OA_ARGSPREC_CODE(oap) ], OA_ARGSPREC_CODE(oap) );
		(*report_func)(QSP_ARG  buf);
	} else if(  OA_ARGSPREC_PTR(oap)  == NULL ){
		(*report_func)(QSP_ARG  "\targsprec not set");
	} else {
		sprintf(buf,"\targsprec:  garbage value (%d)", OA_ARGSPREC_CODE(oap) );
		(*report_func)(QSP_ARG  buf);
	}

	if( /* OA_ARGSTYPE(oap) >= 0 && */ OA_ARGSTYPE(oap) < N_ARGSET_TYPES ){
		sprintf(buf,"\targstype:  %s (%d)",argset_type_name[OA_ARGSTYPE(oap)],OA_ARGSTYPE(oap));
		(*report_func)(QSP_ARG  buf);
	} else if( OA_ARGSTYPE(oap) == INVALID_ARGSET_TYPE ){
		(*report_func)(QSP_ARG  "\targstype not set");
	} else {
		sprintf(buf,"\targstype:  garbage value (%d)",OA_ARGSTYPE(oap));
		(*report_func)(QSP_ARG  buf);
	}

	/* BUG uninitialized functype generates garbage values */
	if( OA_FUNCTYPE(oap) == -1 ){
		(*report_func)(QSP_ARG  "\tfunctype not set");
	} else {
		argset_prec_t ap;
		//prec_t p;
		int dt;
		const char *ap_string;

		/* these are reported, but not used!?!? */
		ap = (argset_prec_t) (OA_FUNCTYPE(oap) % N_ARGSET_PRECISIONS);
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
					warn("CAUTIOUS:  private_show_obj_args:  bad argset precision!?");
					ap_string="(invalid)";
					break;
#endif /* CAUTIOUS */
			}
		}
		sprintf(buf,"\tfunctype:  %d (0x%x), argset_prec = %s (%d), argset type = %d",
			OA_FUNCTYPE(oap),OA_FUNCTYPE(oap),
			ap_string, ap,dt);
		(*report_func)(QSP_ARG  buf);
	}
} // end private_show_obj_args

void _show_obj_args(QSP_ARG_DECL  const Vec_Obj_Args *oap)
{
	private_show_obj_args(ERROR_STRING,oap,_advise);
}

void set_obj_arg_flags(Vec_Obj_Args *oap)
{
	 SET_OA_ARGSPREC_CODE(oap, ARGSET_PREC( OBJ_PREC( OA_DEST(oap) ) ) );

	if( IS_COMPLEX(OA_DEST(oap)) )
		OA_ARGSTYPE(oap) = COMPLEX_ARGS;
	else
		OA_ARGSTYPE(oap) = REAL_ARGS;

	SET_OA_FUNCTYPE(oap , FUNCTYPE_FOR( OA_ARGSPREC_CODE(oap),OA_ARGSTYPE(oap)) );
	//TELL_FUNCTYPE( OA_ARGSPREC_CODE(oap),OA_ARGSTYPE(oap))
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

	SET_OA_ARGSPREC_PTR(oap, NULL );
	SET_OA_ARGSTYPE(oap, INVALID_ARGSET_TYPE);

	SET_OA_FUNCTYPE(oap, -1);
}

static Item_Type *argset_prec_itp=NULL;
static ITEM_INIT_FUNC(Argset_Prec,argset_prec,0)
static ITEM_NEW_FUNC(Argset_Prec,argset_prec)
static Argset_Prec * argset_prec_tbl[N_ARGSET_PRECISIONS];

#define init_argset_precs()	_init_argset_precs(SINGLE_QSP_ARG)
#define new_argset_prec(s)	_new_argset_prec(QSP_ARG  s)

#define INIT_ARGSET_PREC(name,code)			\
	ap_p = new_argset_prec(name);		\
	assert(ap_p != NULL );				\
	ap_p->ap_code = code;				\
	assert( code >= 0 && code < N_ARGSET_PRECISIONS );	\
	argset_prec_tbl[code] = ap_p;

void init_argset_objects(SINGLE_QSP_ARG_DECL)
{
	Argset_Prec *ap_p;

#ifdef CAUTIOUS
	int i;
	bzero(argset_prec_tbl,N_ARGSET_PRECISIONS*sizeof(argset_prec_tbl[0]));
#endif // CAUTIOUS

	INIT_ARGSET_PREC( "byte",		BY_ARGS );	
	INIT_ARGSET_PREC( "short",		IN_ARGS );	
	INIT_ARGSET_PREC( "int32",		DI_ARGS );	
	INIT_ARGSET_PREC( "int64",		LI_ARGS );	
	INIT_ARGSET_PREC( "float",		SP_ARGS );	
	INIT_ARGSET_PREC( "double",		DP_ARGS );	
	INIT_ARGSET_PREC( "u_byte",		UBY_ARGS );	
	INIT_ARGSET_PREC( "u_short",		UIN_ARGS );	
	INIT_ARGSET_PREC( "uint32",		UDI_ARGS );	
	INIT_ARGSET_PREC( "uint64",		ULI_ARGS );	
	INIT_ARGSET_PREC( "byte/short",		BYIN_ARGS );	
	INIT_ARGSET_PREC( "short/byte",		INBY_ARGS );	
	INIT_ARGSET_PREC( "short/int32",	INDI_ARGS );	
	INIT_ARGSET_PREC( "float/double",	SPDP_ARGS );	
	INIT_ARGSET_PREC( "bit",		BIT_ARGS );	

#ifdef CAUTIOUS
	// Make sure that all table entries are filled
	for(i=0;i<N_ARGSET_PRECISIONS;i++){
		assert(argset_prec_tbl[i] != NULL );
	}
#endif // CAUTIOUS
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

void set_argset_prec( Vec_Obj_Args *oap, argset_prec_t code )
{
	assert( code >= 0 && code < N_ARGSET_PRECISIONS );

	SET_OA_ARGSPREC_PTR(oap,argset_prec_tbl[code]);
}

const char *name_for_argsprec(argset_prec_t code)
{
	assert( code >= 0 && code < N_ARGSET_PRECISIONS );
	return argset_prec_tbl[code]->ap_item.item_name;
}

