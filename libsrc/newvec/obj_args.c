#include "quip_config.h"

char VersionId_newvec_obj_args[] = QUIP_VERSION_STRING;

#ifdef HAVE_STRING_H
#include <string.h>
#endif

#include "vecgen.h"
#include "nvf.h"

const char *type_strings[N_ARGSET_PRECISIONS]={
"byte",		"short",	"int32",	"int64",	"float",	"double",
"u_byte",	"u_short",	"uint32",	"uint64",
"u_byte/short",	"short/byte",	"u_short/long",	"float/double", "bitmap"
};


#ifdef FOOBAR
void show_obj_args(Vec_Obj_Args *oap)
{
	int i;

	if( oap->oa_dest != NO_OBJ ){
		sprintf(error_string,"Destination object: %s",oap->oa_dest->dt_name);
		advise(error_string);
	}
	for(i=0;i<MAX_N_ARGS; i++){
		if( oap->oa_dp[i] != NO_OBJ ){
			sprintf(error_string,"Source object %d:  %s",i+1,oap->oa_dp[i]->dt_name);
			advise(error_string);
		}
	}
	for(i=0;i<MAX_RETSCAL_ARGS; i++){
		if( oap->oa_sdp[i] != NO_OBJ ){
			sprintf(error_string,"Scalar object %d:  %s",i+1,oap->oa_sdp[i]->dt_name);
			advise(error_string);
		}
	}
	for(i=0;i<MAX_SRCSCAL_ARGS; i++){
		if( oap->oa_svp[i] != NULL ){
			prec_t prec;
			char buf[LLEN];
			prec = PREC_FOR_ARGSET(oap->oa_argsprec);
			if( prec != PREC_NONE )
				format_scalar_value(buf,(void *)oap->oa_svp[i],prec);
			else	strcpy(buf,"(invalid precision)");
				
			sprintf(error_string,"Scalar value at addr 0x%lx = %s",
				(int_for_addr)oap->oa_svp,buf);
			advise(error_string);
			/* argset precision is not the same as regular precision? */
		}
	}
	if( oap->oa_argsprec >= 0 && oap->oa_argsprec < N_ARGSET_PRECISIONS ){
		sprintf(error_string,"\targsprec:  %s (%d)",type_strings[oap->oa_argsprec],oap->oa_argsprec);
		advise(error_string);
	} else if( oap->oa_argsprec == -1 ){
		advise("\targsprec not set");
	} else {
		sprintf(error_string,"\targsprec:  garbage value (%d)",oap->oa_argsprec);
		advise(error_string);
	}

	if( oap->oa_argstype >= 0 && oap->oa_argstype < N_ARGSET_TYPES ){
		sprintf(error_string,"\targstype:  %s (%d)",argset_type_name[oap->oa_argstype],oap->oa_argstype);
		advise(error_string);
	} else if( oap->oa_argstype == -1 ){
		advise("\targstype not set");
	} else {
		sprintf(error_string,"\targstype:  garbage value (%d)",oap->oa_argstype);
		advise(error_string);
	}

	/* BUG uninitialized functype generates garbage values */
	if( oap->oa_functype == -1 ){
		advise("\tfunctype not set");
	} else {
		argset_prec ap;
		//prec_t p;
		int dt;

		/* these are reported, but not used!?!? */
		ap = (argset_prec) (oap->oa_functype % N_ARGSET_PRECISIONS);
		//ap = ARGSET_PREC(p);
		dt = 1 + oap->oa_functype / N_ARGSET_PRECISIONS;

		/* BUG (just ugly and easy to break) this code relies on the argset precisions
		 * being in the same order as the data object machine precisions,
		 * but without the null precision at 0.
		 */
		sprintf(error_string,"\tfunctype:  %d (0x%x), argset_prec = %s (%d), argset type = %d",
			oap->oa_functype,oap->oa_functype,
			name_for_prec(ap+1),ap,dt);
		advise(error_string);
	}
}
#else /* ! FOOBAR */

void show_obj_args(Vec_Obj_Args *oap)
{
	private_show_obj_args(DEFAULT_ERROR_STRING,oap,advise);
}

#endif /* ! FOOBAR */

/* This version takes a buffer, so it can be used by different threads simultaneously */

void private_show_obj_args(char *buf, Vec_Obj_Args *oap, void (*report_func)(const char *))
{
	int i;

	if( oap->oa_dest != NO_OBJ ){
		sprintf(buf,"Destination object: %s",oap->oa_dest->dt_name);
		(*report_func)(buf);
	}
	for(i=0;i<MAX_N_ARGS; i++){
		if( oap->oa_dp[i] != NO_OBJ ){
			sprintf(buf,"Source object %d:  %s",i+1,oap->oa_dp[i]->dt_name);
			(*report_func)(buf);
		}
	}
	for(i=0;i<MAX_RETSCAL_ARGS; i++){
		if( oap->oa_sdp[i] != NO_OBJ ){
			sprintf(buf,"Scalar object %d:  %s",i+1,oap->oa_sdp[i]->dt_name);
			(*report_func)(buf);
		}
	}
	for(i=0;i<MAX_SRCSCAL_ARGS; i++){
		if( oap->oa_svp[i] != NULL ){
			prec_t prec;
			char buf[LLEN];
			prec = PREC_FOR_ARGSET(oap->oa_argsprec);
			if( prec != PREC_NONE )
				format_scalar_value(buf,(void *)oap->oa_svp[i],prec);
			else	strcpy(buf,"(invalid precision)");
				
			sprintf(buf,"Scalar value at addr 0x%lx = %s",
				(int_for_addr)oap->oa_svp,buf);
			(*report_func)(buf);
			/* argset precision is not the same as regular precision? */
		}
	}
	if( oap->oa_argsprec >= 0 && oap->oa_argsprec < N_ARGSET_PRECISIONS ){
		sprintf(buf,"\targsprec:  %s (%d)",type_strings[oap->oa_argsprec],oap->oa_argsprec);
		(*report_func)(buf);
	} else if( oap->oa_argsprec == -1 ){
		(*report_func)("\targsprec not set");
	} else {
		sprintf(buf,"\targsprec:  garbage value (%d)",oap->oa_argsprec);
		(*report_func)(buf);
	}

	if( oap->oa_argstype >= 0 && oap->oa_argstype < N_ARGSET_TYPES ){
		sprintf(buf,"\targstype:  %s (%d)",argset_type_name[oap->oa_argstype],oap->oa_argstype);
		(*report_func)(buf);
	} else if( oap->oa_argstype == -1 ){
		(*report_func)("\targstype not set");
	} else {
		sprintf(buf,"\targstype:  garbage value (%d)",oap->oa_argstype);
		(*report_func)(buf);
	}

	/* BUG uninitialized functype generates garbage values */
	if( oap->oa_functype == -1 ){
		(*report_func)("\tfunctype not set");
	} else {
		argset_prec ap;
		//prec_t p;
		int dt;

		/* these are reported, but not used!?!? */
		ap = (argset_prec) (oap->oa_functype % N_ARGSET_PRECISIONS);
		//ap = ARGSET_PREC(p);
		dt = 1 + oap->oa_functype / N_ARGSET_PRECISIONS;

		/* BUG (just ugly and easy to break) this code relies on the argset precisions
		 * being in the same order as the data object machine precisions,
		 * but without the null precision at 0.
		 */
		sprintf(buf,"\tfunctype:  %d (0x%x), argset_prec = %s (%d), argset type = %d",
			oap->oa_functype,oap->oa_functype,
			name_for_prec(ap+1),ap,dt);
		(*report_func)(buf);
	}
} // end private_show_obj_args


void set_obj_arg_flags(Vec_Obj_Args *oap)
{
	oap->oa_argsprec = ARGSET_PREC(oap->oa_dest->dt_prec);

	if( IS_COMPLEX(oap->oa_dest) )
		oap->oa_argstype = COMPLEX_ARGS;
	else
		oap->oa_argstype = REAL_ARGS;

	oap->oa_functype = FUNCTYPE_FOR(oap->oa_argsprec,oap->oa_argstype);
	//TELL_FUNCTYPE(oap->oa_argsprec,oap->oa_argstype)
}

void clear_obj_args(Vec_Obj_Args *oap)
{
	int i;

	for(i=0;i<MAX_N_ARGS;i++)
		oap->oa_dp[i] = NO_OBJ;

	oap->oa_dest = NO_OBJ;

	for(i=0;i<MAX_RETSCAL_ARGS;i++){
		oap->oa_sdp[i] = NO_OBJ;
	}

	for(i=0;i<MAX_SRCSCAL_ARGS;i++){
		oap->oa_svp[i] = NULL;
	}

	oap->oa_argsprec = INVALID_ARGSET_PREC;
	oap->oa_argstype = INVALID_ARGSET_TYPE;

	oap->oa_functype = -1;
}

