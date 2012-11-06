#include "quip_config.h"

char VersionId_dataf_dataport[] = QUIP_VERSION_STRING;

#include <stdio.h>

#ifdef HAVE_SIGNAL_H
#include <signal.h>
#endif

#ifdef HAVE_STRING_H
#include <string.h>
#endif

#include "nports_api.h"
#include "debug.h"
#include "data_obj.h"
#include "getbuf.h"

/* local prototypes */
static void send_to_port(Port *mpp,char *cp,u_long n);

#ifdef CRAY

void cray2ieee(cbuf,p,n)
void *cbuf; float *p; int n;
{
	int ierr,type,bitoff;

	type = 2;	/* IEEE float - use 3 for double */
			/* 7 16 bit short -> 32 bit short */
			/* 8 64 bit dbl -> 64 bit CRAY float */
			/* 1 32 bit long -> 64 bit CRAY long */
	bitoff = 0;
	ierr = CRAY2IEG(&type,&n,cbuf,&bitoff,p);
	if( ierr < 0 )
		WARN("CRAY2IEG parameter error");
	else if( ierr > 0 ){
		sprintf(error_string,
			"CRAY2IEG:  %d overflows",ierr);
		WARN(error_string);
	}
}

void ieee2cray(p,cbuf,n)
float *p; void *cbuf; int n;
{
	type = 2;	/* IEEE float - use 3 for double */
			/* 7 16 bit short -> 32 bit short */
			/* 8 64 bit dbl -> 64 bit CRAY float */
			/* 1 32 bit long -> 64 bit CRAY long */
	bitoff = 0;
	ierr = IEG2CRAY(&type,&n,cbuf,&bitoff,p);
	if( ierr < 0 )
		WARN("IEG2CRAY parameter error");
	else if( ierr > 0 ){
		sprintf(error_string,
			"IEG2CRAY:  %d overflows",ierr);
		WARN(error_string);
	}
}

#endif /* CRAY */



static void send_to_port( Port *mpp, char *cp, u_long n )
{
	long nsent;

	while( n > 0 ){
		nsent=write_port(mpp,cp,n);
		n-=nsent;
		cp+=nsent;
	}
}

void if_pipe(int i)
{
	/* WARN("SIGPIPE"); */
	NERROR1("SIGPIPE");
}



/** xmit_obj - send a data object */
void xmit_obj( QSP_ARG_DECL  Port *mpp, Data_Obj *dp, int dataflag )
/* if flag is zero, don't xmit the actual image data */
{
	u_long len[2];
	long code;
	int i;

#ifdef SIGPIPE
//advise("signal(SIGPIPE,if_pipe)");
	signal(SIGPIPE,if_pipe);
#endif /* SIGPIPE */

	code=P_DATA;
	if( put_port_int32(mpp,code) == (-1) )
		WARN("xmit_obj:  error sending code");

	len[0]=(u_long)strlen(dp->dt_name)+1;


	if( dataflag ){		/* figure out the size of the data */
#ifdef CRAY

		if( dp->dt_prec == PREC_SP )
			len[1]=dp->dt_n_mach_elts * sizeof(float) ;
		else if( dp->dt_prec == PREC_BY )
			len[1]=dp->dt_n_mach_elts;
		else {
			NERROR1("Sorry, CRAY can only xmit float or byte objs");
		}

#else /* ! CRAY */

		len[1]=dp->dt_n_mach_elts * ELEMENT_SIZE(dp) ;

#endif /* ! CRAY */

	} else {	/* send a dataless header (for network file info) */
		len[1] = 0;
	}


	if( put_port_int32(mpp,len[0]) == -1 )
		WARN("xmit_obj:  error writing first length word");
	if( put_port_int32(mpp,len[1]) == -1 )
		WARN("xmit_obj:  error writing second length word");

	if( put_port_int32(mpp,(long) dp->dt_prec) == -1 ||
	    put_port_int32(mpp,(long) dp->dt_flags) == -1 ||
	    put_port_int32(mpp,(long) dp->dt_n_mach_elts ) == -1 )
		WARN("error sending object header data");
	    
	for(i=0;i<N_DIMENSIONS;i++){
		if( put_port_int32(mpp,(long) dp->dt_mach_dim[i]) == -1 )
			WARN("error sending object dimensions");
	}

	if( write_port(mpp,dp->dt_name,len[0]) == (-1) )
		WARN("xmit_obj:  error writing data object name");
	
	if( len[1]==0 ) return;		/* no data to send */

	/* Now write the data */

#ifdef CRAY
	if( dp->dt_prec == PREC_SP ){	/* convert to IEEE format */
#define CONV_LEN	2048
		char *cbuf;
		float *p;
		int goofed=0;
		int n,nnums;

		nnums = dp->dt_n_mach_elts;
		n=CONV_LEN<nnums?CONV_LEN:nnums;
		cbuf = getbuf( 4 * n );		/* 4 is size of IEEE float */
		if( cbuf == NULL ) mem_err("xmit_obj");
		p = dp->dt_data;

		while( nnums > 0 ){
			n = CONV_LEN<nnums ? CONV_LEN : nnums ;
			cray2ieee(cbuf,p,n);
			p += n;
			nnums-=n;
			send_to_port(mpp,cbuf,n*4);
		}
		return;
	}
#endif /* CRAY */

	send_to_port(mpp,(char *)dp->dt_data,len[1]);
}

Data_Obj *
recv_obj(QSP_ARG_DECL  Port *mpp)			/** recieve a new data object */
{
	long len[2];
	Data_Obj dobj, *dp, *old_obj, *new_obj;
	char *cp;
	int i;
	char namebuf[LLEN];

	if( ram_area == NO_AREA )
		ERROR1("ram data area not initialized");
	if( curr_ap != ram_area )
		ERROR1("current data area not ram area!?");

	dp=(&dobj);
	len[0]=get_port_int32(QSP_ARG  mpp);
	if( len[0] <= 0 ) return(NO_OBJ);
	len[1]=get_port_int32(QSP_ARG  mpp);
	if( len[1] < 0 ) return(NO_OBJ);

#ifdef DEBUG
if( debug & debug_data ){
sprintf(error_string,
"recv_obj:  want %ld name bytes, %ld data bytes",len[0],len[1]);
advise(error_string);
}
#endif /* DEBUG */

	if( (dp->dt_prec = (prec_t)get_port_int32(QSP_ARG  mpp)) == (prec_t)BAD_PORT_LONG ||
	    (dp->dt_flags = (short)get_port_int32(QSP_ARG  mpp)) == BAD_PORT_LONG ||
	    (dp->dt_n_mach_elts = get_port_int32(QSP_ARG  mpp)) == BAD_PORT_LONG ){
		WARN("error getting object header data");
		return(NO_OBJ);
	}

	for(i=0;i<N_DIMENSIONS;i++){
		long l;
		l= get_port_int32(QSP_ARG  mpp);
		if( l < 0 ) {
			WARN("error getting object dimensions");
			return(NO_OBJ);
		}
		dp->dt_mach_dim[i]=l;
	}


	if( len[0] > LLEN ){
		WARN("more than LLEN name chars!?");
		return(NO_OBJ);
	}
	if( read_port(QSP_ARG  mpp,namebuf,len[0]) != len[0] ){
		WARN("recv_obj:  error reading data object name");
		return(NO_OBJ);
	}
	if( (long)strlen( namebuf ) != len[0]-1 ){
		u_int i;

		sprintf(error_string,"name length %ld, expected %ld",
			(long)strlen(namebuf), len[0]-1);
		advise(error_string);
		sprintf(error_string,"name:  \"%s\"",namebuf);
		advise(error_string);
		for(i=0;i<strlen(namebuf);i++){
			sprintf(error_string,"name[%d] = '%c' (0%o)",
				i,namebuf[i],namebuf[i]);
			advise(error_string);
		}
		ERROR1("choked");
	}

	old_obj=dobj_of(QSP_ARG  namebuf);

	if( old_obj != NO_OBJ && ( old_obj->dt_rows   != dp->dt_rows   ||
		   		   old_obj->dt_cols   != dp->dt_cols   ||
		   		   old_obj->dt_frames != dp->dt_frames ||
		   		   old_obj->dt_comps  != dp->dt_comps  ||
		   		   old_obj->dt_prec   != dp->dt_prec      )  ){
		sprintf(error_string,
			"mismatched object %s received, discarding old",
			old_obj->dt_name);
		WARN(error_string);
		delvec(QSP_ARG  old_obj);
		old_obj = NO_OBJ;
	}

	if( old_obj == NO_OBJ ){
		/* BUG? set curr_ap to ram_area? */
		if( len[1] > 0 )
			new_obj=make_dobj(QSP_ARG  namebuf,&dp->dt_type_dimset,dp->dt_prec);
		else {
			new_obj=_make_dp(QSP_ARG  namebuf,&dp->dt_type_dimset,dp->dt_prec);
			new_obj->dt_data=NULL;
		}

		if( new_obj==NO_OBJ ){
			sprintf(error_string,
				"recv_obj:  couldn't create object \"%s\"",
				namebuf);
			WARN(error_string);
			return(NO_OBJ);
		}
	} else {
		new_obj=old_obj;
	}

	/* return now if no data to read */
	if( len[1]==0 ) return(new_obj);

	cp=(char *)new_obj->dt_data;

	if( len[1] !=
		(long)(new_obj->dt_n_mach_elts * ELEMENT_SIZE(new_obj)) ){
		sprintf(error_string,
			"recv_obj:  len[1] = %ld, size mismatch", len[1]);
		WARN(error_string);
		LONGLIST(new_obj);
	}


	while( len[1] ){
		long nb2;

		nb2=read_port(QSP_ARG  mpp,cp,len[1]);
		if( nb2 == 0 ){	/* EOF */
			WARN("recv_obj:  error reading object data");
			return(NO_OBJ);
		}
#ifdef DEBUG
		if( debug & debug_data ){
			sprintf(error_string,"%ld bytes read, wanted %ld",nb2,
				len[1]);
			advise(error_string);
		}
#endif /* DEBUG */
		len[1]-=nb2;
		cp+=nb2;
	}
#ifdef DEBUG
if( debug & debug_data ){
sprintf(error_string,"data object \"%s\" received",new_obj->dt_name);
advise(error_string);
}
#endif /* DEBUG */
	/* alert_view(new_obj); */

	return(new_obj);
}

