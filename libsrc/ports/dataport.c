#include "quip_config.h"

#include <stdio.h>

#ifdef HAVE_SIGNAL_H
#include <signal.h>
#endif

#ifdef HAVE_STRING_H
#include <string.h>
#endif

#include "quip_prot.h"
#include "query_bits.h"	// LLEN - get rid of this!  BUG
#include "ports.h"
#include "data_obj.h"
#include "getbuf.h"


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
		sprintf(ERROR_STRING,
			"CRAY2IEG:  %d overflows",ierr);
		WARN(ERROR_STRING);
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
		sprintf(ERROR_STRING,
			"IEG2CRAY:  %d overflows",ierr);
		WARN(ERROR_STRING);
	}
}

#endif /* CRAY */



static int send_to_port( QSP_ARG_DECL  Port *mpp, char *cp, u_long n )
{
	long nsent;

	while( n > 0 ){
		nsent=write_port(QSP_ARG  mpp,cp,n);
		if( nsent < 0 ) return -1;
		n-=nsent;
		cp+=nsent;
	}
	return 0;
}

#ifdef FOOBAR

void if_pipe(int i)
{
	/* WARN("SIGPIPE"); */
	//NERROR1("SIGPIPE");

	/* This occurs when we write to a port after the program at
	 * the other end has shut down...
	 * How should we handle this error?
	 */
fprintf(stderr,"if_pipe:  received SIGPIPE, arg = %d\n",i);

	/* How can we find out which file descriptor caused the problem? */
}

#endif // FOOBAR


/** xmit_obj - send a data object */
void xmit_obj( QSP_ARG_DECL  Port *mpp, Data_Obj *dp, int dataflag )
/* if flag is zero, don't xmit the actual image data */
{
	uint32_t name_len;
	uint32_t data_len;
	int32_t code;
	int i;

#ifdef SIGPIPE
//advise("signal(SIGPIPE,if_pipe)");
	//signal(SIGPIPE,if_pipe);
	signal(SIGPIPE,SIG_IGN);
#endif /* SIGPIPE */

	if( put_port_int32(QSP_ARG  mpp,PORT_MAGIC_NUMBER) == (-1) ){
		WARN("xmit_obj:  error sending port magic number");
		return;
	}

	code=P_DATA;
	if( put_port_int32(QSP_ARG  mpp,code) == (-1) ){
		WARN("xmit_obj:  error sending code");
		return;
	}

	name_len=(uint32_t)strlen(OBJ_NAME(dp))+1;


	if( dataflag ){		/* figure out the size of the data */
#ifdef CRAY

		if( OBJ_PREC(dp) == PREC_SP )
			data_len=OBJ_N_MACH_ELTS(dp) * sizeof(float) ;
		else if( OBJ_PREC(dp) == PREC_BY )
			data_len=OBJ_N_MACH_ELTS(dp);
		else {
			NERROR1("Sorry, CRAY can only xmit float or byte objs");
		}

#else /* ! CRAY */

		data_len=OBJ_N_MACH_ELTS(dp) * ELEMENT_SIZE(dp) ;

#endif /* ! CRAY */

	} else {	/* send a dataless header (for network file info) */
		data_len = 0;
	}


	if( put_port_int32(QSP_ARG  mpp,name_len) == -1 ){
		WARN("xmit_obj:  error writing first length word");
		return;
	}
	if( put_port_int32(QSP_ARG  mpp,data_len) == -1 ){
		WARN("xmit_obj:  error writing second length word");
		return;
	}

	if( put_port_int32(QSP_ARG  mpp,(int32_t) OBJ_PREC(dp)) == -1 ||
	    put_port_int32(QSP_ARG  mpp,(int32_t) OBJ_N_MACH_ELTS(dp) ) == -1 ){
		WARN("error sending object header data");
		return;
	}
	    
	for(i=0;i<N_DIMENSIONS;i++){
		if( put_port_int32(QSP_ARG  mpp,(int32_t) OBJ_TYPE_DIM(dp,i)) == -1 ){
			WARN("error sending object dimensions");
			return;
		}
	}

	if( write_port(QSP_ARG  mpp,OBJ_NAME(dp),name_len) == (-1) ){
		WARN("xmit_obj:  error writing data object name");
		return;
	}
	
	if( data_len==0 ) return;		/* no data to send */

	/* Now write the data */

#ifdef CRAY
	if( OBJ_PREC(dp) == PREC_SP ){	/* convert to IEEE format */
#define CONV_LEN	2048
		char *cbuf;
		float *p;
		int goofed=0;
		int n,nnums;

		nnums = OBJ_N_MACH_ELTS(dp);
		n=CONV_LEN<nnums?CONV_LEN:nnums;
		cbuf = getbuf( 4 * n );		/* 4 is size of IEEE float */
		if( cbuf == NULL ) mem_err("xmit_obj");
		p = OBJ_DATA_PTR(dp);

		while( nnums > 0 ){
			n = CONV_LEN<nnums ? CONV_LEN : nnums ;
			cray2ieee(cbuf,p,n);
			p += n;
			nnums-=n;
			if( send_to_port(mpp,cbuf,n*4) < 0 ){
				WARN("Error sending object data to port!?");
				return;
			}
		}
		return;
	}
#endif /* CRAY */

	if( send_to_port(QSP_ARG  mpp,(char *)OBJ_DATA_PTR(dp),data_len) < 0 ){
		sprintf(ERROR_STRING,"Error sending object %s data to port %s!?",
			OBJ_NAME(dp),mpp->mp_name);
		WARN(ERROR_STRING);
	}
}

long recv_obj(QSP_ARG_DECL  Port *mpp, Packet *pkp)			/** recieve a new data object */
{
	long name_len, data_len;
	Data_Obj *old_obj, *new_dp;
	Dimension_Set _type_dims;
	dimension_t _n_mach_elts;
	prec_t _prec_code;
	char *cp;
	int i;
	char namebuf[LLEN];
//	uint32_t f;


#ifdef NOT_YET
	if( ram_area == NULL )
		error1("ram data area not initialized");
	if( curr_ap != ram_area )
		error1("current data area not ram area!?");
#endif /* NOT_YET */

	name_len=get_port_int32(QSP_ARG  mpp);
	if( name_len <= 0 ) goto error_return;
	data_len=get_port_int32(QSP_ARG  mpp);
	if( data_len < 0 ) goto error_return;

#ifdef QUIP_DEBUG
if( debug & debug_data ){
sprintf(ERROR_STRING,
"recv_obj:  want %ld name bytes, %ld data bytes",name_len,data_len);
advise(ERROR_STRING);
}
#endif /* QUIP_DEBUG */

	// Get the critical information abou the object

	_prec_code =(prec_t)get_port_int32(QSP_ARG  mpp);
	_n_mach_elts = (dimension_t) get_port_int32(QSP_ARG  mpp); // n_mach_elts
	if( _prec_code == (prec_t)BAD_PORT_LONG ||
		_n_mach_elts == (dimension_t) BAD_PORT_LONG ) {

		WARN("error getting object header data");
		goto error_return;
	}

	for(i=0;i<N_DIMENSIONS;i++){
		long l;
		l= get_port_int32(QSP_ARG  mpp);
		if( l < 0 ) {
			WARN("error getting object dimensions");
			goto error_return;
		}
		SET_DIMENSION(&_type_dims,i,l);
	}


	if( name_len > LLEN ){
		WARN("more than LLEN name chars!?");
		goto error_return;
	}
	if( read_port(QSP_ARG  mpp,namebuf,name_len) != name_len ){
		WARN("recv_obj:  error reading data object name");
		goto error_return;
	}
	if( (long)strlen( namebuf ) != name_len-1 ){
		sprintf(ERROR_STRING,"name length %ld, expected %ld",
			(long)strlen(namebuf), name_len-1);
		advise(ERROR_STRING);
		sprintf(ERROR_STRING,"name:  \"%s\"",namebuf);
		advise(ERROR_STRING);
		for(i=0;i<strlen(namebuf);i++){
			sprintf(ERROR_STRING,"name[%d] = '%c' (0%o)",
				i,namebuf[i],namebuf[i]);
			advise(ERROR_STRING);
		}
		error1("choked");
	}

	old_obj=dobj_of(namebuf);

	if( old_obj != NULL && ( OBJ_ROWS(old_obj)   != DS_ROWS(&_type_dims)   ||
		   		   OBJ_COLS(old_obj)   != DS_COLS(&_type_dims)   ||
		   		   OBJ_FRAMES(old_obj) != DS_FRAMES(&_type_dims) ||
		   		   OBJ_COMPS(old_obj)  != DS_COMPS(&_type_dims)  ||
		   		   OBJ_PREC(old_obj)   != _prec_code      )  ){
		sprintf(ERROR_STRING,
			"mismatched object %s received, discarding old",
			OBJ_NAME(old_obj));
		WARN(ERROR_STRING);
		delvec(old_obj);
		old_obj = NULL;
	}

	if( old_obj == NULL ){
		/* BUG? set curr_ap to ram_area? */
		if( data_len > 0 )
			new_dp=make_dobj(QSP_ARG  namebuf,&_type_dims,
							PREC_FOR_CODE(_prec_code));
		else {
			new_dp=_make_dp(QSP_ARG  namebuf,&_type_dims,
							PREC_FOR_CODE(_prec_code));
			SET_OBJ_DATA_PTR(new_dp,NULL);
		}

		if( new_dp==NULL ){
			sprintf(ERROR_STRING,
				"recv_obj:  couldn't create object \"%s\"",
				namebuf);
			WARN(ERROR_STRING);
			goto error_return;
		}
	} else {
		new_dp=old_obj;
	}

	/* return now if no data to read */
	if( data_len==0 ) goto all_done;

	cp=(char *)OBJ_DATA_PTR(new_dp);

#ifdef CAUTIOUS
	if( data_len !=
		(long)(OBJ_N_MACH_ELTS(new_dp) * ELEMENT_SIZE(new_dp)) ){
		sprintf(ERROR_STRING,
			"recv_obj:  data_len = %ld, size mismatch", data_len);
		WARN(ERROR_STRING);
		longlist(new_dp);
	}
#endif /* CAUTIOUS */


	while( data_len ){
		long nb2;

		nb2=read_port(QSP_ARG  mpp,cp,data_len);
		if( nb2 == 0 ){	/* EOF */
			WARN("recv_obj:  error reading object data");
			goto error_return;
		}
#ifdef QUIP_DEBUG
		if( debug & debug_data ){
			sprintf(ERROR_STRING,"%ld bytes read, wanted %ld",nb2,
				data_len);
			advise(ERROR_STRING);
		}
#endif /* QUIP_DEBUG */
		data_len-=nb2;
		cp+=nb2;
	}
#ifdef QUIP_DEBUG
if( debug & debug_data ){
sprintf(ERROR_STRING,"data object \"%s\" received",OBJ_NAME(new_dp));
advise(ERROR_STRING);
}
#endif /* QUIP_DEBUG */

	/* alert_view(new_dp); */

all_done:
	// The object is entered into the database,
	// So we don't need to pass the pointer in the packet...
	// Not doing so also means we don't have to worry about
	// automatic freeing.

	// BUT how do we know what object was received???
	// Should we add an extra field?

	//pkp->pk_data = (char *) new_dp;
	//pkp->pk_user_data = pkp->pk_data;
	pkp->pk_extra = (char *) new_dp;
	return sizeof(*new_dp);
error_return:
	return -1;
}

