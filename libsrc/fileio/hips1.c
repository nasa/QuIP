#include "quip_config.h"

#include <stdio.h>

#ifdef HAVE_UNISTD_H
#include <unistd.h>
#endif

#ifdef HAVE_STRING_H
#include <string.h>
#endif

#include "fio_prot.h"
#include "quip_prot.h"
#include "getbuf.h"
#include "data_obj.h"
#include "debug.h"
#include "hips/hips1.h"
//#include "get_hdr.h"
#include "img_file/raw.h"
#include "hips/hipl_fmt.h"

/* local prototypes */

static void rewrite_hips1_nf(int fd,dimension_t n);
static void rls_header(Hips1_Header *hd);

FIO_CLOSE_FUNC( hips1 )
{
	/* see if we need to edit the header */
	if( IS_WRITABLE(ifp)
		&& ifp->if_dp != NULL	/* may be closing 'cause of error */
		&& ifp->if_nfrms != ifp->if_frms_to_wt ){
		if( ifp->if_nfrms <= 0 ){
			sprintf(ERROR_STRING, "file %s nframes=%d!?",
				ifp->if_name,ifp->if_nfrms);
			WARN(ERROR_STRING);
		}
		rewrite_hips1_nf(ifp->if_fd,ifp->if_nfrms);
	}

	if( ifp->if_hdr_p != NULL ){
#ifdef QUIP_DEBUG
if( debug & debug_fileio ) advise("freeing hips1 header strings");
#endif
		rls_header(ifp->if_hdr_p);		/* free strings */
#ifdef QUIP_DEBUG
if( debug & debug_fileio ) advise("freeing hips1 header struct");
#endif
		givbuf(ifp->if_hdr_p);
	}
	GENERIC_IMGFILE_CLOSE(ifp);
}

int hips1_to_dp(Data_Obj *dp,Hips1_Header *hd_p)
{
	dimension_t type_dim=1;
	short prec;

	if( hd_p->pixel_format < 0 ){	/* a non-hips extension */
		prec=PREC_SP;
		type_dim= hd_p->pixel_format * -1;
		if( type_dim <= 2 ){
			NWARN("hips1_to_dp: bad multidimensional format");
			return(-1);
		}
	} else switch( hd_p->pixel_format ){
		case PFBYTE:  prec=PREC_UBY; break;
		case PFSHORT: prec=PREC_IN; break;
		case PFFLOAT: prec=PREC_SP; break;
		case PFINT:   prec=PREC_DI; break;
		case PFDBL:   prec=PREC_DP; break;
		case PFCOMPLEX: prec=PREC_SP; type_dim=2; break;
		default:
			sprintf(DEFAULT_ERROR_STRING,
		"hips1_to_dp:  unsupported pixel format code %d",
				hd_p->pixel_format);
			NWARN(DEFAULT_ERROR_STRING);
			return(-1);
	}
	SET_OBJ_SEQS(dp, 1);
	SET_OBJ_FRAMES(dp, hd_p->num_frame);
	SET_OBJ_ROWS(dp, hd_p->rows);
	SET_OBJ_COLS(dp, hd_p->cols);
	SET_OBJ_COMPS(dp, type_dim);
	SET_OBJ_PREC_PTR(dp, prec_for_code(prec));

	return(0);
}

void hdr1_strs(Hips1_Header *hdp)
{
	hdp->orig_name=savestr("\n");
	hdp->seq_name=savestr("\n");
	hdp->orig_date=savestr("\n");
	hdp->bit_packing=0;
	hdp->seq_history=savestr("\n");
	hdp->seq_desc=savestr("\n");
}

FIO_OPEN_FUNC( hips1 )
{
	Image_File *ifp;

#ifdef QUIP_DEBUG
if( debug & debug_fileio ) advise("opening hips1 image file");
#endif /* QUIP_DEBUG */

	ifp = IMG_FILE_CREAT(name,rw,FILETYPE_FOR_CODE(IFT_HIPS1));
	/* img_file_creat creates dummy if_dp only if readable */

	if( ifp==NULL ) return(ifp);

#ifdef QUIP_DEBUG
if( debug & debug_fileio ) advise("allocating hips1 header");
#endif /* QUIP_DEBUG */

	ifp->if_hdr_p = getbuf( sizeof(Hips1_Header) );

	if( rw == FILE_READ ){
		/* BUG: should check for error here */
		if( ftch_header( QSP_ARG  ifp->if_fd, (Header *)ifp->if_hdr_p ) == 0 ){
			if( ifp->if_hdr_p != NULL ){
				givbuf(ifp->if_hdr_p);
				ifp->if_hdr_p=NULL;
			}
			hips1_close(QSP_ARG  ifp);
			return(NULL);
		}
		hips1_to_dp(ifp->if_dp,ifp->if_hdr_p);
	} else {
		hdr1_strs(ifp->if_hdr_p);		/* make null strings */
	}
	return(ifp);
}


int dp_to_hips1(Hips1_Header *hd_p,Data_Obj *dp)
{
	dimension_t size;

	/* BUG questionable cast */
	hd_p->rows = (int)OBJ_ROWS(dp);
	hd_p->cols = (int)OBJ_COLS(dp);
	hd_p->num_frame = (int)OBJ_FRAMES(dp);
	if( OBJ_PREC(dp) != PREC_SP && OBJ_COMPS(dp) > 1 ){
NWARN("HIPS1 extension does not support non-float multicomponent pixels");
		return(-1);
	}
	switch( OBJ_PREC(dp) ){
		case PREC_BY:
		case PREC_UBY:
			hd_p->pixel_format = PFBYTE;
			size=1;
			break;
		case PREC_IN:
			hd_p->pixel_format = PFSHORT;
			size=2;
			break;
		case PREC_DI:
			hd_p->pixel_format = PFINT;
			size=4;
			break;
		case PREC_DP:
			hd_p->pixel_format = PFDBL;
			size=8;
			break;
		case PREC_SP:
//#ifdef CAUTIOUS
//			if( OBJ_COMPS(dp)==0 ){
//				NWARN("CAUTIOUS:  Zero tdim!?");
//				return(-1);
//			}
//#endif /* CAUTIOUS */
			assert( OBJ_COMPS(dp) != 0 );

			if( OBJ_COMPS(dp) == 1 ){
				hd_p->pixel_format = PFFLOAT;
				size=sizeof(float);
			} else if( OBJ_COMPS(dp) == 2 ){
				hd_p->pixel_format = PFCOMPLEX;
				size=2*sizeof(float);
			} else {	/* OBJ_COMPS(dp) > 2 */
				/* This is a jbm extension to hips1:
				 * negative values for pixel_format
				 * are used to code multidimensional
				 * pixels with the dimension stored here.
				 * PFFLOAT is assumed.
				 */
				hd_p->pixel_format = (int)(OBJ_COMPS(dp) * -1);
				size=(OBJ_COMPS(dp)*sizeof(float));
			}
			break;
		default:
			NWARN("dp_to_hips1:  unsupported pixel format");
			return(-1);
	}
	hd_p->bits_per_pixel=(int)(8*size);
	return(0);
}


#define fpts(s,fd)	_fpts(QSP_ARG  s,fd)

static void _fpts(QSP_ARG_DECL  const char *s,int fd)
{
	size_t n;
	ssize_t n2;
	n=strlen(s);
	if( (n2=write(fd,s,n)) != n ){
		if( n2 < 0 ) tell_sys_error("write");

		sprintf(DEFAULT_ERROR_STRING,"error writing %ld header bytes (%s)",(long)n,s);
		NWARN(DEFAULT_ERROR_STRING);
	}
}

static void pt_header(QSP_ARG_DECL  int fd,Hips1_Header *hd)
{
	char str[256];

	if( *(hd->orig_name) == 0 ) fpts("\n",fd);
	else fpts(hd->orig_name,fd);
	if( *(hd->seq_name) == 0 ) fpts("\n",fd);
	else fpts(hd->seq_name,fd);

	/* make sure to print extra spaces with the number of
	 * frames, so we can possibly go in later and edit it
	 */
	sprintf(str,"%6d\n",hd->num_frame);
	fpts(str,fd);

	if( *(hd->orig_date) == 0 ) fpts("\n",fd);
	else fpts(hd->orig_date,fd);
	sprintf(str,"%d\n",hd->rows);
	fpts(str,fd);
	sprintf(str,"%d\n",hd->cols);
	fpts(str,fd);
	sprintf(str,"%d\n",hd->bits_per_pixel);
	fpts(str,fd);
	sprintf(str,"%d\n",hd->bit_packing);
	fpts(str,fd);
	sprintf(str,"%d\n",hd->pixel_format);
	fpts(str,fd);
	if( *(hd->seq_history) == 0 ) fpts("\n",fd);
	else fpts(hd->seq_history,fd);
	if( *(hd->seq_desc) == 0 ) fpts("\n",fd);
	else fpts(hd->seq_desc,fd);
	fpts("\n.\n",fd);
}

static int set_hdr(QSP_ARG_DECL  Image_File *ifp)		/* set header fields from image object */
{
	if( dp_to_hips1(ifp->if_hdr_p,ifp->if_dp) < 0 ){
		hips1_close(QSP_ARG  ifp);
		return(-1);
	}
	pt_header(QSP_ARG  ifp->if_fd,ifp->if_hdr_p);		/* write it out */
	return(0);
}

FIO_WT_FUNC(hips1)
{
	if( ifp->if_dp == NULL ){
		setup_dummy(ifp);
		copy_dimensions(ifp->if_dp , dp );
		/* reset nframes */
		SET_OBJ_FRAMES(ifp->if_dp, ifp->if_frms_to_wt);
		if( set_hdr(QSP_ARG  ifp) < 0 ) return(-1);
	} else if( !same_type(QSP_ARG  dp,ifp) ) return(-1);

	wt_raw_data(QSP_ARG  dp,ifp);
	return(0);
}

FIO_UNCONV_FUNC(hips1)
{
	Hips1_Header **hdr_pp;

	hdr_pp = (Hips1_Header **)hd_pp;

	/* allocate space for new header */

	*hdr_pp = (Hips1_Header *)getbuf( sizeof(Hips1_Header) );
	if( *hdr_pp == NULL ) return(-1);

	dp_to_hips1(*hdr_pp,dp);

	return(0);
}

FIO_CONV_FUNC(hips1)
{
	NWARN("hips1_conv not implemented");
	return(-1);
}

/* stuff from former put_hdr.c */

/* rewrite the number of frames in this header */

static void rewrite_hips1_nf(int fd,dimension_t n)
{
	int i;
	char c;
	char str[16];

	/* seek to beginning of file */
	if( lseek(fd,0L,SEEK_SET) != 0 ){
		NWARN("error seeking in header");
		return;
	}

	/* eat up the first two lines */

	for(i=0;i<2;i++){
		do {
			if( read(fd,&c,1) != 1 ){
				NWARN("error reading header char");
				return;
			}
		} while(c!='\n');
	}

	/* print the new nframes string */

	sprintf(str,"%6d",n);
	if( write(fd,str,6) != 6 ){
		NWARN("error overwriting header nframes");
		return;
	}
}

static void rls_header(Hips1_Header *hd)		/* free saved strings */
{
	rls_str(hd->orig_name);
	rls_str(hd->seq_name);
	rls_str(hd->orig_date);

	givbuf((void *)(hd->seq_history));
	givbuf((void *)(hd->seq_desc));
}

