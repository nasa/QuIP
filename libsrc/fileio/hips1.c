#include "quip_config.h"

char VersionId_fio_hips1[] = QUIP_VERSION_STRING;

#include <stdio.h>

#ifdef HAVE_STRING_H
#include <string.h>
#endif

#include "fio_prot.h"
#include "filetype.h"
#include "getbuf.h"
#include "data_obj.h"
#include "debug.h"
#include "hips1.h"
#include "get_hdr.h"
#include "savestr.h"
#include "raw.h"
#include "hipl_fmt.h"
#include "uio.h"

#define HDR_P(ifp)	((Image_File_Hdr *)ifp->if_hd)->ifh_u.hips1_hd_p


/* local prototypes */

static void fpts(const char *,int);
static void rewrite_hips1_nf(int fd,dimension_t n);
static void pt_header(int fd,Hips1_Header *hd);
static void rls_header(Hips1_Header *hd);

FIO_CLOSE_FUNC( hips1_close )
{
	/* see if we need to edit the header */
	if( IS_WRITABLE(ifp)
		&& ifp->if_dp != NO_OBJ	/* may be closing 'cause of error */
		&& ifp->if_nfrms != ifp->if_frms_to_wt ){
		if( ifp->if_nfrms <= 0 ){
			sprintf(error_string, "file %s nframes=%d!?",
				ifp->if_name,ifp->if_nfrms);
			WARN(error_string);
		}
		rewrite_hips1_nf(ifp->if_fd,ifp->if_nfrms);
	}

	if( ifp->if_hd != NULL ){
#ifdef DEBUG
if( debug & debug_fileio ) advise("freeing hips1 header strings");
#endif
		rls_header(ifp->if_hd);		/* free strings */
#ifdef DEBUG
if( debug & debug_fileio ) advise("freeing hips1 header struct");
#endif
		givbuf(ifp->if_hd);
	}
	GENERIC_IMGFILE_CLOSE(ifp);
}

int hips1_to_dp(Data_Obj *dp,Hips1_Header *hd_p)
{
	short type_dim=1;
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
	dp->dt_seqs = 1;
	dp->dt_frames = hd_p->num_frame;
	dp->dt_rows = hd_p->rows;
	dp->dt_cols = hd_p->cols;
	dp->dt_comps = type_dim;
	dp->dt_prec = prec;

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

FIO_OPEN_FUNC( hips1_open )
{
	Image_File *ifp;

#ifdef DEBUG
if( debug & debug_fileio ) advise("opening hips1 image file");
#endif /* DEBUG */

	ifp = IMAGE_FILE_OPEN(name,rw,IFT_HIPS1);
	/* image_file_open creates dummy if_dp only if readable */

	if( ifp==NO_IMAGE_FILE ) return(ifp);

#ifdef DEBUG
if( debug & debug_fileio ) advise("allocating hips1 header");
#endif /* DEBUG */

	ifp->if_hd = getbuf( sizeof(Hips1_Header) );

	if( rw == FILE_READ ){
		/* BUG: should check for error here */
		if( ftch_header( ifp->if_fd, (Header *)ifp->if_hd ) == 0 ){
			if( ifp->if_hd != NULL ){
				givbuf(ifp->if_hd);
				ifp->if_hd=NULL;
			}
			hips1_close(QSP_ARG  ifp);
			return(NO_IMAGE_FILE);
		}
		hips1_to_dp(ifp->if_dp,ifp->if_hd);
	} else {
		hdr1_strs(ifp->if_hd);		/* make null strings */
	}
	return(ifp);
}


int dp_to_hips1(Hips1_Header *hd_p,Data_Obj *dp)
{
	dimension_t size;

	/* BUG questionable cast */
	hd_p->rows = (int)dp->dt_rows;
	hd_p->cols = (int)dp->dt_cols;
	hd_p->num_frame = (int)dp->dt_frames;
	if( dp->dt_prec != PREC_SP && dp->dt_comps > 1 ){
NWARN("HIPS1 extension does not support non-float multicomponent pixels");
		return(-1);
	}
	switch( dp->dt_prec ){
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
#ifdef CAUTIOUS
			if( dp->dt_comps==0 ){
				NWARN("CAUTIOUS:  Zero tdim!?");
				return(-1);
			}
#endif /* CAUTIOUS */
			if( dp->dt_comps == 1 ){
				hd_p->pixel_format = PFFLOAT;
				size=sizeof(float);
			} else if( dp->dt_comps == 2 ){
				hd_p->pixel_format = PFCOMPLEX;
				size=2*sizeof(float);
			} else {	/* dp->dt_comps > 2 */
				/* This is a jbm extension to hips1:
				 * negative values for pixel_format
				 * are used to code multidimensional
				 * pixels with the dimension stored here.
				 * PFFLOAT is assumed.
				 */
				hd_p->pixel_format = (int)(dp->dt_comps * -1);
				size=(dp->dt_comps*sizeof(float));
			}
			break;
		default:
			NWARN("dp_to_hips1:  unsupported pixel format");
			return(-1);
	}
	hd_p->bits_per_pixel=(int)(8*size);
	return(0);
}

int set_hdr(QSP_ARG_DECL  Image_File *ifp)		/* set header fields from image object */
{
	if( dp_to_hips1(ifp->if_hd,ifp->if_dp) < 0 ){
		hips1_close(QSP_ARG  ifp);
		return(-1);
	}
	pt_header(ifp->if_fd,ifp->if_hd);		/* write it out */
	return(0);
}

int hips1_wt(QSP_ARG_DECL  Data_Obj *dp,Image_File *ifp)		/** output next frame */
{
	if( ifp->if_dp == NO_OBJ ){
		setup_dummy(ifp);
		copy_dimensions(ifp->if_dp , dp );
		/* reset nframes */
		ifp->if_dp->dt_frames = ifp->if_frms_to_wt;
		if( set_hdr(QSP_ARG  ifp) < 0 ) return(-1);
	} else if( !same_type(QSP_ARG  dp,ifp) ) return(-1);

	wt_raw_data(QSP_ARG  dp,ifp);
	return(0);
}

int hips1_unconv(void *hdr_pp,Data_Obj *dp)
{
	Hips1_Header **hd_pp;

	hd_pp = (Hips1_Header **)hdr_pp;

	/* allocate space for new header */

	*hd_pp = (Hips1_Header *)getbuf( sizeof(Hips1_Header) );
	if( *hd_pp == NULL ) return(-1);

	dp_to_hips1(*hd_pp,dp);

	return(0);
}

int hips1_conv(Data_Obj *dp,void *hd_pp)
{
	NWARN("hips1_conv not implemented");
	return(-1);
}

/* stuff from former put_hdr.c */


static void fpts(const char *s,int fd)
{
	int n,n2;
	n=strlen(s);
	if( (n2=write(fd,s,n)) != n ){
		if( n2 < 0 ) tell_sys_error("write");

		sprintf(DEFAULT_ERROR_STRING,"error writing %d header bytes (%s)",n,s);
		NWARN(DEFAULT_ERROR_STRING);
	}
}

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

static void pt_header(int fd,Hips1_Header *hd)
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

static void rls_header(Hips1_Header *hd)		/* free saved strings */
{
	rls_str(hd->orig_name);
	rls_str(hd->seq_name);
	rls_str(hd->orig_date);

	givbuf(hd->seq_history);
	givbuf(hd->seq_desc);
}

