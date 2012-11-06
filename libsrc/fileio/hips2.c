#include "quip_config.h"

char VersionId_fio_hips2[] = QUIP_VERSION_STRING;

#include <stdio.h>

#ifdef HAVE_STRING_H
#include <string.h>
#endif


#include "fio_prot.h"
#include "filetype.h"
#include "getbuf.h"
#include "data_obj.h"
#include "debug.h"
#include "savestr.h"
#include "hips2.h"
#include "raw.h"
#include "readhdr.h"

#define HDR_P(ifp)	((Image_File_Hdr *)ifp->if_hd)->ifh_u.hips2_hd_p

static int num_color=1;

static void rewrite_hips2_nf(FILE *fp,dimension_t n);

int hips2_to_dp(Data_Obj *dp,Hips2_Header *hd_p)
{
	short type_dim=1;
	short prec;

	switch( hd_p->pixel_format ){
		case PFBYTE:  prec=PREC_UBY; break;
		case PFSHORT: prec=PREC_IN; break;
		case PFFLOAT: prec=PREC_SP; break;
		case PFINT:   prec=PREC_DI; break;
		case PFDOUBLE:prec=PREC_DP; break;
		case PFCOMPLEX: prec=PREC_SP; type_dim=2; break;
		default:
			sprintf(DEFAULT_ERROR_STRING,
		"hips2_to_dp:  unsupported pixel format code %d",
				hd_p->pixel_format);
			NWARN(DEFAULT_ERROR_STRING);
			return(-1);
	}
	dp->dt_prec = prec;

	dp->dt_comps = type_dim;
	dp->dt_cols = hd_p->cols;
	dp->dt_rows = hd_p->rows;
	if( hd_p->numcolor != 1 ){
		if( type_dim != 1 )
			NWARN("Sorry, no complex pixels with multiple color planes");

		/* This header describes the layout of the data in the file,
		 * But now how we really want to represent it...
		 * We'd like to have better scheme!
		 */

		dp->dt_frames = hd_p->numcolor;
		dp->dt_seqs = hd_p->num_frame/hd_p->numcolor;
		/* what we'd really like to do is transpose these
		 * to go to our standard interlaced color format
		 */
	} else {
		dp->dt_frames = hd_p->num_frame;
		dp->dt_seqs = 1;
	}

	dp->dt_cinc = 1;
	dp->dt_pinc = 1;
	dp->dt_rowinc = dp->dt_pinc * (incr_t)hd_p->ocols ;
	dp->dt_finc = dp->dt_rowinc * (incr_t)dp->dt_rows;
	dp->dt_sinc = dp->dt_finc * (incr_t)dp->dt_frames;

	dp->dt_parent = NO_OBJ;
	dp->dt_children = NO_LIST;

	dp->dt_ap = ram_area;		/* the default */
	dp->dt_data = hd_p->image;
	dp->dt_n_type_elts = dp->dt_comps * dp->dt_cols * dp->dt_rows
			* dp->dt_frames * dp->dt_seqs;

	if( hd_p->pixel_format == PFCOMPLEX )
		dp->dt_flags |= DT_COMPLEX;

	set_shape_flags(&dp->dt_shape,dp);

	return(0);
}

void hdr2_strs(Hips2_Header *hdp)
{
	hdp->orig_name=savestr("\n");
	hdp->seq_name=savestr("\n");
	hdp->orig_date=savestr("\n");
	hdp->seq_history=savestr("\n");
	hdp->seq_desc=savestr("\n");
	hdp->params = NULL;
	hdp->sizehist = strlen(hdp->seq_history)+1;
	hdp->sizedesc = strlen(hdp->seq_desc)+1;
}

FIO_OPEN_FUNC(hips2_open)
//Image_File *		/**/
//hips2_open(const char *name,int rw)		/**/
{
	Image_File *ifp;

	ifp = IMAGE_FILE_OPEN(name,rw,IFT_HIPS2);
	if( ifp==NO_IMAGE_FILE ) return(ifp);

	ifp->if_hd = (Hips2_Header *)getbuf( sizeof(Hips2_Header) );

	null_hips2_hd(ifp->if_hd);

	if( IS_READABLE(ifp) ){
		/* BUG: should check for error here */

		/* An error can occur here if the file exists,
		 * but is empty...  we therefore initialize
		 * the header strings (above) with nulls, so
		 * that the program won't try to free nonsense
		 * addresses
		 */

		if( rd_hips2_hdr( ifp->if_fp, (Hips2_Header *)ifp->if_hd,
			ifp->if_name ) != HIPS_OK ){
			hips2_close(QSP_ARG  ifp);
			return(NO_IMAGE_FILE);
		}
		if( hips2_to_dp(ifp->if_dp,ifp->if_hd) < 0 )
			NWARN("error converting hips2 header");
	} else {	/* write file */
		hdr2_strs(ifp->if_hd);		/* make null strings */
	}
	return(ifp);
}


FIO_CLOSE_FUNC( hips2_close )
{
	/* see if we need to edit the header */
	if( IS_WRITABLE(ifp)
		&& ifp->if_dp != NO_OBJ	/* may be closing 'cause of error */
		&& ifp->if_nfrms != ifp->if_frms_to_wt ){
		if( ifp->if_nfrms <= 0 ){
			sprintf(error_string, "file %s nframes=%d!?",
				ifp->if_name,ifp->if_nfrms);
			NWARN(error_string);
		}
		rewrite_hips2_nf(ifp->if_fp,ifp->if_nfrms);
	}

	if( ifp->if_hd != NULL ){
		rls_hips2_hd(ifp->if_hd);		/* free strings */
		givbuf(ifp->if_hd);
	}

	GENERIC_IMGFILE_CLOSE(ifp);
}

int dp_to_hips2(Hips2_Header *hd_p,Data_Obj *dp)
{
	dimension_t size;

	/* num_frame set when when write request given */

	/*
	 * the following is a kludge, but it's based on an even worse
	 * kludge in the way HIPS2 treats color sequences
	 */

	if( dp->dt_seqs > 1 || num_color > 1 )	/* the kludge is on! */
		hd_p->numcolor = (int)dp->dt_frames;
	else
		hd_p->numcolor = 1;

	/* BUG questionable cast */
	hd_p->num_frame = (int)(dp->dt_frames * dp->dt_seqs);
	hd_p->rows = (int)dp->dt_rows;
	hd_p->cols = (int)dp->dt_cols;
	hd_p->orows = (int)dp->dt_rows;
	hd_p->ocols = (int)dp->dt_cols;
	hd_p->frow = 0;
	hd_p->fcol = 0;
	hd_p->numpix = (int)(dp->dt_rows*dp->dt_cols);

	hd_p->firstpix = 
	hd_p->image = (h_byte *) dp->dt_data;

	/* BUG should do something more sensible here... */
	if( hd_p->orig_name != NULL ){
		rls_str(hd_p->orig_name);
	}
	hd_p->orig_name = savestr(dp->dt_name);
	if( hd_p->seq_name != NULL ){
		rls_str(hd_p->seq_name);
	}
	hd_p->seq_name = savestr(dp->dt_name);

	/*
	hd_p->orig_date = savestr("\n");
	hd_p->seq_history = savestr("\n");
	hd_p->seq_desc = savestr("\n");
	*/

	switch( dp->dt_prec ){
		case PREC_BY:
		case PREC_UBY:
			hd_p->pixel_format = PFBYTE;
			size=sizeof(char);
			break;
		case PREC_IN:
		case PREC_UIN:
			hd_p->pixel_format = PFSHORT;
			size=sizeof(short);
			break;
		case PREC_DI:
		case PREC_UDI:
			hd_p->pixel_format = PFINT;
			size=sizeof(long);
			break;
		case PREC_DP:
			hd_p->pixel_format = PFDOUBLE;
			size=sizeof(double);
			if( IS_COMPLEX(dp) )
		NWARN("sorry, hips2 doesn't support double complex");
			break;
		case PREC_SP:
			if( IS_COMPLEX(dp) ){
				hd_p->pixel_format = PFCOMPLEX;
				size=2*sizeof(float);
			} else {
				hd_p->pixel_format = PFFLOAT;
				size=sizeof(float);
			}
			break;
		default:
			NWARN("dp_to_hips2:  unsupported pixel format");
			return(-1);
	}
	hd_p->sizepix=size;
	hd_p->sizeimage = hd_p->numpix * size;
	hd_p->numparam = 0;

	hd_p->imdealloc = 0;
	hd_p->histdealloc = 0;
	hd_p->seqddealloc = 0;
	hd_p->paramdealloc = 0;

	return(0);
}

int set_hips2_hdr(QSP_ARG_DECL  Image_File *ifp)		/* set header fields from image object */
{
	if( dp_to_hips2(ifp->if_hd,ifp->if_dp) < 0 ){
		hips2_close(QSP_ARG  ifp);
		return(-1);
	}
	wt_hips2_hdr(ifp->if_fp,ifp->if_hd,ifp->if_name);	/* write it out */
	return(0);
}

FIO_WT_FUNC( hips2_wt )
{
	Data_Obj dobj;

	/*
	 * since HIPS2 doesn't support interleaved color images,
	 * twiddle the increments to make it come out sequentially.
	 */

	num_color=1;	/* the default */

	if( dp->dt_comps > 1 ){
		if( dp->dt_seqs > 1 ){
	NWARN("can't write color hypersequences in HIPS2 format");
			return(-1);
		}
		dobj = *dp;	/* copy the thing */

		num_color=(int)dp->dt_comps;

		gen_xpose(&dobj,4,3);		/* seqs <--> frames */
		gen_xpose(&dobj,0,3);		/* comps <--> frames */
		dp = &dobj;
	}

	if( ifp->if_dp == NO_OBJ ){	/* first time? */

		/* set the rows & columns in our file struct */
		setup_dummy(ifp);
		copy_dimensions(ifp->if_dp, dp);

		/*
		 * remember # of frames from the HIPS header
		 *
		 * the number of frames we want to write to the file is
		 * in the hips header, but it needs to be multiplied
		 * by the number of components.  The number of frames
		 * (seqs) in the if_dp may not match, since we may be
		 * writing a frame at a time.
		 */

		if( num_color > 1 ){
			ifp->if_dp->dt_frames = num_color;
			ifp->if_dp->dt_seqs = ifp->if_frms_to_wt;
			ifp->if_frms_to_wt *= num_color;
		} else {
			ifp->if_dp->dt_frames = ifp->if_frms_to_wt;
			ifp->if_dp->dt_seqs = 1;
		}

		if( set_hips2_hdr(QSP_ARG  ifp) < 0 ) return(-1);

	} else if( !same_type(QSP_ARG  dp,ifp) ) return(-1);

	wt_raw_data(QSP_ARG  dp,ifp);
	return(0);
}

FIO_RD_FUNC( hips2_rd )
{
	Data_Obj *rd_dp;

	if( dp->dt_comps > 1 ){	/* color kludge */
		if( dp->dt_seqs > 1 ){
			NWARN("Sorry, can't convert color hyperseqs from HIPS2");
			return;
		}
if( verbose ) advise("transposing data to make interleaved components");
		rd_dp = dup_obj(QSP_ARG  dp,"tmp_xpose_obj");
		gen_xpose(rd_dp,3,0);
		gen_xpose(rd_dp,3,4);
	} else rd_dp = dp;

	raw_rd(QSP_ARG  rd_dp,ifp,x_offset,y_offset,t_offset);

	if( rd_dp != dp ){
		gen_xpose(rd_dp,4,3);		/* seqs <--> frames */
		gen_xpose(rd_dp,0,3);		/* comps <--> frames */
		dp_copy(QSP_ARG  dp,rd_dp);
		delvec(QSP_ARG  rd_dp);
	}
}

int hips2_unconv(void *hdr_pp,Data_Obj *dp)
{
	Hips2_Header **hd_pp;

	hd_pp = (Hips2_Header **) hdr_pp;

	/* allocate space for new header */

	*hd_pp = (Hips2_Header *)getbuf( sizeof(Hips2_Header) );
	if( *hd_pp == NULL ) return(-1);

	dp_to_hips2(*hd_pp,dp);

	return(0);
}

int hips2_conv(Data_Obj *dp,void *hd_pp)
{
	NWARN("hips2_conv not implemented");
	return(-1);
}

/* rewrite the number of frames in this header */

static void rewrite_hips2_nf(FILE *fp,dimension_t n)
{
	int i;
	int c;
	char str[16];

	/* seek to beginning of file */
	if( fseek(fp,0L,0) != 0 ){
		NWARN("error seeking to hips2 header");
		return;
	}

	/* eat up the first THREE lines;
	 * hips2 has a HIPS\n magic number line,
	 * and then the two name lines, and then the frame count
	 */

	for(i=0;i<3;i++){
		do {
			if( (c=getc(fp)) == EOF ){
				NWARN("error reading header char");
				return;
			}
		} while(c!='\n');
	}

	/* print the new nframes string */

	sprintf(str,"%6d",n);
	if( fwrite(str,1,6,fp) != 6 ){
		NWARN("error overwriting header nframes");
		return;
	}
}
