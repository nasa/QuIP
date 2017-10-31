#include "quip_config.h"

#include <stdio.h>

#ifdef HAVE_STRING_H
#include <string.h>
#endif


#include "fio_prot.h"
#include "quip_prot.h"
#include "getbuf.h"
#include "data_obj.h"
#include "debug.h"
#include "img_file/raw.h"
#include "hips/hips2.h"
#include "hips/readhdr.h"

static int num_color=1;

// Apparently, hips2 cannot handle multi-component pixels?
// Color is handled with separate color frames???

int _hips2_to_dp(QSP_ARG_DECL  Data_Obj *dp,Hips2_Header *hd_p)
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
			sprintf(ERROR_STRING,
		"hips2_to_dp:  unsupported pixel format code %d",
				hd_p->pixel_format);
			warn(ERROR_STRING);
			return(-1);
	}
	SET_OBJ_PREC_PTR(dp,prec_for_code(prec));

	SET_OBJ_COMPS(dp, type_dim);
	SET_OBJ_COLS(dp, hd_p->cols);
	SET_OBJ_ROWS(dp, hd_p->rows);
	if( hd_p->numcolor != 1 ){
		if( type_dim != 1 )
			warn("Sorry, no complex pixels with multiple color planes");

		/* This header describes the layout of the data in the file,
		 * But now how we really want to represent it...
		 * We'd like to have better scheme!
		 */

		SET_OBJ_FRAMES(dp, hd_p->numcolor);
		SET_OBJ_SEQS(dp, hd_p->num_frame/hd_p->numcolor);
		/* what we'd really like to do is transpose these
		 * to go to our standard interlaced color format
		 */
	} else {
		SET_OBJ_FRAMES(dp, hd_p->num_frame);
		SET_OBJ_SEQS(dp, 1);
	}

	SET_OBJ_COMP_INC(dp, 1);
	SET_OBJ_PXL_INC(dp, 1);
	SET_OBJ_ROW_INC(dp, OBJ_PXL_INC(dp) * (incr_t)hd_p->ocols );
	SET_OBJ_FRM_INC(dp, OBJ_ROW_INC(dp) * (incr_t)OBJ_ROWS(dp) );
	SET_OBJ_SEQ_INC(dp, OBJ_FRM_INC(dp) * (incr_t)OBJ_FRAMES(dp));

	dp->dt_parent = NULL;
	dp->dt_children = NULL;

	// BUG fix ram_area
	//dp->dt_ap = ram_area;		/* the default */

	SET_OBJ_DATA_PTR(dp, hd_p->image);
	SET_OBJ_N_TYPE_ELTS(dp, OBJ_COMPS(dp) * OBJ_COLS(dp) * OBJ_ROWS(dp)
			* OBJ_FRAMES(dp) * OBJ_SEQS(dp) );

	if( hd_p->pixel_format == PFCOMPLEX )
		SET_OBJ_FLAG_BITS(dp, DT_COMPLEX);

	auto_shape_flags(OBJ_SHAPE(dp));

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
	hdp->sizehist = (int) strlen(hdp->seq_history)+1;
	hdp->sizedesc = (int) strlen(hdp->seq_desc)+1;
}

FIO_OPEN_FUNC(hips2)
{
	Image_File *ifp;

	ifp = IMG_FILE_CREAT(name,rw,FILETYPE_FOR_CODE(IFT_HIPS2));
	if( ifp==NULL ) return(ifp);

	ifp->if_hdr_p = (Hips2_Header *)getbuf( sizeof(Hips2_Header) );

	null_hips2_hd(ifp->if_hdr_p);

	if( IS_READABLE(ifp) ){
		/* BUG: should check for error here */

		/* An error can occur here if the file exists,
		 * but is empty...  we therefore initialize
		 * the header strings (above) with nulls, so
		 * that the program won't try to free nonsense
		 * addresses
		 */

		if( rd_hips2_hdr( ifp->if_fp, (Hips2_Header *)ifp->if_hdr_p,
			ifp->if_name ) != HIPS_OK ){
			hips2_close(QSP_ARG  ifp);
			return(NULL);
		}
		if( hips2_to_dp(ifp->if_dp,ifp->if_hdr_p) < 0 )
			warn("error converting hips2 header");
	} else {	/* write file */
		hdr2_strs(ifp->if_hdr_p);		/* make null strings */
	}
	return(ifp);
}


/* rewrite the number of frames in this header */

#define rewrite_hips2_nf(fp,n) _rewrite_hips2_nf(QSP_ARG  fp,n)

static void _rewrite_hips2_nf(QSP_ARG_DECL  FILE *fp,dimension_t n)
{
	int i;
	int c;
	char str[16];

	/* seek to beginning of file */
	if( fseek(fp,0L,0) != 0 ){
		warn("error seeking to hips2 header");
		return;
	}

	/* eat up the first THREE lines;
	 * hips2 has a HIPS\n magic number line,
	 * and then the two name lines, and then the frame count
	 */

	for(i=0;i<3;i++){
		do {
			if( (c=getc(fp)) == EOF ){
				warn("error reading header char");
				return;
			}
		} while(c!='\n');
	}

	/* print the new nframes string */

	sprintf(str,"%6d",n);
	if( fwrite(str,1,6,fp) != 6 ){
		warn("error overwriting header nframes");
		return;
	}
}

FIO_CLOSE_FUNC( hips2 )
{
	/* see if we need to edit the header */
	if( IS_WRITABLE(ifp)
		&& ifp->if_dp != NULL	/* may be closing 'cause of error */
		&& ifp->if_nfrms != ifp->if_frms_to_wt ){
		if( ifp->if_nfrms <= 0 ){
			sprintf(ERROR_STRING, "file %s nframes=%d!?",
				ifp->if_name,ifp->if_nfrms);
			warn(ERROR_STRING);
		}
		rewrite_hips2_nf(ifp->if_fp,ifp->if_nfrms);
	}

	if( ifp->if_hdr_p != NULL ){
		rls_hips2_hd(ifp->if_hdr_p);		/* free strings */
		givbuf(ifp->if_hdr_p);
	}

	GENERIC_IMGFILE_CLOSE(ifp);
}

int _dp_to_hips2(QSP_ARG_DECL  Hips2_Header *hd_p,Data_Obj *dp)
{
	dimension_t size;

	/* num_frame set when when write request given */

	/*
	 * the following is a kludge, but it's based on an even worse
	 * kludge in the way HIPS2 treats color sequences
	 */

	if( OBJ_SEQS(dp) > 1 || num_color > 1 )	/* the kludge is on! */
		hd_p->numcolor = (int)OBJ_FRAMES(dp);
	else
		hd_p->numcolor = 1;

	/* BUG questionable cast */
	hd_p->num_frame = (int)(OBJ_FRAMES(dp) * OBJ_SEQS(dp));
	hd_p->rows = (int)OBJ_ROWS(dp);
	hd_p->cols = (int)OBJ_COLS(dp);
	hd_p->orows = (int)OBJ_ROWS(dp);
	hd_p->ocols = (int)OBJ_COLS(dp);
	hd_p->frow = 0;
	hd_p->fcol = 0;
	hd_p->numpix = (int)(OBJ_ROWS(dp)*OBJ_COLS(dp));

	hd_p->firstpix = 
	hd_p->image = (h_byte *) OBJ_DATA_PTR(dp);

	/* BUG should do something more sensible here... */
	if( hd_p->orig_name != NULL ){
		rls_str(hd_p->orig_name);
	}
	hd_p->orig_name = savestr(OBJ_NAME(dp));
	if( hd_p->seq_name != NULL ){
		rls_str(hd_p->seq_name);
	}
	hd_p->seq_name = savestr(OBJ_NAME(dp));

	/*
	hd_p->orig_date = savestr("\n");
	hd_p->seq_history = savestr("\n");
	hd_p->seq_desc = savestr("\n");
	*/

	switch( OBJ_PREC(dp) ){
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
		warn("sorry, hips2 doesn't support double complex");
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
			warn("dp_to_hips2:  unsupported pixel format");
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
	if( dp_to_hips2(ifp->if_hdr_p,ifp->if_dp) < 0 ){
		hips2_close(QSP_ARG  ifp);
		return(-1);
	}
	wt_hips2_hdr(ifp->if_fp,ifp->if_hdr_p,ifp->if_name);	/* write it out */
	return(0);
}

FIO_WT_FUNC( hips2 )
{
	Data_Obj dobj;

	/*
	 * since HIPS2 doesn't support interleaved color images,
	 * twiddle the increments to make it come out sequentially.
	 */

	num_color=1;	/* the default */

	if( OBJ_COMPS(dp) > 1 ){
		if( OBJ_SEQS(dp) > 1 ){
	warn("can't write color hypersequences in HIPS2 format");
			return(-1);
		}
		dobj = *dp;	/* copy the thing */

		num_color=(int)OBJ_COMPS(dp);

		gen_xpose(&dobj,4,3);		/* seqs <--> frames */
		gen_xpose(&dobj,0,3);		/* comps <--> frames */
		dp = &dobj;
	}

	if( ifp->if_dp == NULL ){	/* first time? */

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
			SET_OBJ_FRAMES(ifp->if_dp, num_color);
			SET_OBJ_SEQS(ifp->if_dp, ifp->if_frms_to_wt);
			ifp->if_frms_to_wt *= num_color;
		} else {
			SET_OBJ_FRAMES(ifp->if_dp, ifp->if_frms_to_wt);
			SET_OBJ_SEQS(ifp->if_dp, 1);
		}

		if( set_hips2_hdr(QSP_ARG  ifp) < 0 ) return(-1);

	} else if( !same_type(QSP_ARG  dp,ifp) ) return(-1);

	wt_raw_data(QSP_ARG  dp,ifp);
	return(0);
}

FIO_RD_FUNC( hips2 )
{
	Data_Obj *rd_dp;

	if( OBJ_COMPS(dp) > 1 ){	/* color kludge */
		if( OBJ_SEQS(dp) > 1 ){
			warn("Sorry, can't convert color hyperseqs from HIPS2");
			return;
		}
if( verbose ) advise("transposing data to make interleaved components");
		rd_dp = dup_obj(QSP_ARG  dp,"tmp_xpose_obj");
		gen_xpose(rd_dp,3,0);
		gen_xpose(rd_dp,3,4);
	} else rd_dp = dp;

	FIO_RD_FUNC_NAME(raw)(QSP_ARG  rd_dp,ifp,x_offset,y_offset,t_offset);

	if( rd_dp != dp ){
		gen_xpose(rd_dp,4,3);		/* seqs <--> frames */
		gen_xpose(rd_dp,0,3);		/* comps <--> frames */
		dp_copy(dp,rd_dp);
		delvec(rd_dp);
	}
}

int _hips2_unconv(QSP_ARG_DECL  void *hdr_pp,Data_Obj *dp)
{
	Hips2_Header **hd_pp;

	hd_pp = (Hips2_Header **) hdr_pp;

	/* allocate space for new header */

	*hd_pp = (Hips2_Header *)getbuf( sizeof(Hips2_Header) );
	if( *hd_pp == NULL ) return(-1);

	dp_to_hips2(*hd_pp,dp);

	return(0);
}

int _hips2_conv(QSP_ARG_DECL  Data_Obj *dp,void *hd_pp)
{
	warn("hips2_conv not implemented");
	return(-1);
}

