
#include "quip_config.h"

#include <stdio.h>

#ifdef HAVE_STRING_H
#include <string.h>
#endif

#include "quip_prot.h"
#include "fio_prot.h"
#include "data_obj.h"
#include "img_file/jbm_ppm.h"
#include "img_file/raw.h"

FIO_FT_TO_DP_FUNC(ppm,Ppm_Header)
{
	short type_dim=1;
	Precision * prec_p;

	hd_p = (Ppm_Header *)hd_p;
	prec_p=PREC_FOR_CODE(PREC_UBY);
	switch( hd_p->format ){
		/* BUG should use symbolic constants here */
		case 5: type_dim=1; break;
		case 6: type_dim=3; break;
		default:
			sprintf(DEFAULT_ERROR_STRING,
		"ppm_to_dp:  unsupported pixel format code %d",
				hd_p->format);
			NWARN(DEFAULT_ERROR_STRING);
			return(-1);
	}
	SET_OBJ_PREC_PTR(dp, prec_p);

	SET_OBJ_COMPS(dp, type_dim);
	SET_OBJ_COLS(dp, hd_p->cols);
	SET_OBJ_ROWS(dp, hd_p->rows);
/*
//sprintf(DEFAULT_ERROR_STRING,"ppm_to_dp:  c = %d, r = %d",hd_p->cols,hd_p->rows);
//advise(DEFAULT_ERROR_STRING);
*/
	SET_OBJ_FRAMES(dp, 1);
	SET_OBJ_SEQS(dp, 1);

	SET_OBJ_COMP_INC(dp, 1);
	SET_OBJ_PXL_INC(dp, 1);
	SET_OBJ_ROW_INC(dp, OBJ_PXL_INC(dp) * (incr_t)OBJ_COLS(dp) );
	SET_OBJ_FRM_INC(dp, OBJ_ROW_INC(dp) * (incr_t)OBJ_ROWS(dp) );
	SET_OBJ_SEQ_INC(dp, OBJ_FRM_INC(dp) * (incr_t)OBJ_FRAMES(dp) );

	SET_OBJ_PARENT(dp, NULL);
	SET_OBJ_CHILDREN(dp, NULL);

	SET_OBJ_AREA(dp, ram_area_p);		/* the default */
	SET_OBJ_DATA_PTR(dp, hd_p->img_data);
	SET_OBJ_N_TYPE_ELTS(dp, OBJ_COMPS(dp) * OBJ_COLS(dp) * OBJ_ROWS(dp)
			* OBJ_FRAMES(dp) * OBJ_SEQS(dp) );

	auto_shape_flags(OBJ_SHAPE(dp),dp);

	return(0);
}

/* read a line, if it's a comment then keep reading...
 */

#define MAX_HDR_LLEN	128
static char hdr_line[MAX_HDR_LLEN];

static char *next_header_line(FILE *fp)
{
	do {
		if( fgets(hdr_line,MAX_HDR_LLEN,fp) == NULL ) return(NULL);
	} while( hdr_line[0] == '#' );
	return(hdr_line);
}

int rd_ppm_hdr(FILE *fp,Ppm_Header *hdp,const char *filename)
{
	int f;
	int r,c;
	int n;
	char *s;

	s=next_header_line(fp);
	if( s == NULL ){
		NWARN("missing ppm format code line");
		return(-1);
	}

	if( sscanf(s,"P%d",&f) != 1 ){
		sprintf(DEFAULT_ERROR_STRING,"error reading ppm format code, file %s",filename);
		NWARN(DEFAULT_ERROR_STRING);
		return(-1);
	}

	s=next_header_line(fp);
	if( s == NULL ){
		NWARN("missing ppm image size line");
		return(-1);
	}

	/* BUG???  ppm written by open office has the extra number on the same
	 * line as the row and column counts...
	 * For now, we hack it (breaking our old files), this needs to be done
	 * properly...
	 */
#ifdef FOOBAR
	if( sscanf(s,"%d %d",&c,&r) != 2 ){
		sprintf(DEFAULT_ERROR_STRING,"error reading ppm image size, file %s",filename);
		NWARN(DEFAULT_ERROR_STRING);
		return(-1);
	}

	s=next_header_line(fp);
	if( s == NULL ){
		NWARN("missing ppm extra number line");
		return(-1);
	}
#else
	if( sscanf(s,"%d %d %d",&c,&r,&n) != 3 ){
		sprintf(DEFAULT_ERROR_STRING,"error reading ppm image size + extra number, file %s",filename);
		NWARN(DEFAULT_ERROR_STRING);
		return(-1);
	}
	
#endif /* FOOBAR */

	if( sscanf(s,"%d",&n) != 1 ){
		sprintf(DEFAULT_ERROR_STRING,"error reading ppm extra number, file %s",filename);
		NWARN(DEFAULT_ERROR_STRING);
		return(-1);
	}
	hdp->format = f;
	hdp->rows = r;
	hdp->cols = c;
	hdp->somex = n;
	return(0);
}

FIO_OPEN_FUNC( ppm )
{
	Image_File *ifp;

	ifp = IMG_FILE_CREAT(name,rw,FILETYPE_FOR_CODE(IFT_PPM));
	if( ifp==NULL ) return(ifp);

	ifp->if_hdr_p = getbuf( sizeof(Ppm_Header) );

	if( IS_READABLE(ifp) ){
		if( rd_ppm_hdr( ifp->if_fp, (Ppm_Header *)ifp->if_hdr_p,
			ifp->if_name ) < 0 ){
			ppm_close(QSP_ARG  ifp);
			return(NULL);
		}
		ppm_to_dp(ifp->if_dp,ifp->if_hdr_p);
	}
	return(ifp);
}



FIO_CLOSE_FUNC( ppm )
{
	if( ifp->if_hdr_p != NULL ){
		givbuf(ifp->if_hdr_p);
	}
	GENERIC_IMGFILE_CLOSE(ifp);
}

FIO_DP_TO_FT_FUNC(ppm,Ppm_Header)
{
	if( OBJ_PREC(dp) != PREC_UBY ){
		sprintf(DEFAULT_ERROR_STRING,
		"Sorry, can only write unsigned byte images to PPM, object %s has prec %s",
			OBJ_NAME(dp),PREC_NAME(OBJ_PREC_PTR(dp)));
		NWARN(DEFAULT_ERROR_STRING);
		return(-1);
	}
	if( OBJ_FRAMES(dp)>1 || OBJ_SEQS(dp)>1 ){
		sprintf(DEFAULT_ERROR_STRING,
		"Sorry, object %s has more than 1 frame/seq, can only write 1 to PPM",
			OBJ_NAME(dp));
		NWARN(DEFAULT_ERROR_STRING);
		return(-1);
	}

	if( OBJ_COMPS(dp) == 1 ) 
		hd_p->format = 5;
	else if( OBJ_COMPS(dp) == 3 )
		hd_p->format = 6;
	else {
		NWARN("Sorry, PPM only supports type dimensions 1 and 3");
		return(-1);
	}
		
	hd_p->rows = (int)OBJ_ROWS(dp);
	hd_p->cols = (int)OBJ_COLS(dp);
	hd_p->somex = 255;

	hd_p->img_data = OBJ_DATA_PTR(dp);

	return(0);
}

void wt_ppm_hdr(FILE *fp,Ppm_Header *hdp,const char *filename)
{
	fprintf(fp,"P%d\n",hdp->format);
	/*
	fprintf(fp,"%d %d\n",hdp->cols,hdp->rows);
	fprintf(fp,"%d\n",hdp->somex);
	*/
	fprintf(fp,"%d %d %d\n",hdp->cols,hdp->rows,hdp->somex);
	fflush(fp);
}

FIO_SETHDR_FUNC( ppm ) /* set header fields from image object */
{
	if( FIO_DP_TO_FT_FUNC_NAME(ppm)(ifp->if_hdr_p,ifp->if_dp) < 0 ){
		ppm_close(QSP_ARG  ifp);
		return(-1);
	}
	wt_ppm_hdr(ifp->if_fp,ifp->if_hdr_p,ifp->if_name);	/* write it out */
	return(0);
}

FIO_RD_FUNC( ppm )
{
	raw_rd( QSP_ARG  dp, ifp, x_offset, y_offset, t_offset );
}

FIO_SEEK_FUNC( ppm )
{
	return std_seek_frame( QSP_ARG  ifp, n );
}

FIO_INFO_FUNC( ppm )
{
	// nop
}


FIO_WT_FUNC( ppm )	/** output next frame */
{
	/* PPM wants color images interleaved */

	if( ifp->if_dp == NULL ){	/* first time check, always true for ppm */
		setup_dummy(ifp);	/* create if_dp */
		copy_dimensions(ifp->if_dp, dp);
		if( set_ppm_hdr(QSP_ARG  ifp) < 0 ) return(-1);
	}

	wt_raw_data(QSP_ARG  dp,ifp);
	return(0);
}

int ppm_unconv(void *hdr_pp,Data_Obj *dp)
{
	Ppm_Header **hd_pp;

	hd_pp = (Ppm_Header **) hdr_pp;

	/* allocate space for new header */

	*hd_pp = (Ppm_Header *)getbuf( sizeof(Ppm_Header) );
	if( *hd_pp == NULL ) return(-1);

	//dp_to_ppm(*hd_pp,dp);
	FIO_DP_TO_FT_FUNC_NAME(ppm)(*hd_pp,dp);

	return(0);
}

int ppm_conv(Data_Obj *dp,void *hd_pp)
{
	NWARN("ppm_conv not implemented");
	return(-1);
}

/* eof */

/* Beau's dis (digital image sequence) format is like ppm,
 * but with an additional nframes field
 */

int dis_to_dp(Data_Obj *dp,Dis_Header *hd_p)
{
	short type_dim=1;
	Precision * prec_p;

	prec_p=PREC_FOR_CODE(PREC_UBY);
	switch( hd_p->format ){
		/* BUG should use symbolic constants here */
		case 5: type_dim=1; break;
		case 6: type_dim=3; break;
		default:
			sprintf(DEFAULT_ERROR_STRING,
		"ppm_to_dp:  unsupported pixel format code %d",
				hd_p->format);
			NWARN(DEFAULT_ERROR_STRING);
			return(-1);
	}
	SET_OBJ_PREC_PTR(dp, prec_p);

	SET_OBJ_COMPS(dp, type_dim);
	SET_OBJ_COLS(dp, hd_p->cols);
	SET_OBJ_ROWS(dp, hd_p->rows);
	SET_OBJ_FRAMES(dp, hd_p->frames);
	SET_OBJ_SEQS(dp, 1);

	SET_OBJ_COMP_INC(dp, 1);
	SET_OBJ_PXL_INC(dp, 1);
	SET_OBJ_ROW_INC(dp, OBJ_PXL_INC(dp) * (incr_t)OBJ_COLS(dp) );
	SET_OBJ_FRM_INC(dp, OBJ_ROW_INC(dp) * (incr_t)OBJ_ROWS(dp) );
	SET_OBJ_SEQ_INC(dp, OBJ_FRM_INC(dp) * (incr_t)OBJ_FRAMES(dp) );

	SET_OBJ_PARENT(dp, NULL);
	SET_OBJ_CHILDREN(dp, NULL);

	SET_OBJ_AREA(dp, ram_area_p);		/* the default */
	SET_OBJ_DATA_PTR(dp, hd_p->img_data);
	SET_OBJ_N_TYPE_ELTS(dp, OBJ_COMPS(dp) * OBJ_COLS(dp) * OBJ_ROWS(dp)
			* OBJ_FRAMES(dp) * OBJ_SEQS(dp) );

	auto_shape_flags(OBJ_SHAPE(dp),dp);

	return(0);
}

int rd_dis_hdr(FILE *fp,Dis_Header *hdp,const char *filename)
{
	int f;
	int r,c;
	int nf;
	int n;
	int ch;

	if( fscanf(fp,"P%d",&f) != 1 ){
		sprintf(DEFAULT_ERROR_STRING,"error reading dis format code, file %s",filename);
		NWARN(DEFAULT_ERROR_STRING);
		return(-1);
	}
	if( fscanf(fp,"%d %d %d",&c,&r,&nf) != 3 ){
		sprintf(DEFAULT_ERROR_STRING,"error reading dis sizes, file %s",filename);
		NWARN(DEFAULT_ERROR_STRING);
		return(-1);
	}
	if( fscanf(fp,"%d",&n) != 1 ){
		sprintf(DEFAULT_ERROR_STRING,"error reading dis extra number, file %s",filename);
		NWARN(DEFAULT_ERROR_STRING);
		return(-1);
	}
	ch = getc(fp);
	if( ch != '\r' && ch != '\n' ){
		NWARN("Bad char at end of .dis header!?");
		return(-1);
	}
	hdp->format = f;
	hdp->rows = r;
	hdp->cols = c;
	hdp->frames = nf;
	hdp->somex = n;
	return(0);
}

FIO_OPEN_FUNC( dis )
{
	Image_File *ifp;

	ifp = IMG_FILE_CREAT(name,rw,FILETYPE_FOR_CODE(IFT_PPM));
	if( ifp==NULL ) return(ifp);

	ifp->if_hdr_p = (Dis_Header *)getbuf( sizeof(Dis_Header) );

	if( IS_READABLE(ifp) ){
		if( rd_dis_hdr( ifp->if_fp, ifp->if_hdr_p,
			ifp->if_name ) < 0 ){
			dis_close(QSP_ARG  ifp);
			return(NULL);
		}
		dis_to_dp(ifp->if_dp,ifp->if_hdr_p);
	}
	return(ifp);
}



FIO_CLOSE_FUNC( dis )
{
	if( ifp->if_hdr_p != NULL ){
		givbuf(ifp->if_hdr_p);
	}
	GENERIC_IMGFILE_CLOSE(ifp);
}

//int dp_to_dis(Dis_Header *hd_p,Data_Obj *dp)
FIO_DP_TO_FT_FUNC(dis,Dis_Header)
{
	if( OBJ_PREC(dp) != PREC_UBY ){
		sprintf(DEFAULT_ERROR_STRING,
		"Sorry, can only write unsigned byte images to PPM, object %s has prec %s",
			OBJ_NAME(dp),PREC_NAME(OBJ_PREC_PTR(dp)));
		NWARN(DEFAULT_ERROR_STRING);
		return(-1);
	}
	if( OBJ_SEQS(dp)>1 ){
		sprintf(DEFAULT_ERROR_STRING,
		"Sorry, object %s has more than 1 seq, can only write 1 to .dis",
			OBJ_NAME(dp));
		NWARN(DEFAULT_ERROR_STRING);
		return(-1);
	}

	if( OBJ_COMPS(dp) == 1 ) 
		hd_p->format = 5;
	else if( OBJ_COMPS(dp) == 3 )
		hd_p->format = 6;
	else {
		NWARN("Sorry, .dis only supports type dimensions 1 and 3");
		return(-1);
	}
		
	hd_p->rows = (int)OBJ_ROWS(dp);
	hd_p->cols = (int)OBJ_COLS(dp);
	hd_p->frames = (int)OBJ_FRAMES(dp);
	hd_p->somex = 255;

	hd_p->img_data = OBJ_DATA_PTR(dp);

	return(0);
}

void wt_dis_hdr(FILE *fp,Dis_Header *hdp,const char *filename)
{
	fprintf(fp,"P%d\n",hdp->format);
	fprintf(fp,"%d %d %d\n",hdp->cols,hdp->rows,hdp->frames);
	fprintf(fp,"%d\n",hdp->somex);
	fflush(fp);
}

FIO_SETHDR_FUNC( dis )		/* set header fields from image object */
{
	if( FIO_DP_TO_FT_FUNC_NAME(dis)(ifp->if_hdr_p,ifp->if_dp) < 0 ){
		dis_close(QSP_ARG  ifp);
		return(-1);
	}
	wt_dis_hdr(ifp->if_fp,ifp->if_hdr_p,ifp->if_name);	/* write it out */
	return(0);
}

FIO_WT_FUNC( dis )
{
	/* PPM wants color images interleaved */

	if( ifp->if_dp == NULL ){	/* first time check, always true for ppm */
		setup_dummy(ifp);	/* create if_dp */
		copy_dimensions(ifp->if_dp, dp);
		if( set_dis_hdr(QSP_ARG  ifp) < 0 ) return(-1);
	}

	wt_raw_data(QSP_ARG  dp,ifp);
	return(0);
}

int dis_unconv(void *hdr_pp,Data_Obj *dp)
{
	Dis_Header **hd_pp;

	hd_pp = (Dis_Header **) hdr_pp;

	/* allocate space for new header */

	*hd_pp = (Dis_Header *)getbuf( sizeof(Dis_Header) );
	if( *hd_pp == NULL ) return(-1);

	FIO_DP_TO_FT_FUNC_NAME(dis)(*hd_pp,dp);

	return(0);
}

int dis_conv(Data_Obj *dp,void *hd_pp)
{
	NWARN("dis_conv not implemented");
	return(-1);
}

/* eof */

