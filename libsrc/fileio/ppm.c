
#include "quip_config.h"

char VersionId_fio_ppm[] = QUIP_VERSION_STRING;


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
#include "jbm_ppm.h"
#include "raw.h"
#include "readhdr.h"

#define HDR_P(ifp)	((Image_File_Hdr *)ifp->if_hd)->ifh_u.ppm_hd_p
#define DHDR_P(ifp)	((Image_File_Hdr *)ifp->if_hd)->ifh_u.dis_hd_p

int ppm_to_dp(Data_Obj *dp,Ppm_Header *hd_p)
{
	short type_dim=1;
	short prec;

	hd_p = (Ppm_Header *)hd_p;
	prec=PREC_UBY;
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
	dp->dt_prec = prec;

	dp->dt_comps = type_dim;
	dp->dt_cols = hd_p->cols;
	dp->dt_rows = hd_p->rows;
/*
//sprintf(DEFAULT_ERROR_STRING,"ppm_to_dp:  c = %d, r = %d",hd_p->cols,hd_p->rows);
//advise(DEFAULT_ERROR_STRING);
*/
	dp->dt_frames = 1;
	dp->dt_seqs = 1;

	dp->dt_cinc = 1;
	dp->dt_pinc = 1;
	dp->dt_rowinc = dp->dt_pinc * (incr_t)dp->dt_cols;
	dp->dt_finc = dp->dt_rowinc * (incr_t)dp->dt_rows;
	dp->dt_sinc = dp->dt_finc * (incr_t)dp->dt_frames;

	dp->dt_parent = NO_OBJ;
	dp->dt_children = NO_LIST;

	dp->dt_ap = ram_area;		/* the default */
	dp->dt_data = hd_p->img_data;
	dp->dt_n_type_elts = dp->dt_comps * dp->dt_cols * dp->dt_rows
			* dp->dt_frames * dp->dt_seqs;

	set_shape_flags(&dp->dt_shape,dp,AUTO_SHAPE);

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

FIO_OPEN_FUNC( ppm_open )
{
	Image_File *ifp;

	ifp = IMAGE_FILE_OPEN(name,rw,IFT_PPM);
	if( ifp==NO_IMAGE_FILE ) return(ifp);

	ifp->if_hd = getbuf( sizeof(Ppm_Header) );

	if( IS_READABLE(ifp) ){
		if( rd_ppm_hdr( ifp->if_fp, (Ppm_Header *)ifp->if_hd,
			ifp->if_name ) < 0 ){
			ppm_close(QSP_ARG  ifp);
			return(NO_IMAGE_FILE);
		}
		ppm_to_dp(ifp->if_dp,ifp->if_hd);
	}
	return(ifp);
}



FIO_CLOSE_FUNC( ppm_close )
{
	if( ifp->if_hd != NULL ){
		givbuf(ifp->if_hd);
	}
	GENERIC_IMGFILE_CLOSE(ifp);
}

int dp_to_ppm(Ppm_Header *hd_p,Data_Obj *dp)
{
	if( dp->dt_prec != PREC_UBY ){
		sprintf(DEFAULT_ERROR_STRING,
		"Sorry, can only write unsigned byte images to PPM, object %s has prec %s",
			dp->dt_name,prec_name[dp->dt_prec]);
		NWARN(DEFAULT_ERROR_STRING);
		return(-1);
	}
	if( dp->dt_frames>1 || dp->dt_seqs>1 ){
		sprintf(DEFAULT_ERROR_STRING,
		"Sorry, object %s has more than 1 frame/seq, can only write 1 to PPM",
			dp->dt_name);
		NWARN(DEFAULT_ERROR_STRING);
		return(-1);
	}

	if( dp->dt_comps == 1 ) 
		hd_p->format = 5;
	else if( dp->dt_comps == 3 )
		hd_p->format = 6;
	else {
		NWARN("Sorry, PPM only supports type dimensions 1 and 3");
		return(-1);
	}
		
	hd_p->rows = (int)dp->dt_rows;
	hd_p->cols = (int)dp->dt_cols;
	hd_p->somex = 255;

	hd_p->img_data = dp->dt_data;

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

FIO_SETHDR_FUNC( set_ppm_hdr ) /* set header fields from image object */
{
	if( dp_to_ppm(ifp->if_hd,ifp->if_dp) < 0 ){
		ppm_close(QSP_ARG  ifp);
		return(-1);
	}
	wt_ppm_hdr(ifp->if_fp,ifp->if_hd,ifp->if_name);	/* write it out */
	return(0);
}

FIO_WT_FUNC( ppm_wt )	/** output next frame */
{
	/* PPM wants color images interleaved */

	if( ifp->if_dp == NO_OBJ ){	/* first time check, always true for ppm */
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

	dp_to_ppm(*hd_pp,dp);

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
	short prec;

	prec=PREC_UBY;
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
	dp->dt_prec = prec;

	dp->dt_comps = type_dim;
	dp->dt_cols = hd_p->cols;
	dp->dt_rows = hd_p->rows;
	dp->dt_frames = hd_p->frames;
	dp->dt_seqs = 1;

	dp->dt_cinc = 1;
	dp->dt_pinc = 1;
	dp->dt_rowinc = dp->dt_pinc * (incr_t)dp->dt_cols ;
	dp->dt_finc = dp->dt_rowinc * (incr_t)dp->dt_rows;
	dp->dt_sinc = dp->dt_finc * (incr_t)dp->dt_frames;

	dp->dt_parent = NO_OBJ;
	dp->dt_children = NO_LIST;

	dp->dt_ap = ram_area;		/* the default */
	dp->dt_data = hd_p->img_data;
	dp->dt_n_type_elts = dp->dt_comps * dp->dt_cols * dp->dt_rows
			* dp->dt_frames * dp->dt_seqs;

	set_shape_flags(&dp->dt_shape,dp,AUTO_SHAPE);

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

FIO_OPEN_FUNC( dis_open )
{
	Image_File *ifp;

	ifp = IMAGE_FILE_OPEN(name,rw,IFT_PPM);
	if( ifp==NO_IMAGE_FILE ) return(ifp);

	ifp->if_hd = (Dis_Header *)getbuf( sizeof(Dis_Header) );

	if( IS_READABLE(ifp) ){
		if( rd_dis_hdr( ifp->if_fp, ifp->if_hd,
			ifp->if_name ) < 0 ){
			dis_close(QSP_ARG  ifp);
			return(NO_IMAGE_FILE);
		}
		dis_to_dp(ifp->if_dp,ifp->if_hd);
	}
	return(ifp);
}



FIO_CLOSE_FUNC( dis_close )
{
	if( ifp->if_hd != NULL ){
		givbuf(ifp->if_hd);
	}
	GENERIC_IMGFILE_CLOSE(ifp);
}

int dp_to_dis(Dis_Header *hd_p,Data_Obj *dp)
{
	if( dp->dt_prec != PREC_UBY ){
		sprintf(DEFAULT_ERROR_STRING,
		"Sorry, can only write unsigned byte images to PPM, object %s has prec %s",
			dp->dt_name,prec_name[dp->dt_prec]);
		NWARN(DEFAULT_ERROR_STRING);
		return(-1);
	}
	if( dp->dt_seqs>1 ){
		sprintf(DEFAULT_ERROR_STRING,
		"Sorry, object %s has more than 1 seq, can only write 1 to .dis",
			dp->dt_name);
		NWARN(DEFAULT_ERROR_STRING);
		return(-1);
	}

	if( dp->dt_comps == 1 ) 
		hd_p->format = 5;
	else if( dp->dt_comps == 3 )
		hd_p->format = 6;
	else {
		NWARN("Sorry, .dis only supports type dimensions 1 and 3");
		return(-1);
	}
		
	hd_p->rows = (int)dp->dt_rows;
	hd_p->cols = (int)dp->dt_cols;
	hd_p->frames = (int)dp->dt_frames;
	hd_p->somex = 255;

	hd_p->img_data = dp->dt_data;

	return(0);
}

void wt_dis_hdr(FILE *fp,Dis_Header *hdp,const char *filename)
{
	fprintf(fp,"P%d\n",hdp->format);
	fprintf(fp,"%d %d %d\n",hdp->cols,hdp->rows,hdp->frames);
	fprintf(fp,"%d\n",hdp->somex);
	fflush(fp);
}

FIO_SETHDR_FUNC( set_dis_hdr )		/* set header fields from image object */
{
	if( dp_to_dis(ifp->if_hd,ifp->if_dp) < 0 ){
		dis_close(QSP_ARG  ifp);
		return(-1);
	}
	wt_dis_hdr(ifp->if_fp,ifp->if_hd,ifp->if_name);	/* write it out */
	return(0);
}

FIO_WT_FUNC( dis_wt )
{
	/* PPM wants color images interleaved */

	if( ifp->if_dp == NO_OBJ ){	/* first time check, always true for ppm */
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

	dp_to_dis(*hd_pp,dp);

	return(0);
}

int dis_conv(Data_Obj *dp,void *hd_pp)
{
	NWARN("dis_conv not implemented");
	return(-1);
}

/* eof */

