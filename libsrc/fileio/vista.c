#include "quip_config.h"

char VersionId_fio_vista[] = QUIP_VERSION_STRING;

#include <stdio.h>

#include "img_file_hdr.h"
#include "fio_prot.h"
#include "debug.h"
#include "filetype.h"
#include "getbuf.h"

#define HDR_P(ifp)		((Vista_Hdr *) &(((Image_File_Hdr *)ifp->if_hd)->ifh_u.vista_hd))

void vista_close(QSP_ARG_DECL  Image_File *ifp)
{
	if( ifp->if_hd != NULL )
		givbuf(ifp->if_hd);
	GENERIC_IMGFILE_CLOSE(ifp);
}

void bswap(short *sp)
{
	unsigned char *b1,*b2,tmp;

	b1=(unsigned char *)sp;
	b2=b1+1;
	tmp = *b1;
	*b1 = *b2;
	*b2 = tmp;
}

void swap_hdr(Vista_Hdr *hd_p)
{
	short *sp;
	dimension_t i;

	sp = (short *) hd_p;
	for(i=0;i<N_VH_WORDS;i++) bswap(sp++);
}

int vista_to_dp(Data_Obj *dp,Vista_Hdr *hdr_p)
{
	short prec;

	switch( (hdr_p->pixel_size) & 0xff ){
		case 8:  prec=PREC_BY; break;
		case 32:   prec=PREC_DI; break;
		default:
			sprintf(DEFAULT_ERROR_STRING,
		"vista_to_dp:  unrecognized pixel size %d",
				hdr_p->pixel_size);
			NWARN(DEFAULT_ERROR_STRING);
			return(-1);
	}
	dp->dt_seqs = 1;
	dp->dt_frames = 1;
	dp->dt_rows = hdr_p->nrows;
	dp->dt_cols = hdr_p->ncols;
	dp->dt_comps = 1;
	dp->dt_prec = prec;

	return(0);
}
	
FIO_OPEN_FUNC( vista_open )
{
	Image_File *ifp;

	ifp = IMAGE_FILE_OPEN(name,rw,IFT_VISTA);
	if( ifp==NO_IMAGE_FILE ) return(ifp);

	ifp->if_hd = getbuf( sizeof(Vista_Hdr) );
	if( ifp->if_hd == NULL ){
		WARN("error allocating vista header");
		vista_close(QSP_ARG  ifp);
	}

	if( rw == FILE_READ ){
		if( fread(ifp->if_hd,sizeof(Vista_Hdr),1,ifp->if_fp) != 1 ){
			WARN("error reading vista header");
			vista_close(QSP_ARG  ifp);
			return(NO_IMAGE_FILE);
		}
		swap_hdr(ifp->if_hd);
		if( (vista_to_dp(ifp->if_dp,(Vista_Hdr *)ifp->if_hd) < 0) ){
			vista_close(QSP_ARG  ifp);
			return(NO_IMAGE_FILE);
		}
	}
	return(ifp);
}

int vista_wt(QSP_ARG_DECL  Data_Obj *dp,Image_File *ifp)	/** write image in vista format */
{
	int i;
	int cols;
	int ros;
	int b_per_p;

	if( dp->dt_prec != PREC_BY && dp->dt_prec != PREC_DI ){
		WARN("can only write byte or long images in vista format");
		return(-1);
	}

	/* write the header !? */

	HDR_P(ifp)->dummy1=0;
	HDR_P(ifp)->dummies[0]=0;
	HDR_P(ifp)->dummies[1]=0;
	HDR_P(ifp)->dummies[2]=0;
	HDR_P(ifp)->dummies[3]=0;

	HDR_P(ifp)->always_2=2;
	HDR_P(ifp)->nrows=(short)dp->dt_rows;

	if( dp->dt_prec == PREC_BY && dp->dt_comps == 1 ) {
		HDR_P(ifp)->pixel_size=8;
		b_per_p=1;
		cols = (int)dp->dt_cols;
	} else if( (dp->dt_prec == PREC_DI && dp->dt_comps==1 ) ||
		   (dp->dt_prec == PREC_BY && dp->dt_comps==4 ) ){
		HDR_P(ifp)->pixel_size=8;
		b_per_p=1;
		cols = 4 * (int)dp->dt_cols;
	} else {
		WARN("bad pixel type for vista file");
		return(-1);
	}

	HDR_P(ifp)->ncols=cols;

	if( !IS_CONTIGUOUS(dp) ){
		WARN("data object must be contiguous for vista format write");
		return(-1);
	}

	swap_hdr(ifp->if_hd);

	if( fwrite(ifp->if_hd,sizeof(Vista_Hdr),1,ifp->if_fp) != 1 ){
		WARN("error writing vista header");
		vista_close(QSP_ARG  ifp);
		return(-1);
	}

	ros = (int)(dp->dt_rowinc * siztbl[dp->dt_prec]);

	for(i=(int)dp->dt_rows-1;i>=0;i--){

		if( fwrite(((char *)dp->dt_data)+i*ros,b_per_p,cols,ifp->if_fp)
			!= (size_t)cols ){
			WARN("error writing row pixel data");
			vista_close(QSP_ARG  ifp);
			i = -1;
		}
	}
	vista_close(QSP_ARG  ifp);
	return(0);
}

int vista_unconv(void *hdr_pp,Data_Obj *dp)
{
#ifdef FOOBAR
	Vista_Hdr **hd_pp;

	hd_pp = (Vista_Hdr **) hdr_pp;

	/* allocate space for new header */

	*hd_pp = getbuf( sizeof(Vista_Hdr) );
	if( *hd_pp == NULL ) return(-1);

	dp_to_vista(*hd_pp,dp);

	return(0);
#endif
	NWARN("vista unconv not implemented");
	return(-1);
}

int vista_conv(Data_Obj *dp,void *hd_pp)
{
	NWARN("vista_conv not implemented");
	return(-1);
}

