#include "quip_config.h"

char VersionId_fio_rgb[] = QUIP_VERSION_STRING;

#ifdef HAVE_RGB

#include <stdio.h>

#ifdef HAVE_STDLIB_H
#include <stdlib.h>		/* malloc() */
#endif

#include "filetype.h"
#include "getbuf.h"
#include "data_obj.h"
#include "debug.h"
#include "rgb.h"
#include "savestr.h"
#include "glimage.h"

#define hdr	if_hd.rgb_ip

void rgb_close(Image_File *ifp)
{
	if( ifp->hdr != NULL ){
		/* should we do something here??? */
	}
	GENERIC_IMGFILE_CLOSE(ifp);
}

int rgb_to_dp(Data_Obj *dp,IMAGE *ip)
{
	short type_dim=1;
	short prec;

	if( ip->zsize == 3 ){		/* rgb */
		type_dim=3;
	} else if( ip->zsize == 1 ){	/* gray */
		type_dim=1;
	} else {
		warn("unrecognized rgb file zsize");
		return(-1);
	}

	/* are these always short? */

	/* prec=PREC_BY; */
	prec=PREC_IN;

	dp->dt_seqs = 1;
	dp->dt_frames = 1;
	dp->dt_rows = ip->ysize;
	dp->dt_cols = ip->xsize;
	dp->dt_tdim = type_dim;
	dp->dt_prec = prec;

	return(0);
}

Image_File *			/**/
rgb_open(const char *name,int rw)		/**/
{
	Image_File *ifp;

#ifdef DEBUG
if( debug ) advise("opening image file");
#endif /* DEBUG */

	ifp = IMAGE_FILE_OPEN(name,rw,IFT_RGB);
	if( ifp==NO_IMAGE_FILE ) return(ifp);

	if( rw == FILE_READ ){
		/* image_file_open creates dummy if_dp only if readable */
		rgb_to_dp(ifp->if_dp,ifp->hdr);
	}
	return(ifp);

}


int dp_to_rgb(IMAGE *ip,Data_Obj *dp)
{
	ip->ysize = (unsigned short)dp->dt_rows;
	ip->xsize = (unsigned short)dp->dt_cols;
	/* BUG should check type_dim for 1 or 3 components... */
	/* ip->zsize ??? */

	switch( dp->dt_prec ){
		case PREC_IN:
			/* we know how to do this... */
			break;
		case PREC_BY:
		case PREC_DI:
		case PREC_SP:
		default:
			warn("dp_to_rgb:  only does short pixels");
			return(-1);
	}
	return(0);
}

int rgb_wt(Data_Obj *dp,Image_File *ifp)	/** output next frame */
{
	short *base,*rowbase, *data;
	dimension_t x,y, channel;
	short *scratch;

	/* we don't have to test if it's the first frame since rgb only has 1 */

	setup_dummy(ifp);
	copy_dimensions(ifp->if_dp, dp);
	ifp->if_dp->dt_frames = 1;

	/* BUG? check for error here??? */
	ifp_iopen(ifp);		/* open the file now */

	/* now write the data */

	scratch=(short*)malloc((size_t)(dp->dt_cols*sizeof(short)));
	base=(short *)dp->dt_data;
	for(y=0;y<dp->dt_rows;y++){
		rowbase=base+y*dp->dt_rowinc;
		for(channel=0;channel<dp->dt_tdim;channel++){
			data=rowbase+(channel*dp->dt_cinc);
			for(x=0;x<dp->dt_cols;x++){
				scratch[x] = *data;
				data+=dp->dt_pinc;
				putrow((IMAGE *)ifp->hdr,(unsigned short *)scratch,
             (unsigned int)(dp->dt_rows-(y+1)),(unsigned int)channel);
			}
		}
	}
	free(scratch);

	return(0);
}

void read_rgb(Data_Obj *dp,Image_File *ifp)
{
	short *base,*rowbase, *data;
	dimension_t x,y,channel;
	short *scratch;

	scratch=(short*)malloc((size_t)(dp->dt_cols*sizeof(short)));
	base=dp->dt_data;
	
	for(y=0;y<dp->dt_rows;y++){
		rowbase=base+y*dp->dt_rowinc;
		for(channel=0;channel<dp->dt_tdim;channel++){
			getrow((IMAGE *)ifp->hdr,(unsigned short *)scratch,
				 (unsigned int)(dp->dt_rows-(y+1)),(unsigned int)channel);
			data=rowbase+(channel*dp->dt_cinc);
			for(x=0;x<dp->dt_cols;x++){
	*data=scratch[x];
	data+=dp->dt_pinc;
			}
		}
	}
	free(scratch);
}

void rgb_rd(Data_Obj *dp,Image_File *ifp,index_t x_offset,index_t y_offset,index_t t_offset)
{
	if( !same_type(dp,ifp) ) return;

	if( t_offset >= dp->dt_frames ){
		sprintf(error_string,
		"raw_rd:  ridiculous frame offset %d (max %d)",
			t_offset,dp->dt_frames-1);
		warn(error_string);
		return;
	}

	t_offset *=	   (dp->dt_rows
			 * dp->dt_cols
			 * dp->dt_tdim
			 * siztbl[dp->dt_prec]);

	if( 	dp->dt_rows==ifp->if_dp->dt_rows &&
		dp->dt_cols==ifp->if_dp->dt_cols &&
		x_offset==0 && y_offset==0 ){

		if( dp->dt_frames == ifp->if_dp->dt_frames ){
			/* read it all at once */
			read_rgb(dp,ifp);
		} else if( dp->dt_frames == 1 ){
			warn("can't do partial rgb reads");
			goto readerr;
		}
	} else {
		warn("can't do partial rgb reads");
		goto readerr;
	}
	ifp->if_nfrms++;
	if( ifp->if_nfrms == ifp->if_dp->dt_frames ){
		if( verbose ){
			sprintf(error_string,
			"closing file \"%s\" after reading %d frames",
			ifp->if_name,ifp->if_nfrms);
			advise(error_string);
		}
		rgb_close(ifp);
	}
	return;
readerr:
	sprintf(error_string, "error reading pixel data from file \"%s\"",
		ifp->if_name);
	warn(error_string);
	SET_ERROR(ifp);
	rgb_close(ifp);
	return;
}

int rgb_unconv(void *hdr_pp,Data_Obj *dp)
{
	IMAGE **hd_pp;

	hd_pp = (IMAGE **) hdr_pp;

	/* allocate space for new header */

	*hd_pp = (IMAGE *)getbuf( sizeof(IMAGE) );
	if( *hd_pp == NULL ) return(-1);

	dp_to_rgb(*hd_pp,dp);

	return(0);
}

int rgb_conv(Data_Obj *dp,void *hd_pp)
{
	warn("rgb_conv not implemented");
	return(-1);
}

#endif /* HAVE_RGB */

