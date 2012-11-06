#include "quip_config.h"

char VersionId_fio_sunras[] = QUIP_VERSION_STRING;

/* read in a sun raster file */

#include <stdio.h>

#include "img_file_hdr.h"

#include "debug.h"
#include "getbuf.h"
#include "filetype.h"
#include "fio_prot.h"

#define HDR_P(ifp)		((struct rasterfile *)&(((Image_File_Hdr *)ifp->if_hd)->ifh_u.rf_hd))

void sunras_close(QSP_ARG_DECL  Image_File *ifp)
{
	if( ifp->if_hd != NULL ){
		givbuf(ifp->if_hd);
	}
	GENERIC_IMGFILE_CLOSE(ifp);
}

void sunras_rd(QSP_ARG_DECL  Data_Obj *dp,Image_File *ifp,index_t x_offset,index_t y_offset,index_t t_offset)		/**/
{
	long nb;

	if( !IS_CONTIGUOUS(dp) ){
		WARN("sorry, can only read rasterfiles into contiguous data objects");
		return;
	}

	/* skip over color map */
	/*
	if( ifp->hdr_p->ras_maptype == RMT_NONE ){
		WARN("no color map information!");
	} else
	*/
       if( HDR_P(ifp)->ras_maptype == RMT_RAW ){

		/* BUG should read in this data somewhere else;
		   could cause a problem for a very small image */

		if( fread(dp->dt_data,1,HDR_P(ifp)->ras_maplength,ifp->if_fp) !=
			(size_t)HDR_P(ifp)->ras_maplength ){
			WARN("error reading color map");
			SET_ERROR(ifp);
			goto stoppp;
		}
		WARN("not retaining color map");
	} else if( HDR_P(ifp)->ras_maptype == RMT_EQUAL_RGB ){
		if( verbose ) advise("equal rgb format");
		if( fread(dp->dt_data,1,HDR_P(ifp)->ras_maplength,ifp->if_fp) !=
			(size_t)HDR_P(ifp)->ras_maplength ){
			WARN("error reading color map");
			SET_ERROR(ifp);
			goto stoppp;
		}
		if( verbose ) advise("not retaining color map");
	}

	/* now read the data */

	nb=HDR_P(ifp)->ras_width*HDR_P(ifp)->ras_height*(HDR_P(ifp)->ras_depth/8);
	if( HDR_P(ifp)->ras_length == nb ){
		if( fread(dp->dt_data,1,HDR_P(ifp)->ras_length,ifp->if_fp)
			!= (size_t)HDR_P(ifp)->ras_length ){

			WARN("error reading pixel data");
			SET_ERROR(ifp);
			goto stoppp;
		}
	} else {	/* encoded image */
		unsigned short count;
#define US_NONE	0xffff		/* unsigned negative one for count */
		unsigned char bdata;
		register unsigned char *cp;
		int nput=0;

		cp=(unsigned char *)dp->dt_data;

		while( HDR_P(ifp)->ras_length ){
			if( fread(&count,sizeof(count),1,ifp->if_fp) != 1 ){
				WARN("error reading count");
				SET_ERROR(ifp);
				goto stoppp;
			}
			if( fread(&bdata,1,1,ifp->if_fp) != 1 ){
				WARN("error reading byte data");
				SET_ERROR(ifp);
				goto stoppp;
			}
			HDR_P(ifp)->ras_length -= 1+sizeof(count);
			while( count-- && nb > 0 ){
				*cp++ = bdata;
				nb --;
				nput++;
			}
			if( count != US_NONE ){
				WARN("error decoding image");
				SET_ERROR(ifp);
				goto stoppp;
			}
		}
		if( nb != 0 )
			WARN("WARNING:  too few bytes decoded!?");
	}
stoppp:
	sunras_close(QSP_ARG  ifp);
	return;
}


Image_File *		/**/
sunras_open(QSP_ARG_DECL  const char *name,int rw)		/**/
{
	Image_File *ifp;

	ifp = IMAGE_FILE_OPEN(name,rw,IFT_SUNRAS);
	if( ifp==NO_IMAGE_FILE ) return(ifp);

	ifp->if_hd = getbuf( sizeof(struct rasterfile) );

	if( fread(ifp->if_hd,sizeof(struct rasterfile),1,ifp->if_fp) != 1 ){
		WARN("error reading rasterfile header");
		goto stoppit;
	}
	if( HDR_P(ifp)->ras_magic != RAS_MAGIC ){
		WARN("not a rasterfile");
		goto stoppit;
	}
	ifp->if_dp->dt_prec = PREC_BY;
	if( HDR_P(ifp)->ras_depth == 8 )
		ifp->if_dp->dt_comps = 1;
	else if( HDR_P(ifp)->ras_depth == 24 )
		ifp->if_dp->dt_comps = 3;
	else {
		sprintf(error_string,"depth = %d",HDR_P(ifp)->ras_depth);
		advise(error_string);
		WARN("Sorry, can only read 8 or 24 bit rasterfiles now");
		goto stoppit;
	}
	ifp->if_dp->dt_cols = HDR_P(ifp)->ras_width;
	ifp->if_dp->dt_rows = HDR_P(ifp)->ras_height;
	ifp->if_dp->dt_frames = 1;
	ifp->if_dp->dt_seqs = 1;

	ifp->if_nfrms = 0;
	ifp->if_flags = rw;

	return(ifp);
stoppit:
	sunras_close(QSP_ARG  ifp);
	return(NO_IMAGE_FILE);
}

int sunras_unconv(void *hd_pp,Data_Obj *dp)
{
	NWARN("sunras_unconv not implemented");
	return(-1);
}

int sunras_conv(Data_Obj *dp,void *hd_pp)
{
	NWARN("sunras_conv not implemented");
	return(-1);
}

