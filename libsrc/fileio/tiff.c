
#include "quip_config.h"

char VersionId_fio_tiff[] = QUIP_VERSION_STRING;


#ifdef HAVE_TIFF

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
#include "fiotiff.h"
#include "raw.h"
#include "readhdr.h"

/* static int num_color=1; */
/* static void rewrite_tiff_nf(TIFF* tiff,dimension_t n); */

void tiff_info(QSP_ARG_DECL  Image_File *ifp)
{
}

int tiff_to_dp(Data_Obj *dp,TIFF * tiff)
{
	uint32 w,h,d;
	//TIFFDataType dtype;
	uint16 dtype;
	/* uint16 */ short bps;

	if( TIFFGetField(tiff,TIFFTAG_BITSPERSAMPLE,&bps) != 1 ){
		NWARN("tiff_to_dp:  No bits_per_sample tag");
		bps=(-1);
	} else {
		if( verbose ){
			sprintf(DEFAULT_ERROR_STRING,"Tiff bits per sample = %d",bps);
			advise(DEFAULT_ERROR_STRING);
		}
	}

	if( TIFFGetField(tiff,TIFFTAG_SAMPLEFORMAT,&dtype) == 1 ){
		switch(dtype){
			case SAMPLEFORMAT_UINT:
				switch(bps){
					case 8: dp->dt_prec = PREC_UBY; break;
					case 16: dp->dt_prec = PREC_UIN; break;
					case 32: dp->dt_prec = PREC_UDI; break;
					default:
						sprintf(DEFAULT_ERROR_STRING,"Bad uint bits_per_pixel:  %d",bps);
						NWARN(DEFAULT_ERROR_STRING);
						break;
				}
				break;

			case SAMPLEFORMAT_INT:
				switch(bps){
					case 8: dp->dt_prec = PREC_BY; break;
					case 16: dp->dt_prec = PREC_IN; break;
					case 32: dp->dt_prec = PREC_DI; break;
					default:
						sprintf(DEFAULT_ERROR_STRING,"Bad int bits_per_pixel:  %d",bps);
						NWARN(DEFAULT_ERROR_STRING);
						break;
				}
				break;

			case SAMPLEFORMAT_IEEEFP:
				switch(bps){
					case 32: dp->dt_prec = PREC_SP; break;
					case 64: dp->dt_prec = PREC_DP; break;
					default:
						sprintf(DEFAULT_ERROR_STRING,"Bad ieeefp bits_per_pixel:  %d",bps);
						NWARN(DEFAULT_ERROR_STRING);
						break;
				}
				break;

			default:
				sprintf(DEFAULT_ERROR_STRING,"Unrecognized TIFF sample format %d",dtype);
				NWARN(DEFAULT_ERROR_STRING);
				break;
		}
	} else {
		if( verbose )
			advise("TIFFTAG_SAMPLEFORMAT is not present, trying TIFFTAG_DATATYPE");

		if( TIFFGetField(tiff,TIFFTAG_DATATYPE,&dtype) == 1 ){
sprintf(DEFAULT_ERROR_STRING,"dtype = %d",dtype);
advise(DEFAULT_ERROR_STRING);
			switch(dtype){
				case TIFF_BYTE: advise("tiff_byte"); dp->dt_prec = PREC_UBY; break;
				case TIFF_SHORT: advise("tiff short"); dp->dt_prec = PREC_UIN; break;
				case TIFF_LONG: advise("tiff long"); dp->dt_prec = PREC_UDI; break;

				case TIFF_SBYTE: advise("tiff_signed_byte"); dp->dt_prec = PREC_BY; break;
				case TIFF_SSHORT: advise("tiff signed short"); dp->dt_prec = PREC_IN; break;
				case TIFF_SLONG: advise("tiff signed long"); dp->dt_prec = PREC_DI; break;

				case TIFF_FLOAT: advise("tiff float"); dp->dt_prec = PREC_SP; break;
				case TIFF_DOUBLE: advise("tiff float"); dp->dt_prec = PREC_DP; break;

				case TIFF_ASCII:
				case TIFF_RATIONAL:
				case TIFF_SRATIONAL:
				case TIFF_UNDEFINED:
					sprintf(DEFAULT_ERROR_STRING,"Unhandled TIFF data type %d",dtype);
					NWARN(DEFAULT_ERROR_STRING);
					break;
				default:
					sprintf(DEFAULT_ERROR_STRING,"Unrecognized TIFF data type %d",dtype);
					NWARN(DEFAULT_ERROR_STRING);
					break;
			}
		} else {
			if( bps < 0 ){	/* bits per sample not specified */
				/* assume byte */

				sprintf(DEFAULT_ERROR_STRING, "TIFFTAG_DATATYPE not present, assuming default TIFF data type for %s",dp->dt_name);
				advise(DEFAULT_ERROR_STRING);
				dp->dt_prec = PREC_BY;
			} else if( bps >0 && bps <= 8 ){
				dp->dt_prec = PREC_UBY;
			} else if( bps >8 && bps <=16 ){
				dp->dt_prec = PREC_UIN;
			} else {
				sprintf(DEFAULT_ERROR_STRING,"Not sure what to do with tiff bps = %d",bps);
				NWARN(DEFAULT_ERROR_STRING);
			}
		}
	}


	if( TIFFGetField(tiff,TIFFTAG_IMAGEWIDTH,&w) == 1 ){
		//sprintf(DEFAULT_ERROR_STRING,"width = %d",w);
		//advise(DEFAULT_ERROR_STRING);
		dp->dt_cols = w;
	} else NWARN("error getting TIFF width tag");

	if( TIFFGetField(tiff,TIFFTAG_IMAGELENGTH,&h) == 1 ){
		//sprintf(DEFAULT_ERROR_STRING,"height = %d",h);
		//advise(DEFAULT_ERROR_STRING);
		dp->dt_rows = h;
	} else NWARN("error getting TIFF length tag");

	if( TIFFGetField(tiff,TIFFTAG_IMAGEDEPTH,&d) == 1 ){
		sprintf(DEFAULT_ERROR_STRING,"depth = %ld",(u_long)d);
		advise(DEFAULT_ERROR_STRING);
	} else {
		/* assume monochrome */
		dp->dt_comps = 1;
	}

	dp->dt_frames = 1;
	dp->dt_seqs = 1;

	dp->dt_cinc = 1;
	dp->dt_pinc = 1;
	dp->dt_rowinc = dp->dt_cols*dp->dt_pinc;
	dp->dt_finc = dp->dt_rowinc * dp->dt_rows;
	dp->dt_sinc = dp->dt_finc * dp->dt_frames;

	dp->dt_parent = NO_OBJ;
	dp->dt_children = NO_LIST;

	dp->dt_ap = ram_area;		/* the default */
	dp->dt_n_type_elts = dp->dt_comps * dp->dt_cols * dp->dt_rows
			* dp->dt_frames * dp->dt_seqs;

	set_shape_flags(&dp->dt_shape,dp);
//longlist(dp);

	return(0);
}

FIO_OPEN_FUNC( tiff_open )
{
	Image_File *ifp;
	char *modestr;

	ifp = IMAGE_FILE_OPEN(name,rw,IFT_TIFF);
	if( ifp==NO_IMAGE_FILE ) return(ifp);

	if( IS_READABLE(ifp) )	modestr="r";
	else			modestr="w";

	ifp->if_tiff = TIFFOpen(ifp->if_pathname,modestr);
	if( ifp->if_tiff == NULL ){
		sprintf(error_string,"error opening TIFF file %s",ifp->if_pathname);
		NWARN(error_string);
		DEL_IMG_FILE(name);
		return(NO_IMAGE_FILE);
	}


	if( IS_READABLE(ifp) ){
		/* BUG: should check for error here */

		/* An error can occur here if the file exists,
		 * but is empty...  we therefore initialize
		 * the header strings (above) with nulls, so
		 * that the program won't try to free nonsense
		 * addresses
		 */

		/* here we would read the header,
		 * but doesn't the tiff lib open routine get this info?
		 */
		tiff_to_dp(ifp->if_dp,ifp->if_tiff);
	}
#ifdef FOO
	else {
		hdr2_strs(ifp->hdr);		/* make null strings */
	}
#endif
	return(ifp);
}


FIO_CLOSE_FUNC( tiff_close )
{
	/* can we write multiple frames to tiff??? */

	TIFFClose(ifp->if_tiff);
	GENERIC_IMGFILE_CLOSE(ifp);
}

int dp_to_tiff(TIFF *tiff,Data_Obj *dp)
{
#ifdef FOOBAR
	dimension_t size;
#endif /* FOOBAR */

	uint32 w,h,d;
	uint16 pc=1;
	uint16 ph=PHOTOMETRIC_MINISBLACK;
	uint16 dtype
		=0	/* elim compiler warning */
	;
	uint16 bps
		=0	/* elim compiler warning */
	;
	/* COMPRESSION_LZW not available due to Unisys patent enforcement */
	/* uint16 comp=COMPRESSION_LZW; */


	w = dp->dt_cols;
	h = dp->dt_rows;
	d = dp->dt_comps;
sprintf(DEFAULT_ERROR_STRING,"dp_to_tiff:  dimensions are %ld x %ld x %ld",(u_long)h,(u_long)w,(u_long)d);
advise(DEFAULT_ERROR_STRING);

	/* num_frame set when when write request given */

	switch( MACHINE_PREC(dp) ){
		case PREC_UBY: bps=8; goto set_uint;
		case PREC_UIN: bps =16; goto set_uint;
		case PREC_UDI: bps=32;
set_uint:
			dtype = SAMPLEFORMAT_UINT;
			break;
		case PREC_BY: bps=8; goto set_int;
		case PREC_IN: bps =16; goto set_int;
		case PREC_DI: bps=32;
set_int:
			dtype = SAMPLEFORMAT_INT;
			break;
		case PREC_SP:  bps=32; goto set_flt;
		case PREC_DP:  bps=64;
set_flt:
			dtype = SAMPLEFORMAT_IEEEFP;
			break;
		default:
			NWARN("dp_to_tiff:  Unhandled precision");
			break;
	}

	if( TIFFSetField(tiff,TIFFTAG_SAMPLEFORMAT,dtype) != 1 )
		NWARN("error setting TIFF sample format");

	if( TIFFSetField(tiff,TIFFTAG_BITSPERSAMPLE,bps) != 1 )
		NWARN("error setting TIFF bits per sample");

	if( TIFFSetField(tiff,TIFFTAG_IMAGEWIDTH,w) != 1 )
		NWARN("error setting TIFF width tag");

	if( TIFFSetField(tiff,TIFFTAG_IMAGELENGTH,h) != 1 )
		NWARN("error setting TIFF length tag");

//	if( TIFFSetField(tiff,TIFFTAG_IMAGEDEPTH,d) != 1 )
//		NWARN("error setting TIFF depth tag");

	if( TIFFSetField(tiff,TIFFTAG_SAMPLESPERPIXEL,d) != 1 )
		NWARN("error setting TIFF depth tag");

	if( TIFFSetField(tiff,TIFFTAG_PLANARCONFIG,pc) != 1 )
		NWARN("error setting TIFF planar_config tag");

	if( TIFFSetField(tiff,TIFFTAG_PHOTOMETRIC,ph) != 1 )
		NWARN("error setting TIFF photometric tag");

	/*
	if( TIFFSetField(tiff,TIFFTAG_COMPRESSION,comp) != 1 )
		NWARN("error setting TIFF bits per sample");
	*/

	return(0);
}

int set_tiff_hdr(QSP_ARG_DECL  Image_File *ifp)		/* set header fields from image object */
{
	if( dp_to_tiff(ifp->if_tiff,ifp->if_dp) < 0 ){
		tiff_close(QSP_ARG  ifp);
		return(-1);
	}
	return(0);
}

FIO_WT_FUNC( tiff_wt )
{
	dimension_t row;
	char *datap;

	if( ifp->if_dp == NO_OBJ ){	/* first time? */

		/* set the rows & columns in our file struct */
		setup_dummy(ifp);
		copy_dimensions(ifp->if_dp, dp);

		ifp->if_dp->dt_frames = ifp->if_frms_to_wt;
		ifp->if_dp->dt_seqs = 1;

		if( set_tiff_hdr(QSP_ARG  ifp) < 0 ) return(-1);

	} else if( !same_type(QSP_ARG  dp,ifp) ) return(-1);

	/* now write the data */
// We get an error from TIFFWriteScanline about setting PlanarConfig...
	datap = (char *)dp->dt_data;
	for(row=0;row<dp->dt_rows;row++){
		if( TIFFWriteScanline(ifp->if_tiff,datap,row,0) != 1 )
			NWARN("error writing TIFF scanline");
		datap += siztbl[MACHINE_PREC(dp)] * dp->dt_rowinc;
	}

	ifp->if_nfrms ++ ;
	check_auto_close(QSP_ARG  ifp);
	return(0);
}

FIO_RD_FUNC( tiff_rd )
{
	dimension_t row;
	char *datap;

	if( dp->dt_prec != ifp->if_dp->dt_prec ){
		sprintf(error_string,"Destination object %s has %s precision, but file %s has %s!?",
			dp->dt_name,name_for_prec(dp->dt_prec),
			ifp->if_name,name_for_prec(ifp->if_dp->dt_prec));
		NWARN(error_string);
		return;
	}

	for(row=0;row<ifp->if_dp->dt_rows;row++){
		datap = (char *)dp->dt_data;
		datap += siztbl[MACHINE_PREC(dp)]*
			(x_offset+(y_offset+row)*dp->dt_rowinc);
		if( TIFFReadScanline(ifp->if_tiff,datap,row,0) != 1 )
			NWARN("error in TIFFReadScanline");
	}
}

int tiff_unconv(void *hdr_pp,Data_Obj *dp)
{
	NWARN("tiff_unconv not implemented");
	return(-1);
}

int tiff_conv(Data_Obj *dp,void *hd_pp)
{
	NWARN("tiff_conv not implemented");
	return(-1);
}

#ifdef FOOBAR
/* rewrite the number of frames in this header */

static void rewrite_tiff_nf(TIFF *tiff,dimension_t n)
{
	int i;
	char c;
	char str[16];

	/* seek to beginning of file */
	/*
	if( fseek(fp,0L,0) != 0 ){
		NWARN("error seeking to tiff header");
		return;
	}

	for(i=0;i<3;i++){
		do {
			if( (c=getc(fp)) == EOF ){
				NWARN("error reading header char");
				return;
			}
		} while(c!='\n');
	}
	*/

}
#endif /* FOOBAR */


#endif /* HAVE_TIFF */

