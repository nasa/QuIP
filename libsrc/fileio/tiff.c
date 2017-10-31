
#include "quip_config.h"

#ifdef HAVE_TIFF

#include <stdio.h>

#ifdef HAVE_STRING_H
#include <string.h>
#endif

#include "quip_prot.h"
#include "fio_prot.h"
#include "data_obj.h"
#include "img_file/fiotiff.h"
#include "img_file/raw.h"
//#include "readhdr.h"

/* static int num_color=1; */
/* static void rewrite_tiff_nf(TIFF* tiff,dimension_t n); */

FIO_INFO_FUNC(tiff)
{
}

FIO_FT_TO_DP_FUNC(tiff,TIFF)
{
	uint32 w,h,d;
	//TIFFDataType dtype;
	uint16 dtype;
	/* uint16 */ short bps;

	if( TIFFGetField(hd_p,TIFFTAG_BITSPERSAMPLE,&bps) != 1 ){
		warn("tiff_to_dp:  No bits_per_sample tag");
		bps=(-1);
	} else {
		if( verbose ){
			sprintf(ERROR_STRING,"Tiff bits per sample = %d",bps);
			advise(ERROR_STRING);
		}
	}

	if( TIFFGetField(hd_p,TIFFTAG_SAMPLEFORMAT,&dtype) == 1 ){
		switch(dtype){
			case SAMPLEFORMAT_UINT:
				switch(bps){
					case 8: SET_OBJ_PREC_PTR(dp, PREC_FOR_CODE( PREC_UBY )); break;
					case 16: SET_OBJ_PREC_PTR(dp, PREC_FOR_CODE( PREC_UIN )); break;
					case 32: SET_OBJ_PREC_PTR(dp, PREC_FOR_CODE( PREC_UDI )); break;
					default:
						sprintf(ERROR_STRING,"Bad uint bits_per_pixel:  %d",bps);
						warn(ERROR_STRING);
						break;
				}
				break;

			case SAMPLEFORMAT_INT:
				switch(bps){
					case 8: SET_OBJ_PREC_PTR(dp, PREC_FOR_CODE( PREC_BY )); break;
					case 16: SET_OBJ_PREC_PTR(dp, PREC_FOR_CODE( PREC_IN )); break;
					case 32: SET_OBJ_PREC_PTR(dp, PREC_FOR_CODE( PREC_DI )); break;
					default:
						sprintf(ERROR_STRING,"Bad int bits_per_pixel:  %d",bps);
						warn(ERROR_STRING);
						break;
				}
				break;

			case SAMPLEFORMAT_IEEEFP:
				switch(bps){
					case 32: SET_OBJ_PREC_PTR(dp, PREC_FOR_CODE( PREC_SP )); break;
					case 64: SET_OBJ_PREC_PTR(dp, PREC_FOR_CODE( PREC_DP )); break;
					default:
						sprintf(ERROR_STRING,"Bad ieeefp bits_per_pixel:  %d",bps);
						warn(ERROR_STRING);
						break;
				}
				break;

			default:
				sprintf(ERROR_STRING,"Unrecognized TIFF sample format %d",dtype);
				warn(ERROR_STRING);
				break;
		}
	} else {
		if( verbose )
			advise("TIFFTAG_SAMPLEFORMAT is not present, trying TIFFTAG_DATATYPE");

		if( TIFFGetField(hd_p,TIFFTAG_DATATYPE,&dtype) == 1 ){
sprintf(ERROR_STRING,"dtype = %d",dtype);
advise(ERROR_STRING);
			switch(dtype){
				case TIFF_BYTE: advise("tiff_byte"); SET_OBJ_PREC_PTR(dp, PREC_FOR_CODE( PREC_UBY )); break;
				case TIFF_SHORT: advise("tiff short"); SET_OBJ_PREC_PTR(dp, PREC_FOR_CODE( PREC_UIN )); break;
				case TIFF_LONG: advise("tiff long"); SET_OBJ_PREC_PTR(dp, PREC_FOR_CODE( PREC_UDI )); break;

				case TIFF_SBYTE: advise("tiff_signed_byte"); SET_OBJ_PREC_PTR(dp, PREC_FOR_CODE( PREC_BY )); break;
				case TIFF_SSHORT: advise("tiff signed short"); SET_OBJ_PREC_PTR(dp, PREC_FOR_CODE( PREC_IN )); break;
				case TIFF_SLONG: advise("tiff signed long"); SET_OBJ_PREC_PTR(dp, PREC_FOR_CODE( PREC_DI )); break;

				case TIFF_FLOAT: advise("tiff float"); SET_OBJ_PREC_PTR(dp, PREC_FOR_CODE( PREC_SP )); break;
				case TIFF_DOUBLE: advise("tiff float"); SET_OBJ_PREC_PTR(dp, PREC_FOR_CODE( PREC_DP )); break;

				case TIFF_ASCII:
				case TIFF_RATIONAL:
				case TIFF_SRATIONAL:
				case TIFF_UNDEFINED:
					sprintf(ERROR_STRING,"Unhandled TIFF data type %d",dtype);
					warn(ERROR_STRING);
					break;
				default:
					sprintf(ERROR_STRING,"Unrecognized TIFF data type %d",dtype);
					warn(ERROR_STRING);
					break;
			}
		} else {
			if( bps < 0 ){	/* bits per sample not specified */
				/* assume byte */

				sprintf(ERROR_STRING, "TIFFTAG_DATATYPE not present, assuming default TIFF data type for %s",OBJ_NAME(dp));
				advise(ERROR_STRING);
				SET_OBJ_PREC_PTR(dp, PREC_FOR_CODE( PREC_BY ));
			} else if( bps >0 && bps <= 8 ){
				SET_OBJ_PREC_PTR(dp, PREC_FOR_CODE( PREC_UBY ));
			} else if( bps >8 && bps <=16 ){
				SET_OBJ_PREC_PTR(dp, PREC_FOR_CODE( PREC_UIN ));
			} else {
				sprintf(ERROR_STRING,"Not sure what to do with tiff bps = %d",bps);
				warn(ERROR_STRING);
			}
		}
	}


	if( TIFFGetField(hd_p,TIFFTAG_IMAGEWIDTH,&w) == 1 ){
		//sprintf(ERROR_STRING,"width = %d",w);
		//advise(ERROR_STRING);
		SET_OBJ_COLS(dp, w);
	} else warn("error getting TIFF width tag");

	if( TIFFGetField(hd_p,TIFFTAG_IMAGELENGTH,&h) == 1 ){
		//sprintf(ERROR_STRING,"height = %d",h);
		//advise(ERROR_STRING);
		SET_OBJ_ROWS(dp, h);
	} else warn("error getting TIFF length tag");

	if( TIFFGetField(hd_p,TIFFTAG_IMAGEDEPTH,&d) == 1 ){
		sprintf(ERROR_STRING,"depth = %ld",(u_long)d);
		advise(ERROR_STRING);
	} else {
		/* assume monochrome */
		SET_OBJ_COMPS(dp, 1);
	}

	SET_OBJ_FRAMES(dp, 1);
	SET_OBJ_SEQS(dp, 1);

	SET_OBJ_COMP_INC(dp, 1);
	SET_OBJ_PXL_INC(dp, 1);
	SET_OBJ_ROW_INC(dp, OBJ_COLS(dp)*OBJ_PXL_INC(dp) );
	SET_OBJ_FRM_INC(dp, OBJ_ROW_INC(dp) * OBJ_ROWS(dp));
	SET_OBJ_SEQ_INC(dp, OBJ_FRM_INC(dp) * OBJ_FRAMES(dp));

	SET_OBJ_PARENT(dp, NULL);
	SET_OBJ_CHILDREN(dp, NULL);

	SET_OBJ_AREA(dp, ram_area_p);		/* the default */
	SET_OBJ_N_TYPE_ELTS(dp, OBJ_COMPS(dp) * OBJ_COLS(dp) * OBJ_ROWS(dp)
			* OBJ_FRAMES(dp) * OBJ_SEQS(dp) );

	auto_shape_flags(OBJ_SHAPE(dp));
//longlist(dp);

	return(0);
}

FIO_OPEN_FUNC( tiff )
{
	Image_File *ifp;
	char *modestr;

	ifp = IMG_FILE_CREAT(name,rw,FILETYPE_FOR_CODE(IFT_TIFF));
	if( ifp==NULL ) return(ifp);

	if( IS_READABLE(ifp) )	modestr="r";
	else			modestr="w";

	ifp->if_tiff = TIFFOpen(ifp->if_pathname,modestr);
	if( ifp->if_tiff == NULL ){
		sprintf(ERROR_STRING,"error opening TIFF file %s",ifp->if_pathname);
		warn(ERROR_STRING);
		del_img_file(ifp);
		// call rls_str here???  BUG?
		return(NULL);
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


FIO_CLOSE_FUNC( tiff )
{
	/* can we write multiple frames to tiff??? */

	TIFFClose(ifp->if_tiff);
	GENERIC_IMGFILE_CLOSE(ifp);
}

//int dp_to_tiff(TIFF *tiff,Data_Obj *dp)
FIO_DP_TO_FT_FUNC(tiff,TIFF)
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


	w = OBJ_COLS(dp);
	h = OBJ_ROWS(dp);
	d = OBJ_COMPS(dp);
//sprintf(ERROR_STRING,"dp_to_tiff:  dimensions are %ld x %ld x %ld",(u_long)h,(u_long)w,(u_long)d);
//advise(ERROR_STRING);

	/* num_frame set when when write request given */

	switch( OBJ_MACH_PREC(dp) ){
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
			warn("dp_to_tiff:  Unhandled precision");
			break;
	}

	if( TIFFSetField(hd_p,TIFFTAG_SAMPLEFORMAT,dtype) != 1 )
		warn("error setting TIFF sample format");

	if( TIFFSetField(hd_p,TIFFTAG_BITSPERSAMPLE,bps) != 1 )
		warn("error setting TIFF bits per sample");

	if( TIFFSetField(hd_p,TIFFTAG_IMAGEWIDTH,w) != 1 )
		warn("error setting TIFF width tag");

	if( TIFFSetField(hd_p,TIFFTAG_IMAGELENGTH,h) != 1 )
		warn("error setting TIFF length tag");

//	if( TIFFSetField(hd_p,TIFFTAG_IMAGEDEPTH,d) != 1 )
//		warn("error setting TIFF depth tag");

	if( TIFFSetField(hd_p,TIFFTAG_SAMPLESPERPIXEL,d) != 1 )
		warn("error setting TIFF depth tag");

	if( TIFFSetField(hd_p,TIFFTAG_PLANARCONFIG,pc) != 1 )
		warn("error setting TIFF planar_config tag");

	if( TIFFSetField(hd_p,TIFFTAG_PHOTOMETRIC,ph) != 1 )
		warn("error setting TIFF photometric tag");

	/*
	if( TIFFSetField(hd_p,TIFFTAG_COMPRESSION,comp) != 1 )
		warn("error setting TIFF bits per sample");
	*/

	return(0);
}

int set_tiff_hdr(QSP_ARG_DECL  Image_File *ifp)		/* set header fields from image object */
{
	if( FIO_DP_TO_FT_FUNC_NAME(tiff)(QSP_ARG  ifp->if_tiff,ifp->if_dp) < 0 ){
		tiff_close(QSP_ARG  ifp);
		return(-1);
	}
	return(0);
}

FIO_WT_FUNC( tiff )
{
	dimension_t row;
	char *datap;

	if( ifp->if_dp == NULL ){	/* first time? */

		/* set the rows & columns in our file struct */
		setup_dummy(ifp);
		copy_dimensions(ifp->if_dp, dp);

		SET_OBJ_FRAMES(ifp->if_dp, ifp->if_frms_to_wt);
		SET_OBJ_SEQS(ifp->if_dp, 1);

		if( set_tiff_hdr(QSP_ARG  ifp) < 0 ) return(-1);

	} else if( !same_type(QSP_ARG  dp,ifp) ) return(-1);

	/* now write the data */
// We get an error from TIFFWriteScanline about setting PlanarConfig...
	datap = (char *)OBJ_DATA_PTR(dp);
	for(row=0;row<OBJ_ROWS(dp);row++){
		if( TIFFWriteScanline(ifp->if_tiff,datap,row,0) != 1 )
			warn("error writing TIFF scanline");
		datap += PREC_SIZE(OBJ_MACH_PREC_PTR(dp)) * OBJ_ROW_INC(dp);
	}

	ifp->if_nfrms ++ ;
	check_auto_close(QSP_ARG  ifp);
	return(0);
}

FIO_RD_FUNC( tiff )
{
	dimension_t row;
	char *datap;

	if( OBJ_PREC(dp) != OBJ_PREC(ifp->if_dp) ){
		sprintf(ERROR_STRING,"Destination object %s has %s precision, but file %s has %s!?",
			OBJ_NAME(dp),PREC_NAME(OBJ_PREC_PTR(dp)),
			ifp->if_name,PREC_NAME(OBJ_PREC_PTR(ifp->if_dp)));
		warn(ERROR_STRING);
		return;
	}

	for(row=0;row<OBJ_ROWS(ifp->if_dp);row++){
		datap = (char *)OBJ_DATA_PTR(dp);
		datap += PREC_SIZE(OBJ_MACH_PREC_PTR(dp))*
			(x_offset+(y_offset+row)*OBJ_ROW_INC(dp));
		if( TIFFReadScanline(ifp->if_tiff,datap,row,0) != 1 )
			warn("error in TIFFReadScanline");
	}
}

FIO_UNCONV_FUNC(tiff)
{
	warn("tiff_unconv not implemented");
	return(-1);
}

FIO_CONV_FUNC(tiff)
{
	warn("tiff_conv not implemented");
	return(-1);
}

FIO_SEEK_FUNC(tiff)
{
	warn("tiff_seek_frame:  not implemented!?");
	return(0);
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
		warn("error seeking to tiff header");
		return;
	}

	for(i=0;i<3;i++){
		do {
			if( (c=getc(fp)) == EOF ){
				warn("error reading header char");
				return;
			}
		} while(c!='\n');
	}
	*/

}
#endif /* FOOBAR */


#endif /* HAVE_TIFF */

