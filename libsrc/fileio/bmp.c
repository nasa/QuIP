#include "quip_config.h"

char VersionId_fio_bmp[] = QUIP_VERSION_STRING;

/* BUG BUG BUG  check for memory leaks! */

#include <stdio.h>

#ifdef HAVE_STRING_H
#include <string.h>
#endif

#ifdef HAVE_STDLIB_H
#include <stdlib.h>
#endif

#include "fio_prot.h"
#include "bmp.h"
#include "filetype.h"
#include "getbuf.h"
#include "data_obj.h"
#include "debug.h"
#include "savestr.h"
#include "raw.h"
#include "uio.h"
#include "vl.h"

typedef enum {
	BMP_BI_RGB = 0,
	BMP_BI_RLE8,
	BMP_BI_RLE4
} CompressionType;

//#define HDR_P(ifp)	((Image_File_Hdr *)ifp->if_hd)->ifh_u.bmp_hd_p
#define HDR_P(ifp)	((BMP_Header *)&(((Image_File_Hdr *)ifp->if_hd)->ifh_u.bmp_hd))

int bmp_to_dp(Data_Obj *dp, BMP_Header *hd_p)
{
	dp->dt_prec = PREC_UBY;
	dp->dt_comps = 1;
	dp->dt_cols = hd_p->bmp_cols;
	dp->dt_rows = hd_p->bmp_rows;
	dp->dt_frames = 1;
	dp->dt_seqs = 1;

	dp->dt_cinc = 1;
	dp->dt_pinc = 1;
	dp->dt_rowinc = dp->dt_pinc * (incr_t)dp->dt_cols ;
	dp->dt_finc = dp->dt_rowinc * (incr_t)dp->dt_rows;
	dp->dt_sinc = dp->dt_finc * (incr_t)dp->dt_frames;

	dp->dt_parent = NO_OBJ;
	dp->dt_children = NO_LIST;

	dp->dt_ap = ram_area;		/* the default */
	/* dp->dt_data = hd_p->image; */
	dp->dt_n_type_elts = dp->dt_comps * dp->dt_cols * dp->dt_rows
			* dp->dt_frames * dp->dt_seqs;

	set_shape_flags(&dp->dt_shape,dp);

	return(0);
}

void bmp_info(QSP_ARG_DECL  Image_File *ifp)
{
	sprintf(msg_str,"\tcompression %d (0x%x)",HDR_P(ifp)->bmp_compression,HDR_P(ifp)->bmp_compression);
	sprintf(msg_str,"\tbit_count %d (0x%x)",HDR_P(ifp)->bmp_bit_count,HDR_P(ifp)->bmp_bit_count);
	prt_msg(msg_str);
	/* BUG do the rest of these */
}

static long read32BitValue(Image_File *ifp)
{
	long l;

	int c1 = fgetc(ifp->if_fp);
	int c2 = fgetc(ifp->if_fp);
	int c3 = fgetc(ifp->if_fp);
	int c4 = fgetc(ifp->if_fp);


	l=c1 + (c2 << 8) + (c3 << 16) + (c4 << 24);

	HDR_P(ifp)->bmp_bytes_read += 4;

	return l;
}

static short read16BitValue(Image_File *ifp)
{
	short s;

	int c1 = fgetc(ifp->if_fp);
	int c2 = fgetc(ifp->if_fp);

	s = c1 + (c2 << 8);


	HDR_P(ifp)->bmp_bytes_read += 2;

	return s;
}

static unsigned char read8BitValue(Image_File *ifp)
{
	unsigned int c1 = fgetc(ifp->if_fp);

	HDR_P(ifp)->bmp_bytes_read += 1;

	return (unsigned char)c1;
}


FIO_CLOSE_FUNC( bmp_close )
{
	GENERIC_IMGFILE_CLOSE(ifp);
}

FIO_OPEN_FUNC( bmp_open )
{
	Image_File *ifp;

#ifdef DEBUG
if( debug ) advise("opening image file");
#endif /* DEBUG */

	ifp = IMAGE_FILE_OPEN(name,rw,IFT_BMP);

	/* image_file_open creates dummy if_dp only if readable */

	if( ifp==NO_IMAGE_FILE ) return(ifp);

	ifp->if_hd = getbuf( sizeof(BMP_Header) );

#ifdef DEBUG
if( debug ) advise("allocating hips header");
#endif /* DEBUG */

	if( rw == FILE_READ ){
		long l;
		short s1,s2;
		/*unsigned int sizeOfInfoHeader;*/

		HDR_P(ifp)->bmp_bytes_read = 0;

		if (read8BitValue(ifp) != 66 || read8BitValue(ifp) != 77){
			sprintf(error_string,"bmp_open %s:  bad magic number",ifp->if_name);
			NWARN(error_string);
			return(NO_IMAGE_FILE);
		}

		l=read32BitValue(ifp);
		s1=read16BitValue(ifp);
		s2=read16BitValue(ifp);

		HDR_P(ifp)->bmp_byte_offset = read32BitValue(ifp);
		HDR_P(ifp)->bmp_info_size = (unsigned int)read32BitValue(ifp);

		if (HDR_P(ifp)->bmp_info_size < 40){
			sprintf(error_string,"bmp_open %s:  bad info header size (%d)",
				ifp->if_name,HDR_P(ifp)->bmp_info_size);
			NWARN(error_string);
			return(NO_IMAGE_FILE);
		}

		HDR_P(ifp)->bmp_cols = read32BitValue(ifp);

		HDR_P(ifp)->bmp_rows = read32BitValue(ifp);

		if (read16BitValue(ifp) != 1){
			sprintf(error_string,"bmp_open %s: expected word after height to be 1!?",
				ifp->if_name);
			NWARN(error_string);
			return(NO_IMAGE_FILE);
		}

		HDR_P(ifp)->bmp_bit_count = read16BitValue(ifp);

		if (HDR_P(ifp)->bmp_bit_count != 1 && HDR_P(ifp)->bmp_bit_count != 4 && HDR_P(ifp)->bmp_bit_count != 8 && HDR_P(ifp)->bmp_bit_count != 24){
			sprintf(error_string,"bmp_open %s: bad bit count (%d)!?",
				ifp->if_name,HDR_P(ifp)->bmp_bit_count);
			NWARN(error_string);
			return(NO_IMAGE_FILE);
		}

		HDR_P(ifp)->bmp_compression = (CompressionType)read32BitValue(ifp);

		if (HDR_P(ifp)->bmp_compression != BMP_BI_RGB && HDR_P(ifp)->bmp_compression != BMP_BI_RLE8 && 
	    		HDR_P(ifp)->bmp_compression != BMP_BI_RLE4){

			sprintf(error_string,"bmp_open %s: bad compression code (%d)!?",
				ifp->if_name,HDR_P(ifp)->bmp_compression);
			NWARN(error_string);
			return(NO_IMAGE_FILE);
		}

		/* what are these five values??? */

		read32BitValue(ifp);
		read32BitValue(ifp);
		read32BitValue(ifp);
		read32BitValue(ifp);
		read32BitValue(ifp);

		HDR_P(ifp)->bmp_info_size -= 40;
		while (HDR_P(ifp)->bmp_info_size > 0) {
			read8BitValue(ifp); 
			HDR_P(ifp)->bmp_info_size --;
		} 
		HDR_P(ifp)->bmp_palette_p = NULL;
    
		if (HDR_P(ifp)->bmp_bit_count != 24){
			int numColors, i, j;

			/* read the palette */
sprintf(error_string,"allocating palette, bit_count = %d",HDR_P(ifp)->bmp_bit_count);
advise(error_string);
			HDR_P(ifp)->bmp_palette_p = (u_char *)getbuf( 3 * 256 * sizeof(u_char) );

			numColors = 1 << HDR_P(ifp)->bmp_bit_count;
			for (i = 0;i < numColors; i++) {
				/* Read RGB. */
				for (j = 2;j >= 0; j--)
					HDR_P(ifp)->bmp_palette_p[ (j<<8) + i] = read8BitValue(ifp);

				/* Skip reversed byte. */
				read8BitValue(ifp);
			}
		}

		bmp_to_dp(ifp->if_dp,ifp->if_hd);
	} else {
		NERROR1("CAUTIOUS:  Sorry, don't know how to write BMP files");
	}
	return(ifp);
} /* end bmp_open */

int bmp_unconv(void *hdr_pp,Data_Obj *dp)
{
	NWARN("bmp_unconv not implemented");
	return(-1);
}

int bmp_conv(Data_Obj *dp,void *hd_pp)
{
	NWARN("bmp_conv not implemented");
	return(-1);
}

FIO_RD_FUNC( bmp_rd_24bit_image )
{
	dimension_t x, y;

	/*for (y = 0; y < dp->dt_rows; y++) { */
	for (y = dp->dt_rows-1; y >= 0 ; y--) {
		u_char *p;

		p = (u_char *)dp->dt_data;
		p += y * dp->dt_rowinc;
		for (x = 0; x < dp->dt_cols; x++) {
			/* BGR order, or display is wacky?? */
			*(p + 2 * dp->dt_cinc) = read8BitValue(ifp);
			*(p + dp->dt_cinc) = read8BitValue(ifp);
			*p = read8BitValue(ifp);
			p += dp->dt_pinc;
		}

		/* Pad to 32-bit boundery. */
		while(HDR_P(ifp)->bmp_bytes_read % 4 != 0)
			read8BitValue(ifp);
	}
}

void bmp_rd_bit_image(Data_Obj *dp,Image_File *ifp,index_t x_offset,index_t y_offset,index_t t_offset)
{
	dimension_t x, y;
	unsigned char color;
	unsigned char byteRead
		=0	/* elim compiler warning */
		;

	/* 1-bit format cannot be compressed */
	if (HDR_P(ifp)->bmp_compression != BMP_BI_RGB){
		sprintf(DEFAULT_ERROR_STRING,"bmp_rd %s:  compression code (%d) is not BMP+BI_RGB (%d)!?",
			ifp->if_name,HDR_P(ifp)->bmp_compression,BMP_BI_RGB);
		NWARN(DEFAULT_ERROR_STRING);
		return;
	}

	for (y = 0; y < dp->dt_rows; y++) {
		u_char *p;

		p = (u_char *)dp->dt_data;
		p += y * dp->dt_rowinc;
		for (x = 0; x < dp->dt_cols; x++) {
			if (x % 8 == 0)
				byteRead = read8BitValue(ifp);

			color = (byteRead >> (7 -(x % 8))) & 1;

			*p = HDR_P(ifp)->bmp_palette_p[color];
			*(p + dp->dt_cinc) = HDR_P(ifp)->bmp_palette_p[256+color];
			p += dp->dt_cinc;
			*p = HDR_P(ifp)->bmp_palette_p[512+color]; 
			p += dp->dt_cinc;
		}

		/* Pad to 32-bit boundery. */
		while(HDR_P(ifp)->bmp_bytes_read % 4 != 0)
			read8BitValue(ifp);
	}
}

FIO_RD_FUNC( bmp_rd )
{
	if( HDR_P(ifp)->bmp_bit_count == 24 )
		bmp_rd_24bit_image(QSP_ARG  dp,ifp,x_offset,y_offset,t_offset);
	else NWARN("Sorry, only 24 bit image reads...");
}

