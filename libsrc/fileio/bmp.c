char VersionId_fio_bmp[] = "$RCSfile: bmp.c,v $ $Revision: 1.10 $ $Date: 2011/10/25 17:00:16 $";

/* BUG BUG BUG  check for memory leaks! */

#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include "fio_prot.h"
#include "quip_prot.h"
#include "getbuf.h"
#include "data_obj.h"
#include "debug.h"
#include "img_file/raw.h"
#include "hips/hips2.h"
#include "bmp.h"

typedef enum {
	BMP_BI_RGB = 0,
	BMP_BI_RLE8,
	BMP_BI_RLE4
} CompressionType;

#define HDR_P(ifp)	((BMP_Header *)ifp->if_hdr_p)

int bmp_to_dp(Data_Obj *dp, BMP_Header *hd_p)
{
	SET_OBJ_PREC_PTR(dp,prec_for_code(PREC_UBY));
	SET_OBJ_COMPS(dp,1);
	SET_OBJ_COLS(dp, hd_p->bmp_cols);
	SET_OBJ_ROWS(dp, hd_p->bmp_rows);
	SET_OBJ_FRAMES(dp, 1);
	SET_OBJ_SEQS(dp, 1);

	SET_OBJ_COMP_INC(dp, 1);
	SET_OBJ_PXL_INC(dp, 1);
	SET_OBJ_ROW_INC(dp, OBJ_PXL_INC(dp) * (incr_t)OBJ_COLS(dp) );
	SET_OBJ_FRM_INC(dp, OBJ_ROW_INC(dp) * (incr_t)OBJ_ROWS(dp));
	SET_OBJ_SEQ_INC(dp, OBJ_FRM_INC(dp) * (incr_t)OBJ_FRAMES(dp));

	SET_OBJ_PARENT(dp, NULL);
	SET_OBJ_CHILDREN(dp, NULL);

	SET_OBJ_AREA(dp, ram_area_p);		/* the default */
	/* dp->dt_data = hd_p->image; */
	SET_OBJ_N_TYPE_ELTS(dp, OBJ_COMPS(dp) * OBJ_COLS(dp) * OBJ_ROWS(dp)
			* OBJ_FRAMES(dp) * OBJ_SEQS(dp));

	set_shape_flags(OBJ_SHAPE(dp),dp,AUTO_SHAPE);

	return(0);
}

//void bmp_info(QSP_ARG_DECL  Image_File *ifp)
FIO_INFO_FUNC( bmp )
{
	sprintf(msg_str,"\tcompression %d (0x%x)",HDR_P(ifp)->bmp_compression,HDR_P(ifp)->bmp_compression);
	sprintf(msg_str,"\tbit_count %d (0x%x)",HDR_P(ifp)->bmp_bit_count,HDR_P(ifp)->bmp_bit_count);
	prt_msg(msg_str);
	/* BUG do the rest of these */
}

static int32_t read32BitValue(Image_File *ifp)
{
	int32_t l;

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

	s = (short)(c1 + (c2 << 8));


	HDR_P(ifp)->bmp_bytes_read += 2;

	return s;
}

static unsigned char read8BitValue(Image_File *ifp)
{
	unsigned int c1 = fgetc(ifp->if_fp);

	HDR_P(ifp)->bmp_bytes_read += 1;

	return (unsigned char)c1;
}


//void bmp_close(Image_File *ifp)
FIO_CLOSE_FUNC( bmp )
{
	generic_imgfile_close(QSP_ARG  ifp);
}

FIO_OPEN_FUNC( bmp )
{
	Image_File *ifp;

#ifdef DEBUG
if( debug ) advise("opening image file");
#endif /* DEBUG */

	ifp = IMG_FILE_CREAT(name,rw,FILETYPE_FOR_CODE(IFT_BMP));
	if( ifp==NO_IMAGE_FILE ) return(ifp);

	/* img_file_creat creates dummy if_dp only if readable */

	ifp->if_hdr_p = (BMP_Header *)getbuf( sizeof(BMP_Header) );
	// here we sometimes clear the header...

#ifdef DEBUG
if( debug ) advise("allocating hips header");
#endif /* DEBUG */

	if( rw == FILE_READ ){
		//long l;
		//short s1,s2;
		/*unsigned int sizeOfInfoHeader;*/

		HDR_P(ifp)->bmp_bytes_read = 0;

		if (read8BitValue(ifp) != 66 || read8BitValue(ifp) != 77){
			sprintf(ERROR_STRING,"bmp_open %s:  bad magic number",ifp->if_name);
			WARN(ERROR_STRING);
			return(NO_IMAGE_FILE);
		}

		/*l=*/  read32BitValue(ifp);
		/*s1=*/ read16BitValue(ifp);
		/*s2=*/ read16BitValue(ifp);

		HDR_P(ifp)->bmp_byte_offset = read32BitValue(ifp);
		HDR_P(ifp)->bmp_info_size = (unsigned int)read32BitValue(ifp);

		if (HDR_P(ifp)->bmp_info_size < 40){
			sprintf(ERROR_STRING,"bmp_open %s:  bad info header size (%d)",
				ifp->if_name,HDR_P(ifp)->bmp_info_size);
			WARN(ERROR_STRING);
			return(NO_IMAGE_FILE);
		}

		HDR_P(ifp)->bmp_cols = read32BitValue(ifp);

		HDR_P(ifp)->bmp_rows = read32BitValue(ifp);

		if (read16BitValue(ifp) != 1){
			sprintf(ERROR_STRING,"bmp_open %s: expected word after height to be 1!?",
				ifp->if_name);
			WARN(ERROR_STRING);
			return(NO_IMAGE_FILE);
		}

		HDR_P(ifp)->bmp_bit_count = read16BitValue(ifp);

		if (HDR_P(ifp)->bmp_bit_count != 1 && HDR_P(ifp)->bmp_bit_count != 4 && HDR_P(ifp)->bmp_bit_count != 8 && HDR_P(ifp)->bmp_bit_count != 24){
			sprintf(ERROR_STRING,"bmp_open %s: bad bit count (%d)!?",
				ifp->if_name,HDR_P(ifp)->bmp_bit_count);
			WARN(ERROR_STRING);
			return(NO_IMAGE_FILE);
		}

		HDR_P(ifp)->bmp_compression = (CompressionType)read32BitValue(ifp);

		if (HDR_P(ifp)->bmp_compression != BMP_BI_RGB && HDR_P(ifp)->bmp_compression != BMP_BI_RLE8 && 
	    		HDR_P(ifp)->bmp_compression != BMP_BI_RLE4){

			sprintf(ERROR_STRING,"bmp_open %s: bad compression code (%d)!?",
				ifp->if_name,HDR_P(ifp)->bmp_compression);
			WARN(ERROR_STRING);
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
//sprintf(ERROR_STRING,"allocating palette, bit_count = %d",HDR_P(ifp)->bmp_bit_count);
//advise(ERROR_STRING);
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

		bmp_to_dp(ifp->if_dp,HDR_P(ifp));
	} else {
		ERROR1("Sorry, don't know how to write BMP files");
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

static void bmp_rd_24bit_image(Data_Obj *dp,Image_File *ifp,index_t x_offset,index_t y_offset,index_t t_offset)
{
	dimension_t x, y;

	for(y=0;y<OBJ_ROWS(dp);y++){
		u_char *p;

		p = (u_char *)OBJ_DATA_PTR(dp);
		p += (OBJ_ROWS(dp)-(1+y)) * OBJ_ROW_INC(dp);
		for (x = 0; x < OBJ_COLS(dp); x++) {
			/* BGR order, or display is wacky?? */
			*(p + 2 * OBJ_COMP_INC(dp)) = read8BitValue(ifp);
			*(p + OBJ_COMP_INC(dp)) = read8BitValue(ifp);
			*p = read8BitValue(ifp);
			p += OBJ_PXL_INC(dp);
		}

		/* Pad to 32-bit boundery. */
		while(HDR_P(ifp)->bmp_bytes_read % 4 != 0)
			read8BitValue(ifp);
	}
}

static void bmp_rd_8bit_image(Data_Obj *dp,Image_File *ifp,index_t x_offset,index_t y_offset,index_t t_offset)
{
	dimension_t x, y;	// an unsigned type...

	for(y=0;y<OBJ_ROWS(dp);y++){
		u_char *p;

		p = (u_char *)OBJ_DATA_PTR(dp);
		p += (OBJ_ROWS(dp)-(1+y)) * OBJ_ROW_INC(dp);
		for (x = 0; x < OBJ_COLS(dp); x++) {
			*p = read8BitValue(ifp);
			p += OBJ_PXL_INC(dp);
		}

		/* Pad to 32-bit boundery. */
		while(HDR_P(ifp)->bmp_bytes_read % 4 != 0)
			read8BitValue(ifp);
	}
}

#ifdef NOT_USED

static void bmp_rd_bit_image(Data_Obj *dp,Image_File *ifp,index_t x_offset,index_t y_offset,index_t t_offset)
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

	for (y = 0; y < OBJ_ROWS(dp); y++) {
		u_char *p;

		p = (u_char *)OBJ_DATA_PTR(dp);
		p += y * OBJ_ROW_INC(dp);
		for (x = 0; x < OBJ_COLS(dp); x++) {
			if (x % 8 == 0)
				byteRead = read8BitValue(ifp);

			color = (byteRead >> (7 -(x % 8))) & 1;

			*p = HDR_P(ifp)->bmp_palette_p[color];
			*(p + OBJ_COMP_INC(dp)) = HDR_P(ifp)->bmp_palette_p[256+color];
			p += OBJ_COMP_INC(dp);
			*p = HDR_P(ifp)->bmp_palette_p[512+color]; 
			p += OBJ_COMP_INC(dp);
		}

		/* Pad to 32-bit boundery. */
		while(HDR_P(ifp)->bmp_bytes_read % 4 != 0)
			read8BitValue(ifp);
	}
}

#endif // NOT_USED

FIO_RD_FUNC( bmp )
{
	if( HDR_P(ifp)->bmp_bit_count == 24 )
		bmp_rd_24bit_image(dp,ifp,x_offset,y_offset,t_offset);
	if( HDR_P(ifp)->bmp_bit_count == 8 )
		bmp_rd_8bit_image(dp,ifp,x_offset,y_offset,t_offset);
	else WARN("Sorry, only 24 or 8 bit image reads...");
}


