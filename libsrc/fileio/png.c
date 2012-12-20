
#include "quip_config.h"

char VersionId_fio_png[] = QUIP_VERSION_STRING;

#ifdef HAVE_PNG

/******************************************************************************

	sq: Copyright notice from decoder:

	Copyright (c) 1998-2000 Greg Roelofs.  All rights reserved.

	This software is provided "as is," without warranty of any kind,
	express or implied.  In no event shall the author or contributors
	be held liable for any damages arising in any way from the use of
	this software.

	Permission is granted to anyone to use this software for any purpose,
	including commercial applications, and to alter it and redistribute
	it freely, subject to the following restrictions:

	1. Redistributions of source code must retain the above copyright
	   notice, disclaimer, and this list of conditions.
	2. Redistributions in binary form must reproduce the above copyright
	   notice, disclaimer, and this list of conditions in the documenta-
	   tion and/or other materials provided with the distribution.
	3. All advertising materials mentioning features or use of this
	   software must display the following acknowledgment:

		This product includes software developed by Greg Roelofs
		and contributors for the book, "PNG: The Definitive Guide,"
		published by O'Reilly and Associates.

********************************************************************************/

#include "fio_prot.h"
#include "debug.h"

#include <stdio.h>

#ifdef HAVE_STDLIB_H
#include <stdlib.h>
#endif /* HAVE_STDLIB_H */

#include "fio_png.h"
#include "filetype.h" /* ft_tbl */

#define HDR_P	((Png_Hdr *)&(((Image_File_Hdr *)ifp->if_hd)->ifh_u.png_hd))

/* sq: The original decoder defines this. */
#define NO_24BIT_MASKS

/* could just include png.h, but this macro is the only thing we need
 * (name and typedefs changed to local versions); note that side effects
 * only happen with alpha (which could easily be avoided with
 * "ush acopy = (alpha);") */

#define alpha_composite(composite, fg, alpha, bg) {			\
	u_short temp = ((u_short)(fg)*(u_short)(alpha) +		\
			(u_short)(bg)*(u_short)(255 - (u_short)(alpha))	\
						+ (u_short)128);	\
	(composite) = (u_char)((temp + (temp >> 8)) >> 8);		\
}


//png_structp png_ptr = NULL;

/* some globals */
/* BUG: some of these are avoidable */
#ifdef FOOBAR
static int RShift, GShift, BShift;
static u_long RMask, GMask, BMask;
#endif/* FOOBAR */

static	u_char bg_red=0, bg_green=0, bg_blue=0;

static int color_type_to_write = -1;

#ifdef OLD_PNG_LIB
int png_to_dp( Data_Obj *dp, png_infop info_ptr )
#else
int png_to_dp( Data_Obj *dp, Png_Hdr *hdr_p )
#endif
{

// With new version of libpng, need to use get functions...
#ifdef OLD_PNG_LIB
	dp->dt_cols = info_ptr->width;
	dp->dt_rows = info_ptr->height;
	dp->dt_comps = info_ptr->channels;
#else
	dp->dt_cols = png_get_image_width(hdr_p->png_ptr,hdr_p->info_ptr);
	dp->dt_rows = png_get_image_height(hdr_p->png_ptr,hdr_p->info_ptr);
	dp->dt_comps = png_get_channels(hdr_p->png_ptr,hdr_p->info_ptr);
#endif /* ! OLD_PNG_LIB */

	/* prec will always get converted to 8 bits */
	dp->dt_prec = PREC_UBY;

	dp->dt_frames = 1;
	dp->dt_seqs = 1;

	dp->dt_cinc = 1;
	dp->dt_pinc = 1;
	dp->dt_rinc = dp->dt_pinc*dp->dt_cols;
	dp->dt_finc = dp->dt_rinc*dp->dt_rows;
	dp->dt_sinc = dp->dt_finc*dp->dt_frames;

	dp->dt_parent = NO_OBJ;
	dp->dt_children = NO_LIST;

	dp->dt_ap = ram_area;		/* the default */
	dp->dt_data = NULL;
	dp->dt_n_type_elts = dp->dt_comps * dp->dt_cols * dp->dt_rows
			* dp->dt_frames * dp->dt_seqs;

	set_shape_flags(&dp->dt_shape,dp,AUTO_SHAPE);

	return(0);
}


#ifdef FOOBAR
static void fill_hdr(Png_Hdr *png_hp, png_infop info_ptr)
#else
static void fill_hdr(Png_Hdr *png_hp)
#endif /* OLD_PNG_LIB */
{
	//printf("fill_hdr: IN\n");

#ifdef OLD_PNG_LIB
	png_hp->width = info_ptr->width;
	png_hp->height = info_ptr->height;
	png_hp->channels = info_ptr->channels;
	png_hp->color_type = info_ptr->color_type;

	png_hp->bit_depth = info_ptr->bit_depth;
	png_hp->compression_type = info_ptr->compression_type;
	png_hp->filter_type = info_ptr->filter_type;
	png_hp->interlace_type = info_ptr->interlace_type;
	png_hp->pixel_depth = info_ptr->pixel_depth;
#else

#define PNG_GET_ARGS	png_hp->png_ptr,png_hp->info_ptr

	png_hp->width = png_get_image_width(PNG_GET_ARGS);
	png_hp->height = png_get_image_height(PNG_GET_ARGS);
	png_hp->channels = png_get_channels(PNG_GET_ARGS);
	png_hp->color_type = png_get_color_type(PNG_GET_ARGS);

	png_hp->bit_depth = png_get_bit_depth(PNG_GET_ARGS);
	png_hp->compression_type = png_get_compression_type(PNG_GET_ARGS);
	png_hp->filter_type = png_get_filter_type(PNG_GET_ARGS);
	png_hp->interlace_type = png_get_interlace_type(PNG_GET_ARGS);
	png_hp->pixel_depth = png_hp->channels * png_hp->bit_depth;
#endif /* ! OLD_PNG_LIB */

	//printf("fill_hdr: OUT\n");
}


FIO_CLOSE_FUNC( pngfio_close )
{
	/* First do the png library cleanup */
	if( IS_READABLE(ifp) ){
		png_destroy_read_struct(&HDR_P->png_ptr, &HDR_P->info_ptr, NULL);

	} else {
//		WARN("pngfio_close:  don't know how to finish up file writing operation");
	}

	/* Shouldn't we also free the info struct that we allocated? */
	if( ifp->if_hd != NULL )
		givbuf(ifp->if_hd);

	GENERIC_IMGFILE_CLOSE(ifp);
}

/*
 * init_png :   rewind the file, verify magic number.
 *	Create the library struct, and the info struct.
 */

static int init_png(Image_File *ifp /* , png_infop info_ptr */ )
{
	u_char sig[8];
	png_infop info_ptr;

	//png_infop orig_info_ptr;


	//orig_info_ptr = info_ptr;

	rewind(ifp->if_fp);
	fread(sig, 1, 8, ifp->if_fp);

	if (!png_check_sig(sig,8)) {
		NWARN("init_png: not a valid PNG file (bad signature)");
		return(-1);
	}

	HDR_P->png_ptr = png_create_read_struct(PNG_LIBPNG_VER_STRING,	NULL, NULL, NULL);

	if (!HDR_P->png_ptr){
		NWARN("error creating png read struct");
		return(-1);   /* out of memory */
	}

	info_ptr = png_create_info_struct(HDR_P->png_ptr);

	if (!info_ptr) {
		NWARN("error creating png info struct");
		png_destroy_read_struct(&HDR_P->png_ptr, NULL, NULL);
		return(-1);   /* out of memory */
	}

		/* setjmp() must be called in every function
		 * that calls a PNG-reading libpng function */

	if (setjmp(png_jmpbuf(HDR_P->png_ptr))) {
		png_destroy_read_struct(&HDR_P->png_ptr, &info_ptr, NULL);
		return(-1);
	}

	png_init_io(HDR_P->png_ptr, ifp->if_fp);

	/* we have already read the 8 signature bytes */
	png_set_sig_bytes(HDR_P->png_ptr, 8);

	/* read all PNG info up to image data */
	png_read_info(HDR_P->png_ptr, info_ptr);

	HDR_P->info_ptr = info_ptr;
	//orig_info_ptr = (png_infop)memcpy(orig_info_ptr, info_ptr, sizeof(png_info));

	/* what about freeing our new struct? */

	//printf("init_png: OUT\n");
	return 0;
}


/* Expand palette images to RGB, low-bit-depth grayscale
 * images to 8 bits, transparency chunks to full alpha
 * channel; strip 16-bit-per-sample images to 8 bits
 * per sample; and convert grayscale to RGB[A].
 */
static int expand_image(Image_File *ifp)
{
	png_infop info_ptr;
	double display_exponent = 2.2;	/* default */
	double gamma;
	png_byte color_type, bit_depth, channels, pixel_depth;

	int init_n_of_channels = -1;
	int init_color_type = -1;
	int init_pixel_depth = -1;
	int init_bit_depth = -1;

	info_ptr = HDR_P->info_ptr;

//	printf("expand_image: IN\n");

	if(verbose) {
#ifdef OLD_PNG_LIB
		init_n_of_channels = info_ptr->channels;
		init_color_type = info_ptr->color_type;
		init_pixel_depth = info_ptr->pixel_depth;
		init_bit_depth = info_ptr->bit_depth;
#else
		init_n_of_channels = png_get_channels(HDR_P->png_ptr,HDR_P->info_ptr);
		//init_color_type = info_ptr->color_type;
		//init_pixel_depth = info_ptr->pixel_depth;
		//init_bit_depth = info_ptr->bit_depth;
#endif
	}

	if (setjmp(png_jmpbuf(HDR_P->png_ptr))) {
		png_destroy_read_struct(&HDR_P->png_ptr, &info_ptr, NULL);
		printf("expand_image: error in setjmp\n");
		return -1;
	}

#ifdef OLD_PNG_LIB
	color_type = info_ptr->color_type;
	bit_depth = info_ptr->bit_depth;
	channels = info_ptr->channels;
	pixel_depth = info_ptr->pixel_depth;
#else
	color_type = png_get_color_type(HDR_P->png_ptr,HDR_P->info_ptr);
	bit_depth = png_get_bit_depth(HDR_P->png_ptr,HDR_P->info_ptr);
	channels = png_get_channels(HDR_P->png_ptr,HDR_P->info_ptr);;
	pixel_depth = bit_depth * channels;
#endif /* ! OLD_PNG_LIB */

	if (color_type == PNG_COLOR_TYPE_PALETTE)
		png_set_expand(HDR_P->png_ptr);

	if (color_type == PNG_COLOR_TYPE_GRAY && bit_depth < 8)
		png_set_expand(HDR_P->png_ptr);

	if (png_get_valid(HDR_P->png_ptr, info_ptr, PNG_INFO_tRNS))
		png_set_expand(HDR_P->png_ptr);

	if (bit_depth == 16)
		png_set_strip_16(HDR_P->png_ptr);

	if (color_type == PNG_COLOR_TYPE_GRAY
			|| color_type == PNG_COLOR_TYPE_GRAY_ALPHA)
		png_set_gray_to_rgb(HDR_P->png_ptr);

	if (png_get_gAMA(HDR_P->png_ptr, info_ptr, &gamma))
		png_set_gamma(HDR_P->png_ptr, display_exponent, gamma);


	/* all transformations have been registered.
	 * Now update info_ptr data.
	 */
	png_read_update_info(HDR_P->png_ptr, info_ptr);

	if(verbose) {
		if(init_n_of_channels != channels)
			printf("image component(s) have been expanded from %d to %d\n",
				init_n_of_channels, channels);

		if(init_color_type != color_type)
			printf("color type %d has been changed to %d\n",
				init_color_type, color_type);

		if(init_pixel_depth!= pixel_depth)
			printf("pixel_depth %d has been changed to %d\n",
				init_pixel_depth, pixel_depth);

		if(init_bit_depth!= bit_depth)
			printf("bit_depth %d has been changed to %d\n",
				init_bit_depth, bit_depth);
	}

//	printf("expand_image: OUT\n");

	return 0;
}


/* I tried to predict the header info by looking at the routines in
 * expand_image, but it seems that when images get expanded rowbytes
 * are also taken into consideration (I don't know how!).
 * This hack expands the images and picks up the hdr info.
 *
 * (who wrote that comment?  doesn't sound like me (jbm) ...
 */

static int get_hdr_info(Image_File *ifp)
{
	//png_infop info_ptr;

	//info_ptr = HDR_P->info_ptr;

	if( init_png(ifp /*, info_ptr*/ ) < 0)
		return(-1);

	if( expand_image(ifp) < 0)
		return(-1);

#ifdef OLD_PNG_LIB
	if( png_to_dp(ifp->if_dp, HDR_P->info_ptr))
		return(-1);
	fill_hdr(HDR_P, HDR_P->info_ptr);
#else
	if( png_to_dp(ifp->if_dp, HDR_P ))
		return(-1);
	fill_hdr(HDR_P);
#endif /* ! OLD_PNG_LIB */


//advise("get_hdr_info calling png_destroy_read_struct");
	//png_destroy_read_struct(&HDR_P->png_ptr, &info_ptr, NULL);
//advise("get_hdr_info setting png_ptr to NULL");
	//HDR_P->png_ptr = NULL;
	//info_ptr = NULL;
	//free(info_ptr);

	//givbuf(ifp->if_hd);

	//ifp->if_hd = getbuf( sizeof(Png_Hdr) );
	//HDR_P->info_ptr = (png_infop)getbuf(sizeof(png_info));

	return 0;
}



FIO_OPEN_FUNC( pngfio_open )
{
	Image_File *ifp;

	ifp = IMAGE_FILE_OPEN(name,rw,IFT_PNG);
	if( ifp==NO_IMAGE_FILE ) return(ifp);

	ifp->if_hd = getbuf( sizeof(Png_Hdr) );
	/* We should zero the contents here!? */
	HDR_P->info_ptr = NULL;

	if( IS_READABLE(ifp) ) {

		if(get_hdr_info(ifp) < 0)
			return(NO_IMAGE_FILE);

	} else {
		/* We've only opened a file for writing, the header info
		 * will be written when we get an object to write to file.
		 */
	}

	return(ifp);
}



static int get_bgcolor(QSP_ARG_DECL  Image_File *ifp, u_char *red, u_char *green, u_char *blue)
{
	png_color_16p pBackground;
	png_infop info_ptr;
	png_byte bit_depth,color_type;

	info_ptr = HDR_P->info_ptr;

#ifdef OLD_PNG_LIB
	bit_depth = info_ptr->bit_depth;
	color_type = info_ptr->color_type;
#else
	bit_depth = png_get_bit_depth(HDR_P->png_ptr,HDR_P->info_ptr);
	color_type = png_get_color_type(HDR_P->png_ptr,HDR_P->info_ptr);
#endif

	/* setjmp() must be called in every function that calls a PNG-reading
	 * libpng function */

	if (setjmp(png_jmpbuf(HDR_P->png_ptr))) {
		pngfio_close(QSP_ARG  ifp);
		return 2;
	}

	if (!png_get_valid(HDR_P->png_ptr, info_ptr, PNG_INFO_bKGD))
		return 1;

	/* it is not obvious from the libpng documentation, but this function
	 * takes a pointer to a pointer, and it always returns valid red, green
	 * and blue values, regardless of color_type: */

	png_get_bKGD(HDR_P->png_ptr, info_ptr, &pBackground);

	/* however, it always returns the raw bKGD data, regardless of any
	 * bit-depth transformations, so check depth and adjust if necessary */

	if (bit_depth == 16) {
		*red   = pBackground->red   >> 8;
		*green = pBackground->green >> 8;
		*blue  = pBackground->blue  >> 8;
	} else if (color_type == PNG_COLOR_TYPE_GRAY && bit_depth < 8) {
		if (bit_depth == 1)
			*red = *green = *blue = pBackground->gray? 255 : 0;
		else if (bit_depth == 2)
			*red = *green = *blue = (255/3) * pBackground->gray;
		else /* bit_depth == 4 */
			*red = *green = *blue = (255/15) * pBackground->gray;
	} else {
		*red   = (u_char)pBackground->red;
		*green = (u_char)pBackground->green;
		*blue  = (u_char)pBackground->blue;
	}

	return 0;
}

static u_char *get_image( QSP_ARG_DECL  Image_File *ifp, u_long *pRowbytes )
{
	png_uint_32  i, rowbytes;
	png_bytepp  row_pointers = NULL;
	u_char *png_image_data;

	/* setjmp() must be called in every function that calls a PNG-reading
	* libpng function */

	if (setjmp(png_jmpbuf(HDR_P->png_ptr))) {
		pngfio_close(QSP_ARG  ifp);
		return (u_char *)NULL;
	}

	*pRowbytes = rowbytes = png_get_rowbytes(HDR_P->png_ptr, HDR_P->info_ptr);

	if ((png_image_data = (u_char *)getbuf(rowbytes*HDR_P->height)) == NULL) {
		WARN("get_image:  getbuf error #1!?");
		pngfio_close(QSP_ARG  ifp);
		return (u_char *)NULL;
	}

	if ((row_pointers = (png_bytepp)getbuf(HDR_P->height*sizeof(png_bytep))) == NULL) {
		WARN("get_image:  getbuf error #2!?");
		pngfio_close(QSP_ARG  ifp);
		givbuf(png_image_data);
		png_image_data = NULL;
		return (u_char *)NULL;
	}

	/* set the individual row_pointers to point at the correct offsets */

	for (i = 0;  i < HDR_P->height;  ++i)
		row_pointers[i] = png_image_data + i*rowbytes;

	/* now we can go ahead and just read the whole image */
	png_read_image(HDR_P->png_ptr, row_pointers);

	/* and we're done!  (png_read_end() can be omitted if no processing of
	* post-IDAT text/time/etc. is desired) */

	givbuf(row_pointers);
	row_pointers = NULL;

	png_read_end(HDR_P->png_ptr, NULL);

	/* note that the image data needs to be freed sometime!? */

	return png_image_data;
}


FIO_RD_FUNC( pngfio_rd )
{
	// u_char *data_ptr;
	u_long rowbytes;
	u_char *png_image_data;
	u_char *src;
	u_char *dst;
	dimension_t row,col,comp;

	if(ifp->if_nfrms) {
		advise("ERROR: png format does not have a stream of frames!");
		exit(1);
	}

//advise("png_rd calling expand_image");
//	if(expand_image(ifp) < 0)
//		error1("error in expand_image");

	/* make sure that the sizes match */
	if( ! dp_same_dim(QSP_ARG  dp,ifp->if_dp,0,"png_rd") ) return;	/* same # components? */
	if( ! dp_same_dim(QSP_ARG  dp,ifp->if_dp,1,"png_rd") ) return;	/* same # columns? */
	if( ! dp_same_dim(QSP_ARG  dp,ifp->if_dp,2,"png_rd") ) return;	/* same # rows? */

	// data_ptr = dp->dt_data;

	png_image_data = (u_char *)NULL;

	if( (png_image_data = get_image( QSP_ARG  ifp, &rowbytes )) == NULL ) {
		advise("error returned from get_image!");
		pngfio_close(QSP_ARG  ifp);
		return;
	}

	/* This function returns the background info. In its absence we can
	 * specify the background color ourselves or just leave it alone to
	 * default to black.
	 */
	get_bgcolor(QSP_ARG  ifp, &bg_red, &bg_green, &bg_blue);

	if( HDR_P->channels != dp->dt_comps ){
		sprintf(error_string,"png_rd:  file %s has %d channels, but object %s has depth %d!?",
			ifp->if_name,HDR_P->channels,dp->dt_name,dp->dt_comps);
		WARN(error_string);
		return;
	}

	for (row = 0;  row < dp->dt_rows;  row++ ) {
		src = png_image_data + row*rowbytes;
		dst = ((u_char *)dp->dt_data) + row*dp->dt_rowinc;
		for (col = 0;  col < dp->dt_cols;  col++ ) {
			for(comp=0;comp<dp->dt_comps;comp++)
				*dst++ = *src++;
		}
	}

	/* Now free the source image data ...
	 * BUG:  it would be much more efficient to simply pass the data_obj's data
	 * address to get_image, eliminating another frame allocation and the above copy...
	 */
	givbuf(png_image_data);

	ifp->if_nfrms = 1;

	if( FILE_FINISHED(ifp) ){
		if( verbose ){
			sprintf(error_string,
				"closing file \"%s\" after reading %d frames",
				ifp->if_name,ifp->if_nfrms);
			advise(error_string);
		}
		pngfio_close(QSP_ARG  ifp);
	}
}


static short prec_to_bitdepth(QSP_ARG_DECL  mach_prec prec)
{
	short bit_depth;

	switch(prec) {
		case PREC_BY:
		case PREC_UBY:
			bit_depth = 8;
			break;

		case PREC_IN:
		case PREC_UIN:
			bit_depth = 16;
			break;

		default:
			sprintf(ERROR_STRING,"prec_to_bitdepth:  unexpected precision %s",
				name_for_prec(prec));
			WARN(ERROR_STRING);
			return -1;
	}

	return bit_depth;
}


FIO_WT_FUNC( pngfio_wt )
{
	png_infop png_info_ptr;
	int bit_depth;
	int color_type;
	dimension_t k;
	int bytes_per_pixel = dp->dt_comps;
	png_bytep row_pointers[dp->dt_rows];


	if( ifp->if_dp == NO_OBJ ){	/* first time? */
		/* what should be here? */
	} else {
		advise("png_wt:  obj exists");
		/* BUG need to make sure that this image matches if_dp */
	}

	/* Create and initialize the png_struct with the desired error handler
	* functions.  If you want to use the default stderr and longjump method,
	* you can supply NULL for the last three parameters.  We also check that
	* the library version is compatible with the one used at compile time,
	* in case we are using dynamically linked libraries.
	*/

	HDR_P->png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
 
	if (!HDR_P->png_ptr)	return(-1);

#ifdef FOOBAR
	/* this sets the zlib compression level (this is slow and the effect is
	 * not that great, eg. 148.1K got compressed to 146.7K and 87.4K went down
	 * to 76.8K).
	 */
	png_set_compression_level(HDR_P->png_ptr, Z_BEST_COMPRESSION);
#endif /* FOOBAR */

	/* Allocate/initialize the image information data. */
	png_info_ptr = png_create_info_struct(HDR_P->png_ptr);

	if (!png_info_ptr) {
		png_destroy_read_struct(&HDR_P->png_ptr, NULL, NULL);
		return(-1);   /* out of memory */
	}

	/* Set error handling. Required since we are not supplying our own
	 * error handling functions in the png_create_write_struct() call.
	 */
	if (setjmp(png_jmpbuf(HDR_P->png_ptr))) {
		png_destroy_read_struct(&HDR_P->png_ptr, &png_info_ptr, NULL);
		return(-1);
	}

	/* An I/O initialization function is required */
	/* Since we are using standard C streams we need to set up output
	 * control. */
	png_init_io(HDR_P->png_ptr, ifp->if_fp);

	if((bit_depth = prec_to_bitdepth(QSP_ARG  (mach_prec)dp->dt_prec)) < 0) {
		return(-1);
	}


	/* if dp->dt_comps==3 grey,rgb,palette
	 * if dp->dt_comps==4 grey_alpha,rgb_alpha */

	/* BUG:
	 * right now we'll just assume that
	 * dp->dt_comps==3 implies rgb
	 * and
	 * dp->dt_comps==4 implies rgb_alpha */

	/* we do ini mini mina mo if color type hasn't been given */
	if(color_type_to_write < 0) {

		if( dp->dt_comps == 3 )
			color_type = PNG_COLOR_TYPE_RGB;

		else if( dp->dt_comps == 4 )
			color_type = PNG_COLOR_TYPE_RGB_ALPHA;
		else {
			sprintf(error_string,
				"Object %s has bad number of components (%d) for png",
							dp->dt_name,dp->dt_comps);
			WARN(error_string);
			return(-1);
		}

	} else {
		color_type = color_type_to_write;
	}


	/* The image information is set here.  Width and height are up to 2^31,
	 * bit_depth is one of 1, 2, 4, 8, or 16, but valid values also depend on
	 * the color_type selected. color_type is one of PNG_COLOR_TYPE_GRAY,
	 * PNG_COLOR_TYPE_GRAY_ALPHA, PNG_COLOR_TYPE_PALETTE, PNG_COLOR_TYPE_RGB,
	 * or PNG_COLOR_TYPE_RGB_ALPHA. interlace is either PNG_INTERLACE_NONE or
	 * PNG_INTERLACE_ADAM7, and the compression_type and filter_type MUST
	 * currently be PNG_COMPRESSION_TYPE_BASE and PNG_FILTER_TYPE_BASE.
	 */
	png_set_IHDR(HDR_P->png_ptr, png_info_ptr, dp->dt_cols,
		dp->dt_rows, bit_depth, color_type, PNG_INTERLACE_NONE,
		PNG_COMPRESSION_TYPE_BASE, PNG_FILTER_TYPE_BASE);

	//png_info_ptr->channels = dp->dt_comps;

#if 0 /* HAVE_PALETTE */
	/* set the palette if there is one.  REQUIRED for indexed-color images */
	palette = (png_colorp)png_malloc(HDR_P->png_ptr, PNG_MAX_PALETTE_LENGTH
							* sizeof (png_color));
	/* ... set palette colors ... */
	png_set_PLTE(HDR_P->png_ptr, info_ptr, palette, PNG_MAX_PALETTE_LENGTH);

#endif /* HAVE_PALETTE */

	/* Write the file header information. */
	png_write_info(HDR_P->png_ptr, png_info_ptr);

	for (k = 0; k < dp->dt_rows; k++)
		row_pointers[k] = ((png_bytep)dp->dt_data) + k*dp->dt_cols*bytes_per_pixel;

	/* the output method */
	png_write_image(HDR_P->png_ptr, row_pointers);

#if 0 /* FOOBAR */
	/* we could also use the following output method
	 * (it doesn't make a difference here) */
	for (k = 0; k < dp->dt_rows; k++) {
		png_write_rows(HDR_P->png_ptr, &row_pointers[k], 1);
	}
#endif /* FOOBAR */

	/* finish writing the rest of the file */
	png_write_end(HDR_P->png_ptr, png_info_ptr);

	/* clean up after the write, and free any memory allocated */
	png_destroy_write_struct(&HDR_P->png_ptr, &png_info_ptr);

	/* close the file */
//	pngfio_close(QSP_ARG  ifp);

	ifp->if_nfrms = 1;
	/* or should I do ifp->if_nfrms ++; */

	return(0);
}


void pngfio_info(QSP_ARG_DECL  Image_File *ifp)
{
	sprintf(msg_str,"\tnumber of rows in header %ld", HDR_P->height);
	prt_msg(msg_str);

	sprintf(msg_str,"\tnumber of channels %d", HDR_P->channels);
	prt_msg(msg_str);

	sprintf(msg_str,"\tbits per channel %d", HDR_P->bit_depth);
	prt_msg(msg_str);

	sprintf(msg_str,"\tbits per pixel %d", HDR_P->pixel_depth);
	prt_msg(msg_str);

	switch(HDR_P->color_type) {

		case PNG_COLOR_TYPE_GRAY:
			prt_msg("\tcolor type PNG_COLOR_TYPE_GRAY");
			break;

		case PNG_COLOR_TYPE_PALETTE:
			prt_msg("\tcolor type PNG_COLOR_TYPE_PALETTE");
			break;

		case PNG_COLOR_TYPE_RGB:
			prt_msg("\tcolor type PNG_COLOR_TYPE_RGB");
			break;

		case PNG_COLOR_TYPE_RGB_ALPHA:
			prt_msg("\tcolor type PNG_COLOR_TYPE_RGB_ALPHA");
			break;

		case PNG_COLOR_TYPE_GRAY_ALPHA:
			prt_msg("\tcolor type PNG_COLOR_TYPE_GRAY_ALPHA");
			break;

		default:
			sprintf(msg_str,"\tunknown color type %d", HDR_P->color_type);
			prt_msg(msg_str);
	}

	switch(HDR_P->interlace_type) {
		case PNG_INTERLACE_NONE:
			sprintf(msg_str,"\tinterlace type PNG_INTERLACE_NONE");
			prt_msg(msg_str);
			break;

		case PNG_INTERLACE_ADAM7:
			sprintf(msg_str,"\tinterlace type PNG_INTERLACE_ADAM7");
			prt_msg(msg_str);
			break;
	}
}


/* set_bg_color() & set_color_type() called from pngmenu.c */
void set_bg_color(int bg_color)
{
	bg_red   = (bg_color & 0xFF0000) >> 16;
	bg_green = (bg_color & 0x00FF00) >> 8;
	bg_blue  = (bg_color & 0x0000FF);
}

void set_color_type(int color_type)
{
	switch(color_type) {
		case 0:
			color_type_to_write = PNG_COLOR_TYPE_GRAY;
			break;

		case 1:
			color_type_to_write = PNG_COLOR_TYPE_PALETTE;
			break;

		case 2:
			color_type_to_write = PNG_COLOR_TYPE_RGB;
			break;

		case 3:
			color_type_to_write = PNG_COLOR_TYPE_RGB_ALPHA;
			break;

		case 4:
			color_type_to_write = PNG_COLOR_TYPE_GRAY_ALPHA;
			break;

		default:
			sprintf(msg_str,"unknown color type %d", color_type);
			NWARN(msg_str);
	}
}


int png_seek_frame(Image_File *ifp,dimension_t n)
{
	NWARN("png_seek_frame:  not implemented!?");
	return(0);
}


/* the unconvert routine creates a disk header */

int pngfio_unconv(void *hdr_pp,Data_Obj *dp)
{
	NWARN("png_unconv() not implemented!?");
	return(-1);
}


int pngfio_conv(Data_Obj *dp,void *hd_pp)
{
	NWARN("png_conv not implemented");
	return(-1);
}

#endif /* HAVE_PNG */
