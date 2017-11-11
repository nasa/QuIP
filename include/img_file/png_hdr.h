
/* png_info is defined in png.h */

/* png_info is a structure that holds the information in a PNG file so
 * that the application can find out the characteristics of the image.
 * If you are reading the file, this structure will tell you what is
 * in the PNG file.  If you are writing the file, fill in the information
 * you want to put into the PNG file, then call png_write_info().
 * The names chosen should be very close to the PNG specification, so
 * consult that document for information about the meaning of each field.
 */

//typedef png_info Png_Hdr;

typedef struct png_hdr {
	
	dimension_t	n_frames;	// jbm extension
	long		frame_size;

	png_uint_32 width;       /* width of image in pixels */
	png_uint_32 height;      /* height of image in pixels */

	png_byte bit_depth;      /* 1, 2, 4, 8, or 16 bits/channel */
	png_byte color_type;
	png_byte compression_type; /* must be PNG_COMPRESSION_TYPE_BASE */
	png_byte filter_type;    /* must be PNG_FILTER_TYPE_BASE */
	png_byte interlace_type; /* One of PNG_INTERLACE_NONE, PNG_INTERLACE_ADAM7 */

	png_byte channels;       /* number of data channels per pixel (1, 2, 3, 4)*/
	png_byte pixel_depth;    /* number of bits per pixel */

	
	png_infop info_ptr;
	png_structp png_ptr;

} Png_Hdr;


