
#ifndef BMP_HEADER

typedef struct bmp_header {
	int		bmp_compression;
	int		bmp_bytes_read;
	int		bmp_byte_offset;
	int		bmp_bit_count;
	int		bmp_rows;
	int		bmp_cols;
	u_char *	bmp_palette_p;
	int		bmp_info_size;
} BMP_Header;

#define NO_BMP_HEADER		((BMP_Header *)NULL)

#endif /* BMP_HEADER */

