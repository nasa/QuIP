
#ifdef INC_VERSION
char VersionId_inc_lshdr[] = QUIP_VERSION_STRING;
#endif /* INC_VERSION */

/* lumisys header is 2048 bytes */

/* from toto@fovea.pndr.upenn.edu:

The Lumisys header is 2048 bytes long and is followed by 12 bit image
data (0-4095).  The 0 (zero) value in the image file represents the
WHITE on the original film, so the image will appear with black bones
unless you invert the greyscale by subtracting image values from 4095.

	newvalue = 4095 - oldvalue

The x and y dimensions of the image are in bytes 806 and 808 respectively,
each consuming two bytes.  Since the image was scanned on a Little
Endian (80486), if you are reading the image on a Big Endian (Sparc, M68000,
HP, etc.) you will need to byteswap every word (16 bits) of data that you read,
including the data in the header.

*/

typedef struct ls_header {
	char	ls_stuff[806];
	short	ls_width;		/* 808 */
	short	ls_height;		/* 810 */
	char	ls_rest[1238];		/* 1238+806+4=2048 */
} Lumisys_Hdr;

