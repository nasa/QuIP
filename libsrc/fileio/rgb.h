
#ifndef NO_RGB
#define NO_RGB

#include "quip_config.h"

#ifdef HAVE_RGB

#include "img_file.h"

/* prototypes for SGI functions
 *
 * For some reason these are in a comment in /usr/include/gl/image.h, so we can
 * have them here unconditionally (even on SGI)
 */

//extern int putrow(IMAGE *image, unsigned short *buffer, unsigned int y, unsigned int z);
//extern int getrow(IMAGE *image, unsigned short *buffer, unsigned int y, unsigned int z);

extern	void rgb_close(Image_File *ifp);
extern	int rgb_to_dp(Data_Obj *dp,IMAGE *ip);
extern	Image_File *rgb_open(const char *name,int rw);
extern	int dp_to_rgb(IMAGE *ip,Data_Obj *dp);
extern	int rgb_wt(Data_Obj *dp,Image_File *ifp);
extern	void read_rgb(Data_Obj *dp,Image_File *ifp);
extern	void rgb_rd(Data_Obj *dp,Image_File *ifp,
		index_t x_offset,index_t y_offset,index_t t_offset);
extern	int rgb_unconv(void *hd_pp ,Data_Obj *dp);
extern	int rgb_conv(Data_Obj *dp, void *hd_pp);

#endif /* HAVE_RGB */

#endif /* NO_RGB */
