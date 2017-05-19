#include "quip_config.h"

#include <stdio.h>

#ifdef HAVE_UNISTD_H
#include <unistd.h>
#endif

#ifdef HAVE_STRING_H
#include <string.h>
#endif

#ifdef HAVE_SYS_TYPES_H
#include <sys/types.h>
#endif

#ifdef HAVE_SYS_TIME_H
#include <sys/time.h>		/* gettimeofday */
#endif

#ifdef HAVE_SYS_STAT_H
#include <sys/stat.h>
#endif

#ifdef HAVE_SYS_IOCTL_H
#include <sys/ioctl.h>
#endif

#ifdef HAVE_SYS_MMAN_H
#include <sys/mman.h>
#endif

#ifdef HAVE_FCNTL_H
#include <fcntl.h>
#endif

#ifdef HAVE_LINUX_FB_H
#include <linux/fb.h>
#endif

#include "data_obj.h"
//#include "dataprot.h"
#include "xsupp.h"
//#include "img_file.h"
#include "my_fb.h"
#include "view_cmds.h"

ITEM_INTERFACE_DECLARATIONS(FB_Info,fbi,0)

static FB_Info *curr_fbip=NO_FBI;

#define DEFAULT_FB_DEVICE	"/dev/fb0"

#define INSURE_FB(subrt_name)						\
									\
	if( curr_fbip == NO_FBI ){					\
		sprintf(ERROR_STRING,					\
		"%s:  no frame buffer open, trying %s",subrt_name,	\
			DEFAULT_FB_DEVICE);				\
		WARN(ERROR_STRING);					\
		fb_open(QSP_ARG DEFAULT_FB_DEVICE);			\
		if( curr_fbip == NO_FBI ){				\
			sprintf(ERROR_STRING,				\
			"unable to open default frame buffer %s",	\
				DEFAULT_FB_DEVICE);			\
			ERROR1(ERROR_STRING);				\
		}							\
	}


/* fb_open assumes that this frame buffer has not been opened before, and so creates
 * a new item.
 */

static void fb_open(QSP_ARG_DECL const char *fb_name)
{
#ifdef HAVE_FB_DEV

	Dimension_Set dimset;
	FB_Info *fbip;

	long nbytes;

	fbip = new_fbi(QSP_ARG  fb_name);
	if( fbip == NO_FBI ) return;

	fbip->fbi_fd = open(fb_name,O_RDWR);
	if( fbip->fbi_fd < 0 ){
		perror(fb_name);
		sprintf(ERROR_STRING,"couldn't open device %s",fb_name);
		WARN(ERROR_STRING);
		/* BUG? - do we need any more cleanup? */
		del_fbi(QSP_ARG  fbip);
		return;
	}

	if (ioctl(fbip->fbi_fd,FBIOGET_VSCREENINFO, &fbip->fbi_var_info)<0) {
		perror("ioctl error getting variable screeninfo");
		return;
	}	
	/* see if the vbl sync flags are set... */

	if (ioctl(fbip->fbi_fd,FBIOGET_FSCREENINFO, &fbip->fbi_fix_info)<0) {
		perror("ioctl error getting fixed screeninfo");
		return;
	}	

	SET_DS_COMPS(&dimset, fbip->fbi_var_info.bits_per_pixel/8 );
	SET_DS_COLS(&dimset, fbip->fbi_var_info.xres_virtual );
	SET_DS_ROWS(&dimset, fbip->fbi_var_info.yres_virtual );
	SET_DS_FRAMES(&dimset, 1 );
	SET_DS_SEQS(&dimset, 1 );

	fbip->fbi_dp = _make_dp(QSP_ARG  fb_name,&dimset,PREC_FOR_CODE(PREC_UBY));
	if( fbip->fbi_dp == NO_OBJ ){
		sprintf(ERROR_STRING,"Unable to create data object structure for %s",fb_name);
		WARN(ERROR_STRING);
		close(fbip->fbi_fd);
		del_fbi(QSP_ARG  fbip);
		return;
	}

	nbytes = OBJ_ROWS(fbip->fbi_dp) * OBJ_COLS(fbip->fbi_dp)
		* OBJ_COMPS(fbip->fbi_dp);

sprintf(ERROR_STRING,"mapping frame buffer device, %ld (0x%lx) bytes (%ld Mb)",nbytes,nbytes,nbytes/(1024*1024));
advise(ERROR_STRING);

	if( (OBJ_DATA_PTR(fbip->fbi_dp)=mmap(0,nbytes,PROT_READ|PROT_WRITE,MAP_SHARED,fbip->fbi_fd,0)) == MAP_FAILED ){
		perror("mmap /dev/fb");
		close(fbip->fbi_fd);
		fbip->fbi_fd = -1;
		del_fbi(QSP_ARG  fbip);
		return;
	}

	curr_fbip = fbip;
#else /* ! HAVE_FB_DEV */
	WARN("No frame buffer device present at config time!?");
#endif /* ! HAVE_FB_DEV */
}

static COMMAND_FUNC( do_open_fb_dev )
{
	const char *s;
	FB_Info *save_fbip, *fbip;

	s=NAMEOF("frame buffer device");

	/* See if requested frame buffer is the current frame buffer */
	if( curr_fbip != NO_FBI && !strcmp(s,curr_fbip->fbi_name) ){
		sprintf(ERROR_STRING,"Frame buffer device %s is already the current frame buffer.",s);
		advise(ERROR_STRING);
		return;
	}

	/* See if requested frame buffer is already open*/
	fbip = fbi_of(QSP_ARG  s);
	if( fbip != NO_FBI ){
		curr_fbip = fbip;
		sprintf(ERROR_STRING,"Frame buffer device %s is already open, making current.",s);
		advise(ERROR_STRING);
		return;
	}

	save_fbip = curr_fbip;		/* save */
	fb_open(QSP_ARG s);

	if( curr_fbip == NO_FBI ){
		sprintf(ERROR_STRING,"unable to open frame buffer device %s",s);
		WARN(ERROR_STRING);
		curr_fbip = save_fbip;	/* un-save */
		if( curr_fbip != NO_FBI ){
			sprintf(ERROR_STRING,"Reverting to previous frame buffer device %s",curr_fbip->fbi_name);
			advise(ERROR_STRING);
		}
	}
}

static COMMAND_FUNC( do_select_fb_dev )
{
	FB_Info *fbip;

	fbip = PICK_FBI("frame buffer device");
	if( fbip == NO_FBI ) return;

	curr_fbip = fbip;
}

static void fb_load(QSP_ARG_DECL Data_Obj *dp,int x, int y)
{
#ifdef HAVE_FB_DEV
	dimension_t i,j;
	/* char *p,*q; */	/* BUG probably a lot faster if we cast to long! */
	long *p,*q;
	u_long bytes_per_row, words_per_row;

	/* BUG assume dp is the right kind of object */

	if( ! IS_CONTIGUOUS(dp) ){
		sprintf(ERROR_STRING,"fb_load:  object %s must be contiguous",
			OBJ_NAME(dp));
		WARN(ERROR_STRING);
		return;
	}

	INSURE_FB("fb_load");

	p=(long *)OBJ_DATA_PTR(dp);
	q=(long *)OBJ_DATA_PTR(curr_fbip->fbi_dp);

	bytes_per_row = OBJ_COLS(dp) * OBJ_COMPS(dp);
	words_per_row = bytes_per_row / sizeof(long);

	for(i=0;i<OBJ_ROWS(dp);i++){
		/* BUG we need to correct the row ptr if dp is narrower than the display */
		for(j=0;j<words_per_row;j++)
			*q++ = *p++;
	}
#endif /* HAVE_FB_DEV */
}

static void fb_save(QSP_ARG_DECL Data_Obj *dp,int x, int y)
{
#ifdef HAVE_FB_DEV
	dimension_t i,j,k;
	char *p,*q;

	/* BUG assume dp is the right kind of object */

	if( ! IS_CONTIGUOUS(dp) ){
		sprintf(ERROR_STRING,"fb_save:  object %s must be contiguous",
			OBJ_NAME(dp));
		WARN(ERROR_STRING);
		return;
	}

	INSURE_FB("fb_save");

	p=(char *)OBJ_DATA_PTR(dp);
	q=(char *)OBJ_DATA_PTR(curr_fbip->fbi_dp);

	/* BUG this byte-at-a-time copy is horribly inefficient */
	for(i=0;i<OBJ_ROWS(dp);i++)
		for(j=0;j<OBJ_COLS(dp);j++)
			for(k=0;k<OBJ_COMPS(dp);k++)
				*p++ = *q++;
#endif /* HAVE_FB_DEV */
}

static COMMAND_FUNC( do_save_fb )
{
	Data_Obj *dp;
	int x,y;

	dp=PICK_OBJ("");
	x=(int)HOW_MANY("x origin");
	y=(int)HOW_MANY("y origin");

	if( dp == NO_OBJ ) return;

	INSIST_RAM_OBJ(dp,"save_fb")

	fb_save(QSP_ARG dp,x,y);
}

static COMMAND_FUNC( do_load_fb )
{
	Data_Obj *dp;
	int x,y;

	dp=PICK_OBJ("");
	x=(int)HOW_MANY("x origin");
	y=(int)HOW_MANY("y origin");

	if( dp == NO_OBJ ) return;

	INSIST_RAM_OBJ(dp,"load_fb")

	fb_load(QSP_ARG dp,x,y);
}

/* BUG this should be allocated dynamically */

static COMMAND_FUNC( do_test_clock )
{
	int i,n;
	struct timeval *tv_tbl;

	n = (int)HOW_MANY("number of times to read the clock");

	tv_tbl = (struct timeval *) getbuf(n*sizeof(struct timeval));

	/* We've observed on poisson that vblank loops have gaps of around 80 msec which double the elapsed time!?
	 * Can we observe this effect with a simple clock read?
	 * The number of values we have to store depends on the time per read...
	 * If it's 1 usec, then we need 100k reads to take 100 msec...
	 */
	for(i=0;i<n;i++){
		gettimeofday(&tv_tbl[i],NULL);
	}
	/* Now show */
	for(i=0;i<n;i++){
		sprintf(msg_str,"%ld\t%ld",(long)tv_tbl[i].tv_sec,(long)tv_tbl[i].tv_usec);
		prt_msg(msg_str);
	}

	givbuf(tv_tbl);
}

#if defined(FBIOGET_VBLANK)

static int vbl_capability_flags=0;

#define VBL_CAPABILITY_CHECKED_BIT	1
#define HAS_VBL_CAPABILITY_BIT		2

#define VBL_CAPABILITY_CHECKED		(vbl_capability_flags&VBL_CAPABILITY_CHECKED_BIT)
#define HAS_VBL_CAPABILITY		(vbl_capability_flags&HAS_VBL_CAPABILITY_BIT)

static COMMAND_FUNC( do_fb_vblank )
{
	struct fb_vblank vbl_info;
	__u32 initial_count, current_count;


	INSURE_FB("do_fb_vblank")

	if( !VBL_CAPABILITY_CHECKED ){
		/* The first call is to fetch the flags to make sure we have the capability */
		if(ioctl(curr_fbip->fbi_fd, FBIOGET_VBLANK, &vbl_info)<0) {
			perror("ioctl");
			WARN("ioctl FBIOGET_VBLANK failed!\n");
			vbl_capability_flags |= VBL_CAPABILITY_CHECKED;
			return;
		}
		if( (vbl_info.flags & FB_VBLANK_HAVE_VBLANK) == 0 ){
			sprintf(ERROR_STRING,"Sorry, %s does not support vertical blanking detection",curr_fbip->fbi_name);
			WARN(ERROR_STRING);
			vbl_capability_flags |= VBL_CAPABILITY_CHECKED;
			return;
		}
		vbl_capability_flags |= VBL_CAPABILITY_CHECKED_BIT;
		vbl_capability_flags |= HAS_VBL_CAPABILITY_BIT;
	}

	if( ! HAS_VBL_CAPABILITY ){
		WARN("device does not have VBL capability!?");
		return;
	}

	/* vcount has the number of lines, count has the number of frames */


	if(ioctl(curr_fbip->fbi_fd, FBIOGET_VBLANK, &vbl_info)<0) {
		perror("ioctl");
		WARN("ioctl FBIOGET_VBLANK failed!\n");
		return;
	}
	initial_count = vbl_info.count;
	do {
		if(ioctl(curr_fbip->fbi_fd, FBIOGET_VBLANK, &vbl_info)<0) {
			perror("ioctl");
			WARN("ioctl FBIOGET_VBLANK failed!\n");
			return;
		}
		current_count = vbl_info.count;
	} while( current_count == initial_count );

/*
sprintf(ERROR_STRING,"initial_count = %d, final count = %d",initial_count,current_count);
advise(ERROR_STRING);
*/
	if( (current_count - initial_count) != 1 ){
		/* wrap around? */
		if( initial_count == 0xffffffff && current_count == 0 ){
			/* do nothing */
		} else if( current_count > initial_count ){
			sprintf(ERROR_STRING,"do_fb_vblank:  seem to have missed %d frames!?",current_count-initial_count-1);
			WARN(ERROR_STRING);
		} else {
			sprintf(ERROR_STRING,"do_fb_vblank:  seem to have some frames, current_count = %d, initial_count = %d.",
				current_count,initial_count);
			WARN(ERROR_STRING);
		}
	}

#ifdef FOOBAR
//sprintf(ERROR_STRING,"vbl data = 0x%x, vcount = %d",vbl_info.flags,vbl_info.vcount);
//advise(ERROR_STRING);
/*
sprintf(ERROR_STRING,"vbl data = 0x%lx, FB_VBLANK_VBLANKING = 0x%lx",vbl_info.flags,FB_VBLANK_VBLANKING);
advise(ERROR_STRING);
*/

	while( vbl_info.flags & FB_VBLANK_VBLANKING ){	/* wait for current interval to end */
		if(ioctl(curr_fbip->fbi_fd, FBIOGET_VBLANK, &vbl_info)<0) {
			perror("ioctl");
			WARN("ioctl FBIOGET_VBLANK failed!\n");
			return;
		}
sprintf(ERROR_STRING,"vbl data = 0x%x, vcount = %d",vbl_info.flags,vbl_info.vcount);
advise(ERROR_STRING);
	}

	while( (vbl_info.flags & FB_VBLANK_VBLANKING) == 0 ){	/* wait for next interval to start */
		/* for fastest response, don't sleep here */
		if(ioctl(curr_fbip->fbi_fd, FBIOGET_VBLANK, &vbl_info)<0) {
			perror("ioctl");
			WARN("ioctl FBIOGET_VBLANK failed!\n");
			return;
		}
sprintf(ERROR_STRING,"vbl data = 0x%x, count = %u, vcount = %u, hcount = %u",vbl_info.flags,
	vbl_info.count,vbl_info.vcount,vbl_info.hcount);
advise(ERROR_STRING);
	}
#endif /* FOOBAR */
#ifdef FOOBAR
	if(ioctl(curr_fbip->fbi_fd, FBIO_WAITFORVSYNC, &p)<0) {
		perror("ioctl");
		WARN("ioctl FBIO_WAITFORVSYNC failed!\n");
		return (void *) -1;
	}
#endif /* FOOBAR */
}

#endif /* define(FBIOGET_VBLANK) */



#ifdef FOOBAR
static void dump_screen_info( struct fb_var_screeninfo *fbvp )
{
#ifdef HAVE_FB_DEV
	printf("nonstd:%d activate:%d\n",
			fbvp->nonstd, fbvp->activate);
	printf("\nres:%dx%d  bpp:%d\n",
			fbvp->xres, fbvp->yres, fbvp->bits_per_pixel);
	printf("y_os:%d x_os:%d\n",
			fbvp->xoffset, fbvp->yoffset);
	printf("grayscale:%d\nvirt_res:%dx%d\nmm_ht:%d, mm_width:%d\n",
			fbvp->grayscale, fbvp->xres_virtual,
			fbvp->yres_virtual, 
			fbvp->height, fbvp->width);
	printf("\nred: os:%d len:%d msb_right:%d \n",
			fbvp->red.offset, fbvp->red.length,
			fbvp->red.msb_right);
	printf("green: os:%d len:%d msb:%d \n",
			fbvp->green.offset, fbvp->green.length,
			fbvp->green.msb_right);
	printf("blue: os:%d len:%d msb:%d \n",
			fbvp->blue.offset, fbvp->blue.length,
			fbvp->blue.msb_right);
	printf("transp: os:%d len:%d msb:%d \n",
			fbvp->transp.offset, fbvp->transp.length,
			fbvp->transp.msb_right);
#endif /* HAVE_FB_DEV */

}
#endif /* FOOBAR */

static COMMAND_FUNC( do_fb_pan )
{
#ifdef HAVE_FB_DEV
	__u32 dx,dy;

	INSURE_FB("do_fb_pan")

	dx = HOW_MANY("xoffset");
	dy = HOW_MANY("yoffset"); 

	/* BUG make sure values are valid */

	/* with genlock running, we need to lock... */
	curr_fbip->fbi_var_info.xoffset = dx;
	curr_fbip->fbi_var_info.yoffset = dy;
	
	/* mutex might be needed for multi-threaded operation */
	/* if( genlock_active ) get_genlock_mutex(); */

	if(ioctl(curr_fbip->fbi_fd, FBIOPAN_DISPLAY, &curr_fbip->fbi_var_info)<0) {
		tell_sys_error("ioctl");
		sprintf(ERROR_STRING,"do_fb_pan:  ioctl iopan %d %d error\n",dx,dy);
	        WARN(ERROR_STRING);
	        return;
	}

	/* if( genlock_active ) rls_genlock_mutex(); */

	return;
#else /* ! HAVE_FB_DEV */
	ERROR1("do_fb_pan:  Program not configured with framebuffer device support.");
#endif /* ! HAVE_FB_DEV */
}

static COMMAND_FUNC( do_new_fb_pan )
{
#ifdef HAVE_FB_DEV
	__u32 dx,dy;

	INSURE_FB("do_fb_pan")

	dx = HOW_MANY("xoffset");
	dy = HOW_MANY("yoffset"); 

	/* BUG make sure values are valid */

	curr_fbip->fbi_var_info.xoffset = dx;
	curr_fbip->fbi_var_info.yoffset = dy;
	curr_fbip->fbi_var_info.activate |= FB_ACTIVATE_VBL;
	
	if(ioctl(curr_fbip->fbi_fd, FBIOPAN_DISPLAY, &curr_fbip->fbi_var_info)<0) {
		tell_sys_error("ioctl");
	        sprintf(ERROR_STRING,"do_new_fb_pan:  ioctl iopan %d %d error\n",dx,dy);
		WARN(ERROR_STRING);
	        return;
	}

	return;
#else /* ! HAVE_FB_DEV */
	ERROR1("do_new_fb_pan:  Program not configured with framebuffer device support.");
#endif /* ! HAVE_FB_DEV */
}

#ifdef HAVE_FB_DEV
static void show_var_info(QSP_ARG_DECL  FB_Info *fbip)
{
	/* Now display the contents */
	sprintf(msg_str,"Frame buffer %s:",fbip->fbi_name);			prt_msg(msg_str);

	sprintf(msg_str,"\tResolution:\t%ld x %ld",	CFBVAR(fbip,xres),CFBVAR(fbip,yres));	prt_msg(msg_str);
	sprintf(msg_str,"\tVirtual:\t%ld x %ld",	CFBVAR(fbip,xres_virtual),CFBVAR(fbip,yres_virtual));	prt_msg(msg_str);
	sprintf(msg_str,"\tOffset:\t%ld , %ld",		CFBVAR(fbip,xoffset),CFBVAR(fbip,yoffset));	prt_msg(msg_str);
	sprintf(msg_str,"\tBitsPerPixel:\t%ld",		CFBVAR(fbip,bits_per_pixel));	prt_msg(msg_str);
	sprintf(msg_str,"\tGrayscale:\t%ld",		CFBVAR(fbip,grayscale));		prt_msg(msg_str);
	/* bitfields for red,green,blue,transp - ? */
	if( FBVAR(fbip,nonstd) ){
	sprintf(msg_str,"\tNon-standard pixel format:\t%ld",	CFBVAR(fbip,nonstd));	prt_msg(msg_str);
	}
	sprintf(msg_str,"\tActivate:\t%ld",		CFBVAR(fbip,activate));		prt_msg(msg_str);
	sprintf(msg_str,"\tSize (mm):\t%ld x %ld",	CFBVAR(fbip,width),CFBVAR(fbip,height));	prt_msg(msg_str);
	sprintf(msg_str,"\tPixclock:\t%ld",		CFBVAR(fbip,pixclock));		prt_msg(msg_str);
	sprintf(msg_str,"\tLeft margin:\t%ld",		CFBVAR(fbip,left_margin));		prt_msg(msg_str);
	sprintf(msg_str,"\tRight margin:\t%ld",		CFBVAR(fbip,right_margin));		prt_msg(msg_str);
	sprintf(msg_str,"\tUpper margin:\t%ld",		CFBVAR(fbip,upper_margin));		prt_msg(msg_str);
	sprintf(msg_str,"\tLower margin:\t%ld",		CFBVAR(fbip,lower_margin));		prt_msg(msg_str);
	sprintf(msg_str,"\tHsync len:\t%ld",		CFBVAR(fbip,hsync_len));		prt_msg(msg_str);
	sprintf(msg_str,"\tVsync len:\t%ld",		CFBVAR(fbip,vsync_len));		prt_msg(msg_str);
	sprintf(msg_str,"\tSync:\t%ld",			CFBVAR(fbip,sync));			prt_msg(msg_str);
	sprintf(msg_str,"\tVMode:\t%ld",		CFBVAR(fbip,vmode));			prt_msg(msg_str);
	/* rotate field not present on purkinje - different kernel version? */
	/*
	sprintf(msg_str,"\tRotate:\t%ld",		CFBVAR(fbip,rotate));			prt_msg(msg_str);
	*/
}
#endif /* HAVE_FB_DEV */

static COMMAND_FUNC( do_get_var )
{
	INSURE_FB("do_get_var")

#ifdef HAVE_FB_DEV
	if(ioctl(curr_fbip->fbi_fd, FBIOGET_VSCREENINFO, &curr_fbip->fbi_var_info)<0) {
		perror("ioctl FBIOGET_VSCREENINFO");
	        return;
	}
	/*
	nc_show_var_info(curr_fbip);
	*/
	show_var_info(QSP_ARG  curr_fbip);
#endif /* HAVE_FB_DEV */
}

#ifdef HAVE_FB_DEV
static void write_back_var_info(SINGLE_QSP_ARG_DECL)
{
	INSURE_FB("do_get_var")

/*
	curr_fbip->fbi_var_info.activate |= FB_ACTIVATE_VBL;
*/
	curr_fbip->fbi_var_info.activate |= FB_ACTIVATE_NOW;
	/*
	curr_fbip->fbi_var_info.vmode |= FB_VMODE_CONUPDATE;
	*/

	if(ioctl(curr_fbip->fbi_fd, FBIOPUT_VSCREENINFO, &curr_fbip->fbi_var_info)<0) {
		perror("ioctl FBIOPUT_VSCREENINFO");
	        return;
	}
}
#endif /* HAVE_FB_DEV */

#define N_FB_LUT_ENTRIES	256
#define N_FB_LUT_COMPONENTS	1

static int good_for_lut(QSP_ARG_DECL  Data_Obj *dp)
{
	/* check for proper size and type - should be short... */
	if( dp == NULL ) return 0;

	if( OBJ_PREC(dp) != PREC_IN && OBJ_PREC(dp) != PREC_UIN ){
		sprintf(ERROR_STRING,
			"Object %s (%s) should have %s or %s precision!?",
			OBJ_NAME(dp),PREC_NAME(OBJ_PREC_PTR(dp)),
				PREC_NAME(PREC_FOR_CODE(PREC_IN)),
				PREC_NAME(PREC_FOR_CODE(PREC_UIN)) );
		WARN(ERROR_STRING);
		return 0;
	}

	if( OBJ_COLS(dp) != N_FB_LUT_ENTRIES ){
		sprintf(ERROR_STRING,
			"Object %s (%ld) should have %d columns!?",
			OBJ_NAME(dp),(long)OBJ_COLS(dp),N_FB_LUT_ENTRIES);
		WARN(ERROR_STRING);
		return 0;
	}

	if( OBJ_COMPS(dp) != N_FB_LUT_COMPONENTS ){
		sprintf(ERROR_STRING,
			"Object %s (%ld) should have %d components!?",
			OBJ_NAME(dp),(long)OBJ_COLS(dp),N_FB_LUT_COMPONENTS);
		WARN(ERROR_STRING);
		return 0;
	}
	return 1;
}

static COMMAND_FUNC( do_get_cmap )
{
	Data_Obj *red_dp, *green_dp, *blue_dp;
#ifdef HAVE_FB_DEV
	struct fb_cmap fbcm;
#endif // HAVE_FB_DEV

	red_dp = PICK_OBJ("color map RED data object");
	green_dp = PICK_OBJ("color map GREEN data object");
	blue_dp = PICK_OBJ("color map BLUE data object");

	if( ! good_for_lut(QSP_ARG  red_dp) ) return;
	if( ! good_for_lut(QSP_ARG  green_dp) ) return;
	if( ! good_for_lut(QSP_ARG  blue_dp) ) return;

	INSIST_RAM_OBJ(red_dp,"get_cmap")
	INSIST_RAM_OBJ(green_dp,"get_cmap")
	INSIST_RAM_OBJ(blue_dp,"get_cmap")

#ifdef HAVE_FB_DEV
	INSURE_FB("do_get_cmap")

	fbcm.start=0;
	fbcm.len=256;

	fbcm.red = (__u16 *)OBJ_DATA_PTR(red_dp);
	fbcm.green = (__u16 *)OBJ_DATA_PTR(green_dp);
	fbcm.blue = (__u16 *)OBJ_DATA_PTR(blue_dp);
	fbcm.transp = NULL;

	if(ioctl(curr_fbip->fbi_fd, FBIOGETCMAP, &fbcm)<0) {
		perror("ioctl FBIOGETCMAP");
	        return;
	}
#endif // HAVE_FB_DEV
}

static COMMAND_FUNC( do_set_cmap )
{
	Data_Obj *red_dp, *green_dp, *blue_dp;
#ifdef HAVE_FB_DEV
	struct fb_cmap fbcm;
#endif // HAVE_FB_DEV

	red_dp = PICK_OBJ("color map RED data object");
	green_dp = PICK_OBJ("color map GREEN data object");
	blue_dp = PICK_OBJ("color map BLUE data object");
	if( red_dp == NO_OBJ || green_dp == NO_OBJ || blue_dp == NO_OBJ ) return;

	INSIST_RAM_OBJ(red_dp,"set_cmap")
	INSIST_RAM_OBJ(green_dp,"set_cmap")
	INSIST_RAM_OBJ(blue_dp,"set_cmap")

	/* BUG check for proper size and type - should be short... */

#ifdef HAVE_FB_DEV
	INSURE_FB("do_get_cmap")

	fbcm.start=0;
	fbcm.len=256;

	fbcm.red = (__u16 *) OBJ_DATA_PTR(red_dp);
	fbcm.green = (__u16 *) OBJ_DATA_PTR(green_dp);
	fbcm.blue = (__u16 *) OBJ_DATA_PTR(blue_dp);
	fbcm.transp = NULL;

	if(ioctl(curr_fbip->fbi_fd, FBIOPUTCMAP, &fbcm)<0) {
		perror("ioctl FBIOPUTCMAP");
	        return;
	}
#endif // HAVE_FB_DEV
}

static COMMAND_FUNC( do_set_xoffset )
{

#ifdef HAVE_FB_DEV
	//__u32 os;
	uint32_t os;
    
	os = (uint32_t)HOW_MANY("x offset");
	/* BUG do range checking here */
	FBVAR(curr_fbip,xoffset) = os;

	write_back_var_info(SINGLE_QSP_ARG);
#else // ! HAVE_FB_DEV
    HOW_MANY("dummy word");
#endif // ! HAVE_FB_DEV
}

static COMMAND_FUNC( do_set_yoffset )
{
#ifdef HAVE_FB_DEV
	uint32_t os;
	
	os = (uint32_t)HOW_MANY("y offset");
	/* BUG do range checking here */
	FBVAR(curr_fbip,yoffset) = os;

	write_back_var_info(SINGLE_QSP_ARG);
#else // ! HAVE_FB_DEV
	HOW_MANY("y offset (ineffective)");
#endif // HAVE_FB_DEV
}

static COMMAND_FUNC( do_set_vxres )
{
#ifdef HAVE_FB_DEV
	uint32_t res;
    
	res = (uint32_t)HOW_MANY("virtual x resolution");
	/* BUG check range here */
	FBVAR(curr_fbip,xres_virtual) = res;

	write_back_var_info(SINGLE_QSP_ARG);
#else // ! HAVE_FB_DEV
	HOW_MANY("dummy value");
#endif // ! HAVE_FB_DEV
}

static COMMAND_FUNC( do_set_vyres )
{
#ifdef HAVE_FB_DEV
	//__u32 res;
	uint32_t res;
	
	res = (uint32_t)HOW_MANY("virtual y resolution");
	/* BUG check range here */
	
	FBVAR(curr_fbip,yres_virtual) = res;

	write_back_var_info(SINGLE_QSP_ARG);
#else // ! HAVE_FB_DEV
	HOW_MANY("virtual y resolution (ineffective)");
#endif // ! HAVE_FB_DEV
}

#define ADD_CMD(s,f,h)	ADD_COMMAND(var_info_menu,s,f,h)

MENU_BEGIN(var_info)
ADD_CMD( show,		do_get_var,	get variable fb info		)
ADD_CMD( xoffset,	do_set_xoffset,	set x offset			)
ADD_CMD( yoffset,	do_set_yoffset,	set y offset			)
ADD_CMD( xres_virtual,	do_set_vxres,	set virtual x resolution	)
ADD_CMD( yres_virtual,	do_set_vyres,	set virtual y resolution	)
MENU_END(var_info)

static COMMAND_FUNC( do_var_info )
{
	PUSH_MENU(var_info);
}

static COMMAND_FUNC( do_init_gl )
{
#ifdef HAVE_GENLOCK
	init_genlock(SINGLE_QSP_ARG);
#endif /* HAVE_GENLOCK */
}

static COMMAND_FUNC( do_gl_vbl )
{
#ifdef HAVE_GENLOCK
	genlock_vblank(&gli1);
#endif /* HAVE_GENLOCK */
}

#undef ADD_CMD
#define ADD_CMD(s,f,h)	ADD_COMMAND(fb_menu,s,f,h)

MENU_BEGIN(fb)
ADD_CMD( open,		do_open_fb_dev,		open frame buffer device )
ADD_CMD( select,	do_select_fb_dev,	select fb device for subsequent operations )
ADD_CMD( save,		do_save_fb,		save mmap frame buffer to data object )
ADD_CMD( load,		do_load_fb,		load mmap frame buffer from data object )
ADD_CMD( test_clock,	do_test_clock,		test reading the system clock )
#if defined(FBIOGET_VBLANK)
ADD_CMD( vblank,	do_fb_vblank,		wait for vertical blanking interval )
#endif /* define(FBIOGET_VBLANK) */
ADD_CMD( pan,		do_fb_pan,		pans the screen according to x-y )
ADD_CMD( var_info,	do_var_info,		variable screen parameter submenu )
ADD_CMD( vbl_pan,	do_new_fb_pan,		pans the screen at next vbl )
ADD_CMD( get_cmap,	do_get_cmap,		get framebuffer colormap )
ADD_CMD( set_cmap,	do_set_cmap,		set framebuffer colormap )
ADD_CMD( monitor_pair,	do_fbpair_monitor,	monitor the timing of a pair of framebuffers )
ADD_CMD( gl_vbl,	do_gl_vbl,		wait for v. blanking w/ genlock adjustment )
ADD_CMD( init_genlock,	do_init_gl,		initialize genlock subsystem )
ADD_CMD( test_parport,	do_test_parport,	test reading sync on pport line )
ADD_CMD( start_genlock,	do_genlock,		genlock framebuffers to external signal )
ADD_CMD( genlock_status,	do_report_genlock_status,	report genlock status )
ADD_CMD( halt_genlock,	halt_genlock,		genlock framebuffers to external signal )
MENU_END(fb)

COMMAND_FUNC( do_fb_menu )
{
	/* insure_x11_server(); */	/* not clear we really need this for *this* menu!? BUG? */
	PUSH_MENU(fb);
}



