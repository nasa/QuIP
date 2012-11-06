#include "quip_config.h"

char VersionId_fio_dsk[] = QUIP_VERSION_STRING;

#include <stdio.h>

#ifdef HAVE_STRING_H
#include <string.h>
#endif

#ifdef HAVE_MALLOC_H
#include <malloc.h>		/* memalign() */
#endif

#ifdef HAVE_STDLIB_H
#include <stdlib.h>	/* memalign */
#endif

#ifdef HAVE_UNISTD_H
#include <unistd.h>	/* read */
#endif

#include "fio_prot.h"
#include "filetype.h"
#include "getbuf.h"
#include "data_obj.h"
#include "debug.h"
#include "savestr.h"
#include "sir_disk.h"
#include "raw.h"
#include "readhdr.h"

#include "vldefs.h"

/* local prototypes */
static void msg_num(const char *str,int n);

//#define HDR_P	((Image_File_Hdr *)ifp->if_hd)->ifh_u.sir_dsk_hd_p
#define HDR_P	((Sir_Disk_Hdr *)&(((Image_File_Hdr *)ifp->if_hd)->ifh_u.sir_dsk_hd))

#ifdef FOOBAR
static Data_Obj sobj, *sir_dsk_dp=NO_OBJ;

static void init_sddp()
{
	dimension_t dims[N_DIMENSIONS];

	if( sir_dsk_dp != NO_OBJ ) return;

	sir_dsk_dp = &sobj;

	/* we don't need to use savestr() to store the name;
	 * this object is not in the database and therefore
	 * cannot be removed.
	 */

	sir_dsk_dp->dt_name = "sir_disk_obj";

	dims[0]=4;
	/* BUG should get these size from .h file?  from video device? */
	dims[1]=646;
	dims[2]=243;
	dims[3]=1;	/* a dummy... */
	dims[4]=1;
	if( init_dp(sir_dsk_dp,dims,PREC_BY) == NO_OBJ ){
		NWARN("can't initialize sir_dsk_dp");
		return;
	}
}
#endif /* FOOBAR */

#ifdef LINUX

typedef struct fourb {
	char b[4];
} Swap_Buf;

static void swap_long(long *lp)
{
	char b;
	Swap_Buf *sbp;

	sbp = (Swap_Buf *) lp;

	b=sbp->b[0];
	sbp->b[0] = sbp->b[3];
	sbp->b[3] = b;

	b=sbp->b[1];
	sbp->b[1] = sbp->b[2];
	sbp->b[2] = b;
}

static void swap_header(Sir_Disk_Hdr *hd_p)
{
	swap_long(&hd_p->magic);
	swap_long(&hd_p->version);
	swap_long(&hd_p->video_width);
	swap_long(&hd_p->video_height);
	swap_long(&hd_p->video_packing);
	swap_long(&hd_p->video_format);
	swap_long(&hd_p->video_timing);
	swap_long((long *)&hd_p->video_capture_type);
	swap_long(&hd_p->video_field_dominance);
	swap_long(&hd_p->block_size);
	swap_long(&hd_p->blocks_per_image);
	swap_long(&hd_p->video_start_block);
	swap_long(&hd_p->video_n_blocks);
	swap_long(&hd_p->audio_n_blocks);
}

#endif /* LINUX */

int dsk_to_dp(Data_Obj *dp,Sir_Disk_Hdr *hd_p)
{
	dp->dt_prec = PREC_BY;

	dp->dt_comps = 4;
	dp->dt_cols = hd_p->video_width;
	dp->dt_rows = hd_p->video_height;

	/* number of fields */
	dp->dt_frames = hd_p->video_n_blocks / hd_p->blocks_per_image;

	if( dp->dt_frames % 1 ){
		NWARN("Number of fields should be even for video disk format!?");
		return(-1);
	}
	dp->dt_seqs = 1;

	dp->dt_cinc = 1;
	dp->dt_pinc = 4;
	dp->dt_rowinc = hd_p->video_width*dp->dt_pinc;

	/* the frames are rounded up to block boundaries, (plus an extra block?) */
#define EXTRABLOCK	512		/* make this 0 if no extra block */
	dp->dt_finc = (incr_t)((dp->dt_comps * dp->dt_cols * dp->dt_rows) + EXTRABLOCK + 511 ) & ~511;

	dp->dt_sinc = dp->dt_finc * (incr_t)dp->dt_frames;

	dp->dt_parent = NO_OBJ;
	dp->dt_children = NO_LIST;

	dp->dt_ap = ram_area;		/* the default */
	/* dp->dt_data = hd_p->image; */
	dp->dt_data = NULL;
	dp->dt_n_type_elts = dp->dt_comps * dp->dt_cols * dp->dt_rows
			* dp->dt_frames * dp->dt_seqs;

	set_shape_flags(&dp->dt_shape,dp);

	return(0);
}

FIO_OPEN_FUNC( dsk_open )
{
	Image_File *ifp;

	ifp = IMAGE_FILE_OPEN(name,rw,IFT_DISK);

	if( ifp==NO_IMAGE_FILE ) return(ifp);

#ifdef HAVE_MEMALIGN
	ifp->if_hd = memalign(512,512); /* alignment , size */
#elif HAVE_POSIX_MEMALIGN
	if( posix_memalign((void **)(&ifp->if_hd),512,512) < 0 ){
		tell_sys_error("posix_memalign");
		NERROR1("memory allocation error");
	}
#else
	NERROR1("dsk.c:  need to find a replacement for memalign in this configuration.");
#endif
	memset(ifp->if_hd,0,512);

	if( IS_READABLE(ifp) ){
		int n;
		if( (n=read( ifp->if_fd, ifp->if_hd, 512 )) != 512 ){
			if( n < 0 ) tell_sys_error("read");
			NWARN("error reading disk header");
			close_image_file(QSP_ARG  ifp);
			return(NO_IMAGE_FILE);
		}

		/* because this is a binary file,
		 * the ordering of the long words
		 * may get messed up on some machines.
		 * The most sensible thing would be to have
		 * the data stored in network order, but
		 * since the format is native to SGI,
		 * we will stick with whatever order they use.
		 * We got lucky and weren't bit by this
		 * on the sun, but on our pentium linux box
		 * it's another story.
		 */
#ifdef LINUX
		swap_header(ifp->if_hd);
#endif /* LINUX */

		if( HDR_P->magic != SIR_DISK_MAGIC ){
			sprintf(error_string,
		"File %s has wrong magic number for disk format",
				ifp->if_name);
			NWARN(error_string);
			/* we probably should do something here */
		}

#ifdef CAUTIOUS
if( ifp->if_dp == NO_OBJ )
NERROR1("CAUTIOUS:  null if_dp for readfile");
#endif /* CAUTIOUS */

		dsk_to_dp(ifp->if_dp,ifp->if_hd);
	} else {
		/*
		Data_Obj *dp;

		if( sir_dsk_dp == NO_OBJ ) init_sddp();
		ifp->if_dp = dp = sir_dsk_dp;
		*/

		/* We set this up from sirius code
		 * so we can get the proper size from
		 * the video path pointer
		 */

		ifp->if_dp = NO_OBJ;

		/* BUG? should we do rest of header set-up??? */
	}
	return(ifp);
}

int dp_to_dsk(Sir_Disk_Hdr *hd_p,Data_Obj *dp)
{
	int bpi;

#ifdef CAUTIOUS
	if( dp == NO_OBJ ) {
		NWARN("CAUTIOUS:  dp_to_dsk:  null dp");
		return(-1);
	}
#endif /* CAUTIOUS */

	/* zero the header (it's a whole block!) */

	/* num_frame set when when write request given */

	hd_p->video_height = (int) dp->dt_rows;
	hd_p->video_width = (int) dp->dt_cols;

	/* set the other header fields */
	hd_p->magic = SIR_DISK_MAGIC;
	hd_p->audio_n_blocks = 0;
	hd_p->version = 1;
	hd_p->video_start_block = 1;

	/* should fs blocksize be soft? */
	hd_p->block_size = 512;

	bpi = (int)(((siztbl[dp->dt_prec]*dp->dt_comps * dp->dt_cols * dp->dt_rows)
			+ 511 ) / 512 );

	hd_p->blocks_per_image = bpi;

	hd_p->video_n_blocks = (int)( dp->dt_frames * bpi );

	/* we have to get the include's right to work on sun ! */
	hd_p->video_packing = VL_PACKING_RGBA_8;
	hd_p->video_format = VL_FORMAT_RGB;
	hd_p->video_capture_type = VL_CAPTURE_NONINTERLEAVED;

	/* normally timing is gotten from the device!? */
	/* normally dominance is gotten from the device!? */
	/* these values match what v2d produces... HACK */
	hd_p->video_timing = 0;
	hd_p->video_field_dominance = 1;

	return(0);
}

FIO_SETHDR_FUNC( set_dsk_hdr )
{
	int n;

#define BSIZE	512

	if( dp_to_dsk(ifp->if_hd,ifp->if_dp) < 0 ){
		/* where does this come from?? */
		GENERIC_IMGFILE_CLOSE(ifp);
		return(-1);
	}

#ifdef LINUX
	swap_header(ifp->if_hd);
#endif /* LINUX */

	if( (n=write(ifp->if_fd,ifp->if_hd,BSIZE) ) != BSIZE ){
		if( n < 0 ) tell_sys_error("write");
		else {
			sprintf(error_string,
				"%d bytes written",n);
			advise(error_string);
		}
		NWARN("error writing dsk header");
		/* BUG? clean up? close file? */
		return(-1);
	}

	return(0);
}

FIO_WT_FUNC( dsk_wt )
{
	long bpi, n;

	if( ifp->if_dp == NO_OBJ ){	/* first time? */

		/* set the rows & columns in our file struct */
		setup_dummy(ifp);
		copy_dimensions(ifp->if_dp, dp);

		ifp->if_dp->dt_frames = ifp->if_frms_to_wt;
		ifp->if_dp->dt_seqs = 1;

		if( set_dsk_hdr(QSP_ARG  ifp) < 0 ) return(-1);

	} else if( !same_type(QSP_ARG  dp,ifp) ) return(-1);

	/* BUG?  in v2d, there is a seek first, which does an extra block? */

	/* for sgi dsk files, the data must be aligned!? */

	bpi = (long)( ((dp->dt_comps * dp->dt_cols * dp->dt_rows) + 511 ) & ~511);
	if( (n=write(ifp->if_fd,dp->dt_data,bpi)) != bpi ){
		if( n < 0 ) tell_sys_error("write");
		NWARN("error writing dsk file data");
		/* BUG? do something sensible here to clean up? */
		return(-1);
	}
	return(0);
}

/* read a single frame (field) */

FIO_RD_FUNC( dsk_rd )
{
	/* dp->dt_data needs to be aligned for direct i/o (on SGI) !? */
	/* BUG? should we verify that dp is only one frame? */

	int n,nwant;

	nwant = 512 * HDR_P->blocks_per_image;

	if( x_offset !=0 || y_offset != 0 || t_offset != 0 ){
		NWARN("dsk_rd:  non-zero offsets not supported");
	}

#ifdef SGI
	/* check alignment */
	if( ( ((u_long)dp->dt_data) & 511) != 0 )
		NWARN("data buffer not aligned on block boundary for DIRECT_IO");
#endif /* SGI */

	if( (n=read(ifp->if_fd,dp->dt_data,nwant)) != nwant ){
		if( n < 0 ) tell_sys_error("read");
		NWARN("error reading diskfile data");
	}

	ifp->if_nfrms++;
}

/* the unconvert routine creates a disk header */

int dsk_unconv(void *hdr_pp,Data_Obj *dp)
{
	Sir_Disk_Hdr **hd_pp;

	hd_pp = (Sir_Disk_Hdr **) hdr_pp;

	/* allocate space for new header */

	*hd_pp = (Sir_Disk_Hdr *)getbuf( 512 );
	if( *hd_pp == NULL ) return(-1);

	dp_to_dsk(*hd_pp,dp);

	return(0);
}

int dsk_conv(Data_Obj *dp,void *hd_pp)
{
	NWARN("dsk_conv not implemented");
	return(-1);
}

static void msg_num(const char *str,int n)
{
	sprintf(msg_str,"\t%s  %d",str,n);
	prt_msg(msg_str);
}

void dsk_info(QSP_ARG_DECL  Image_File *ifp)
{
	/* print out the non-standard fields */
	/* bug should print out symbolic names for enum's? */
	msg_num("version:     ",HDR_P->version);
	msg_num("start block: ",HDR_P->video_start_block);
	msg_num("block size:  ",HDR_P->block_size);
	msg_num("packing:     ",HDR_P->video_packing);
	msg_num("format:      ",HDR_P->video_format);
	msg_num("capture type:",HDR_P->video_capture_type);
	msg_num("timing:      ",HDR_P->video_timing);
	msg_num("dominance:   ",HDR_P->video_field_dominance);
	msg_num("blks/image:  ",HDR_P->blocks_per_image);
	msg_num("video_blks:  ",HDR_P->video_n_blocks);
}

