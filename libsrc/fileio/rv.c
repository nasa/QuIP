
#include "quip_config.h"

char VersionId_fio_rv[] = QUIP_VERSION_STRING;

#include <stdio.h>

#ifdef HAVE_SYS_TIME_H
#include <sys/time.h>
#endif

#ifdef HAVE_STRING_H
#include <string.h>
#endif

#ifdef HAVE_UNISTD_H
#include <unistd.h>
#endif

#include "fio_prot.h"
#include "filetype.h"
#include "getbuf.h"
#include "data_obj.h"
#include "debug.h"
#include "savestr.h"
#include "rv_api.h"
#include "rv.h"

static int n_rv_disks=1;

//#define HDR_P(ifp) ((RV_Inode *)&(((Image_File_Hdr *)ifp->if_hd)->ifh_u.rv_ino))
#define HDR_P(ifp) ((RV_Inode *)ifp->if_hd)
#define HDR_P_LVAL(ifp) ifp->if_hd

static int rv_fd_arr[MAX_DISKS];

/* local prototypes */
//static int rv_to_dp(Data_Obj *dp,RV_Inode *inp);
//static int dp_to_rv(RV_Inode *inp,Data_Obj *dp);

int rvfio_seek_frame(QSP_ARG_DECL  Image_File *ifp, dimension_t n )
{
	if( rv_frame_seek(HDR_P(ifp),n) < 0 )
		return(-1);
	ifp->if_nfrms = n;
	return(0);
}

static int rv_to_dp(Data_Obj *dp,RV_Inode *inp)
{
	u_long blks_per_frame;

	dp->dt_shape = inp->rvi_shape;

	/* The increments should be ok, except for the frame increment... */

	blks_per_frame = (((dp->dt_comps * dp->dt_cols * dp->dt_rows + inp->rvi_extra_bytes)
				+ (BLOCK_SIZE-1)) & ~(BLOCK_SIZE-1))/ BLOCK_SIZE;
	dp->dt_cinc=1;
	dp->dt_pinc=(incr_t)dp->dt_comps;
	dp->dt_rinc=(incr_t)(dp->dt_comps*dp->dt_cols);
	dp->dt_finc=((incr_t)blks_per_frame) * BLOCK_SIZE;
	dp->dt_sinc=dp->dt_finc * (incr_t)dp->dt_frames;

	dp->dt_parent = NO_OBJ;
	dp->dt_children = NO_LIST;

	dp->dt_ap = ram_area;		/* the default */
	dp->dt_data = NULL;
	dp->dt_n_type_elts = dp->dt_comps * dp->dt_cols * dp->dt_rows
			* dp->dt_frames * dp->dt_seqs;

	set_shape_flags(&dp->dt_shape,dp,AUTO_SHAPE);

	return(0);
}

FIO_OPEN_FUNC( rvfio_open )
{
	Image_File *ifp;
	RV_Inode *inp;

	if( ! legal_rv_filename(name) ){
		sprintf(error_string,"rv_open:  \"%s\" is not a legal filename",name);
		NWARN(error_string);
		return(NO_IMAGE_FILE);
	}

	inp = rv_inode_of(QSP_ARG  name);

	if( rw == FILE_WRITE ){
		u_long size;
		/* need to know the total number of blocks!?
		 * We fudge it for now by assuming the default video frame size.
		 * and a single frame...
		 * BUG?? do we set it correctly later???
		 * This is wrong if we are in field mode!?
		 */
		size = (640*480*4)/BLOCK_SIZE;	/* 1200 blocks (1 frame) */

		if( inp != NO_INODE ){
			/* overwrite of an existing file.
			 * destroy the old one to make sure we get the size right.
			 */
			rv_rmfile(QSP_ARG  name);
		}
		n_rv_disks = creat_rv_file(QSP_ARG  name,size,rv_fd_arr);
		if( n_rv_disks < 0 ) return(NO_IMAGE_FILE);
		inp = rv_inode_of(QSP_ARG  name);
	} else if( inp == NO_INODE ){	/* FILE_READ */
		sprintf(error_string,"File %s does not exist, can't read",name);
		NWARN(error_string);
		return(NO_IMAGE_FILE);
	}

	if( rw == FILE_READ ){
		/* check for file struct already existing */
		ifp = img_file_of(QSP_ARG  name);
		/* BUG make sure that it is type RV here! */
		if( ifp != NO_IMAGE_FILE ){
			if( ! IS_READABLE(ifp) ){
				/*
				sprintf(error_string,"Setting READABLE flag on rv file %s",
						ifp->if_name);
				advise(error_string);
				*/
				ifp->if_flags |= FILE_READ;
			}
			if( IS_READABLE(ifp) ){
				if( (n_rv_disks=queue_rv_file(inp,rv_fd_arr)) < 0 ){
			sprintf(error_string,"Error queueing file %s",ifp->if_name);
					NWARN(error_string);
				}
				return(ifp);
			}
			/* If we've just recorded this file, go ahead and change it */ 
			/* BUG?  what if we are assembling it? */

			/* NOTREACHED */
			sprintf(error_string,"File %s is not readable!?",ifp->if_name);
			NWARN(error_string);
			return(NO_IMAGE_FILE);
		}
	}

	ifp = new_img_file(QSP_ARG  name);
	if( ifp==NO_IMAGE_FILE ) return(ifp);

	ifp->if_flags = rw;
	ifp->if_nfrms = 0;			/* number of frames written or read */

	ifp->if_pathname = ifp->if_name;	/* default */
	/* update_pathname(ifp); */

	ifp->if_dp = NO_OBJ;
	ifp->if_type = IFT_RV;

	HDR_P_LVAL(ifp) = inp;

	/* We might need to swap to be portable (see dsk.c)
	 * but since the raw disk is only
	 * accessible on the host machine, let's not worry about
	 * it now.
	 */

	if( IS_READABLE(ifp) ){
		char tnam[LLEN];
		ifp->if_dp = (Data_Obj *)getbuf(sizeof(Data_Obj));
		/* not hashed into database! */
		/* give it a name so it has a name to print in case of accident */
		sprintf(tnam,"dp.%s",ifp->if_name);
		ifp->if_dp->dt_name = savestr(tnam);
		rv_to_dp(ifp->if_dp,inp);

		/* queue the disk to this file.
		 *
		 * BUG - things will not work properly if we have more than
		 * one file open at a time, because we are not keeping
		 * per-file seek offsets around...  but this should work
		 * if we are reading from a single file.
		 */
		if( (n_rv_disks=queue_rv_file(inp,rv_fd_arr)) < 0 ){
			sprintf(error_string,"Error queueing file %s",ifp->if_name);
			NWARN(error_string);
		}
	} else {
		ifp->if_dp = NO_OBJ;
	}
	return(ifp);
}

static int dp_to_rv(RV_Inode *inp,Data_Obj *dp)
{
#ifdef CAUTIOUS
	if( dp == NO_OBJ ) {
		NWARN("CAUTIOUS:  dp_to_rv:  null dp");
		return(-1);
	}
#endif /* CAUTIOUS */

	/* num_frame set when when write request given */

	inp->rvi_shape = dp->dt_shape;
	set_shape_flags(&inp->rvi_shape,NO_OBJ,AUTO_SHAPE);

	return(0);
}

FIO_SETHDR_FUNC( set_rvfio_hdr ) /* set header fields from image object */
{
	/* BUG  For write files, we need to request a certain number of
	 * blocks at file creation time...  we do this using the default
	 * frame size of 640*480*4/1024 = 640*480/256 = 10*480/4 = 10*120
	 * = 1200 blocks per frame.
	 * If the actual frame size is less than this, we should reallocate
	 * the disk blocks so as not to waste...
	 */

	if( dp_to_rv(ifp->if_hd,ifp->if_dp) < 0 ){
		/* where does this come from?? */
		GENERIC_IMGFILE_CLOSE(ifp);
		return(-1);
	}
	return(0);
}

FIO_WT_FUNC( rvfio_wt )
{
	long bpi;		/* bytes per image */
	long bpf;		/* bytes per frame (padded to block boundary) */
	u_long f2a;		/* total frames to allocate */
	/* u_long n; */
	long nw;
	int disk_index;
	int n_disks;

	n_disks = HDR_P(ifp)->rvi_sbp->rv_ndisks;

	if( ifp->if_dp == NO_OBJ ){	/* first time? */
		u_long size;	/* file size in blocks */
		RV_Inode *inp;

		/* set the rows & columns in our file struct */
		setup_dummy(ifp);
		copy_dimensions(ifp->if_dp, dp);
		ifp->if_dp->dt_prec = dp->dt_prec;

		ifp->if_dp->dt_frames =  ifp->if_frms_to_wt;
		ifp->if_dp->dt_seqs = 1;
		set_shape_flags(&ifp->if_dp->dt_shape,ifp->if_dp,AUTO_SHAPE);
		rv_set_shape(QSP_ARG  ifp->if_name,&ifp->if_dp->dt_shape);

		/* This used to be after the call to rv_realloc()...
		 * Does the object exist now?  it should...
		 * We can check the len and decide whether or
		 * not we need to reallocate.
		 */
		inp = get_rv_inode(QSP_ARG  ifp->if_name);
		HDR_P_LVAL(ifp) = inp;

		/* Now we need to reallocate the blocks for this file */
		/* Sometimes the correct size is already allocated??? */
		/* When should this really be done??? */

		size = ifp->if_dp->dt_comps * ifp->if_dp->dt_cols * ifp->if_dp->dt_rows
			+ inp->rvi_extra_bytes;

		/* if the size is not an integral number of blocks, round up... */
		size += (BLOCK_SIZE-1);
		size /= BLOCK_SIZE;

		/* where does ifp->if_dp->dt_frames get set??? */
		f2a = FRAMES_TO_ALLOCATE(ifp->if_dp->dt_frames,n_disks);
		size *= f2a;

		if( rv_realloc(QSP_ARG  ifp->if_name,size) < 0 ){
			sprintf(error_string,
		"error allocating %ld disk blocks for file %s",
				size,ifp->if_name);
			NWARN(error_string);
		}

		if( set_rvfio_hdr(QSP_ARG  ifp) < 0 ) return(-1);
	} else if( !same_type(QSP_ARG  dp,ifp) ) return(-1);

	/* Because we are writing each frame to a separate disk,
	 * we need to know the index of this frame to know which
	 * disk to write it to...
	 */

	/* BUG should store bytes per image in inp struct... */
	/* BUG what about rvi_extra_bytes???... */

	bpi = (dp->dt_comps * dp->dt_cols * dp->dt_rows );
	bpf = (bpi +  BLOCK_SIZE - 1) & ~(BLOCK_SIZE-1);

	disk_index = (int)(ifp->if_nfrms % n_disks);

	if( (nw=write(rv_fd_arr[disk_index],dp->dt_data,bpi)) != bpi ){
		if( nw < 0 ) tell_sys_error("write");
		sprintf(error_string,
	"write error on disk %d (fd=%d), %ld bytes requested, %ld written",disk_index,
		rv_fd_arr[disk_index],bpi,nw);
		NWARN(error_string);
		/* BUG? do something sensible here to clean up? */
		return(-1);
	}
	/* write the pad bytes so we don't have to seek... */
	if( bpf > bpi ){
		if( (nw=write(rv_fd_arr[disk_index],dp->dt_data,bpf-bpi)) != bpf-bpi ){
			if( nw < 0 ) tell_sys_error("write");
			sprintf(error_string,
	"write error on disk %d (fd=%d), %ld bytes requested, %ld written",disk_index,
			rv_fd_arr[disk_index],bpf-bpi,nw);
			NWARN(error_string);
			/* BUG? do something sensible here to clean up? */
			return(-1);
		}
	}

#ifdef CAUTIOUS
	if( dp->dt_frames != 1 ){
		sprintf(error_string,"CAUTIOUS:  rvfio_wt:  object %s has %d frames, expected 1!?",
			dp->dt_name,dp->dt_frames);
		NWARN(error_string);
	}
#endif /* CAUTIOUS */

	ifp->if_nfrms ++;

	return(0);
}

struct timeval *rv_time_ptr(Image_File *ifp, index_t frame)
{
	static struct timeval tv,*tvp;
	char *buf;
	long bpi,n_to_read;	/* bytes per image */
	int disk_index;
	int n;

	disk_index = (int)( frame % HDR_P(ifp)->rvi_sbp->rv_ndisks );

	if( rv_frame_seek(HDR_P(ifp),frame) < 0 )
		return(NULL);

	bpi = ifp->if_dp->dt_comps * ifp->if_dp->dt_cols * ifp->if_dp->dt_rows ;

	if( HDR_P(ifp)->rvi_extra_bytes != sizeof(struct timeval) ){
		/* sizeof is long in ia64? */
		sprintf(DEFAULT_ERROR_STRING,"rv_time_ptr:  expected rvi_extra_bytes (%d) to equal sizeof(struct timeval) (%d) !?",
				HDR_P(ifp)->rvi_extra_bytes,(int)sizeof(struct timeval));
		NWARN(DEFAULT_ERROR_STRING);
		return(NULL);
	}
	n_to_read = bpi + HDR_P(ifp)->rvi_extra_bytes;

	/* round up to block size */
	n_to_read = (n_to_read + BLOCK_SIZE -1 ) & ( ~ (BLOCK_SIZE-1) );

	buf = (char *)getbuf(n_to_read);

	if( (n=read(rv_fd_arr[disk_index],buf,n_to_read)) != n_to_read ){
		if( n < 0 ) tell_sys_error("read");
		sprintf(DEFAULT_ERROR_STRING,
	"error reading RV data from disk %d (fd=%d), %d bytes read (%ld requested)",
			disk_index,rv_fd_arr[disk_index],n,n_to_read);
		NWARN(DEFAULT_ERROR_STRING);
	}
	tvp = (struct timeval *)(buf + bpi);
	tv = *tvp;
	givbuf(buf);

	return(&tv);
}


/* read a single frame (field) */

FIO_RD_FUNC( rvfio_rd )
{
	int n;
	long bpi;	/* bytes per image */
	char *data_ptr;
	int disk_index;

	/* BUG? should we verify that dp is only one frame? */

	if( x_offset !=0 || y_offset != 0 || t_offset != 0 ){
		NWARN("rvfio_rd:  non-zero offsets not supported");
		return;
	}

	/* we don't read any timestamp data here ... */
	bpi = (ifp->if_dp->dt_comps * ifp->if_dp->dt_cols * ifp->if_dp->dt_rows );
	disk_index = (int)( ifp->if_nfrms % HDR_P(ifp)->rvi_sbp->rv_ndisks );

	data_ptr = (char *)dp->dt_data;

	/* We need to seek before we read; we might have read another file between
	 * this read and a previous one.
	 */

if( verbose ){
sprintf(error_string,"rvfio_rd:  file %s seeking to frame %d, will read from disk %d",
ifp->if_name,ifp->if_nfrms,disk_index);
advise(error_string);
}
	if( rv_frame_seek(HDR_P(ifp),ifp->if_nfrms) < 0 ){
		NWARN("rvfio_rd:  seek failed, not reading data");
		return;
	}

	if( (n=read(rv_fd_arr[disk_index],data_ptr,bpi)) != bpi ){
		if( n < 0 ) tell_sys_error("read");
		sprintf(error_string,
	"error reading RV data from disk %d (fd=%d), %d bytes read (%ld requested)",
			disk_index,rv_fd_arr[disk_index],n,bpi);
		NWARN(error_string);
	}

	ifp->if_nfrms++;
}

/* the unconvert routine creates a disk header */

int rvfio_unconv(void *hdr_pp,Data_Obj *dp)
{
	RV_Inode **in_pp;

	in_pp = (RV_Inode **) hdr_pp;

	/* allocate space for new header */

	*in_pp = (RV_Inode *)getbuf( sizeof(RV_Inode) );
	if( *in_pp == NULL ) return(-1);

	dp_to_rv(*in_pp,dp);

	return(0);
}

int rvfio_conv(Data_Obj *dp,void *hd_pp)
{
	NWARN("rv_conv not implemented");
	return(-1);
}

FIO_INFO_FUNC( rvfio_info )
{
	rv_info(QSP_ARG  HDR_P(ifp));
}

/*
 * Closing the file deletes it from the database.
 */

FIO_CLOSE_FUNC( rvfio_close )
{
	/* this is where we should check the number of frames written?? */
	GENERIC_IMGFILE_CLOSE(ifp);
}

double get_rv_seconds(Image_File *ifp,dimension_t frame)
{
	struct timeval *tvp;

	tvp = rv_time_ptr(ifp,frame);	/* BUG how do we pass the frame index? */
	if( tvp == NULL ) return(-1.0);
	return((double)tvp->tv_sec);
}

double get_rv_milliseconds(Image_File *ifp,dimension_t frame)
{
	struct timeval *tvp;

	tvp = rv_time_ptr(ifp,frame);	/* BUG how do we pass the frame index? */
	if( tvp == NULL ) return(-1.0);
	return(((double)tvp->tv_usec/1000.0));
}

double get_rv_microseconds(Image_File *ifp,dimension_t frame)
{
	struct timeval *tvp;

	tvp = rv_time_ptr(ifp,frame);	/* BUG how do we pass the frame index? */
	if( tvp == NULL ) return(-1.0);
	return((double)(tvp->tv_usec%1000));
}

