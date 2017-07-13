
#include "quip_config.h"

#ifdef HAVE_RAWVOL

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

#ifdef HAVE_FCNTL_H
#include <fcntl.h>	// definition of O_DIRECT is here...
#endif // HAVE_FCNTL_H

#include "quip_prot.h"
#include "fio_prot.h"
#include "data_obj.h"
#include "rv_api.h"
#include "img_file/rv.h"
#include "llseek.h"

static int _n_disks=1;

#define HDR_P(ifp) ((RV_Inode *)ifp->if_hdr_p)
#define HDR_P_LVAL(ifp) ifp->if_hdr_p

static int rv_fd_arr[MAX_DISKS];

/* local prototypes */
//static int rv_to_dp(Data_Obj *dp,RV_Inode *inp);
//static int dp_to_rv(RV_Inode *inp,Data_Obj *dp);

int rvfio_seek_frame(QSP_ARG_DECL  Image_File *ifp, dimension_t n )
{
	if( rv_frame_seek(QSP_ARG  HDR_P(ifp),n) < 0 )
		return(-1);
	ifp->if_nfrms = n;
	return(0);
}

static int rv_to_dp(Data_Obj *dp,RV_Inode *inp)
{
	u_long blks_per_frame;

//fprintf(stderr,"rv_to_dp 0x%lx  0x%lx BEGIN\n",(long)dp,(long)inp);
//describe_shape(DEFAULT_QSP_ARG  &inp->rvi_shape);
//longlist(dp);

	// Does this do mach dims as well as type dims???
	copy_shape( OBJ_SHAPE(dp), rv_movie_shape(inp) );
	//COPY_SHAPE( OBJ_SHAPE(dp), rv_movie_shape(inp) );

	/* The increments should be ok, except for the frame increment... */

	blks_per_frame = (((OBJ_COMPS(dp) * OBJ_COLS(dp) * OBJ_ROWS(dp) + rv_movie_extra( inp ))
				+ (BLOCK_SIZE-1)) & ~(BLOCK_SIZE-1))/ BLOCK_SIZE;
	SET_OBJ_COMP_INC(dp,1);
	SET_OBJ_PXL_INC(dp,(incr_t)OBJ_COMPS(dp) );
	SET_OBJ_ROW_INC(dp,(incr_t)(OBJ_COMPS(dp)*OBJ_COLS(dp)) );
	SET_OBJ_FRM_INC(dp,((incr_t)blks_per_frame) * BLOCK_SIZE );
	SET_OBJ_SEQ_INC(dp,OBJ_FRM_INC(dp) * (incr_t)OBJ_FRAMES(dp) );

	SET_OBJ_PARENT(dp, NULL);
	SET_OBJ_CHILDREN(dp, NULL);

	SET_OBJ_AREA(dp, ram_area_p);		/* the default */
	SET_OBJ_DATA_PTR(dp, NULL);
	SET_OBJ_N_TYPE_ELTS(dp, OBJ_COMPS(dp) * OBJ_COLS(dp) * OBJ_ROWS(dp)
			* OBJ_FRAMES(dp) * OBJ_SEQS(dp) );

	auto_shape_flags(OBJ_SHAPE(dp));

	return(0);
}

FIO_OPEN_FUNC( rvfio )
{
	Image_File *ifp;
	RV_Inode *inp;

	if( ! legal_rv_filename(name) ){
		sprintf(ERROR_STRING,"rv_open:  \"%s\" is not a legal filename",name);
		NWARN(ERROR_STRING);
		return(NULL);
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

		if( inp != NULL ){
			/* overwrite of an existing file.
			 * destroy the old one to make sure we get the size right.
			 */
			rv_rmfile(QSP_ARG  name);
		}
		_n_disks = creat_rv_file(QSP_ARG  name,size,rv_fd_arr);
		if( _n_disks < 0 ) return(NULL);
		inp = rv_inode_of(QSP_ARG  name);
	} else {			/* FILE_READ */
		if( inp == NULL ){
			sprintf(ERROR_STRING,"File %s does not exist, can't read",name);
			NWARN(ERROR_STRING);
			return(NULL);
		}

		/* check for file struct already existing */
		ifp = img_file_of(QSP_ARG  name);
		/* BUG make sure that it is type RV here! */
		if( ifp != NULL ){
			if( ! IS_READABLE(ifp) ){
				/*
				sprintf(ERROR_STRING,"Setting READABLE flag on rv file %s",
						ifp->if_name);
				advise(ERROR_STRING);
				*/
				ifp->if_flags |= FILE_READ;
			}
			if( IS_READABLE(ifp) ){
				if( (_n_disks=queue_rv_file(QSP_ARG  inp,rv_fd_arr)) < 0 ){
			sprintf(ERROR_STRING,"Error queueing file %s",ifp->if_name);
					NWARN(ERROR_STRING);
				}
				return(ifp);
			}
			/* If we've just recorded this file, go ahead and change it */ 
			/* BUG?  what if we are assembling it? */

			/* NOTREACHED */
			sprintf(ERROR_STRING,"File %s is not readable!?",ifp->if_name);
			NWARN(ERROR_STRING);
			return(NULL);
		}
	}

	ifp = new_img_file(QSP_ARG  name);
	if( ifp==NULL ) return(ifp);

	ifp->if_flags = rw;
	ifp->if_nfrms = 0;			/* number of frames written or read */

	ifp->if_pathname = ifp->if_name;	/* default */
	/* update_pathname(ifp); */

	ifp->if_dp = NULL;
	SET_IF_TYPE(ifp,FILETYPE_FOR_CODE(IFT_RV));

	HDR_P_LVAL(ifp) = inp;

	/* We might need to swap to be portable (see dsk.c)
	 * but since the raw disk is only
	 * accessible on the host machine, let's not worry about
	 * it now.
	 */

	if( IS_READABLE(ifp) ){
		char tnam[LLEN];
		ifp->if_dp = (Data_Obj *)getbuf(sizeof(Data_Obj));
		// Need to allocate shape too!
		// BUG make sure we free this!  Memory leak!
		// Should we use the dummy object???
		SET_OBJ_SHAPE( ifp->if_dp, ALLOC_SHAPE );

		// Need to set precision...
		SET_OBJ_PREC_PTR( ifp->if_dp, PREC_FOR_CODE( PREC_UBY ) );
//fprintf(stderr,"obj prec at 0x%lx\n",(long)OBJ_PREC_PTR(ifp->if_dp));

		/* not hashed into database! */
		/* give it a name so it has a name to print in case of accident */
		sprintf(tnam,"dp.%s",ifp->if_name);
		SET_OBJ_NAME(ifp->if_dp, savestr(tnam));
		rv_to_dp(ifp->if_dp,inp);

		/* queue the disk to this file.
		 *
		 * BUG - things will not work properly if we have more than
		 * one file open at a time, because we are not keeping
		 * per-file seek offsets around...  but this should work
		 * if we are reading from a single file.
		 */
		if( (_n_disks=queue_rv_file(QSP_ARG  inp,rv_fd_arr)) < 0 ){
			sprintf(ERROR_STRING,"Error queueing file %s",ifp->if_name);
			NWARN(ERROR_STRING);
		}
	} else {
		ifp->if_dp = NULL;
	}
	return(ifp);
}

static int dp_to_rv(RV_Inode *inp,Data_Obj *dp)
{
	assert( dp != NULL );

	/* num_frame set when when write request given */

	//inp->rvi_shape = (* OBJ_SHAPE(dp) );
	// Has the shape been allocated???
	COPY_SHAPE( rv_movie_shape(inp), OBJ_SHAPE(dp) );
	// Should we copy to the on-disk rawvol things as well???

	auto_shape_flags(rv_movie_shape(inp));

	return(0);
}

FIO_SETHDR_FUNC( rvfio ) /* set header fields from image object */
{
	/* BUG  For write files, we need to request a certain number of
	 * blocks at file creation time...  we do this using the default
	 * frame size of 640*480*4/1024 = 640*480/256 = 10*480/4 = 10*120
	 * = 1200 blocks per frame.
	 * If the actual frame size is less than this, we should reallocate
	 * the disk blocks so as not to waste...
	 */

	if( dp_to_rv(ifp->if_hdr_p,ifp->if_dp) < 0 ){
		/* where does this come from?? */
		GENERIC_IMGFILE_CLOSE(ifp);
		return(-1);
	}
	return(0);
}

FIO_WT_FUNC( rvfio )
{
	long bpi;		/* bytes per image */
	long bpf;		/* bytes per frame (padded to block boundary) */
	u_long f2a;		/* total frames to allocate */
	/* u_long n; */
	long nw;
	int disk_index;
	int n_disks;
off64_t retoff;

#ifdef O_DIRECT
	// the object must be aligned!
	// BUG - use BLOCKSIZE instead of hard-coding 1024
	if( ((u_long)(OBJ_DATA_PTR(dp))) & (BLOCK_SIZE-1) ){
		sprintf(ERROR_STRING,
	"Object %s must have data block-aligned (%d bytes) to write to a raw volume!?",
			OBJ_NAME(dp),BLOCK_SIZE);
		WARN(ERROR_STRING);
		return -1;
	}
#endif // O_DIRECT

	n_disks = n_rv_disks();

	if( ifp->if_dp == NULL ){	/* first time? */
		u_long size;	/* file size in blocks */
		RV_Inode *inp;

		/* set the rows & columns in our file struct */
		setup_dummy(ifp);
		copy_dimensions(ifp->if_dp, dp);
		SET_OBJ_PREC_PTR(ifp->if_dp, OBJ_PREC_PTR(dp) );

		SET_OBJ_FRAMES(ifp->if_dp,  ifp->if_frms_to_wt );
		SET_OBJ_SEQS(ifp->if_dp, 1);
		auto_shape_flags(OBJ_SHAPE(ifp->if_dp));
		rv_set_shape(QSP_ARG  ifp->if_name,OBJ_SHAPE(ifp->if_dp));

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

		size = OBJ_COMPS(ifp->if_dp) * OBJ_COLS(ifp->if_dp) * OBJ_ROWS(ifp->if_dp)
			+ rv_movie_extra( inp );

		/* if the size is not an integral number of blocks, round up... */
		size += (BLOCK_SIZE-1);
		size /= BLOCK_SIZE;

		/* where does OBJ_FRAMES(ifp->if_dp) get set??? */
		f2a = rv_frames_to_allocate(OBJ_FRAMES(ifp->if_dp));
		size *= f2a;

		if( rv_realloc(QSP_ARG  ifp->if_name,size) < 0 ){
			sprintf(ERROR_STRING,
		"error allocating %ld disk blocks for file %s",
				size,ifp->if_name);
			NWARN(ERROR_STRING);
		}

		if( set_rvfio_hdr(QSP_ARG  ifp) < 0 ) return(-1);
	} else if( !same_type(QSP_ARG  dp,ifp) ) return(-1);

	/* Because we are writing each frame to a separate disk,
	 * we need to know the index of this frame to know which
	 * disk to write it to...
	 */

	/* BUG should store bytes per image in inp struct... */
	/* BUG what about rvi_extra_bytes???... */

	bpi = (OBJ_COMPS(dp) * OBJ_COLS(dp) * OBJ_ROWS(dp) );	// bytes per image
	bpf = (bpi +  BLOCK_SIZE - 1) & ~(BLOCK_SIZE-1);	// bytes per frame
fprintf(stderr,"bpi = %ld (0x%lx), bpf = %ld (0x%lx)\n",bpi,bpi,bpf,bpf);

	disk_index = (int)(ifp->if_nfrms % n_disks);

retoff = my_lseek64(rv_fd_arr[disk_index],(off64_t) 0,SEEK_CUR);
fprintf(stderr,"Current file position is 0x%lx\n",retoff);

fprintf(stderr,"writing %d (0x%lx) bytes of data from 0x%lx\n",bpi,bpi,(u_long)OBJ_DATA_PTR(dp));
	if( (nw=write(rv_fd_arr[disk_index],OBJ_DATA_PTR(dp),bpi)) != bpi ){
		if( nw < 0 ) tell_sys_error("write");
		sprintf(ERROR_STRING,
	"write error on disk %d (fd=%d), %ld bytes requested, %ld written",disk_index,
		rv_fd_arr[disk_index],bpi,nw);
		NWARN(ERROR_STRING);
		/* BUG? do something sensible here to clean up? */
		return(-1);
	}

	/* write the pad bytes so we don't have to seek... */
	/* BUT with O_DIRECT flag set, we have to write an integral number of blocks...
	 * better to just seek!
	 * Note that if this test is true then the frame size must not be an integral number
	 * of blocks, so we are already in trouble!
	 */

	if( bpf > bpi ){
fprintf(stderr,"writing %d pad bytes of data from 0x%lx\n",bpf-bpi,(u_long)OBJ_DATA_PTR(dp));
		if( (nw=write(rv_fd_arr[disk_index],OBJ_DATA_PTR(dp),bpf-bpi)) != bpf-bpi ){
			if( nw < 0 ) tell_sys_error("write");
			sprintf(ERROR_STRING,
	"write error on disk %d (fd=%d), %ld bytes requested, %ld written",disk_index,
			rv_fd_arr[disk_index],bpf-bpi,nw);
			NWARN(ERROR_STRING);
			/* BUG? do something sensible here to clean up? */
			return(-1);
		}
	}

	assert( OBJ_FRAMES(dp) == 1 );

	ifp->if_nfrms ++;

	return(0);
}

static struct timeval *rv_time_ptr(QSP_ARG_DECL  Image_File *ifp, index_t frame)
{
	static struct timeval tv,*tvp;
	char *buf;
	long bpi,n_to_read;	/* bytes per image */
	int disk_index;
	int n;

	disk_index = (int)( frame % n_rv_disks() );

	if( rv_frame_seek(QSP_ARG  HDR_P(ifp),frame) < 0 )
		return(NULL);

	bpi = OBJ_COMPS(ifp->if_dp) * OBJ_COLS(ifp->if_dp) * OBJ_ROWS(ifp->if_dp) ;

	if( rv_movie_extra( HDR_P(ifp) ) != sizeof(struct timeval) ){
		/* sizeof is long in ia64? */
		sprintf(DEFAULT_ERROR_STRING,"rv_time_ptr:  expected rvi_extra_bytes (%d) to equal sizeof(struct timeval) (%d) !?",
				rv_movie_extra( HDR_P(ifp) ),(int)sizeof(struct timeval));
		NWARN(DEFAULT_ERROR_STRING);
		return(NULL);
	}
	n_to_read = bpi + rv_movie_extra( HDR_P(ifp) );

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

FIO_RD_FUNC( rvfio )
{
	int n;
	long bpi;	/* bytes per image */
	char *data_ptr;
	int disk_index;

#ifdef O_DIRECT
	// the object must be aligned!
	// BUG - use BLOCKSIZE instead of hard-coding 1024
	if( ((u_long)(OBJ_DATA_PTR(dp))) & (BLOCK_SIZE-1) ){
		sprintf(ERROR_STRING,
	"Object %s must be block-aligned (%d bytes) to read from a raw volume!?",
			OBJ_NAME(dp),BLOCK_SIZE);
		WARN(ERROR_STRING);
		return;
	}
#endif // O_DIRECT

	/* BUG? should we verify that dp is only one frame? */

	if( x_offset !=0 || y_offset != 0 || t_offset != 0 ){
		NWARN("rvfio_rd:  non-zero offsets not supported");
		return;
	}

	/* we don't read any timestamp data here ... */
	bpi = (OBJ_COMPS(ifp->if_dp) * OBJ_COLS(ifp->if_dp) * OBJ_ROWS(ifp->if_dp) );
	disk_index = (int)( ifp->if_nfrms % n_rv_disks() );

	data_ptr = (char *)OBJ_DATA_PTR(dp);

	/* We need to seek before we read; we might have read another file between
	 * this read and a previous one.
	 */

if( verbose ){
sprintf(ERROR_STRING,"rvfio_rd:  file %s seeking to frame %d, will read from disk %d",
ifp->if_name,ifp->if_nfrms,disk_index);
advise(ERROR_STRING);
}
	if( rv_frame_seek(QSP_ARG  HDR_P(ifp),ifp->if_nfrms) < 0 ){
		NWARN("rvfio_rd:  seek failed, not reading data");
		return;
	}

	if( (n=read(rv_fd_arr[disk_index],data_ptr,bpi)) != bpi ){
		if( n < 0 ) tell_sys_error("read");
		sprintf(ERROR_STRING,
	"error reading RV data from disk %d (fd=%d), %d bytes read (%ld requested)",
			disk_index,rv_fd_arr[disk_index],n,bpi);
		NWARN(ERROR_STRING);
	}

	ifp->if_nfrms++;
}

/* the unconvert routine creates a disk header */

int rvfio_unconv(void *hdr_pp,Data_Obj *dp)
{
	RV_Inode **in_pp;

	in_pp = (RV_Inode **) hdr_pp;

	/* allocate space for new header */

	*in_pp = rv_inode_alloc();
	if( *in_pp == NULL ) return(-1);

	dp_to_rv(*in_pp,dp);

	return(0);
}

int rvfio_conv(Data_Obj *dp,void *hd_pp)
{
	NWARN("rv_conv not implemented");
	return(-1);
}

FIO_INFO_FUNC( rvfio )
{
	rv_info(QSP_ARG  HDR_P(ifp));
}

/*
 * Closing the file deletes it from the database.
 */

FIO_CLOSE_FUNC( rvfio )
{
	/* this is where we should check the number of frames written?? */
	GENERIC_IMGFILE_CLOSE(ifp);
}

double get_rv_seconds(QSP_ARG_DECL  Image_File *ifp,dimension_t frame)
{
	struct timeval *tvp;

	tvp = rv_time_ptr(QSP_ARG  ifp,frame);	/* BUG how do we pass the frame index? */
	if( tvp == NULL ) return(-1.0);
	return((double)tvp->tv_sec);
}

double get_rv_milliseconds(QSP_ARG_DECL  Image_File *ifp,dimension_t frame)
{
	struct timeval *tvp;

	tvp = rv_time_ptr(QSP_ARG  ifp,frame);	/* BUG how do we pass the frame index? */
	if( tvp == NULL ) return(-1.0);
	return(((double)tvp->tv_usec/1000.0));
}

double get_rv_microseconds(QSP_ARG_DECL  Image_File *ifp,dimension_t frame)
{
	struct timeval *tvp;

	tvp = rv_time_ptr(QSP_ARG  ifp,frame);	/* BUG how do we pass the frame index? */
	if( tvp == NULL ) return(-1.0);
	return((double)(tvp->tv_usec%1000));
}

#endif /* HAVE_RAWVOL */
