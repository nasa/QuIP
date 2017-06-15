
/* Utility routines to read/write from raw volume
 *
 * These comments are from the original version written for sirius/fusion;
 * the current version runs on lagrange w/ meteor...
 *
 * We want our raw volume layout to be compatible with sir_vidtodisk,
 * which writes to the first block of the raw volume.  Therefore,
 * we put our "superblock" at the end of the volume.
 *
 * How many blocks do we need to index?  Our striped filesystem
 * is 54Gb...  divide by 512 bytes/block, we get about 100Mblocks
 * 1M = 2^20, 100~2^7, so we need 27 bits...  a long should do it!
 */

#include "quip_config.h"

#ifdef HAVE_SYS_TYPES_H
#include <sys/types.h>
#endif

#ifdef HAVE_SYS_STAT_H
#include <sys/stat.h>
#endif

#ifdef HAVE_SYS_TIME_H
#include <sys/time.h>
#endif

#ifdef HAVE_UNISTD_H
#include <unistd.h>		/* lseek() */
#endif

#ifdef HAVE_MATH_H
#include <math.h>		/* floor */
#endif

#ifdef HAVE_FCNTL_H
#include <fcntl.h>
#endif

#ifdef HAVE_STDLIB_H
#include <stdlib.h>
#endif

#ifdef HAVE_STRING_H
#include <string.h>
#endif

#ifdef HAVE_PWD_H
#include <pwd.h>
#endif

#ifdef HAVE_GRP_H
#include <grp.h>
#endif

#ifdef HAVE_TIME_H
#include <time.h>
#endif


#include "quip_prot.h"
#include "rawvol.h"
#include "llseek.h"
#include "fio_api.h"

ITEM_INTERFACE_DECLARATIONS(RV_Inode,rv_inode,0)

#ifdef HAVE_RAWVOL

// Used to be  (off_ret<0), but now off64_t is unsigned?
//#define BAD_OFFSET64(o)		( (o) & 0x80000000 )
#define BAD_OFFSET64(o)		( (o) == (off64_t)(-1) )

#define ROOT_DIR_NAME	"/"

static int n_extra_bytes=0;
static RV_Inode *rv_in_tbl=NULL;
static RV_Super *rv_sbp=NULL;	// only one at a time?
static int use_osync=0;			// synchronous writes by default

/* We use this flag, so that the first time we create the volume,
 * we can open it to create the root directory, and the missing root
 * directory will not be considered an error.
 */

static int in_mkfs=0;

static uint32_t string_bytes_per_disk;
static uint32_t inode_bytes_per_disk;
static uint32_t inodes_per_disk;
static uint32_t total_string_bytes;

static char rv_pathname[LLEN];

static char *rv_stp;		/* string table pointer */

#define MAX_STRING_CHUNKS	1024
#define MAX_INODE_CHUNKS	1024
#define MAX_DATA_CHUNKS		1024

#define LONG_ALIGN_SLOP		(sizeof(uint32_t)-1)
#define LONG_ALIGN_MASK		(~LONG_ALIGN_SLOP)
#define ALIGN( index )		( ((index)+LONG_ALIGN_SLOP) & LONG_ALIGN_MASK )

#define CHECK_VOLUME(s)							\
									\
	if( rv_sbp == NULL ){					\
		sprintf(ERROR_STRING,"%s:  no raw volume open",s);	\
		WARN(ERROR_STRING);					\
		return;							\
	}


FreeList rv_st_freelist;
FreeList rv_inode_freelist;
FreeList rv_data_freelist;

/* this macro transforms a requested size (in blocks) to blocks per disk */
#define BLOCKS_PER_DISK(s)	(((s)+rv_sbp->rv_ndisks-1)/rv_sbp->rv_ndisks)

/* In the old days, only SGI could perform direct i/o,
 * and mem_get here was defined to memalign...
 * Now we use posix_memalign.
 */

#define mem_release	givbuf
#define mem_get		getbuf



static size_t block_size=BLOCK_SIZE;
int rawvol_debug=0;

#define MAX_DIR_STACK_DEPTH	20
static const char *dir_stack[MAX_DIR_STACK_DEPTH];
static int dir_stack_depth=0;

// We used to do this with structure copies, but that caused problems
// because of dynamic allocation of shape_info...

static void copy_rv_inode(RV_Inode *dst_inp, RV_Inode *src_inp)
{
	//SET_RV_MOVIE_SHAPE(dst_inp,NULL);	// take care of this later!

	SET_RV_NAME(dst_inp,RV_NAME(src_inp));	// ok to copy pointer???

	SET_RV_NAME_IDX		( dst_inp, RV_NAME_IDX	(src_inp));
	SET_RV_INODE_IDX	( dst_inp, RV_INODE_IDX	(src_inp));
	SET_RV_ADDR		( dst_inp, RV_ADDR	(src_inp));
	SET_RV_N_BLOCKS		( dst_inp, RV_N_BLOCKS	(src_inp));
	SET_RV_FLAGS		( dst_inp, RV_FLAGS	(src_inp));

	SET_RV_MODE		( dst_inp, RV_MODE	(src_inp));
	SET_RV_UID		( dst_inp, RV_UID	(src_inp));
	SET_RV_GID		( dst_inp, RV_GID	(src_inp));
	SET_RV_ATIME		( dst_inp, RV_ATIME	(src_inp));
	SET_RV_MTIME		( dst_inp, RV_MTIME	(src_inp));
	SET_RV_CTIME		( dst_inp, RV_CTIME	(src_inp));

	SET_RV_INODE		( dst_inp, RV_INODE	(src_inp));
	SET_RV_PARENT		( dst_inp, RV_PARENT	(src_inp));

	// BUG?  we might clear the union to 0's just for safety...
	if( IS_DIRECTORY(src_inp) ){
		SET_RV_DIR_ENTRIES	( dst_inp, RV_DIR_ENTRIES	(src_inp));
		// Dir_Info has parent, but do we need?
	} else if( IS_LINK(src_inp) ){
		SET_RV_LINK_INODE_PTR	( dst_inp, RV_LINK_INODE_PTR	(src_inp));
		SET_RV_LINK_INODE_IDX	( dst_inp, RV_LINK_INODE_IDX	(src_inp));
	} else {	// regular movie file
		int i;

		SET_RV_MOVIE_SHAPE	( dst_inp, NULL);	// take care of this later!
		COPY_DIMS( & RV_MOVIE_DIMS(dst_inp), &RV_MOVIE_DIMS(src_inp) );
		COPY_INCS( & RV_MOVIE_INCS(dst_inp), &RV_MOVIE_INCS(src_inp) );
		SET_RV_MOVIE_PREC_CODE	( dst_inp, RV_MOVIE_PREC_CODE	(src_inp));
		//SET_RV_MOVIE_SUPER(	( dst_inp, RV_MOVIE_SUPER	(src_inp));
		for(i=0;i<N_RV_FRAMEINFOS;i++){
			SET_RV_MOVIE_FRAME_INFO(dst_inp,i,RV_MOVIE_FRAME_INFO(src_inp,i));
		}
		SET_RV_MOVIE_EXTRA	( dst_inp, RV_MOVIE_EXTRA	(src_inp));
	}
} // copy_rv_inode

int rv_is_open(void)
{
	if( rv_sbp == NULL )
		return 0;
	else
		return 1;
}

static int rv_pushd(QSP_ARG_DECL  const char *dirname)
{
	if( dir_stack_depth>=MAX_DIR_STACK_DEPTH ){
		WARN("rv_pushd:  directory stack overflow");
		return(-1);
	}
	dir_stack[dir_stack_depth] = savestr(rv_pathname);
	dir_stack_depth++;
	/* BUG check for stack overflow */
	return( rv_cd(QSP_ARG  dirname) );
}

static void rv_popd(SINGLE_QSP_ARG_DECL)
{
	if( dir_stack_depth <= 0 ){
		WARN("rv_popd:  directory stack underflow");
		return;
	}
	dir_stack_depth--;
	rv_cd(QSP_ARG  dir_stack[dir_stack_depth]);
	/*givbuf( (void *) dir_stack[dir_stack_depth]);*/
	rls_str( dir_stack[dir_stack_depth] );
}

static void close_disk_files(QSP_ARG_DECL  int ndisks, int *fd_arr)
{
	int i;

	for( i=0; i<ndisks; i++){
		if( close(fd_arr[i]) < 0 ){
			tell_sys_error("close");
			sprintf(ERROR_STRING,
		"(rawvol) close_disk_files:  error closing descriptor %d!?",i);
			WARN(ERROR_STRING);
		}
	}
}

static int open_disk_files(QSP_ARG_DECL  int ndisks,const char **disknames,int *fd_arr,blk_t *siz_arr)
{
	int i;

	// We would like exclusive use of disks if we will be recording!

	// O_EXCL used on linux 2.6 or greater will cause open to fail
	// if the device is in use by the system (e.g. mounted)


	int oflags= O_RDWR | (use_osync? O_SYNC : 0 ) | O_EXCL ;

#ifdef O_DIRECT
	/* use direct i/o ! */
//fprintf(stderr,"rawvol:  Using direct i/o...\n");
	oflags |= O_DIRECT;
#endif // ! O_DIRECT

	if( ndisks > MAX_DISKS ){
		sprintf(ERROR_STRING,
	"open_disk_files:  maximum number of disks is %d (%d requested)",
			MAX_DISKS,ndisks);
		NWARN(ERROR_STRING);
		return(-1);
	}

	for(i=0;i<ndisks;i++){
		fd_arr[i] = open(disknames[i],oflags);
		if( fd_arr[i] < 0 ){
			perror("open");
			sprintf(ERROR_STRING,
		"Error opening raw disk \"%s\"",disknames[i]);
			NWARN(ERROR_STRING);
			return(-1);
		}

		/* first find out what the offset of the end is */
		if( get_device_size(QSP_ARG  disknames[i],BLOCK_SIZE,&siz_arr[i]) < 0 ){
			NWARN("get_device_size() returned an error!?");
			return(-1);
		} else if( verbose ){
			sprintf(ERROR_STRING,"Device %s has %ld blocks",
				disknames[i],siz_arr[i]);
			advise(ERROR_STRING);
		}
	}
/* This is a complete hack to get this stuff working on donders...
 * For some reason, we get no space on device errors trying to write
 * the last 4 blocks, and invalid ioctl errors trying to write just before that!?
 * When we make /dev/sdf1 the first disk, (followed by c d e) we can make this 2,
 * but when b is first we have to make it 3.
 * How wacky!?
 */

#define N_RESERVED_BLOCKS	3
	if( verbose ){
sprintf(ERROR_STRING,"open_disk_files:  Reserving %d blocks on every disk (FIXME)",N_RESERVED_BLOCKS);
advise(ERROR_STRING);
	}

	for(i=0;i<ndisks;i++)
		siz_arr[i] -= N_RESERVED_BLOCKS;

	return(0);
}

int insure_default_rv(SINGLE_QSP_ARG_DECL)
{
	const char *default_name;

	if (!rawvol_debug)
		rawvol_debug =  add_debug_module(QSP_ARG  "rawvol");

	if( rv_sbp != NULL ){
		sprintf(ERROR_STRING,"insure_default_rv:  raw volume already open");
		WARN(ERROR_STRING);
		return(-1);
	}
	/* we used to have per-system defaults, now we assume there
	 * is a symlink /dev/rawvol
	 */
	default_name = "/dev/rawvol";

	if( verbose ){
		sprintf(ERROR_STRING,"%s:  using default raw volume %s",
			tell_progname(),default_name);
		advise(ERROR_STRING);
	}

	read_rv_super(QSP_ARG  default_name);
	/* BUG should return error code from read_rv_super */
	return(0);
}


static void add_path_component(const char *name)
{
	if( strlen(rv_pathname) > 1 )
		strcat(rv_pathname,ROOT_DIR_NAME);
	strcat(rv_pathname,name);
}

static void remove_path_component(void)
{
	char *s;

	s = &rv_pathname[ strlen(rv_pathname)-1 ];	/* pt to last char */
	assert( *s != '/' );

	while( *s != '/' )
		s--;

	if( s == rv_pathname )	/* leave / for root directory */
		*(s+1) = 0;
	else
		*s = 0;
}

static void set_pathname_context(SINGLE_QSP_ARG_DECL)
{
	Item_Context *icp;
	char ctxname[LLEN];

	sprintf(ctxname,"RV_Inode.%s",rv_pathname);
	icp = ctx_of(QSP_ARG  ctxname);
	if( icp == NULL ){
		if( rv_inode_itp == NULL )
			rv_inode_itp = new_item_type(QSP_ARG  "RV_Inode", DEFAULT_CONTAINER_TYPE);

		icp = create_item_context(QSP_ARG  rv_inode_itp,rv_pathname);
		if( icp == NULL ){
			sprintf(ERROR_STRING,"error creating rv context %s",rv_pathname);
			WARN(ERROR_STRING);
			return;
		}
	}
//fprintf(stderr,"Pushing RV inode context %s\n",CTX_NAME(icp));
	PUSH_ITEM_CONTEXT(rv_inode_itp,icp);
}

static void read_rv_data(RV_Inode *inp,char *data,uint32_t size)
{
	int i;
	off64_t offset,retoff;
	int n;

	offset = (off64_t) RV_ADDR(inp) * (off64_t) rv_sbp->rv_blocksize;
	for(i=0;i<rv_sbp->rv_ndisks;i++){
		retoff = my_lseek64(rv_sbp->rv_fd[i],offset,SEEK_SET);
		if( retoff != offset ){
			sprintf(DEFAULT_ERROR_STRING,"read_rv_data:  Error seeking on raw disk %d",i);
			NWARN(DEFAULT_ERROR_STRING);
			return;
		}
	}
	if( size > (uint32_t)(BLOCK_SIZE*rv_sbp->rv_ndisks) ){
		NERROR1("read_rv_data:  too much data");
	}
	i=0;
	while( size ){
		if( (n=read(rv_sbp->rv_fd[i],data,BLOCK_SIZE)) != BLOCK_SIZE ){
			perror("read");
			sprintf(DEFAULT_ERROR_STRING,
				"Tried to read data at 0x%lx",(long)data);
			NADVISE(DEFAULT_ERROR_STRING);
			NWARN("error in read_rv_data");
		}
		data += BLOCK_SIZE;
		size -= BLOCK_SIZE;
	}
}

/* This is like descend_directory, but it works with the on-disk data
 * instead of the working linked list...
 *
 * It gets called with scan_inode, and link_directory, in read_rv_super.
 */

static void scan_directory(QSP_ARG_DECL  RV_Inode *dk_inp,
					void (*func)(QSP_ARG_DECL  RV_Inode *) )
{
	void *data_blocks;
	short *sp;
	int not_root_dir;

	assert( RV_NAME_IDX(dk_inp) >= 0 && RV_NAME_IDX(dk_inp) < total_string_bytes );

	not_root_dir = strcmp( rv_stp+RV_NAME_IDX(dk_inp), ROOT_DIR_NAME );

	/* set the context */
	if( not_root_dir ){
		add_path_component(rv_stp+RV_NAME_IDX(dk_inp));
		set_pathname_context(SINGLE_QSP_ARG);
	}

	// The directory contents are stored as short indices, one block
	// per disk.  This limits the number of directory entries...

#ifdef O_DIRECT
	{
	int err_val;

	if( (err_val=posix_memalign(&data_blocks,BLOCK_SIZE,BLOCK_SIZE*rv_sbp->rv_ndisks)) != 0 ){
	 	ERROR1("Error in posix_memalign!?");
	}
	}
#else // ! O_DIRECT
	data_blocks = getbuf(BLOCK_SIZE*rv_sbp->rv_ndisks);
#endif // ! O_DIRECT

	read_rv_data(dk_inp,data_blocks,BLOCK_SIZE*rv_sbp->rv_ndisks);
	sp=(short *)data_blocks;
	while( *sp > 0 ){
		if( IS_DIRECTORY(&rv_in_tbl[*sp]) )
			scan_directory(QSP_ARG  &rv_in_tbl[*sp],func);
		else
			(*func)(QSP_ARG  &rv_in_tbl[*sp]);
		sp++;
	}
	givbuf(data_blocks);

	if( not_root_dir ){
		pop_item_context(QSP_ARG  rv_inode_itp);
		remove_path_component();
	}

	/* now call the thingy on this directory node */
	(*func)(QSP_ARG  dk_inp);
} // end scan_directory

/*
 * Scan the image of on-disk inodes, creating a working struct for each.
 *
 * If inum is -1, we are just reconverting after a disk sync,
 * and don't need to reallocate the memory structure.
 */

static void scan_inode(QSP_ARG_DECL  RV_Inode *dk_inp)
{
	long len;
	RV_Inode *inp;
	int i;

	assert( RV_INUSE(dk_inp) );
	assert( ! RV_SCANNED(dk_inp) );

	/* allocate this inode block */

	takespace(&rv_inode_freelist,RV_INODE_IDX(dk_inp),1);
	RV_INODE_IDX(dk_inp) = dk_inp-rv_in_tbl;

	/* allocate the string */
	assert( RV_NAME_IDX(dk_inp) >= 0 && RV_NAME_IDX(dk_inp) < (long)(rv_sbp->rv_nsb*BLOCK_SIZE) );

	len = strlen( rv_stp+RV_NAME_IDX(dk_inp) )  + 1;
#ifdef STRING_DEBUG
sprintf(ERROR_STRING,"reserving %d string bytes at offset %d",
len,RV_NAME_IDX(dk_inp));
advise(ERROR_STRING);
#endif /* STRING_DEBUG */
	takespace(&rv_st_freelist,RV_NAME_IDX(dk_inp),len);

	// BUG?  these frame info's are only for movies???
	/* allocate any error frames (also in the string table) */
	for(i=0;i<N_RV_FRAMEINFOS;i++){
		if( dk_inp->rvi_fi[i].fi_nsaved > 0 ){
			len = dk_inp->rvi_fi[i].fi_nsaved * sizeof(uint32_t);
			/* round up to insure alignment */
			len += LONG_ALIGN_SLOP;
#ifdef STRING_DEBUG
sprintf(ERROR_STRING,"reserving %d frame string bytes at offset %d",
len,dk_inp->rvi_fi[i].fi_savei);
advise(ERROR_STRING);
#endif /* STRING_DEBUG */
			takespace(&rv_st_freelist,dk_inp->rvi_fi[i].fi_savei,len);
		}
	}

	/* now allocate the data blocks */
	// Do links and directories have data blocks?
	// I don't think so!

	takespace(&rv_data_freelist,RV_ADDR(dk_inp),RV_N_BLOCKS(dk_inp));

	SET_RV_FLAG_BITS(dk_inp, RVI_SCANNED);

	/* now create a heap struct for this inode */

//fprintf(stderr,"creating new heap rv_inode '%s'\n",rv_stp+RV_NAME_IDX(dk_inp));
	inp = new_rv_inode(QSP_ARG  rv_stp+RV_NAME_IDX(dk_inp));
	if( inp == NULL ){
		sprintf(ERROR_STRING,
			"Couldn't create working copy of raw volume file %s",
			rv_stp+RV_NAME_IDX(dk_inp));
		WARN(ERROR_STRING);
		return;		/* BUG? no cleanup done */
	}

	SET_RV_NAME( dk_inp, RV_NAME(inp) );
	SET_RV_INODE( dk_inp, inp );

	copy_rv_inode(inp,dk_inp);

	//SET_RV_SUPER_P(inp,rv_sbp);

	/* We have to do this after we have copied the information from
	 * the disk inode, and so the call to setup_rv_iofile used to be
	 * conditional on inum>=0.
	 *
	 * But this caused a problem when syncing after a halted recording,
	 * because the iofile ended up with the wrong # of frames...
	 */
	if( IS_DIRECTORY(inp) ){
		inp->rvi_lp = new_list();
	} else if( IS_LINK(inp) ){
		/* advise("scan_inode:  not sure what to do about a link!?"); */
	} else {
		Image_File *ifp;

		// connect the links of the shape struct
		SET_RV_MOVIE_SHAPE(inp,ALLOC_SHAPE);

		SET_SHP_PREC_PTR( RV_MOVIE_SHAPE(inp), PREC_FOR_CODE(RV_MOVIE_PREC_CODE(inp)) );
		SET_SHP_MAXDIM( RV_MOVIE_SHAPE(inp), 0 );
		SET_SHP_MINDIM( RV_MOVIE_SHAPE(inp), 0 );
		SET_SHP_FLAGS( RV_MOVIE_SHAPE(inp), 0 );
//		SET_SHP_LAST_SUBI( RV_MOVIE_SHAPE(inp), 0 );
		// copy data, not pointers...
		COPY_DIMS( SHP_TYPE_DIMS( RV_MOVIE_SHAPE(inp) ), &RV_MOVIE_DIMS(inp));
		COPY_DIMS( SHP_MACH_DIMS( RV_MOVIE_SHAPE(inp) ), &RV_MOVIE_DIMS(inp));
		COPY_INCS( SHP_TYPE_INCS( RV_MOVIE_SHAPE(inp) ), &RV_MOVIE_INCS(inp));
		COPY_INCS( SHP_MACH_INCS( RV_MOVIE_SHAPE(inp) ), &RV_MOVIE_INCS(inp));
		// set the flags
		set_shape_flags(RV_MOVIE_SHAPE(inp),NULL,AUTO_SHAPE);

		/* the image file might be open already if we are rescanning */
		ifp = img_file_of(QSP_ARG  RV_NAME(inp));
		if( ifp != NULL ){
			close_image_file(QSP_ARG  ifp);
		}

		setup_rv_iofile(QSP_ARG  inp);
	}
} /* end scan_inode */



/* we need to do this after all of the inodes have been scanned... */

static void link_directory(QSP_ARG_DECL  RV_Inode *dk_inp)
{
	void *data_blocks;
	short *sp;

	if( ! IS_DIRECTORY(dk_inp) ){
		return;
	}

#ifdef O_DIRECT
	{
	int err_val;

	if( (err_val=posix_memalign(&data_blocks,BLOCK_SIZE,BLOCK_SIZE*rv_sbp->rv_ndisks)) != 0 ){
	 	ERROR1("Error in posix_memalign!?");
	}
	}
#else // ! O_DIRECT
	data_blocks = getbuf(BLOCK_SIZE*rv_sbp->rv_ndisks);
#endif // ! O_DIRECT

	read_rv_data(dk_inp,data_blocks,BLOCK_SIZE*rv_sbp->rv_ndisks);
	sp=(short *)data_blocks;
	while( *sp > 0 ){
		RV_Inode *inp2;
		Node *np;
		inp2 = rv_in_tbl[*sp].rvi_inp; /* look up an inode from its on-disk index */

		assert( inp2 != NULL );

		np = mk_node(inp2);
		addTail(dk_inp->rvi_inp->rvi_lp,np);
		RV_PARENT(inp2) = dk_inp->rvi_inp;
		sp++;
	}
}

/* This is the routine which opens a new volume.
 * Now we just read the info from the 1st disk,
 * it would be nice to have it duplicated on all 4...
 *
 * This routine has been enhanced to automatically register
 * all files with the fileio module...
 */

void read_rv_super(QSP_ARG_DECL  const char *vol_name)
{
	uint32_t max_inodes;
	off64_t offset,off_ret;
	int i;
	int fd_arr[MAX_DISKS];
	blk_t siz_arr[MAX_DISKS];
	const char *disknames[MAX_DISKS];
	off64_t end_offset;
	RV_Inode *inp;
	char *s_ptr;

	if( rv_sbp != NULL ){
		/* check and see if this one is already open! */
		for(i=0;i<rv_sbp->rv_ndisks;i++){
			if( !strcmp(rv_sbp->rv_diskname[i],vol_name) ){
				sprintf(ERROR_STRING,
		"read_rv_super:  Raw volume %s is already open",vol_name);
				advise(ERROR_STRING);
				return;
			}
		}

		sprintf(ERROR_STRING,
			"Closing previously opened raw volume %s",
			rv_sbp->rv_diskname[0]);
		advise(ERROR_STRING);

		rv_close(SINGLE_QSP_ARG);		/* close and free mem */
	}

	/* the volume name will be the name of any of the disks...
	 * they all should have a superblock written.
	 */

//fprintf(stderr,"read_rv_super:  reading first disk, \"%s\"\n",vol_name);
	if( open_disk_files(QSP_ARG  1,&vol_name,fd_arr,siz_arr) < 0 )
		return;

	end_offset = ((off64_t)siz_arr[0]) * BLOCK_SIZE;

	offset = end_offset - BLOCK_SIZE;

	off_ret = my_lseek64(fd_arr[0],offset,SEEK_SET);

	if( BAD_OFFSET64(off_ret) ){
		perror("read_rv_super:  my_lseek64");
		WARN("read_rv_super:  Error #1 seeking to last volume block");
//fprintf(stderr,"off_ret = 0x%lx\n",off_ret);
		goto errorA;
	}

#ifdef O_DIRECT
	{
	int err_val;
	void *ptr;

	if( (err_val=posix_memalign(&ptr,block_size,block_size)) != 0 ){
	 	WARN("Error in posix_memalign!?");
		goto errorA;
	}
	rv_sbp = ptr;
	}
#else // ! O_DIRECT
	rv_sbp = (RV_Super *)mem_get(block_size);

	if( rv_sbp == NULL ){
		WARN("read_rv_super:  Unable to allocate mem for superblock");
		goto errorA;
	}
#endif // ! O_DIRECT

	if( read(fd_arr[0],rv_sbp,block_size) != (int)block_size ){
		tell_sys_error("read");
		WARN("read_rv_super:  error reading superblock");
		goto errorB;
	}

	if( rv_sbp->rv_magic != RV_MAGIC && rv_sbp->rv_magic != RV_MAGIC2 ){
		sprintf(ERROR_STRING,
			"read_rv_super:  Volume file %s is not a raw file system (bad magic number)!?",
			vol_name);
		WARN(ERROR_STRING);
		goto errorB;
	}

	/* if the number of disks is more than 1, open the
	 * rest of the devices.
	 * For now, we assume that the disk which is already
	 * open is the first one, but this is really a BUG.
	 */

	if( strcmp(vol_name,rv_sbp->rv_diskname[0]) ){
		/* This is not an error if we are using a symlink /dev/rawvol */
		/*
		sprintf(ERROR_STRING,
	"read_rv_super:  Volume %s is not the first disk (%s)",vol_name,
			rv_sbp->rv_diskname[0]);
		WARN(ERROR_STRING);
		*/
	}

	if( rv_sbp->rv_ndisks > 1 ){
		for(i=1;i<rv_sbp->rv_ndisks;i++){
//fprintf(stderr,"read_rv_super:  disk %d = \"%s\"\n",i,rv_sbp->rv_diskname[i]);
			disknames[i] = rv_sbp->rv_diskname[i];
		}
		if( open_disk_files(QSP_ARG  rv_sbp->rv_ndisks-1,
			&disknames[1],&fd_arr[1],&siz_arr[1]) < 0 ){
			WARN("read_rv_super:  error opening disk files");
		}
	}

	/* allocate memory for the inodes and strings */

	/* string table */

	string_bytes_per_disk = block_size*rv_sbp->rv_nsb;
	total_string_bytes = rv_sbp->rv_ndisks*string_bytes_per_disk;

#ifdef O_DIRECT
	{
	int err_val;
	void *ptr;

	if( (err_val=posix_memalign(&ptr,block_size,total_string_bytes)) != 0 ){
	 	WARN("Error in posix_memalign!?");
		goto errorB;
	}
	rv_stp = ptr;
	}

#else // ! O_DIRECT
	rv_stp = (char *)mem_get(total_string_bytes);

	if( rv_stp == NULL ){
		WARN("read_rv_super:  failed to allocate string table");
		goto errorB;
	}
#endif // ! O_DIRECT

	offset = BLOCK_SIZE*(off64_t)rv_sbp->rv_ndb;

	s_ptr = rv_stp;

	for(i=0;i<rv_sbp->rv_ndisks;i++){
		off_ret = my_lseek64(fd_arr[i],offset,SEEK_SET);
		if( BAD_OFFSET64(off_ret) ){
			perror("read_rv_super:  my_lseek64");
			sprintf(ERROR_STRING,"read_rv_super:  error seeking to string table, disk %d (%s)",i,rv_sbp->rv_diskname[i]);
			WARN(ERROR_STRING);
			goto errorC;
		}
		if( read(fd_arr[i],s_ptr,string_bytes_per_disk) != (int)string_bytes_per_disk ){
			perror("read_rv_super:  read");
			sprintf(ERROR_STRING,"Tried to read %d (0x%x) bytes at 0x%lx",
				string_bytes_per_disk,string_bytes_per_disk,(long)s_ptr);
			advise(ERROR_STRING);
			sprintf(ERROR_STRING,"read_rv_super:  error reading string blocks, disk %d",i);
			WARN(ERROR_STRING);
			goto errorC;
		}
		s_ptr += string_bytes_per_disk;
	}


if( verbose ){
i=sizeof(RV_Inode);
sprintf(ERROR_STRING,"Inode size is %d bytes",i);
advise(ERROR_STRING);
}

	inode_bytes_per_disk = block_size*rv_sbp->rv_nib;
	inodes_per_disk = floor(inode_bytes_per_disk/sizeof(RV_Inode));

if( verbose ){
sprintf(ERROR_STRING,"%d inodes per disk",inodes_per_disk);
advise(ERROR_STRING);
}

#ifdef O_DIRECT
	{
	int err_val;
	void *ptr;

	if( (err_val=posix_memalign(&ptr,block_size,rv_sbp->rv_ndisks*inode_bytes_per_disk)) != 0 ){
	 	WARN("Error in posix_memalign!?");
		goto errorC;
	}
	rv_in_tbl = ptr;
	}

#else // ! O_DIRECT
	rv_in_tbl=(RV_Inode *)mem_get(rv_sbp->rv_ndisks*inode_bytes_per_disk);

	if( rv_in_tbl==NULL ){
		WARN("failed to allocate inode buffer");
		goto errorC;
	}

#endif // ! O_DIRECT

	inp=rv_in_tbl;

	for(i=0;i<rv_sbp->rv_ndisks;i++){
		/* inodes directly follow strings, so we don't have to seek */
		if( read(fd_arr[i],inp,inode_bytes_per_disk) != (int)inode_bytes_per_disk ){
			perror("read");
			sprintf(ERROR_STRING,"Tried to read 0x%x bytes at 0x%lx",
				inode_bytes_per_disk,(long)inp);
			advise(ERROR_STRING);
			sprintf(ERROR_STRING,"error reading inode blocks, disk %d",i);
			WARN(ERROR_STRING);
			goto errorD;
		}
		inp += inodes_per_disk;
	}

	/* initialize the freelists */

	freeinit(&rv_st_freelist,MAX_STRING_CHUNKS,rv_sbp->rv_nsb*block_size);

	inode_bytes_per_disk = (rv_sbp->rv_nib*block_size);
	inodes_per_disk = inode_bytes_per_disk / sizeof(RV_Inode);
	max_inodes = inodes_per_disk * rv_sbp->rv_ndisks;

	freeinit(&rv_inode_freelist,MAX_INODE_CHUNKS,max_inodes);


	/* strings and data are allocated in bytes, data in blocks */
	freeinit(&rv_data_freelist,MAX_DATA_CHUNKS,rv_sbp->rv_ndb);


	for(i=0;i<rv_sbp->rv_ndisks;i++){
		rv_sbp->rv_fd[i] = fd_arr[i];
	}

	strcpy(rv_pathname,ROOT_DIR_NAME);

	if( ! in_mkfs ){
		scan_directory(QSP_ARG  &rv_in_tbl[0],scan_inode);		/* this will recursively descend all the directories */
		scan_directory(QSP_ARG  &rv_in_tbl[0],link_directory);
		RV_PARENT(rv_in_tbl[0].rvi_inp) = NULL;

		if( rv_sbp->rv_magic == RV_MAGIC2 ){
			RV_Inode *inp;
			inp=rv_inode_of(QSP_ARG  ROOT_DIR_NAME);
			assert( inp != NULL );

			rv_sbp->rv_cwd = inp;
			set_pathname_context(SINGLE_QSP_ARG);
		}
	}

	return;

	/* various stages of error cleanup */

errorD:
	mem_release(rv_in_tbl);
errorC:
	mem_release(rv_stp);
errorB:
	mem_release(rv_sbp);

errorA:
	rv_sbp = NULL;
	close(fd_arr[0]);
} /* end read_rv_super */

static int has_root_access(uid_t uid)
{
	int i;

	if ( uid == 0) return 1;	/* really is root */

	for(i=0;i<rv_sbp->rv_n_super_users;i++)
		if( uid == rv_sbp->rv_root_uid[i] ) return 1;

	return 0;
}


int rv_access_allowed(QSP_ARG_DECL  RV_Inode *inp)
{
	uid_t my_uid;

	my_uid = getuid();

	/* root should be able to do anything */
	if( has_root_access(my_uid) ) return 1;

	if( my_uid != inp->rvi_uid ){
		/* not owner ... check group permissions */
		if( getgid() != inp->rvi_gid ) return(0);;
		/* group member, check mode */
		if( (RV_MODE(inp) & 020) == 0 ) return(0);;
		/* allow group members to delete */
	} else {
		/* if owner clears mode write mode, how can he set back? */
		if( (RV_MODE(inp) & 0200) == 0 ) return(0);
	}
	return(1);
}

/* Update the disk block image from the working heap struct, then convert the
 * disk image to the on-disk format.
 */

static void rls_inode(QSP_ARG_DECL  RV_Inode *inp)			/* convert back to disk format */
{
	long len;
	int i;
	Node *np;

	assert( RV_SCANNED(inp) );

	/* we should have already removed all the child inodes */
	if( IS_DIRECTORY(inp) ){
		dellist(inp->rvi_lp);
		inp->rvi_lp = NULL;
	}

	/* free the space used by the disk image copy.
	 *
	 * We give back the inode, string and data stuff here,
	 * so we can do a blind scan again later
	 */

	givspace(&rv_inode_freelist,1,RV_INODE_IDX(inp));

	len=strlen(&rv_stp[RV_NAME_IDX(inp)])+1;
	givspace(&rv_st_freelist,len,RV_NAME_IDX(inp));

	/* give away any error frames */
	for(i=0;i<N_RV_FRAMEINFOS;i++){
		if( inp->rvi_fi[i].fi_nsaved > 0 ){
			len = inp->rvi_fi[i].fi_nsaved * sizeof(uint32_t);
			/* round up to insure alignment */
			len += LONG_ALIGN_SLOP;
			givspace(&rv_st_freelist,len,inp->rvi_fi[i].fi_savei);
		}
	}

	/* The length SHOULD be 1 for directories... */
	if( RV_N_BLOCKS(inp) > 0 ){
		if( IS_DIRECTORY(inp) ){
			sprintf(ERROR_STRING,"Releasing directory %s, size %d at 0x%x",
					RV_NAME(inp),RV_N_BLOCKS(inp),RV_ADDR(inp));
			advise(ERROR_STRING);
		}
		givspace(&rv_data_freelist,RV_N_BLOCKS(inp),RV_ADDR(inp));
	}

	if( RV_PARENT(inp) != NULL ){
//fprintf(stderr,"rls_inode calling remData, lp = 0x%lx\n",(long)RV_PARENT(inp)->rvi_lp);
		np=remData(RV_PARENT(inp)->rvi_lp,inp);
//fprintf(stderr,"rls_inode back from remData, lp = 0x%lx\n",(long)RV_PARENT(inp)->rvi_lp);
		assert( np != NULL );

		rls_node(np);
	}

	del_rv_inode(QSP_ARG  inp);	/* remove from database */
} /* end rls_inode */

/* This is a bit tricky, because rv files can be created in isolation,
 * or they can be created by the image_file or movie modules...
 * Therefore, if we simply delete the rvi structure, we could leave
 * a dangling pointer.  However, when we close an image_file or movie,
 * we don't need to call rv_rmfile, because these things describe stuff
 * that is on the disk.
 */

static int rm_inode(QSP_ARG_DECL  RV_Inode *inp, int check_permissions)
{
	Image_File *ifp;
	int status=0;

	if( check_permissions && !rv_access_allowed(QSP_ARG  inp) ){
		sprintf(ERROR_STRING,"No permission to modify raw volume file %s",
			RV_NAME(inp));
		WARN(ERROR_STRING);
		return(-1);
	}

	if( IS_REGULAR_FILE(inp) ){

		/* remove from image file database also, if it is there...
		 *
		 * It won't be there though if we request the deletion from
		 * the fileio menu!
		 */

		ifp = img_file_of(QSP_ARG  RV_NAME(inp));
		if( ifp != NULL )
			GENERIC_IMGFILE_CLOSE(ifp);
	}

	if( IS_DIRECTORY(inp) ){
		Node *np;
		rv_pushd(QSP_ARG  RV_NAME(inp));
		while( status == 0 && (np=QLIST_HEAD(inp->rvi_lp)) != NULL ){
			status = rm_inode(QSP_ARG  (RV_Inode *)np->n_data,check_permissions);
		}
		rv_popd(SINGLE_QSP_ARG);
	}

	if( status == 0 ){
		CLEAR_RV_FLAG_BITS(inp, RVI_INUSE);
		rls_inode(QSP_ARG  inp);
	}
	return(status);
}

void rv_close(SINGLE_QSP_ARG_DECL)
{
	int i;
	RV_Inode *inp;

	if( rv_sbp == NULL ){
		WARN("rv_close:  no raw volume open");
		return;
	}

	rv_sync(SINGLE_QSP_ARG);		/* flush data to disk, don't rescan */

	/* now get rid of all the mem structs */
	if( rv_cd(QSP_ARG  ROOT_DIR_NAME) < 0 ) WARN("unable to cd /");
	inp=get_rv_inode(QSP_ARG  ROOT_DIR_NAME);

	assert( inp != NULL );

	/* why rm_inode when all we really want to do is clear the memory? */
	/* rv_sync above flushes to disk */
	if( rm_inode(QSP_ARG  inp,0) < 0 && verbose )
		WARN("rv_close:  error removing inode.");

	mem_release(rv_stp);
	mem_release(rv_in_tbl);
	/* BUG we are still using rv_sbp!? */
	//mem_release(rv_sbp);

	for(i=0;i<rv_sbp->rv_ndisks;i++){
		if( close(rv_sbp->rv_fd[i]) < 0 ){
			perror("close");
			sprintf(ERROR_STRING,
				"error closing raw volume disk %s",
				rv_sbp->rv_diskname[i]);
			WARN(ERROR_STRING);
		}
	}

	/* do this here to fix bug? */
	mem_release(rv_sbp);
	rv_sbp = NULL;

	while( eltcount( CONTEXT_LIST(rv_inode_itp) ) > 1 )
		pop_item_context(QSP_ARG  rv_inode_itp);
} // end rv_close

/* enter a readable file into the fileio database */

void setup_rv_iofile(QSP_ARG_DECL  RV_Inode *inp)
{
	set_filetype(QSP_ARG  FILETYPE_FOR_CODE(IFT_RV));

	/* BUG read_image_file() may change the filetype, if the filename
	 * implies something different...  there should be a way of disabling
	 * file type inference, or else inconsistent filenames should be
	 * disallowed.
	 */

	read_image_file(QSP_ARG  RV_NAME(inp));
}

static void write_rv_data(RV_Inode *inp,char *data,uint32_t size)
{
	int i;
	off64_t offset,retoff;
	int n;

	offset = (off64_t) RV_ADDR(inp) * (off64_t) rv_sbp->rv_blocksize;
	for(i=0;i<rv_sbp->rv_ndisks;i++){
		retoff = my_lseek64(rv_sbp->rv_fd[i],offset,SEEK_SET);
		if( retoff != offset ){
			sprintf(DEFAULT_ERROR_STRING,"write_rv_data:  Error seeking on raw disk %d (%s)",i,
					rv_sbp->rv_diskname[i]);
			NWARN(DEFAULT_ERROR_STRING);
			return;
		}
	}
	if( size > (uint32_t)(BLOCK_SIZE*rv_sbp->rv_ndisks) ){
		NERROR1("write_rv_data:  too much data");
	}
	i=0;
	while( size ){
		if( (n=write(rv_sbp->rv_fd[i],data,BLOCK_SIZE)) != BLOCK_SIZE ){
			perror("write");
			sprintf(DEFAULT_ERROR_STRING,
				"Tried to write data at 0x%lx",(long)data);
			NADVISE(DEFAULT_ERROR_STRING);
			NWARN("error in write_rv_data");
		}
		data += BLOCK_SIZE;
		size -= BLOCK_SIZE;
		i++;
	}
}

static void sync_dir_data(QSP_ARG_DECL  RV_Inode *inp)
{
	Node *np;
	void * data_blocks;	/* BUG get blocksize from superblock */
	short *sp;

	/* now rescan the list and write out the data */
#ifdef O_DIRECT
	/* BUG assumes one block per disk */
	{
	int err_val;

	if( (err_val=posix_memalign(&data_blocks,BLOCK_SIZE,BLOCK_SIZE*rv_sbp->rv_ndisks)) != 0 ){
	 	ERROR1("Error in posix_memalign!?");
	}
	}
#else // ! O_DIRECT
	data_blocks = getbuf(BLOCK_SIZE*rv_sbp->rv_ndisks);	/* BUG assumes one block per disk */
#endif // ! O_DIRECT

	/* now do we find the context? */

	assert( eltcount(inp->rvi_lp) <= (BLOCK_SIZE*rv_sbp->rv_ndisks)/sizeof(short) );

	np=QLIST_HEAD(inp->rvi_lp);
	sp=(short *)data_blocks;
	while(np!=NULL){
		RV_Inode *inp2;
		inp2 = (RV_Inode *)np->n_data;
		*sp++ = (short) inp2->rvi_ini;	/* remember the inode number of this child */
		np = np->n_next;
	}
	while( ( ((char*)sp) - ((char *)data_blocks) ) < BLOCK_SIZE*rv_sbp->rv_ndisks )
		*sp++ = 0;

	/* Now we need to write out the block... */
	write_rv_data(inp,data_blocks,BLOCK_SIZE*rv_sbp->rv_ndisks);
	givbuf(data_blocks);
}

static void sync_inode(QSP_ARG_DECL  RV_Inode *inp)
{
	if( IS_DIRECTORY(inp) ){
		sync_dir_data(QSP_ARG  inp);	/* flush inode numbers to disk */
	}

	/* update the disk image copy */
	// structure copy
	//rv_in_tbl[RV_INODE_IDX(inp)] = *inp;
	copy_rv_inode(&rv_in_tbl[RV_INODE_IDX(inp)],inp);

	/* do we need to do this? */
	rv_in_tbl[RV_INODE_IDX(inp)].rvi_flags = RV_FLAGS(inp) & ~RVI_SCANNED;
}

#ifdef UNUSED

static RV_Inode *search_directory(RV_Inode *inp, int index)
{
	Node *np;

	np=QLIST_HEAD(inp->rvi_lp);
	while(np!=NULL){
		inp=(RV_Inode *)np->n_data;
		if( RV_INODE_IDX(inp) == index ) return(inp);
		if( IS_DIRECTORY(inp) ){
			inp=search_directory(inp,index);
			if( inp != NULL && RV_INODE_IDX(inp) == index )
				return(inp);
		}
		np=np->n_next;
	}
	return(NULL);
}
#endif /* UNUSED */



static int perform_seek( int fd, off64_t offset )
{
	off64_t retoff;
	retoff = my_lseek64(fd,offset,SEEK_SET);
	if( retoff != offset ){
		sprintf(DEFAULT_ERROR_STRING,"perform_write_test:  Error seeking on raw disk");
		NWARN(DEFAULT_ERROR_STRING);
		return(-1);
	}
	return(0);
}

#define N_REPETITIONS 128

void perform_write_test( QSP_ARG_DECL  int i_disk, int n_blocks, int n_reps )
{
	uint32_t i_blk;
	int n_xfer;
	off64_t offset;
	int fd;
	char *buf;
	struct timeval tv1, tv2;
	uint32_t max_block;
	int n_actual;
	int i_rep;
	long delta1,delta2;
	double delta;

	/* BUG?  make sure rv_sbp has valid value? */

	n_xfer = n_blocks * BLOCK_SIZE;
	buf = (char *)getbuf( n_xfer );	/* BUG?  check alignment? */
	max_block = rv_sbp->rv_ndb - 1 ;

	if( i_disk < 0 || i_disk >= rv_sbp->rv_ndisks ){
		sprintf(DEFAULT_ERROR_STRING,"Disk index %d is out of range (0-%d)",
			i_disk,rv_sbp->rv_ndisks-1);
		NWARN(DEFAULT_ERROR_STRING);
		return;
	}

	fd = rv_sbp->rv_fd[i_disk];
	i_blk = 0;
	while( (i_blk+n_blocks-1) <= max_block ){
		offset = (off64_t) i_blk * (off64_t) rv_sbp->rv_blocksize;
		if( perform_seek(fd,offset) < 0 ) return;
		if( (n_actual=read(fd,buf,n_xfer)) != n_xfer ){
			sprintf(DEFAULT_ERROR_STRING,
	"perform_write_test:  tried to read %d bytes, but got %d!?",n_xfer,n_actual);
			NWARN(DEFAULT_ERROR_STRING);
			return;
		}

		gettimeofday(&tv1,NULL);
		/* now try to write many times */
		for(i_rep=0;i_rep<n_reps;i_rep++){
			if( perform_seek(fd,offset) < 0 ) return;
			if( (n_actual=write(fd,buf,n_xfer)) != n_xfer ){
				sprintf(DEFAULT_ERROR_STRING,
		"perform_write_test:  tried to write %d bytes, but got %d!?",n_xfer,n_actual);
				NWARN(DEFAULT_ERROR_STRING);
				return;
			}
		}
		gettimeofday(&tv2,NULL);

		delta1 = tv2.tv_sec - tv1.tv_sec;
		delta2 = tv2.tv_usec - tv1.tv_usec;
		if( delta2 < 0 ){
			delta2 += 1000000;
			delta1 -= 1;
		}
		delta = delta2;
		delta /= 1000000;
		delta += delta1;

		sprintf(msg_str,"disk %d\tblk %d/%d\t\t%g",i_disk,i_blk,max_block,delta);
		prt_msg(msg_str);

		i_blk += n_blocks;
	}
}

RV_Inode * rv_newfile(QSP_ARG_DECL  const char *name,uint32_t size)
{
	RV_Inode *inp;
	int i;

	if( rv_sbp == NULL ){
		sprintf(ERROR_STRING,
			"no raw volume open, can't create new file %s",name);
		WARN(ERROR_STRING);
		return(NULL);
	}

	/* make sure name is unique */

	inp = new_rv_inode(QSP_ARG  name);			/* get a struct from the heap */
	if( inp == NULL ) return(inp);

	/* get space in the disk image for the name */

	RV_NAME_IDX(inp) = getspace(&rv_st_freelist,strlen(name)+1);

	if( RV_NAME_IDX(inp) < 0 ){
		sprintf(ERROR_STRING,"Error getting name space for new rv file %s",name);
		WARN(ERROR_STRING);
		goto errorA;
	}
#ifdef STRING_DEBUG
sprintf(ERROR_STRING,"%d name bytes allocated at offset %d",strlen(name)+1,
RV_NAME_IDX(inp));
advise(ERROR_STRING);
#endif /* STRING_DEBUG */
	strcpy(rv_stp+RV_NAME_IDX(inp),name);

	/* get space in the disk image for the inode */

	RV_INODE_IDX(inp) = getspace(&rv_inode_freelist,1);
	if( RV_INODE_IDX(inp) < 0 ){
		sprintf(ERROR_STRING,"Error getting inode space for new rv file %s",name);
		WARN(ERROR_STRING);
		goto errorB;
	}

	SET_RV_FLAG_BITS(inp, RVI_INUSE | RVI_SCANNED);

	/* divide size by the number of disks, rounding up to nearest int */

	if( size > 0 ){
		size = BLOCKS_PER_DISK(size);
		RV_ADDR(inp) = getspace(&rv_data_freelist,size);
		if( RV_ADDR(inp) < 0 ){
			sprintf(ERROR_STRING,"Error getting %d data blocks for new rv file %s",size,name);
			WARN(ERROR_STRING);
			goto errorC;
		}
		RV_N_BLOCKS(inp) = size;
	} else {
		RV_N_BLOCKS(inp) = 0;
	}

	time(&RV_ATIME(inp));
	RV_MTIME(inp) = RV_ATIME(inp);
	RV_CTIME(inp) = RV_ATIME(inp);

	/* set the mode, uid, gid */
	SET_RV_MODE(inp,0644);			/* default */
	inp->rvi_uid = getuid();		/* should we use real or effective? */
	inp->rvi_gid = getgid();

	//inp->rvi_sbp = rv_sbp;
	SET_RV_MOVIE_EXTRA(inp, n_extra_bytes);

	for(i=0;i<N_RV_FRAMEINFOS;i++)
		inp->rvi_fi[i].fi_nsaved = 0;

	/* make the disk image match our working version */
	rv_in_tbl[RV_INODE_IDX(inp)] = (*inp);

	/* we add this file to the current directory - UNLESS we are creating the root dir */
	/* BUT we also have to worry about the . entry in the root directory... */
	if( (!in_mkfs) && rv_sbp->rv_cwd!=NULL ){
		Node *np;
		np = mk_node(inp);
//fprintf(stderr,"rv_newfile adding inode to directory list 0x%lx...\n",(long)rv_sbp->rv_cwd->rvi_lp);
		addTail(rv_sbp->rv_cwd->rvi_lp,np);
		RV_PARENT(inp) = rv_sbp->rv_cwd;
	} else {
		RV_PARENT(inp) = NULL;
	}

	SET_RV_MOVIE_SHAPE(inp,ALLOC_SHAPE);
	// Need to put some values here?  RV_MOVIE_PREC
	SET_RV_MOVIE_PREC_CODE(inp,PREC_UBY);
	{
		Dimension_Set dimset={{1,1,1,1,1},1};
		Increment_Set incset={{1,0,0,0,0}};
		// BUG set dimset[0] to number of bytes
		COPY_DIMS( & RV_MOVIE_DIMS(inp), &dimset );
		COPY_INCS( & RV_MOVIE_INCS(inp), &incset );
	}

	return(inp);

errorC:
	givspace(&rv_inode_freelist,1,RV_INODE_IDX(inp));
errorB:
	givspace(&rv_st_freelist,strlen(name)+1,RV_NAME_IDX(inp));
errorA:
	del_rv_inode(QSP_ARG  inp);		/* remove from database */
	return(NULL);
} /* end rv_newfile */

void xfer_frame_info(uint32_t *lp,int index,RV_Inode *inp)
{
	uint32_t *src;
	int n;

	assert( inp->rvi_fi[index].fi_nsaved > 0 );

	src = (uint32_t *)( rv_stp + ALIGN(inp->rvi_fi[index].fi_savei) );

	n = inp->rvi_fi[index].fi_nsaved;
	while(n--)
		*lp++ = *src++;
}

int remember_frame_info(RV_Inode *inp,int index,USHORT_ARG n,uint32_t *frames)
{
	uint32_t *lp;
	uint32_t len;
	long i_addr;

	assert( index >= 0 && index < N_RV_FRAMEINFOS );

	/* find space for the frames in the string table */

	len = n*sizeof(uint32_t);
	/* round up to insure alignment */
	len += LONG_ALIGN_SLOP;
	i_addr = getspace(&rv_st_freelist,len);
	if( i_addr < 0 ){
		sprintf(DEFAULT_ERROR_STRING,
			"error allocating space for %d error frames for rv inode %s",
			n,RV_NAME(inp));
		NWARN(DEFAULT_ERROR_STRING);
		return(-1);
	}
	if( i_addr > 0x7fffffff ){	// largest uint32
		// We could just make this be 64 bits...
		sprintf(DEFAULT_ERROR_STRING,
			"Error frame allocation address 0x%lx for rv inode %s exists variable range!?",
			i_addr,RV_NAME(inp));
		NWARN(DEFAULT_ERROR_STRING);
		return(-1);
	}
	inp->rvi_fi[index].fi_savei = i_addr;

#ifdef STRING_DEBUG
sprintf(DEFAULT_ERROR_STRING,"allocated %d frame bytes at offset %d",
len,inp->rvi_fi[index].fi_savei);
advise(DEFAULT_ERROR_STRING);
#endif /* STRING_DEBUG */

	inp->rvi_fi[index].fi_nsaved = n;

	lp = (uint32_t *)( rv_stp + ALIGN(inp->rvi_fi[index].fi_savei) );
	while(n--)
		*lp++ = *frames++;

	return(0);
}

static long uid_from_name(const char *s)
{
	struct passwd *pwp;

	setpwent();	/* rewind file */
	while( (pwp=getpwent()) != NULL ){
		if( !strcmp(s,pwp->pw_name) )
			return((long)pwp->pw_uid);
	}
	sprintf(DEFAULT_ERROR_STRING,"Unable to determine numeric user id for user name %s",s);
	NWARN(DEFAULT_ERROR_STRING);
	return(-1);
}

static char * name_from_uid(uid_t uid)
{
	struct passwd *pwp;

	setpwent();	/* rewind file */
	while( (pwp=getpwent()) != NULL ){
		if( uid == pwp->pw_uid )
			return(pwp->pw_name);
	}
	return(NULL);
}

static void deny_root_access( int index )
{
	int i;

	assert( rv_sbp->rv_n_super_users > 0 );

	/* shift valid id's down to fill space vacated by denied id */
	for(i=index;i<(rv_sbp->rv_n_super_users-1);i++)
		rv_sbp->rv_root_uid[i] = rv_sbp->rv_root_uid[i+1];
	rv_sbp->rv_n_super_users --;
}

/* the super block is the very last block... */

static int flush_super(QSP_ARG_DECL  RV_Super *sbp)
{
	off64_t offset,off_ret;
	int nw;
	uint32_t n_blocks;

	n_blocks = sbp->rv_nblks[0];
	offset = ((off64_t)(n_blocks-1)) * BLOCK_SIZE;

	off_ret = my_lseek64(sbp->rv_fd[0],offset,SEEK_SET);
	if( BAD_OFFSET64(off_ret) ){
		perror("flush_super:  my_lseek64");
		NWARN("flush_super:  Error seeking to last-1 volume block");
		return(-1);
	}

	/* off_ret should be the file offset relative to the start */

	if( (nw=write(sbp->rv_fd[0],sbp,block_size)) != (int)block_size ){
		if( nw != (-1) ){
			sprintf(ERROR_STRING,"%d bytes written",nw);
			advise(ERROR_STRING);
		} else {
			perror("write");
			sprintf(ERROR_STRING,
"Tried to write %ld (0x%lx) bytes at 0x%lx",block_size,block_size,(long)sbp);
			advise(ERROR_STRING);
		}
		WARN("error writing superblock");

		return(-1);
	}
	return(0);
}

static void show_root_users(SINGLE_QSP_ARG_DECL)
{
	int i;
	char *s;

	if( rv_sbp->rv_n_super_users == 0 ){
		prt_msg("\nNo users with root privileges");
		return;
	}
	// CHECK BUG? - changed <= to < in test below...
	assert( rv_sbp->rv_n_super_users >= 0 && rv_sbp->rv_n_super_users < MAX_RV_SUPER_USERS );

	for(i=0;i<rv_sbp->rv_n_super_users;i++){
		if( i == 0 ) prt_msg("\nUsers with root privilege:");
		s = name_from_uid( rv_sbp->rv_root_uid[i] );
		if( s == NULL ){
			sprintf(DEFAULT_ERROR_STRING,"\t(uid %d, unable to locate user name!?)",
				rv_sbp->rv_root_uid[i]);
			NWARN(DEFAULT_ERROR_STRING);
			deny_root_access(i);
			flush_super(QSP_ARG  rv_sbp);
			i--;
		} else {
			sprintf(msg_str,"\t%s",s);
			prt_msg(msg_str);
		}
	}
}

int grant_root_access(QSP_ARG_DECL  const char *user_name)
{
	uid_t uid;
	long l_uid;
	int i;

	if( rv_sbp == NULL ){
		WARN("grant_root_access:  no raw volume open");
		sprintf(ERROR_STRING,"Unable to grant rawvol root access to user %s",user_name);
		advise(ERROR_STRING);
		return(-1);
	}
	/* make sure we have a slot available */
	if( rv_sbp->rv_n_super_users >= MAX_RV_SUPER_USERS ){
		sprintf(ERROR_STRING,"Current rawvol already has %d super users",
			MAX_RV_SUPER_USERS);
		WARN(ERROR_STRING);
		return(-1);
	}
	/* Now look up the user id */
	l_uid = uid_from_name(user_name);

	if( l_uid < 0 ) return(-1);
	uid = (uid_t) l_uid;

	/* make sure that this user is not already on the list */
	for(i=0;i<rv_sbp->rv_n_super_users;i++){
		if( uid == rv_sbp->rv_root_uid[i] ){
			sprintf(ERROR_STRING,"User %s is already on the rawvol root access list!?",
				user_name);
			WARN(ERROR_STRING);
			return(-1);
		}
	}

	/* go ahead and add */

	rv_sbp->rv_root_uid[ rv_sbp->rv_n_super_users ++ ] = uid;
	flush_super(QSP_ARG  rv_sbp);
	return(0);
} // grant_root_access

void rv_chmod(QSP_ARG_DECL  RV_Inode *inp,int mode)
{
	if( !rv_access_allowed(QSP_ARG  inp) ) return;

	SET_RV_MODE(inp,mode);
}

static char path_comp[LLEN];

static char * next_path_component(const char **strp)
{
	const char *s;
	char *buf;

	s=(*strp);

	if( *s == '/' ){
		strcpy(path_comp,ROOT_DIR_NAME);
	} else {
		buf=path_comp;
		while( *s && *s!='/' )
			*buf++ = *s++;
		*buf=0;
	}
	if( *s == '/' ) s++;
	*strp=s;
	return(path_comp);
}

static int rv_step_dir(QSP_ARG_DECL  const char *dirname)
{
	RV_Inode *inp;

	if( !strcmp(dirname,"..") ){
		if( RV_PARENT(rv_sbp->rv_cwd) == NULL ){
			WARN("Current directory has no parent, can't cd ..");
			return(-1);
		} else {
			rv_sbp->rv_cwd = RV_PARENT(rv_sbp->rv_cwd);
			remove_path_component();
			pop_item_context(QSP_ARG  rv_inode_itp);
			set_pathname_context(SINGLE_QSP_ARG);
		}
		return(0);
	} else if( !strcmp(dirname,".") ) return(0);	/* a noop */

	inp = rv_inode_of(QSP_ARG  dirname);
	if( inp==NULL ){
		sprintf(ERROR_STRING,"RV directory \"%s\" does not exist",dirname);
		WARN(ERROR_STRING);
		return(-1);
	}
	rv_sbp->rv_cwd = inp;
	if( !strcmp(RV_NAME(inp),ROOT_DIR_NAME) )
		strcpy(rv_pathname,RV_NAME(inp));
	else
		add_path_component(RV_NAME(inp));

	if( eltcount( CONTEXT_LIST(rv_inode_itp) ) > 1 )
		pop_item_context(QSP_ARG  rv_inode_itp);
	else if( ! in_mkfs ){
		sprintf(ERROR_STRING,"rv_mkdir:  no context to pop!?");
		WARN(ERROR_STRING);
	}
	set_pathname_context(SINGLE_QSP_ARG);
	return(0);
} /* end rv_step_dir */

int rv_rmfile(QSP_ARG_DECL  const char *name)
{
	RV_Inode *inp;
	char *s;
	int pushed=0;
	char tmp_name[LLEN];
	int status=0;

	strcpy(tmp_name,name);
	name=tmp_name;

	while( *name && (s=next_path_component(&name)) != NULL ){
		if( *name == 0 ){	/* last component? */
			if( !strcmp(s,".") || !strcmp(s,"..") ){
				sprintf(ERROR_STRING,"You probably should not be removing directory %s",s);
				WARN(ERROR_STRING);
			}
			inp = get_rv_inode(QSP_ARG  s);
			if( inp==NULL ) return(-1);
			status=rm_inode(QSP_ARG  inp,1);
		} else {
			if( !pushed ){
				if( rv_pushd(QSP_ARG  s) < 0 ) return(-1);
				pushed++;
			} else {
				if( rv_step_dir(QSP_ARG  s) < 0 ) {
					rv_popd(SINGLE_QSP_ARG);
					return(-1);
				}
			}
		}
	}
	if( pushed ) rv_popd(SINGLE_QSP_ARG);
	return(status);
}

void rv_lsfile(QSP_ARG_DECL  const char *name)
{
	RV_Inode *inp;

	inp = rv_inode_of(QSP_ARG  name);
	if( inp==NULL ) return;

	rv_ls_inode(QSP_ARG  inp);
}


static const char *error_name[N_RV_FRAMEINFOS]={
	"frame drop",
	"fifo",
	"dma",
	"fifo/dma"
};

static void rv_ls_extra(QSP_ARG_DECL  RV_Inode *inp)
{
	int i;

	sprintf(msg_str,"index %d",RV_INODE_IDX(inp));
	prt_msg_frag(msg_str);

	sprintf(msg_str,"\t\taddr 0x%x",RV_ADDR(inp));
	prt_msg_frag(msg_str);

	sprintf(msg_str,"\t\tflags:  %s, %s",
		(RV_FLAGS(inp) & RVI_INUSE) ? "inuse" : "unused" ,
		(RV_FLAGS(inp) & RVI_SCANNED) ? "scanned" : "unscanned" );
	prt_msg(msg_str);

	if( IS_REGULAR_FILE(inp) )
		for(i=0;i<N_RV_FRAMEINFOS;i++)
			if( inp->rvi_fi[i].fi_nsaved > 0 ){
				sprintf(msg_str,"%d %s errors",inp->rvi_fi[i].fi_nsaved,
					error_name[i]);
				prt_msg(msg_str);
			}
}

void rv_info(QSP_ARG_DECL  RV_Inode *inp)
{
	rv_ls_inode(QSP_ARG  inp);
	rv_ls_extra(QSP_ARG  inp);
	if( IS_REGULAR_FILE(inp) )
		describe_shape(QSP_ARG  RV_MOVIE_SHAPE(inp));
}

void rv_ls_inode(QSP_ARG_DECL  RV_Inode *inp)
{
	char mode_str[12];
	struct passwd *pwp;
	struct group *grpp;
	char *s;

	if( IS_DIRECTORY(inp) )
		mode_str[0]='d';
	else
		mode_str[0]='-';

	if( RV_MODE(inp) & 0400 ) mode_str[1]='r';
	else			   mode_str[1]='-';
	if( RV_MODE(inp) & 0200 ) mode_str[2]='w';
	else			   mode_str[2]='-';
	if( RV_MODE(inp) & 0100 ) mode_str[3]='x';
	else			   mode_str[3]='-';
	if( RV_MODE(inp) & 040 )  mode_str[4]='r';
	else			   mode_str[4]='-';
	if( RV_MODE(inp) & 020 )  mode_str[5]='w';
	else			   mode_str[5]='-';
	if( RV_MODE(inp) & 010 )  mode_str[6]='x';
	else			   mode_str[6]='-';
	if( RV_MODE(inp) & 04 )   mode_str[7]='r';
	else			   mode_str[7]='-';
	if( RV_MODE(inp) & 02 )   mode_str[8]='w';
	else			   mode_str[8]='-';
	if( RV_MODE(inp) & 01 )   mode_str[9]='x';
	else			   mode_str[9]='-';
	mode_str[10]=0;
	prt_msg_frag(mode_str);

	prt_msg_frag("   ");

	pwp=getpwuid(inp->rvi_uid);
	if( pwp==NULL )
		sprintf(msg_str,"%-12s","(null)");
	else
		sprintf(msg_str,"%-12s",pwp->pw_name);
	prt_msg_frag(msg_str);

	grpp = getgrgid(inp->rvi_gid);
	if( grpp == NULL )
		sprintf(msg_str,"%-12s","(null)");
	else
		sprintf(msg_str,"%-12s",grpp->gr_name);
	prt_msg_frag(msg_str);

	sprintf(msg_str,"%12d  ",RV_N_BLOCKS(inp));
	prt_msg_frag(msg_str);

	s=ctime(&RV_MTIME(inp));
	if( s[strlen(s)-1] == '\n' || s[strlen(s)-1]=='\r' ){
		s[strlen(s)-1]=0;
	}
	sprintf(msg_str,"%-28s",s);
	prt_msg_frag( msg_str );

	prt_msg_frag(RV_NAME(inp));
	prt_msg("");		/* get \n out */
}

/* write a memory image of the on-disk inodes to disk
 *
 * the offset is nsb (number of string blocks?) plus ndb (number of data blocks?)
 */

static int flush_inodes(RV_Super *sbp,RV_Inode *inp)
{
	off64_t offset,off_ret;
	int i;

	offset = BLOCK_SIZE*(off64_t)( sbp->rv_nsb + sbp->rv_ndb );
	for(i=0;i<sbp->rv_ndisks;i++){
		off_ret = my_lseek64(sbp->rv_fd[i],offset,SEEK_SET);
		if( BAD_OFFSET64(off_ret) ){
			perror("flush_inodes:  my_lseek64");
			sprintf(DEFAULT_ERROR_STRING,
	"flush_inodes:  error #3 seeking to inode table, disk %d (%s)",i,rv_sbp->rv_diskname[i]);
			NWARN(DEFAULT_ERROR_STRING);
			return(-1);
		}

		if( write(sbp->rv_fd[i],inp,inode_bytes_per_disk)
			!= (int) inode_bytes_per_disk ){

			perror("write");
			sprintf(DEFAULT_ERROR_STRING,
	"Tried to write %d (0x%x) bytes at 0x%lx",inode_bytes_per_disk,inode_bytes_per_disk,(long)inp);
			NADVISE(DEFAULT_ERROR_STRING);
			sprintf(DEFAULT_ERROR_STRING,
	"flush_inodes:  error writing inode blocks, disk %d",i);
			NWARN(DEFAULT_ERROR_STRING);
			return(-1);
		}
		inp += inodes_per_disk;
	}
	return(0);
} // flush_inodes

void rv_mkfs(QSP_ARG_DECL  int ndisks,const char **disknames,uint32_t nib,uint32_t nsb)
{
	int fd_arr[MAX_DISKS];
	blk_t min_siz,siz_arr[MAX_DISKS];
	long nb,n;
	size_t block_size=BLOCK_SIZE;
	RV_Inode *inp;
	int i;
	RV_Super *sbp;

	/*
	if( geteuid() != 0 ){
		WARN("Sorry, only root can mkfs");
		return;
	}
	*/

	if( open_disk_files(QSP_ARG  ndisks,disknames,fd_arr,siz_arr) < 0 )
		return;

	/* BUG should use the minimum # of blocks of all the disks... */

	/* find minimum size */
	min_siz = siz_arr[0];
	for(i=1;i<ndisks;i++)
		if( siz_arr[i] < min_siz ) min_siz = siz_arr[i];


	nb = min_siz - 1;
	nb -= nib;
	nb -= nsb;

#ifdef O_DIRECT
	{
	int err_val;
	void *ptr;

	if( (err_val=posix_memalign(&ptr,block_size,block_size)) != 0 ){
	 	WARN("Error in posix_memalign!?");
		goto errorA;
	}
	sbp = ptr;
	}
#else // ! O_DIRECT
	sbp = (RV_Super *)mem_get(block_size);

	if( sbp == NULL ){
		WARN("Unable to allocate mem for superblock");
		goto errorA;
	}

#endif // ! O_DIRECT

	/* initialize super-block */

	sbp->rv_magic = RV_MAGIC2;
	sbp->rv_ndisks = ndisks;
	sbp->rv_blocksize = BLOCK_SIZE;
	for(i=0;i<ndisks;i++){
		sbp->rv_nblks[i] = siz_arr[i];
		strcpy(sbp->rv_diskname[i],disknames[i]);
		sbp->rv_fd[i] = fd_arr[i];
	}
	sbp->rv_nib = nib;
	sbp->rv_nsb = nsb;
	sbp->rv_ndb = nb;
	//sbp->rv_flags = 0;
	sbp->rv_n_super_users = 0;
	sbp->rv_cwd = NULL;

	if( flush_super(QSP_ARG  sbp) < 0 ) goto errorB;

	/* allocate memory for the inodes */

	inode_bytes_per_disk = block_size*sbp->rv_nib;

#ifdef O_DIRECT
	{
	int err_val;
	void *ptr;

	if( (err_val=posix_memalign(&ptr,block_size,ndisks*inode_bytes_per_disk)) != 0 ){
	 	WARN("Error in posix_memalign!?");
		goto errorB;
	}
	rv_in_tbl = ptr;
	}

#else // ! O_DIRECT
	rv_in_tbl=(RV_Inode *)mem_get(inode_bytes_per_disk * ndisks);

	if( rv_in_tbl==NULL ){
		WARN("rv_mkfs:  failed to allocate inode buffer");
		goto errorB;
	}
#endif // ! O_DIRECT

	inp=rv_in_tbl;
	n = inode_bytes_per_disk * ndisks / sizeof(*inp);
	while(n--){
		RV_FLAGS(inp)=0;		/* clear RVI_INUSE */
		inp++;
	}

	if( flush_inodes(sbp,rv_in_tbl) < 0 )
		goto errorC;

	/* now we mount, make a root directory, then unmount... */
	close_disk_files(QSP_ARG  ndisks,fd_arr);
	in_mkfs=1;
	read_rv_super(QSP_ARG  disknames[0]);
	rv_mkdir(QSP_ARG  ROOT_DIR_NAME);
	rv_close(SINGLE_QSP_ARG);
	in_mkfs=0;

	return;


	/* BUG, on error we should clear the magic number!? */
	/* Really, we should just write the superblock last... */

errorC:
	mem_release(rv_in_tbl);
errorB:
	mem_release(sbp);
errorA:
	//close(fd_arr[0]);
	// close ALL the drives...
	close_disk_files(QSP_ARG  ndisks,fd_arr);
}

void rv_ls_cwd(SINGLE_QSP_ARG_DECL)
{
	Node *np;
	List *lp;

	CHECK_VOLUME("rv_ls_cwd")

	lp=rv_sbp->rv_cwd->rvi_lp;
	if( lp==NULL ) return;

	np=QLIST_HEAD(lp);
	while( np != NULL ){
		rv_ls_inode(QSP_ARG   (RV_Inode *)np->n_data );
		np = np->n_next;
	}
}

/* Remove all files in the current directory (recursive)
 */

void rv_rm_cwd(SINGLE_QSP_ARG_DECL)
{
	Node *np;
	List *lp;

	CHECK_VOLUME("rv_rm_cwd")

	lp=rv_sbp->rv_cwd->rvi_lp;
	if( lp==NULL ) return;

	np=QLIST_HEAD(lp);
	while( np != NULL ){
		RV_Inode *inp;
		inp = (RV_Inode *)np->n_data ;
#ifdef DEBUG
if( debug & rawvol_debug ){
sprintf(ERROR_STRING,"rv_rm_cwd removing %s",RV_NAME(inp));
advise(ERROR_STRING);
}
#endif /* DEBUG */
		if( rv_rmfile(QSP_ARG  RV_NAME(inp)) < 0 )
			return;

		np=QLIST_HEAD(lp);
	}
}

static void descend_directory( QSP_ARG_DECL  RV_Inode *inp, void (*func)(QSP_ARG_DECL  RV_Inode *) )
{
	Node *np;
	RV_Inode *child_inp;

	assert( IS_DIRECTORY(inp) );

	np=QLIST_HEAD(inp->rvi_lp);
	while( np != NULL ){
		child_inp = (RV_Inode *)np->n_data;
		if( IS_DIRECTORY(child_inp) && strcmp(RV_NAME(child_inp),".")
			&& strcmp(RV_NAME(child_inp),"..") ){

			rv_pushd(QSP_ARG  RV_NAME(child_inp));
			descend_directory(QSP_ARG  child_inp,func);
			rv_popd(SINGLE_QSP_ARG);
		} else
			(*func)( QSP_ARG  child_inp );
		np = np->n_next;
	}
	(*func)(QSP_ARG  inp);
}

void traverse_rv_inodes( QSP_ARG_DECL  void (*func)(QSP_ARG_DECL RV_Inode *) )
{
	RV_Inode *inp;

//fprintf(stderr,"traverse_rv_inodes BEGIN:  QS_SERIAL = %d\n",QS_SERIAL);
//fflush(stderr);
	CHECK_VOLUME("traverse_rv_inodes")

//fprintf(stderr,"traverse_rv_inodes:  listing rv_inodes...\n");
//fflush(stderr);
//list_items( QSP_ARG  rv_inode_itp );
//fprintf(stderr,"traverse_rv_inodes:  DONE listing, looking up root '%s'...\n",ROOT_DIR_NAME);
//fflush(stderr);

	inp = rv_inode_of(QSP_ARG  ROOT_DIR_NAME);
	assert( inp != NULL );

	rv_pushd(QSP_ARG  ROOT_DIR_NAME);
	descend_directory(QSP_ARG  inp,func);
	rv_popd(SINGLE_QSP_ARG);
}

void rv_ls_all(SINGLE_QSP_ARG_DECL)
{
	CHECK_VOLUME("rv_ls_all")

	traverse_rv_inodes( QSP_ARG  rv_ls_inode );
}

static void traverse_list( QSP_ARG_DECL  List *lp, void (*func)(QSP_ARG_DECL  RV_Inode *) )
{
	Node *np;

	if( lp == NULL ) return;

	np=QLIST_HEAD(lp);
	while(np!=NULL ){
		(*func)(QSP_ARG  (RV_Inode *)np->n_data);
		np=np->n_next;
	}
}

void rv_ls_ctx(SINGLE_QSP_ARG_DECL)
{
	traverse_list( QSP_ARG  item_list(QSP_ARG  rv_inode_itp), rv_ls_inode );
}

static void sync_super(void)
{
	RV_Inode *inp;
	off64_t offset,off_ret;
	char *s_ptr;
	int i;

	/* flush strings */

	offset = BLOCK_SIZE * (off64_t)rv_sbp->rv_ndb;
	string_bytes_per_disk = block_size*rv_sbp->rv_nsb;
	s_ptr = rv_stp;
	for(i=0;i<rv_sbp->rv_ndisks;i++){
		off_ret = my_lseek64(rv_sbp->rv_fd[i],offset,SEEK_SET);
		if( BAD_OFFSET64(off_ret) ){
			perror("sync_super:  my_lseek64");
			sprintf(DEFAULT_ERROR_STRING,
				"sync_super:  error seeking to string table, disk %d (%s)",i,
					rv_sbp->rv_diskname[i]);
			NWARN(DEFAULT_ERROR_STRING);
			return;
		}
		if( write(rv_sbp->rv_fd[i],s_ptr,string_bytes_per_disk)
			!= (int) string_bytes_per_disk ){

			perror("write");
			sprintf(DEFAULT_ERROR_STRING,
				"sync_super:  error writing string blocks, disk %d",i);
			NWARN(DEFAULT_ERROR_STRING);
			return;
		}
		s_ptr += string_bytes_per_disk;
	}


	/* flush inodes */

	inode_bytes_per_disk = block_size*rv_sbp->rv_nib;
	inodes_per_disk = inode_bytes_per_disk / sizeof(RV_Inode);

	inp = rv_in_tbl;

	for(i=0;i<rv_sbp->rv_ndisks;i++){
		if( write(rv_sbp->rv_fd[i],inp,inode_bytes_per_disk)
			!= (int) inode_bytes_per_disk ){

			perror("write");
			sprintf(DEFAULT_ERROR_STRING,
				"sync_super:  error writing inode blocks, disk %d",i);
			NWARN(DEFAULT_ERROR_STRING);
			return;
		}
		inp += inodes_per_disk;
	}
}


void rv_sync(SINGLE_QSP_ARG_DECL)
{
	CHECK_VOLUME("rv_sync")

	/* sync_inode updates the disk image of the inode and also
	 * flushes directory data to disk.
	 */

	traverse_rv_inodes(QSP_ARG  sync_inode);

	sync_super();
}

void rv_mkfile(QSP_ARG_DECL  const char *pathname,long total_blocks,long blocks_per_write)
{
	int fd;
	uint32_t i;
	char *buf;
	mode_t mode;
//	int n_writes;

	mode = 0664;

#ifdef O_DIRECT
	fd = open(pathname,O_WRONLY|O_CREAT|O_DIRECT,mode);
#else // ! O_DIRECT
	fd = open(pathname,O_WRONLY|O_CREAT,mode);
#endif // ! O_DIRECT

	if( fd < 0 ){
		perror("open");
		sprintf(ERROR_STRING,"Can't open %s",pathname);
		WARN(ERROR_STRING);
		return;
	}
#ifdef SGI
	get_dio_params(fd);
#endif

	buf= (char *)mem_get(blocks_per_write*block_size);

	if( buf == (char *)NULL ){
		WARN("can't get block buffer");
		goto errorA;
	}

	for(i=0;i<block_size*blocks_per_write;i++) buf[i]=0;

//	n_writes = (total_blocks+blocks_per_write-1)/blocks_per_write;

	i=1;
	while( total_blocks > 0 ){
		int n_written,n_to_write;

		if( blocks_per_write > total_blocks )
			blocks_per_write = total_blocks;
		n_to_write = blocks_per_write * block_size;
		if( (n_written=write(fd,buf,n_to_write)) != n_to_write ){
			perror("write");
			sprintf(ERROR_STRING,
				"%d bytes actually written, %d requested, write %d",
				n_written,n_to_write,i);
			WARN(ERROR_STRING);
			goto errorB;
		}
		total_blocks -= blocks_per_write;
		i++;
	}


errorB:
	mem_release(buf);
errorA:
	close(fd);
	return;

}

#ifdef SGI
static void get_dio_params(int fd)
{
	struct dioattr da;

	/* query the block size and memory alignment */
	if(fcntl(fd, F_DIOINFO, &da) < 0) {
		perror("fcntl");
		WARN("error getting DIO params");
		return;
	}

	/*
	DiskBlockSize = da.d_miniosz;
	MemAlignment = da.d_mem;
	*/

	sprintf(DEFAULT_ERROR_STRING,"Max i/o size is %d bytes (%d blocks)",
		da.d_maxiosz,da.d_maxiosz/block_size);
	advise(DEFAULT_ERROR_STRING);
}
#endif

/* creat_rv_file returns the number of file descriptors, or -1
 */

int creat_rv_file(QSP_ARG_DECL  const char *filename,uint32_t size,int *fd_arr)
{
	RV_Inode *inp;

	inp = rv_inode_of(QSP_ARG  filename);
	if( inp != NULL ){
		sprintf(ERROR_STRING,"Deleting old version of file %s",filename);
		advise(ERROR_STRING);
		rv_rmfile(QSP_ARG  filename);
	}

	inp = rv_newfile(QSP_ARG  filename,size);
	if( inp == NULL ) return(-1);

	/* now queue up the file descriptors */

	return( queue_rv_file(QSP_ARG  inp,fd_arr) );
}

int rv_cd(QSP_ARG_DECL  const char *dirname)
{
	char *s;

	while( *dirname && (s=next_path_component(&dirname)) != NULL )
		if( rv_step_dir(QSP_ARG  s) < 0 ) return(-1);
	return(0);
}

static void make_link(QSP_ARG_DECL  const char *name,RV_Inode *inp)
{
	RV_Inode *new_inp;

	new_inp = rv_newfile(QSP_ARG  name,0);
	if( new_inp != NULL ){
		SET_RV_FLAG_BITS(new_inp, RVI_LINK);
		rv_in_tbl[RV_INODE_IDX(new_inp)].rvi_flags = RV_FLAGS(new_inp);
		new_inp->rvi_u.u_li.li_inp = inp;
		new_inp->rvi_u.u_li.li_ini = RV_INODE_IDX(inp);
	}
}

static void rv_mksubdir(QSP_ARG_DECL  const char *dirname)
{
	RV_Inode *inp;

	CHECK_VOLUME("rv_mksubdir")

	if( rv_sbp->rv_cwd != NULL ){	/* not root directory */
		Node *np;
		/* make sure that this directory does not exist already */
		np = QLIST_HEAD(rv_sbp->rv_cwd->rvi_lp);
		while( np != NULL ){
			inp=(RV_Inode *)np->n_data;
			if( !strcmp(dirname,RV_NAME(inp)) ){
				sprintf(ERROR_STRING,
	"Subdirectory %s already exists in directory %s",dirname,RV_NAME(rv_sbp->rv_cwd));
				WARN(ERROR_STRING);
				return;
			}
			np=np->n_next;
		}
	}
	inp = rv_newfile(QSP_ARG  dirname,1);
	if( inp == NULL ){
		sprintf(ERROR_STRING,"error creating rawvol subdirectory %s",dirname);
		WARN(ERROR_STRING);
		return;
	}
	RV_MODE(inp) |= DIRECTORY_BIT;			/* BUG use a symbolic constant here! */
	rv_in_tbl[RV_INODE_IDX(inp)].rvi_mode |= DIRECTORY_BIT;	/* make the change to the disk image too */
	inp->rvi_lp = new_list();

	if( rv_sbp->rv_cwd == NULL ){
		rv_sbp->rv_cwd = inp;
	}

	rv_pushd(QSP_ARG  dirname);

	make_link(QSP_ARG  ".",inp);
	if( RV_PARENT(inp) != NULL ){
		make_link(QSP_ARG  "..",RV_PARENT(inp));
	}

	rv_popd(SINGLE_QSP_ARG);
} /* end rv_mksubdir */

void rv_mkdir(QSP_ARG_DECL  const char *dirname)
{
	char *s;
	int pushed=0;

	while( *dirname && (s=next_path_component(&dirname)) != NULL ){
		if( *dirname == 0 ){	/* last component? */
			rv_mksubdir(QSP_ARG  s);
		} else {
			if( ! pushed ){
				if( rv_pushd(QSP_ARG  s) < 0 ) return;
				pushed++;
			} else {
				if( rv_step_dir(QSP_ARG  s) < 0 ){
					if( pushed ) rv_popd(SINGLE_QSP_ARG);
					return;
				}
			}
		}
	}
	if( pushed ) rv_popd(SINGLE_QSP_ARG);
}

int queue_rv_file(QSP_ARG_DECL  RV_Inode *inp,int *fd_arr)
{
	int i;
	/* what is the type of off64_t? */
	off64_t offset,retoff;
	int retval;

if( debug & rawvol_debug ){
sprintf(ERROR_STRING,"queueing rawvol disks for file %s, addr = %d (0x%x)",
RV_NAME(inp),RV_ADDR(inp),RV_ADDR(inp));
advise(ERROR_STRING);
}

	offset = (off64_t) RV_ADDR(inp) * (off64_t) rv_sbp->rv_blocksize;
	retval=rv_sbp->rv_ndisks;
	for(i=0;i<rv_sbp->rv_ndisks;i++){
		retoff = my_lseek64(rv_sbp->rv_fd[i],offset,SEEK_SET);
		if( retoff != offset ){
			/* BUG loff_t is not long long on 64 bit architecture!? */
			sprintf(ERROR_STRING,
		"queue_rv_file:  Error seeking on raw disk %d (%s), requested %lld but got %lld",
				i,rv_sbp->rv_diskname[i],(long long)offset,(long long)retoff);
			WARN(ERROR_STRING);
			retval=(-1);
		}
		fd_arr[i]=rv_sbp->rv_fd[i];
if( debug & rawvol_debug ){
sprintf(ERROR_STRING,"disk %d fd=%d seek to offset 0x%x / 0x%x",
i,fd_arr[i],(uint32_t)(offset>>32),(uint32_t)(offset&0xffffffff) /* offset is 64 bits!? */);
advise(ERROR_STRING);
}

	}
	return(retval);
}

int rv_frame_seek(QSP_ARG_DECL  RV_Inode *inp,uint32_t frame_index)
{
	off64_t os,offset,retoff, blks_per_frame;
	int retval=0;
	int disk_index;
	int i;
	uint32_t n_pix_bytes;

	assert( frame_index < SHP_FRAMES(RV_MOVIE_SHAPE(inp)) );

	/* BUG we need a set of utility routines to do these calcs... */

	/* Originally, we had each frame striped across all disks...
	 * Now we write an entire frame to each disk, then move on...
	 */

	n_pix_bytes =	SHP_COLS(RV_MOVIE_SHAPE(inp)) *
			SHP_ROWS(RV_MOVIE_SHAPE(inp)) *
			SHP_COMPS(RV_MOVIE_SHAPE(inp));

	blks_per_frame = (n_pix_bytes + RV_MOVIE_EXTRA(inp) + rv_sbp->rv_blocksize - 1 ) /
				rv_sbp->rv_blocksize;

#ifdef DEBUG
if( debug & rawvol_debug ){
sprintf(ERROR_STRING, "rv_frame_seek: file %s, frame %d, n_pix_bytes 0x%x, extra bytes = %d, blocksize = %d",
		RV_NAME(inp), frame_index, n_pix_bytes, RV_MOVIE_EXTRA(inp), rv_sbp->rv_blocksize);
advise(ERROR_STRING);
}
#endif

	disk_index = frame_index % rv_sbp->rv_ndisks;

	frame_index /= rv_sbp->rv_ndisks;	/* seek all disks to here */

	offset = (off64_t) RV_ADDR(inp);
	offset += blks_per_frame * frame_index;


	/* Don't seek all the disks - just the one containing the frame of interest! */
#ifdef SEEK_ALL_DISKS
	for(i=0;i<rv_sbp->rv_ndisks;i++){
#else
	i = disk_index;
#endif /* ! SEEK_ALL_DISKS */

		os = offset;
		if( i < disk_index ) os += blks_per_frame;

		os *= (off64_t) rv_sbp->rv_blocksize;

		retoff = my_lseek64(rv_sbp->rv_fd[i],os,SEEK_SET);

#ifdef DEBUG
if( debug & rawvol_debug ){
sprintf(ERROR_STRING, "rv_frame_seek: file %s, addr 0x%x, frame/ndisks %d, blks_per_frame = %d (extra bytes = %d)",
		RV_NAME(inp), RV_ADDR(inp), frame_index, (uint32_t)blks_per_frame, RV_MOVIE_EXTRA(inp));
advise(ERROR_STRING);
sprintf(ERROR_STRING, "rv_frame_seek: disk %d seeked to offset %lld (0x%llx)", i, (long long)os,(long long)os);
advise(ERROR_STRING);
}
#endif
		if( retoff != os ){
			sprintf(ERROR_STRING,
		"rv_frame_seek:  error seeking to file %s frame %d (disk %d, %s)",
				RV_NAME(inp),frame_index,i,rv_sbp->rv_diskname[i]);
			WARN(ERROR_STRING);
			retval = -1;
		}
#ifdef SEEK_ALL_DISKS
	}
#endif
	return(retval);
}

int rv_set_shape(QSP_ARG_DECL  const char *filename,Shape_Info *shpp)
{
	RV_Inode *inp;

	inp = get_rv_inode(QSP_ARG  filename);
	if( inp == NULL ) return(-1);

	COPY_SHAPE(RV_MOVIE_SHAPE(inp), shpp);

	set_shape_flags(RV_MOVIE_SHAPE(inp),NULL,AUTO_SHAPE);

	// also update the area that will be written to disk...
	COPY_DIMS( &RV_MOVIE_DIMS(inp), SHP_TYPE_DIMS( RV_MOVIE_SHAPE(inp) ) );
	COPY_INCS( &RV_MOVIE_INCS(inp), SHP_TYPE_INCS( RV_MOVIE_SHAPE(inp) ) );

	SET_RV_MOVIE_PREC_CODE(inp,SHP_PREC(RV_MOVIE_SHAPE(inp)));

	return(0);
}

void set_use_osync(int flag)
{
	use_osync=flag;
}

void rawvol_info(SINGLE_QSP_ARG_DECL)
{
	int i;

	CHECK_VOLUME("rawvol_info")

	sprintf(msg_str,"%d disks, block size = %d:",rv_sbp->rv_ndisks,BLOCK_SIZE);
	prt_msg(msg_str);
	sprintf(msg_str,"\t%d (0x%x) inode blocks",
		rv_sbp->rv_nib, rv_sbp->rv_nib);
	prt_msg(msg_str);
	sprintf(msg_str,"\t%d (0x%x) string blocks",
		rv_sbp->rv_nsb, rv_sbp->rv_nsb);
	prt_msg(msg_str);
	sprintf(msg_str,"\t%ld (0x%lx) data blocks",
		(u_long)rv_sbp->rv_ndb, (u_long)rv_sbp->rv_ndb);
	prt_msg(msg_str);

	for(i=0;i<rv_sbp->rv_ndisks;i++){
		sprintf(msg_str,"\t%s\t%ld (0x%lx) total blocks",
			rv_sbp->rv_diskname[i],
			(u_long)rv_sbp->rv_nblks[i],
			(u_long)rv_sbp->rv_nblks[i]);
		prt_msg(msg_str);
	}
	prt_msg("string freelist:");
	showmap( &rv_st_freelist );
	prt_msg("inode freelist:");
	showmap( &rv_inode_freelist );
	prt_msg("data freelist:");
	showmap( &rv_data_freelist );

	show_root_users(SINGLE_QSP_ARG);
}

void rawvol_get_usage(SINGLE_QSP_ARG_DECL)
{
	int freespace=0, total=0, percent=0;
	FreeBlk *fbp;

	CHECK_VOLUME("rawvol_get_usage")

	sprintf(msg_str,"%d disks, block size = %d:",rv_sbp->rv_ndisks,BLOCK_SIZE);
	/* ASH - the freelist represents free space on ONE DISK only!
	 * (they should all have the same amount of free space per disk)
	 * because allocation is done in units of blocks per disk.
	 */

	freespace = 0;
	fbp = rv_data_freelist.fl_blockp;

	while( fbp->size != 0 ){
		freespace += fbp->size;
		fbp++;
	}
	freespace *= rv_sbp->rv_ndisks;

	total = rv_sbp->rv_ndisks * rv_sbp->rv_ndb;

	percent = (100.0 * (float) freespace) / (float) total;
	sprintf(msg_str,"%d free / %d total (%d %%free)", freespace, total, percent);
	prt_msg(msg_str);
}

/* We do this when we need to resize a file...
 *
 * the size argument should be the number of blocks PER DISK
 * NO NO NO
 * the size arg is the total number of blocks!
 */

int rv_realloc(QSP_ARG_DECL  const char *name,uint32_t size)
{
	RV_Inode *inp;

	inp = rv_inode_of(QSP_ARG  name);
	if( inp == NULL ){
		sprintf(ERROR_STRING,"rv_realloc:  no file %s",name);
		WARN(ERROR_STRING);
		return(-1);
	}

	size = BLOCKS_PER_DISK(size);

	if( size == RV_N_BLOCKS(inp) ){
		if( verbose ){
			sprintf(ERROR_STRING,
		"rv_realloc:  RV file %s already has the requested number of blocks per disk (%d)",
				RV_NAME(inp),RV_N_BLOCKS(inp));
			advise(ERROR_STRING);
		}
		return(0);
	}

	givspace(&rv_data_freelist,RV_N_BLOCKS(inp),RV_ADDR(inp));
	RV_ADDR(inp) = getspace(&rv_data_freelist,size);
	if( RV_ADDR(inp) < 0 ){
		RV_N_BLOCKS(inp) = 0;
		sprintf(ERROR_STRING,"Error reallocating %d blocks per disk for file %s",
			size,RV_NAME(inp));
		WARN(ERROR_STRING);
		return(-1);
	}

	RV_N_BLOCKS(inp) = size;

	return(0);
}

/* We call this after we halt prematurely during a record */

int rv_truncate(RV_Inode *inp,uint32_t new_size)
{
	new_size = BLOCKS_PER_DISK(new_size);
	if( new_size >= RV_N_BLOCKS(inp) ){
		sprintf(DEFAULT_ERROR_STRING,
			"RV file %s has size %d, can't truncate to size %d!?",
			RV_NAME(inp),RV_N_BLOCKS(inp),new_size);
		NWARN(DEFAULT_ERROR_STRING);
		return(-1);
	}
	givspace(&rv_data_freelist, RV_N_BLOCKS(inp) - new_size, RV_ADDR(inp) + new_size);
	RV_N_BLOCKS(inp) = new_size;

	return(0);
}

/* this stuff was put in for debugging, because dd was too slow */

static void print_block(QSP_ARG_DECL  u_char *s)
{
	int addr;
	int i;

	/* now do od -c, 16 bytes per line */
	addr=0;
	for(i=0;i<(BLOCK_SIZE/16);i++){
		sprintf(ERROR_STRING,
"0%-4o %3o %3o %3o %3o %3o %3o %3o %3o %3o %3o %3o %3o %3o %3o %3o %3o",
			addr,
			s[0],s[1],s[2], s[3], s[4], s[5], s[6], s[7],
			s[8],s[9],s[10],s[11],s[12],s[13],s[14],s[15]);
		advise(ERROR_STRING);
		s+=16;
		addr+=16;
	}
}

static u_char blockbuf[BLOCK_SIZE];

void dump_block(QSP_ARG_DECL  int i,uint32_t block)
{
	off64_t offset;

	offset = BLOCK_SIZE * ((off64_t)block);

	/* if( my_lseek64(rv_sbp->rv_fd[i],offset,SEEK_SET) < 0 ) */
	if( my_lseek64(rv_sbp->rv_fd[i],offset,SEEK_SET) & 0x80000000 )
								{
		perror("dump_block:  my_lseek64");
		NWARN("dump_block:  error seeking");
		return;
	}
	if( read(rv_sbp->rv_fd[i],blockbuf,BLOCK_SIZE) != BLOCK_SIZE ){
		perror("read");
		NWARN("error reading block");
		return;
	}
	print_block(QSP_ARG  blockbuf);
}

int rv_get_ndisks(void)
{
	if(rv_sbp == NULL)
	return -1;
	else
		return rv_sbp->rv_ndisks;
}

void rv_pwd(SINGLE_QSP_ARG_DECL)
{
	CHECK_VOLUME("rv_pwd")

	assert( rv_sbp->rv_cwd != NULL );

	sprintf(msg_str,"current working directory is %s",rv_pathname );
	prt_msg(msg_str);
}

#define MAX_EXTRA	32 /* sizeof(struct timeval) */

void rv_set_extra(int n_extra)
{
	if( n_extra < 0 && n_extra > MAX_EXTRA ){
		sprintf(DEFAULT_ERROR_STRING,"rv_set_extra:  argument (%d) should be greater than 0 and less than or equal to %d",
				n_extra,MAX_EXTRA);
		NWARN(DEFAULT_ERROR_STRING);
		return;
	}
	n_extra_bytes = n_extra;
}

int legal_rv_filename(const char *name)
{
	const char *s;

	s=name;
	while(*s){
		if( *s == '/' ) return(0);
		s++;
	}
	return(1);
}

int32_t n_rv_disks(void)
{
	assert( rv_sbp != NULL );

	return rv_sbp->rv_ndisks;
}


#endif // HAVE_RAWVOL
