
#ifndef _RAWVOL_H_
#define _RAWVOL_H_

// This is the private stuff for the module...

#include "quip_config.h"

#ifdef INC_VERSION
char VersionId_inc_rawvol[] = QUIP_VERSION_STRING;
#endif /* INC_VERSION */


#ifdef HAVE_SYS_TYPES_H
#include <sys/types.h>
#endif

#ifdef HAVE_SYS_STAT_H
#include <sys/stat.h>
#endif

#include "node.h"

/* image shape information is not really properly part
 * of the raw file info, but since most of our use of raw
 * volumes are to be used for video clips, we will go ahead
 * and put the shape info in the inode.
 * This way we won't need to write the header info in the
 * data blocks.
 */

#include "data_obj.h"
#include "rv_api.h"
#include "item_type.h"

#define MAX_RV_SUPER_USERS	4

/* this was 16 when we were contend to have things like /dev/sdd1
 * but for testing it is sometimes convenient to fake it with a longer pathname...
 */
#define MAX_DISKNAME_LEN	80	/* /dev/sdd1 is typical... */


// These only need to be 32 bits, but they used to be long
// and now we have a disk with the old setup...

typedef struct rv_super {
	int64_t		rv_magic;		/* so we can be a little bit sure */
	int32_t		rv_blocksize;
	int32_t		rv_ndisks;
	uint64_t	rv_nblks[MAX_DISKS];	/* total device blocks */

						/* the following are all per-disk: */
						/* BUG only disk[0] is used */

	uint32_t	rv_n_inode_blocks;			/* number of "inode" blocks */
	uint32_t	rv_n_string_blocks;			/* number of "string" blocks */
	uint64_t	rv_n_data_blocks;			/* number of data blocks */

	//uint32_t	rv_flags;		// are the flags used???
	int 		rv_fd[MAX_DISKS];
	char		rv_diskname[MAX_DISKS][MAX_DISKNAME_LEN];
	int		rv_n_super_users;
	uid_t		rv_root_uid[MAX_RV_SUPER_USERS];
	struct rv_inode *	rv_cwd;
} RV_Super;


/* This structure is used to save info about dropped/corrupted frames.
 * (see the meteor driver)
 */

typedef struct frame_info {
	u_short		fi_nsaved;
	uint32_t	fi_str_idx;	// index into string table
} Frame_Info;

#define FRAME_INFO_N_SAVED(fi_p)	(fi_p)->fi_nsaved
#define SET_FRAME_INFO_N_SAVED(fi_p,v)	(fi_p)->fi_nsaved = v
#define FRAME_INFO_STR_IDX(fi_p)	(fi_p)->fi_str_idx
#define SET_FRAME_INFO_STR_IDX(fi_p,v)	(fi_p)->fi_str_idx = v

typedef struct rv_link_info {
	struct rv_inode *	li_inp;
	uint32_t		li_ini;
} RV_Link_Info;


typedef struct rv_dir_info {
	List *			di_children;	/* list of entries for directory */
	struct rv_inode *	di_parent;
	// should have its own namespace, so different folders
	// can have files with common names???
} RV_Dir_Info;



typedef struct rv_movie_info {
	/* an extension so we don't need image headers */
	Shape_Info *	mi_shpp;		// dynamically allocated in memory,
						// not saved to disk...

	// These are redundant with the contents of Shape_Info, but 
	// that doesn't get stored to disk.  Here we keep the essential
	// information.

	Dimension_Set	mi_dimset;		// overload for type_dim and mach_dim
	Increment_Set	mi_incset;		// overload for type_incrs and mach_incrs
	prec_t		mi_prec_code;		// how to handle this?

#define N_RV_FRAMEINFOS	4			/* drops, plus 3 HW error types */
	Frame_Info	mi_frame_info[N_RV_FRAMEINFOS];
	int32_t		mi_extra_bytes;		/* to account for timestamp at end of frame... */
} RV_Movie_Info;

struct rv_inode_data {
	char *		rvi_name;	/* pointer to the heap, item mgmt */

	struct rv_inode *	rvi_inp;	/* ptr to working mem struct, for disk inode */
	struct rv_inode *	rvi_parent;	/* ptr to parent directory */

	/* These addresses were u_long, but then that masks the error
	 * return from getspace...  do we need the extra bit?
	 */

	int32_t		rvi_name_idx;	/* on disk, an offset into
					 * the string area for the name */
	int32_t		rvi_inode_idx;
	int32_t		rvi_addr;	/* address of first block for
					 * plain file */

	uint32_t	rvi_n_blocks;	/* number of blocks (per-disk or total???) */
	int32_t		rvi_flags;

	/* these are patterned on stat(2) */
	mode_t		rvi_mode;
	uid_t		rvi_uid;
	gid_t		rvi_gid;
	time_t		rvi_atime;	/* time of last access */
	time_t		rvi_mtime;	/* time of last modification */
	time_t		rvi_ctime;	/* time of last status change */

	union {
		// BUG Movie_Info is a big struct, should be a pointer...
		// But it needs to be on-disk...
		RV_Movie_Info 	u_mi;	/* movie info for movies */
		RV_Dir_Info	u_di;
		RV_Link_Info	u_li;
	} rvi_u;
};

// on 64 bit machine with pad of 112, size was 288...
// So pad should be 112-(288-256) = 112 - 32 = 80
//
struct rv_inode {
	struct rv_inode_data	rvi_inode;
	// We only need the pad if using O_DIRECT...
	// size above is 0xb0, we need a pad of 0x50 to round up to 256 bytes
	//
	// on 32bit machine, size is 144 bytes?
#define N_RV_INODE_PAD_BYTES	80

	char			pad_bytes[N_RV_INODE_PAD_BYTES];
};

/* We use the mode bits from stat(2) */
#define DIRECTORY_BIT			040000
#define IS_DIRECTORY(inp)	((inp)->rvi_inode.rvi_mode&DIRECTORY_BIT)

// On disk, the inodes are a big array, but in memory they are dynamic items

/* disk inode flags */
#define RVI_INUSE	1
#define RVI_SCANNED	2

// general inode flags
#define RVI_LINK	4

#define RV_INUSE(inp)		(( inp )->rvi_inode.rvi_flags & RVI_INUSE)
#define RV_SCANNED(inp)		(( inp )->rvi_inode.rvi_flags & RVI_SCANNED)
#define IS_LINK(inp)		(( inp )->rvi_inode.rvi_flags & RVI_LINK)
#define IS_REGULAR_FILE(inp)	( !IS_LINK(inp) && ! IS_DIRECTORY(inp) )

#define RV_NAME(inp)		((inp)->rvi_inode.rvi_name)
#define SET_RV_NAME(inp,s)	((inp)->rvi_inode.rvi_name) = s
#define RV_INODE(inp)		((inp)->rvi_inode.rvi_inp)
#define SET_RV_INODE(inp,t)	((inp)->rvi_inode.rvi_inp) = t
//#define RV_SHAPE(inp)		((inp)->rvi_inode.rvi_shpp)
//#define SET_RV_SHAPE(inp,shpp)	((inp)->rvi_inode.rvi_shpp) = shpp
#define RV_MODE(inp)		((inp)->rvi_inode.rvi_mode)
#define SET_RV_MODE(inp,m)	((inp)->rvi_inode.rvi_mode) = m
#define RV_UID(inp)		((inp)->rvi_inode.rvi_uid)
#define SET_RV_UID(inp,id)	((inp)->rvi_inode.rvi_uid) = id
#define RV_GID(inp)		((inp)->rvi_inode.rvi_gid)
#define SET_RV_GID(inp,id)	((inp)->rvi_inode.rvi_gid) = id
#define RV_INODE_IDX(inp)	((inp)->rvi_inode.rvi_inode_idx)
#define SET_RV_INODE_IDX(inp,i)	((inp)->rvi_inode.rvi_inode_idx) = i
#define RV_NAME_IDX(inp)	((inp)->rvi_inode.rvi_name_idx)
#define SET_RV_NAME_IDX(inp,i)	((inp)->rvi_inode.rvi_name_idx) = i
#define RV_ADDR(inp)		((inp)->rvi_inode.rvi_addr)
#define SET_RV_ADDR(inp,a)	((inp)->rvi_inode.rvi_addr) = a
#define RV_FLAGS(inp)		((inp)->rvi_inode.rvi_flags)
#define SET_RV_FLAGS(inp,f)	((inp)->rvi_inode.rvi_flags) = f
#define SET_RV_FLAG_BITS(inp,b)	((inp)->rvi_inode.rvi_flags) |= (b)
#define CLR_RV_FLAG_BITS(inp,b)	((inp)->rvi_inode.rvi_flags) &= ~(b)
#define RV_N_BLOCKS(inp)	((inp)->rvi_inode.rvi_n_blocks)
#define SET_RV_N_BLOCKS(inp,n)	((inp)->rvi_inode.rvi_n_blocks) = n
#define RV_PARENT(inp)		((inp)->rvi_inode.rvi_parent)
#define SET_RV_PARENT(inp,p)	((inp)->rvi_inode.rvi_parent) = p
#define RV_MTIME(inp)		((inp)->rvi_inode.rvi_mtime)
#define RV_ATIME(inp)		((inp)->rvi_inode.rvi_atime)
#define RV_CTIME(inp)		((inp)->rvi_inode.rvi_ctime)
#define SET_RV_MTIME(inp,t)	((inp)->rvi_inode.rvi_mtime) = t
#define SET_RV_ATIME(inp,t)	((inp)->rvi_inode.rvi_atime) = t
#define SET_RV_CTIME(inp,t)	((inp)->rvi_inode.rvi_ctime) = t

// macros for Directories
#define rvi_children		rvi_inode.rvi_u.u_di.di_children
#define RV_CHILDREN(inp)	((inp)->rvi_inode.rvi_u.u_di.di_children)

#define SET_RV_CHILDREN(inp,lp)	((inp)->rvi_inode.rvi_u.u_di.di_children) = lp

// macros for links
#define RV_LINK_INODE_PTR(inp)		((inp)->rvi_inode.rvi_u.u_li.li_inp)
#define SET_RV_LINK_INODE_PTR(inp,v)	((inp)->rvi_inode.rvi_u.u_li.li_inp) = v
#define RV_LINK_INODE_IDX(inp)		((inp)->rvi_inode.rvi_u.u_li.li_ini)
#define SET_RV_LINK_INODE_IDX(inp,v)	((inp)->rvi_inode.rvi_u.u_li.li_ini) = v

// macros for movies

#define RV_MOVIE_SHAPE(inp)		((inp)->rvi_inode.rvi_u.u_mi.mi_shpp)
#define SET_RV_MOVIE_SHAPE(inp,shpp)	((inp)->rvi_inode.rvi_u.u_mi.mi_shpp) = shpp
#define RV_MOVIE_DIMS(inp)		((inp)->rvi_inode.rvi_u.u_mi.mi_dimset)
#define SET_RV_MOVIE_DIMS(inp,d)	((inp)->rvi_inode.rvi_u.u_mi.mi_dimset) = d
#define RV_MOVIE_INCS(inp)		((inp)->rvi_inode.rvi_u.u_mi.mi_incset)
#define SET_RV_MOVIE_INCS(inp,v)	((inp)->rvi_inode.rvi_u.u_mi.mi_incset) = v
#define RV_MOVIE_PREC_CODE(inp)		((inp)->rvi_inode.rvi_u.u_mi.mi_prec_code)
#define SET_RV_MOVIE_PREC_CODE(inp,v)	((inp)->rvi_inode.rvi_u.u_mi.mi_prec_code) = v
#define RV_MOVIE_FRAME_INFO(inp,idx)	((inp)->rvi_inode.rvi_u.u_mi.mi_frame_info[idx])
#define SET_RV_MOVIE_FRAME_INFO(inp,idx,v)	((inp)->rvi_inode.rvi_u.u_mi.mi_frame_info[idx]) = v
#define RV_MOVIE_EXTRA(inp)	((inp)->rvi_inode.rvi_u.u_mi.mi_extra_bytes)
#define SET_RV_MOVIE_EXTRA(inp,v)	((inp)->rvi_inode.rvi_u.u_mi.mi_extra_bytes) = v

//#define rvi_shpp		rvi_inode.rvi_u.u_mi.mi_shpp
// Why directory parent???

/* Block size is 1024 under linux...  maybe we should try
 * to determine this some other way.
 */

/* We allocate the same number of frames on all disks;
 * divide down and round up.
 */

#define FRAMES_TO_ALLOCATE( nf, nd )	( ( ( (nf) + (nd) - 1 ) / (nd) ) * (nd) )

/* We name the volume by the first disk (given to the mkfs cmd) */

/* The data blocks are striped... with the housekeeping data,
 * we have a choice:  we can either keep it on one disk or the other,
 * or we can mirror it...
 */

#define RV_MAGIC	0x1fabcdef
#define RV_MAGIC2	0x12abcdef	/* version 2 allows directories */

#define NO_SUPER ((RV_Super *)NULL)
#define N_INODE_PAD	36

#define rvi_frame_info		rvi_inode.rvi_u.u_mi.mi_frame_info

#include "off64_t.h"

/* globals */
extern int rawvol_debug;

/* prototypes */

/* rawvol.c */

extern int rv_is_open(void);
extern void perform_write_test(QSP_ARG_DECL  int i_block, int n_blocks, int n_reps );
extern int legal_rv_filename(const char *);
extern void rv_mkdir(QSP_ARG_DECL  const char *dirname);
extern int rv_cd(QSP_ARG_DECL  const char *dirname);
extern void rv_pwd(SINGLE_QSP_ARG_DECL);
extern int grant_root_access(QSP_ARG_DECL  const char *user_name);
extern void xfer_frame_info(dimension_t *lp,int index,RV_Inode *inp);
extern void dump_block(QSP_ARG_DECL  int i,dimension_t block);

extern void read_rv_super(QSP_ARG_DECL  const char *vol_name);
extern RV_Inode * rv_newfile(QSP_ARG_DECL  const char *name,dimension_t size);
extern void rv_lsfile(QSP_ARG_DECL  const char *name);
extern void rv_ls_inode(QSP_ARG_DECL  RV_Inode *inp);
extern void rv_ls_all(SINGLE_QSP_ARG_DECL);
extern void rv_ls_ctx(SINGLE_QSP_ARG_DECL);
extern void rv_ls_cwd(SINGLE_QSP_ARG_DECL);
extern void rv_rm_cwd(SINGLE_QSP_ARG_DECL);
extern void rv_mkfs(QSP_ARG_DECL  int ndisks, const char **disknames,uint32_t nib,uint32_t nsb);
extern void rv_close(SINGLE_QSP_ARG_DECL);
extern int rv_get_ndisks(void);
extern void rv_chomd(RV_Inode *,int);

extern List *rv_inode_list(SINGLE_QSP_ARG_DECL);
extern void rv_chmod(QSP_ARG_DECL  RV_Inode *,int);
extern void rv_mkfile(QSP_ARG_DECL  const char *s,long total_blocks,long n_per_write);

extern void _rv_set_extra(QSP_ARG_DECL  int n);

#define rv_set_extra(n) _rv_set_extra(QSP_ARG  n)

ITEM_INTERFACE_PROTOTYPES(RV_Inode,rv_inode)
#define pick_rv_inode(p)	_pick_rv_inode(QSP_ARG  p)
#define get_rv_inode(s)		_get_rv_inode(QSP_ARG  s)
#define rv_inode_of(s)		_rv_inode_of(QSP_ARG  s)
#define new_rv_inode(s)		_new_rv_inode(QSP_ARG  s)

extern void set_use_osync(int flag);

extern void	rawvol_info(SINGLE_QSP_ARG_DECL);
extern void     rawvol_get_usage(SINGLE_QSP_ARG_DECL);

// moved to rv_api.h
//extern int	_rv_truncate(QSP_ARG_DECL  RV_Inode *,dimension_t);
//#define rv_truncate(inp,d) _rv_truncate(QSP_ARG  inp,d)

#endif /* undef _RAWVOL_H_ */

