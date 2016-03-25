

#ifndef _RV_API_H_
#define _RV_API_H_

#include "quip_config.h"
#include "item_type.h"
#include "data_obj.h"

#define MAX_RV_SUPER_USERS	4
#define BLOCK_SIZE		1024
#define MAX_DISKS		8

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

	uint32_t	rv_nib;			/* number of "inode" blocks */
	uint32_t	rv_nsb;			/* number of "string" blocks */
	uint64_t	rv_ndb;			/* number of data blocks */

	//uint32_t	rv_flags;		// are the flags used???
	int 		rv_fd[MAX_DISKS];
	char		rv_diskname[MAX_DISKS][MAX_DISKNAME_LEN];
	int		rv_n_super_users;
	uid_t		rv_root_uid[MAX_RV_SUPER_USERS];
	struct rv_inode *
			rv_cwd;
} RV_Super;


/* This structure is used to save info about dropped/corrupted frames.
 * (see the meteor driver)
 */

typedef struct frame_info {
	u_short		fi_nsaved;
	uint32_t	fi_savei;
} Frame_Info;


typedef struct rv_link_info {
	struct rv_inode *	li_inp;
	uint32_t		li_ini;
} RV_Link_Info;


typedef struct rv_dir_info {
	List *			di_lp;	/* list of entries for directory */
	struct rv_inode *	di_parent;
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

	//RV_Super *	mi_sbp;			/* BUG shouldn't be here */
						/* then why is it??? */
#define N_RV_FRAMEINFOS	4			/* drops, plus 3 HW error types */
	Frame_Info	mi_fi[N_RV_FRAMEINFOS];
	int32_t		mi_extra_bytes;		/* to account for timestamp at end of frame... */
} RV_Movie_Info;

//#define rvi_shpp		rvi_u.u_mi.mi_shpp
//#define rvi_extra_bytes		rvi_u.u_mi.mi_extra_bytes
//#define rvi_sbp			rvi_u.u_mi.mi_sbp
//#define rvi_precision		rvi_u.u_mi.mi_prec
//#define rvi_dimset		rvi_u.u_mi.mi_dimset
//#define rvi_incset		rvi_u.u_mi.mi_incset

typedef struct rv_inode {
	char *		rvi_name;	/* pointer to the heap, item mgmt */

	/* These addresses were u_long, but then that masks the error
	 * return from getspace...  do we need the extra bit?
	 */

	int32_t		rvi_nmi;	/* on disk, an offset into
					 * the string area for the name */
	int32_t		rvi_ini;
	int32_t		rvi_addr;	/* address of first block for
					 * plain file */

	uint32_t	rvi_len;	/* number of blocks */
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

	struct rv_inode *	rvi_inp;	/* ptr to working mem struct, for disk inode */
	struct rv_inode *	rvi_parent;	/* ptr to parent directory */

	// We only need the pad if using O_DIRECT...
	// size above is 0xb0, we need a pad of 0x50 to round up to 256 bytes
	char			pad_bytes[80];
} RV_Inode;

/* We use the mode bits from stat(2) */
#define DIRECTORY_BIT			040000
#define IS_DIRECTORY(inp)		((inp)->rvi_mode&DIRECTORY_BIT)

#define NO_INODE ((RV_Inode *)NULL)

/* inode flags */
#define RVI_INUSE	1
#define RVI_SCANNED	2
#define RVI_LINK	4

#define RV_INUSE(inp)		(( inp )->rvi_flags & RVI_INUSE)
#define RV_SCANNED(inp)		(( inp )->rvi_flags & RVI_SCANNED)
#define IS_LINK(inp)		(( inp )->rvi_flags & RVI_LINK)
#define IS_REGULAR_FILE(inp)	( !IS_LINK(inp) && ! IS_DIRECTORY(inp) )

#define RV_NAME(inp)		((inp)->rvi_name)
#define SET_RV_NAME(inp,s)	((inp)->rvi_name) = s
#define RV_INODE(inp)		((inp)->rvi_inp)
#define SET_RV_INODE(inp,t)	((inp)->rvi_inp) = t
//#define RV_SHAPE(inp)		((inp)->rvi_shpp)
//#define SET_RV_SHAPE(inp,shpp)	((inp)->rvi_shpp) = shpp
#define RV_MODE(inp)		((inp)->rvi_mode)
#define SET_RV_MODE(inp,m)	((inp)->rvi_mode) = m
#define RV_UID(inp)		((inp)->rvi_uid)
#define SET_RV_UID(inp,id)	((inp)->rvi_uid) = id
#define RV_GID(inp)		((inp)->rvi_gid)
#define SET_RV_GID(inp,id)	((inp)->rvi_gid) = id
#define RV_INODE_IDX(inp)	((inp)->rvi_ini)
#define SET_RV_INODE_IDX(inp,i)	((inp)->rvi_ini) = i
#define RV_NAME_IDX(inp)	((inp)->rvi_nmi)
#define SET_RV_NAME_IDX(inp,i)	((inp)->rvi_nmi) = i
#define RV_ADDR(inp)		((inp)->rvi_addr)
#define SET_RV_ADDR(inp,a)	((inp)->rvi_addr) = a
#define RV_FLAGS(inp)		((inp)->rvi_flags)
#define SET_RV_FLAGS(inp,f)	((inp)->rvi_flags) = f
#define SET_RV_FLAG_BITS(inp,b)	((inp)->rvi_flags) |= b
#define CLEAR_RV_FLAG_BITS(inp,b)	((inp)->rvi_flags) &= ~(b)
#define RV_N_BLOCKS(inp)		((inp)->rvi_len)
#define SET_RV_N_BLOCKS(inp,n)	((inp)->rvi_len) = n
#define RV_PARENT(inp)		((inp)->rvi_parent)
#define SET_RV_PARENT(inp,p)	((inp)->rvi_parent) = p
#define RV_MTIME(inp)		((inp)->rvi_mtime)
#define RV_ATIME(inp)		((inp)->rvi_atime)
#define RV_CTIME(inp)		((inp)->rvi_ctime)
#define SET_RV_MTIME(inp,t)	((inp)->rvi_mtime) = t
#define SET_RV_ATIME(inp,t)	((inp)->rvi_atime) = t
#define SET_RV_CTIME(inp,t)	((inp)->rvi_ctime) = t


// macros for Directories
#define RV_DIR_ENTRIES(inp)		((inp)->rvi_u.u_di.di_lp)
#define SET_RV_DIR_ENTRIES(inp,lp)	((inp)->rvi_u.u_di.di_lp) = lp

// macros for links
#define RV_LINK_INODE_PTR(inp)		((inp)->rvi_u.u_li.li_inp)
#define SET_RV_LINK_INODE_PTR(inp,v)	((inp)->rvi_u.u_li.li_inp) = v
#define RV_LINK_INODE_IDX(inp)		((inp)->rvi_u.u_li.li_ini)
#define SET_RV_LINK_INODE_IDX(inp,v)	((inp)->rvi_u.u_li.li_ini) = v

// macros for movies

#define RV_MOVIE_SHAPE(inp)		((inp)->rvi_u.u_mi.mi_shpp)
#define SET_RV_MOVIE_SHAPE(inp,shpp)	((inp)->rvi_u.u_mi.mi_shpp) = shpp
#define RV_MOVIE_DIMS(inp)		((inp)->rvi_u.u_mi.mi_dimset)
#define SET_RV_MOVIE_DIMS(inp,d)	((inp)->rvi_u.u_mi.mi_dimset) = d
#define RV_MOVIE_INCS(inp)		((inp)->rvi_u.u_mi.mi_incset)
#define SET_RV_MOVIE_INCS(inp,v)	((inp)->rvi_u.u_mi.mi_incset) = v
#define RV_MOVIE_PREC_CODE(inp)		((inp)->rvi_u.u_mi.mi_prec_code)
#define SET_RV_MOVIE_PREC_CODE(inp,v)	((inp)->rvi_u.u_mi.mi_prec_code) = v
#define RV_MOVIE_FRAME_INFO(inp,idx)	((inp)->rvi_u.u_mi.mi_fi[idx])
#define SET_RV_MOVIE_FRAME_INFO(inp,idx,v)	((inp)->rvi_u.u_mi.mi_fi[idx]) = v
#define RV_MOVIE_EXTRA(inp)	((inp)->rvi_u.u_mi.mi_extra_bytes)
#define SET_RV_MOVIE_EXTRA(inp,v)	((inp)->rvi_u.u_mi.mi_extra_bytes) = v

//#define rvi_shpp		rvi_u.u_mi.mi_shpp
// Why directory parent???

/* Block size is 1024 under linux...  maybe we should try
 * to determine this some other way.
 */

/* We allocate the same number of frames on all disks;
 * divide down and round up.
 */

#define FRAMES_TO_ALLOCATE( nf, nd )	( ( ( (nf) + (nd) - 1 ) / (nd) ) * (nd) )


/* Public prototypes */

/* rawvol.c */
ITEM_INTERFACE_PROTOTYPES(RV_Inode,rv_inode)
extern int	rv_truncate(RV_Inode *,dimension_t);
extern int	rv_set_shape(QSP_ARG_DECL  const char *filename, Shape_Info *shpp);
extern void	rv_info(QSP_ARG_DECL  RV_Inode *);
extern int	creat_rv_file(QSP_ARG_DECL  const char *,dimension_t,int *);
extern int	rv_rmfile(QSP_ARG_DECL  const char *name);
extern int	legal_rv_filename(const char *);
extern void	traverse_rv_inodes( QSP_ARG_DECL  void (*f)(QSP_ARG_DECL  RV_Inode *) );
extern int	insure_default_rv(SINGLE_QSP_ARG_DECL);
extern int	rv_realloc(QSP_ARG_DECL  const char *,dimension_t);
extern int	rv_get_ndisks();
extern void	rv_sync(SINGLE_QSP_ARG_DECL);
extern int	queue_rv_file(QSP_ARG_DECL  RV_Inode *,int *);
extern int	rv_frame_seek(QSP_ARG_DECL  RV_Inode *,dimension_t);
extern int	remember_frame_info(RV_Inode *inp, int index,
				USHORT_ARG nerr, dimension_t *frames);
extern void	setup_rv_iofile(QSP_ARG_DECL  RV_Inode *inp);
extern int32_t	n_rv_disks(void);
extern int	rv_access_allowed(QSP_ARG_DECL  RV_Inode *inp);




#endif /* undef _RV_API_H_ */

