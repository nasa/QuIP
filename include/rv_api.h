

#ifndef _RV_API_H_
#define _RV_API_H_

#include "quip_config.h"
#include "items.h"
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

	uint32_t	rv_flags;
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
	Shape_Info	mi_shape;
	RV_Super *	mi_sbp;			/* BUG shouldn't be here */
						/* then why is it??? */
#define N_RV_FRAMEINFOS	4			/* drops, plus 3 HW error types */
	Frame_Info	mi_fi[N_RV_FRAMEINFOS];
	int32_t		mi_extra_bytes;		/* to account for timestamp at end of frame... */
} RV_Movie_Info;

#define rvi_shape		rvi_u.u_mi.mi_shape
#define rvi_extra_bytes		rvi_u.u_mi.mi_extra_bytes
#define rvi_sbp			rvi_u.u_mi.mi_sbp


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
		RV_Movie_Info 	u_mi;	/* movie info for movies */
		RV_Dir_Info	u_di;
		RV_Link_Info	u_li;
	} rvi_u;

	struct rv_inode *	rvi_inp;	/* ptr to working mem struct, for disk inode */
	struct rv_inode *	rvi_dotdot;	/* ptr to parent directory */
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
extern int insure_default_rv(SINGLE_QSP_ARG_DECL);
extern int	rv_realloc(QSP_ARG_DECL  const char *,dimension_t);
extern int	rv_get_ndisks();
extern void	rv_sync(SINGLE_QSP_ARG_DECL);
extern int	queue_rv_file(RV_Inode *,int *);
extern int	rv_frame_seek(RV_Inode *,dimension_t);
extern int	remember_frame_info(RV_Inode *inp, int index,
				USHORT_ARG nerr, dimension_t *frames);
extern void setup_rv_iofile(QSP_ARG_DECL  RV_Inode *inp);




#endif /* undef _RV_API_H_ */

