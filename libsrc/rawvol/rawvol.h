
#ifndef _RAWVOL_H_
#define _RAWVOL_H_

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

/* #define BLOCK_SIZE	512L */


#define FRAMES_TO_ALLOCATE( nf, nd )	( ( ( (nf) + (nd) - 1 ) / (nd) ) * (nd) )

/* We name the volume by the first disk (given to the mkfs cmd) */

#define LAGRANGE_NAME		"lagrange"
#define LAGRANGE_DEFAULT_VOLUME	"/dev/sdd1"

#define BROCA_NAME		"broca"
#define BROCA_DEFAULT_VOLUME	"/dev/sdb1"

#define FOURIER_NAME		"fourier"
#define FOURIER_DEFAULT_VOLUME	"/export/home/fourier2/mark/disk1"

#define CRAIK_NAME		"craik"
// Inode struct changed, old legacy rawvol on sdb1
//#define CRAIK_DEFAULT_VOLUME	"/dev/sdb1"
#define CRAIK_DEFAULT_VOLUME	"/dev/sdd1"

#define GOETHE_NAME		"goethe"
#define GOETHE_DEFAULT_VOLUME	"/home/plateau3/jbm/ramdisk/disk1"

#define HILBERT_NAME		"hilbert"
/* #define HILBERT_DEFAULT_VOLUME	"/dev/sdb1" */
/* #define HILBERT_DEFAULT_VOLUME	"/home/plateau3/jbm/ramdisk/disk1" */
#define HILBERT_DEFAULT_VOLUME	""

#define DIRAC_NAME		"dirac"
#define DIRAC_DEFAULT_VOLUME	"/dev/sdb1"

#define DONDERS_NAME		"donders"
#define DONDERS_DEFAULT_VOLUME	"/dev/sdb1"

#define PURKINJE_NAME		"purkinje"
#define PURKINJE_DEFAULT_VOLUME	"/dev/rawvol"

/* just for testing now */
#define DALTON_NAME		"dalton"
#define DALTON_DEFAULT_VOLUME	"dev1"

/* The data blocks are striped... with the housekeeping data,
 * we have a choice:  we can either keep it on one disk or the other,
 * or we can mirror it...
 */


#define RV_MAGIC	0x1fabcdef
#define RV_MAGIC2	0x12abcdef	/* version 2 allows directories */

#define NO_SUPER ((RV_Super *)NULL)
#define N_INODE_PAD	36

#ifdef FOOBAR	// these are defined in rv_api.h
#define rvi_shape		rvi_u.u_mi.mi_shape
#define rvi_sbp			rvi_u.u_mi.mi_sbp
#define rvi_extra_bytes		rvi_u.u_mi.mi_extra_bytes

#define rvi_tdim		rvi_shape.si_tdim
#define rvi_rows		rvi_shape.si_rows
#define rvi_cols		rvi_shape.si_cols
#define rvi_frames		rvi_shape.si_frames

#endif // FOOBAR

#define rvi_fi			rvi_u.u_mi.mi_fi
#define rvi_lp		rvi_u.u_di.di_lp

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

#include "off64_t.h"

/* globals */
extern int rawvol_debug;

/* prototypes */

/* rawvol.c */

extern int rv_is_open(void);
extern void perform_write_test(QSP_ARG_DECL  int i_block, int n_blocks, int n_reps );
extern int legal_rv_filename(const char *);
extern void rv_set_extra(int);
extern void rv_mkdir(QSP_ARG_DECL  const char *dirname);
extern int rv_cd(QSP_ARG_DECL  const char *dirname);
extern void rv_pwd(SINGLE_QSP_ARG_DECL);
extern int grant_root_access(QSP_ARG_DECL  const char *user_name);
extern int remember_frame_info(RV_Inode *inp, int index, USHORT_ARG nerr, dimension_t *frames);
extern void xfer_frame_info(dimension_t *lp,int index,RV_Inode *inp);
extern void dump_block(QSP_ARG_DECL  int i,dimension_t block);

extern int	rv_frame_seek(QSP_ARG_DECL  RV_Inode *,dimension_t);

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


ITEM_INTERFACE_PROTOTYPES(RV_Inode,rv_inode)
#define PICK_RV_INODE(p)	pick_rv_inode(QSP_ARG  p)

extern void set_use_osync(int flag);

extern void	rawvol_info(SINGLE_QSP_ARG_DECL);
extern void     rawvol_get_usage(SINGLE_QSP_ARG_DECL);
extern int	rv_truncate(RV_Inode *,dimension_t);

#endif /* undef _RAWVOL_H_ */

