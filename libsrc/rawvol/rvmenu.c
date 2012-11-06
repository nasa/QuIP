
#include "quip_config.h"

char VersionId_rawvol_rvmenu[] = QUIP_VERSION_STRING;

#ifdef HAVE_SYS_TYPES_H
#include <sys/types.h>
#endif

#ifdef HAVE_SYS_STAT_H
#include <sys/stat.h>
#endif

#ifdef HAVE_FCNTL_H
#include <fcntl.h>
#endif

#ifdef HAVE_STRING_H
#include <string.h>
#endif

#ifdef HAVE_UNISTD_H
#include <unistd.h>
#endif

#ifdef HAVE_LINUX_UNISTD_H
#include <linux/unistd.h>
#endif

#include "rawvol.h"
#include "query.h"
#include "debug.h"
#include "version.h"


static COMMAND_FUNC( do_mkfs )
{
	const char *s;
	int ndisks;
	long nib,nsb;
	char disknames[MAX_DISKS][MAX_DISKNAME_LEN];
	const char *str_arr[MAX_DISKS];
	int i;

	ndisks = HOW_MANY("number of disks");
	if( ndisks < 1 || ndisks > MAX_DISKS ){
		sprintf(ERROR_STRING,
	"Number of disks (%d) must be between 1 and %d",
			ndisks,MAX_DISKS);
		WARN(ERROR_STRING);
		ndisks=1;
	}
	for(i=0;i<ndisks;i++){
		s=NAMEOF("disk name");
		if( strlen(s) >= MAX_DISKNAME_LEN ){
			sprintf(ERROR_STRING,
	"Need to increase MAX_DISKNAME_LEN (%d) to accomodate diskname \"%s\"",
				MAX_DISKNAME_LEN,s);
			ERROR1(ERROR_STRING);
		}
		strcpy(disknames[i],s);
		str_arr[i] = disknames[i];
	}

	nib=HOW_MANY("number of inode blocks");
	nsb=HOW_MANY("number of string blocks");

	if( nib <= 0  ){
		sprintf(ERROR_STRING,"number of inode blocks (%ld) must be positive",nib);
		WARN(ERROR_STRING);
		return;
	}
	if( nsb <= 0  ){
		sprintf(ERROR_STRING,"number of string blocks (%ld) must be positive",nsb);
		WARN(ERROR_STRING);
		return;
	}

	rv_mkfs(QSP_ARG  ndisks,str_arr,nib,nsb);
}

static COMMAND_FUNC( do_vol )
{
	const char *s;

	s=NAMEOF("volume file name");

	read_rv_super(QSP_ARG  s);
}

static COMMAND_FUNC( do_new )
{
	const char *s;
	long n;

	s=NAMEOF("filename");
	n=HOW_MANY("size");

	rv_newfile(QSP_ARG  s,n);
}

static COMMAND_FUNC( do_rm )
{
	RV_Inode *inp;

	inp = PICK_RV_INODE("");
	if( inp==NO_INODE ) return;
	rv_rmfile(QSP_ARG  inp->rvi_name);
}

static COMMAND_FUNC( do_chmod )
{
	RV_Inode *inp;
	int mode;

	inp = PICK_RV_INODE("");
	mode = HOW_MANY("integer mode code");
	if( inp==NO_INODE ) return;
	rv_chmod(inp,mode);
}

static COMMAND_FUNC( do_ls_one )
{
	RV_Inode *inp;

	inp = PICK_RV_INODE("");
	if( inp==NO_INODE ) return;
	rv_ls_inode(QSP_ARG  inp);
}

static COMMAND_FUNC( do_mkfile )
{
	const char *s;
	long nt,nw;

	s=NAMEOF("pathname");
	nt=HOW_MANY("total number of blocks");
	nw=HOW_MANY("number of blocks per write");

	if( nt<= 1 || nw <= 1 ){
		WARN("number of blocks must be greater than 1");
		return;
	}

	rv_mkfile(s,nt,nw);
}

static COMMAND_FUNC( do_set_osync )
{
	set_use_osync( ASKIF("open raw volumes with O_SYNC") );
}

static COMMAND_FUNC( do_dump_block )
{
	int i;
	u_long block;

	i=HOW_MANY("disk index");
	block = HOW_MANY("block index");

	dump_block(i,block);
}

/* Root priveleges allow one user to delete another's files.
 * Typically we might want to grant this to the experimenter, so that
 * they can delete files created by other subjects.  The alternative is
 * to make the files world-writable...
 */

static COMMAND_FUNC( do_set_root )
{
	const char *s;

	s=NAMEOF("user name for root privileges");

	if( getuid() != 0 ){
		WARN("Only root can grant rawvol super-user privileges to other users");
		return;
	}

	if( grant_root_access(s) < 0 ){
		sprintf(ERROR_STRING,"Error granting root access to user %s",s);
		WARN(ERROR_STRING);
	}
}

static COMMAND_FUNC( do_wtest )
{
	int n, s, r;

	n = HOW_MANY("index of disk to test");
	s = HOW_MANY("size (in blocks) to test");
	r = HOW_MANY("number of repetitions per write");

	perform_write_test(n,s,r);

}

static COMMAND_FUNC( do_rawvol_info ){ rawvol_info(SINGLE_QSP_ARG); }
static COMMAND_FUNC( do_rawvol_get_usage ){ rawvol_get_usage(SINGLE_QSP_ARG); }

Command admin_ctbl[]={
{ "mkfs",	do_mkfs,	"make a new file system"		},
{ "osync",	do_set_osync,	"set/clear O_SYNC flag (for open(2))"	},
{ "mkfile",	do_mkfile,	"create a big empty file"		},
{ "wtest",	do_wtest,	"write test to a single platter"	},
{ "info",	do_rawvol_info,	"give info about the current volume"	},
{ "dump",	do_dump_block,	"dump a block"				},
{ "get_usage",	do_rawvol_get_usage, "print out the number of free and total bytes" },
{ "root",	do_set_root,	"grant superuser priveleges to a user"	},
{ "quit",	popcmd,		"exit program"				},
{ NULL_COMMAND								}
};


static COMMAND_FUNC( do_admin )
{
	PUSHCMD(admin_ctbl,"rv_admin");
}

static COMMAND_FUNC( do_rv_end )
{
	/* volume may not be open after a mkfs??? */
	if( rv_is_open() )
		rv_sync(SINGLE_QSP_ARG);

	popcmd(SINGLE_QSP_ARG);
}

static COMMAND_FUNC( do_rv_info )
{
	RV_Inode *inp;

	inp = PICK_RV_INODE("filename");
	if( inp == NO_INODE ) return;

	rv_info(QSP_ARG  inp);
}

#define N_ERROR_TYPES	4
static const char *error_type_list[N_ERROR_TYPES]={
	"dropped_frame",
	"fifo",
	"dma",
	"fifo/dma"
};

static COMMAND_FUNC( do_err_frms )
{
	Data_Obj *dp;
	const char *s;
	RV_Inode *inp;
	int i;

	s = NAMEOF("name for data vector");
	inp = PICK_RV_INODE("filename");
	i = WHICH_ONE("error type",N_ERROR_TYPES,error_type_list);

	if( inp==NO_INODE || i < 0 ) return;

	if( inp->rvi_fi[i].fi_nsaved <= 0 ){
		/*if( verbose ){ */
			sprintf(ERROR_STRING,
	"rv file %s has no %s errors",inp->rvi_name,error_type_list[i]);
			advise(ERROR_STRING);
		/*} */
		return;
	}

	dp = dobj_of(QSP_ARG  s);
	if( dp != NO_OBJ ){
		/* object already exists */
		if( verbose ){
			sprintf(ERROR_STRING,
		"deleting previously created object %s",s);
			advise(ERROR_STRING);
		}
		delvec(QSP_ARG  dp);
	}

	dp = mk_vec(QSP_ARG  s,inp->rvi_fi[i].fi_nsaved,1,PREC_DI);
	if( dp == NO_OBJ ){
		sprintf(ERROR_STRING,"do_err_frms:  unable to create data vector %s",s);
		WARN(ERROR_STRING);
		return;
	}
	xfer_frame_info((dimension_t *)dp->dt_data,i,inp);
}

static COMMAND_FUNC( do_mkdir )
{
	const char *s;

	s=NAMEOF("directory name");
	rv_mkdir(QSP_ARG  s);
}

static COMMAND_FUNC( do_cd )
{
	const char *s;

	s=NAMEOF("directory name");
	rv_cd(QSP_ARG  s);
}

static COMMAND_FUNC( do_default_rv )
{
	if( insure_default_rv(SINGLE_QSP_ARG) < 0 )
		WARN("Unable to mount default raw volume");
}

static COMMAND_FUNC(do_rv_sync){ rv_sync(SINGLE_QSP_ARG); }
static COMMAND_FUNC(do_rv_ls_cwd){ rv_ls_cwd(SINGLE_QSP_ARG); }
static COMMAND_FUNC(do_rv_rm_cwd){ rv_rm_cwd(SINGLE_QSP_ARG); }
static COMMAND_FUNC(do_rv_ls_all){ rv_ls_all(SINGLE_QSP_ARG); }
static COMMAND_FUNC(do_rv_ls_ctx){ rv_ls_ctx(SINGLE_QSP_ARG); }
static COMMAND_FUNC(do_rv_pwd){ rv_pwd(SINGLE_QSP_ARG); }
static COMMAND_FUNC(do_rv_close){ rv_close(SINGLE_QSP_ARG); }

Command rv_ctbl[]={
{ "default",	do_default_rv,	"mount default raw volume"	},
{ "volume",	do_vol,		"specify current raw volume"	},
{ "sync",	do_rv_sync,	"update raw volume disk header"	},
{ "ls_cwd",	do_rv_ls_cwd,	"list files in current directory" },
{ "ls_all",	do_rv_ls_all,	"list files on current volume"	},
{ "rm",		do_rm,		"delete raw volume file"	},
{ "rm_cwd",	do_rv_rm_cwd,	"remove all files in current directory" },
{ "ls_ctx",	do_rv_ls_ctx,	"list files in current context" },
{ "ls",		do_ls_one,	"list files on current volume"	},
{ "pwd",	do_rv_pwd,		"print working RV directory"	},
{ "info",	do_rv_info,	"list shape information about a file"	},
{ "chmod",	do_chmod,	"change file mode"		},
{ "new",	do_new,		"create a new raw volume file"	},
{ "mkdir",	do_mkdir,	"create directory"		},
{ "cd",		do_cd,		"change directory"		},
{ "error_frames",do_err_frms,	"transfer error frames to a data object" },
{ "admin",	do_admin,	"perform administrative tasks"	},
{ "close",	do_rv_close,	"close current raw volume"	},
{ "quit",	do_rv_end,	"exit submenu"			},
{ NULL_COMMAND							}
};

COMMAND_FUNC( rv_menu )
{
	static int inited=0;

	if( ! inited ){
		dataobj_init(SINGLE_QSP_ARG);	/* initialize prec_name[] */
		auto_version(QSP_ARG  "RAWVOL","VersionId_rawvol");
		/* insure_default_rv(); */
#ifdef DEBUG
		if( rawvol_debug == 0 )
			rawvol_debug = add_debug_module(QSP_ARG  "rawvol");
#endif /* DEBUG */
		inited++;
	}

	PUSHCMD(rv_ctbl,"rawvol");
}

