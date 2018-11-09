
#include "quip_config.h"

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

#include "quip_prot.h"
#include "rawvol.h"

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
		warn(ERROR_STRING);
		ndisks=1;
	}
	for(i=0;i<ndisks;i++){
		s=NAMEOF("disk name");
		if( strlen(s) >= MAX_DISKNAME_LEN ){
			sprintf(ERROR_STRING,
	"Need to increase MAX_DISKNAME_LEN (%d) to accomodate diskname \"%s\"",
				MAX_DISKNAME_LEN,s);
			error1(ERROR_STRING);
		}
		strcpy(disknames[i],s);
		str_arr[i] = disknames[i];
	}

	nib=HOW_MANY("number of inode blocks");
	nsb=HOW_MANY("number of string blocks");

	if( nib <= 0  ){
		sprintf(ERROR_STRING,"number of inode blocks (%ld) must be positive",nib);
		warn(ERROR_STRING);
		return;
	}
	if( nsb <= 0  ){
		sprintf(ERROR_STRING,"number of string blocks (%ld) must be positive",nsb);
		warn(ERROR_STRING);
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

	inp = pick_rv_inode("");
	if( inp==NULL ) return;
	rv_rmfile(RV_NAME(inp));
}

static COMMAND_FUNC( do_chmod )
{
	RV_Inode *inp;
	int mode;

	inp = pick_rv_inode("");
	mode = HOW_MANY("integer mode code");
	if( inp==NULL ) return;
	rv_chmod(QSP_ARG  inp,mode);
}

static COMMAND_FUNC( do_ls_one )
{
	RV_Inode *inp;

	inp = pick_rv_inode("");
	if( inp==NULL ) return;
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
		warn("number of blocks must be greater than 1");
		return;
	}

	rv_mkfile(QSP_ARG  s,nt,nw);
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

	dump_block(QSP_ARG  i,block);
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
		warn("Only root can grant rawvol super-user privileges to other users");
		return;
	}

	if( grant_root_access(QSP_ARG  s) < 0 ){
		sprintf(ERROR_STRING,"Error granting root access to user %s",s);
		warn(ERROR_STRING);
	}
}

static COMMAND_FUNC( do_wtest )
{
	int n, s, r;

	n = HOW_MANY("index of disk to test");
	s = HOW_MANY("size (in blocks) to test");
	r = HOW_MANY("number of repetitions per write");

	perform_write_test(QSP_ARG  n,s,r);

}

static COMMAND_FUNC( do_rawvol_info )
{
	rawvol_info(SINGLE_QSP_ARG);
}

static COMMAND_FUNC( do_rawvol_get_usage )
{
	rawvol_get_usage(SINGLE_QSP_ARG);
}

#define ADD_CMD(s,f,h)	ADD_COMMAND(admin_menu,s,f,h)

MENU_BEGIN(admin)
ADD_CMD( mkfs,		do_mkfs,	make a new file system )
ADD_CMD( osync,		do_set_osync,	set/clear O_SYNC flag (for open(2)) )
ADD_CMD( mkfile,	do_mkfile,	create a big empty file )
ADD_CMD( wtest,		do_wtest,	write test to a single platter )
ADD_CMD( info,		do_rawvol_info,	give info about the current volume )
ADD_CMD( dump,		do_dump_block,	dump a block )
ADD_CMD( get_usage,	do_rawvol_get_usage, show disk usage )
ADD_CMD( root,		do_set_root,	grant superuser priveleges to a user )
MENU_END(admin)


static COMMAND_FUNC( do_admin )
{
	CHECK_AND_PUSH_MENU(admin);
}

static COMMAND_FUNC( do_rv_end )
{
	/* volume may not be open after a mkfs??? */
	if( rv_is_open() )
		rv_sync(SINGLE_QSP_ARG);

	pop_menu();
}

static COMMAND_FUNC( do_rv_info )
{
	RV_Inode *inp;

	inp = pick_rv_inode("filename");
	if( inp == NULL ) return;

	rv_info(inp);
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
	Frame_Info *fi_p;

	s = NAMEOF("name for data vector");
	inp = pick_rv_inode("filename");
	i = WHICH_ONE("error type",N_ERROR_TYPES,error_type_list);

	if( inp==NULL || i < 0 ) return;

	fi_p = &(RV_MOVIE_FRAME_INFO(inp,i));
	if( FRAME_INFO_N_SAVED(fi_p) <= 0 ){
		/*if( verbose ){ */
			sprintf(ERROR_STRING,
	"rv file %s has no %s errors",RV_NAME(inp),error_type_list[i]);
			advise(ERROR_STRING);
		/*} */
		return;
	}

	dp = dobj_of(s);
	if( dp != NULL ){
		/* object already exists */
		if( verbose ){
			sprintf(ERROR_STRING,
		"deleting previously created object %s",s);
			advise(ERROR_STRING);
		}
		delvec(dp);
	}

	dp = mk_vec(s,FRAME_INFO_N_SAVED(fi_p),1,PREC_FOR_CODE(PREC_DI));
	if( dp == NULL ){
		sprintf(ERROR_STRING,"do_err_frms:  unable to create data vector %s",s);
		warn(ERROR_STRING);
		return;
	}
	xfer_frame_info((dimension_t *)OBJ_DATA_PTR(dp),i,inp);
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
		warn("Unable to mount default raw volume");
}

static COMMAND_FUNC(do_rv_sync)
{
	rv_sync(SINGLE_QSP_ARG);
}

static COMMAND_FUNC(do_rv_ls_cwd)
{
	rv_ls_cwd(SINGLE_QSP_ARG);
}

static COMMAND_FUNC(do_rv_rm_cwd)
{
	rv_rm_cwd(SINGLE_QSP_ARG);
}

static COMMAND_FUNC(do_rv_ls_all)
{
	rv_ls_all(SINGLE_QSP_ARG);
}

static COMMAND_FUNC(do_rv_ls_ctx)
{
	rv_ls_ctx(SINGLE_QSP_ARG);
}

static COMMAND_FUNC(do_rv_pwd)
{
	rv_pwd(SINGLE_QSP_ARG);
}

static COMMAND_FUNC(do_rv_close)
{
	rv_close(SINGLE_QSP_ARG);
}

#undef ADD_CMD
#define ADD_CMD(s,f,h)	ADD_COMMAND(rawvol_menu,s,f,h)

MENU_BEGIN(rawvol)
ADD_CMD( default,	do_default_rv,	mount default raw volume )
ADD_CMD( volume,	do_vol,		specify current raw volume )
ADD_CMD( sync,		do_rv_sync,	update raw volume disk header )
ADD_CMD( ls_cwd,	do_rv_ls_cwd,	list files in current directory )
ADD_CMD( ls_all,	do_rv_ls_all,	list files on current volume )
ADD_CMD( rm,		do_rm,		delete raw volume file )
ADD_CMD( rm_cwd,	do_rv_rm_cwd,	remove all files in current directory )
ADD_CMD( ls_ctx,	do_rv_ls_ctx,	list files in current context )
ADD_CMD( ls,		do_ls_one,	list files on current volume )
ADD_CMD( pwd,		do_rv_pwd,	print working RV directory )
ADD_CMD( info,		do_rv_info,	list shape information about a file )
ADD_CMD( chmod,		do_chmod,	change file mode )
ADD_CMD( new,		do_new,		create a new raw volume file )
ADD_CMD( mkdir,		do_mkdir,	create directory )
ADD_CMD( cd,		do_cd,		change directory )
ADD_CMD( error_frames,	do_err_frms,	transfer error frames to a data object )
ADD_CMD( admin,		do_admin,	perform administrative tasks )
ADD_CMD( close,		do_rv_close,	close current raw volume )
ADD_CMD( quit,		do_rv_end,	exit submenu )
MENU_END(rawvol)

COMMAND_FUNC( do_rv_menu )
{
	static int inited=0;

	if( ! inited ){
		dataobj_init();	/* initialize prec_name[] */
		/* insure_default_rv(); */
#ifdef DEBUG
		if( rawvol_debug == 0 )
			rawvol_debug = add_debug_module("rawvol");
#endif /* DEBUG */
		inited++;
	}

	CHECK_AND_PUSH_MENU(rawvol);
}

