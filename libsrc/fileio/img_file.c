#include "quip_config.h"

char VersionId_fio_img_file[] = QUIP_VERSION_STRING;

#include <stdio.h>

/* these next two includes used to be ifdef SGI */
/* For the old sgi system, we used iopen(), and O_DIRECT ... */
#ifdef HAVE_SYS_TYPES_H
#include <sys/types.h>
#endif

#ifdef HAVE_SYS_STATVFS_H
#include <sys/statvfs.h>
#endif

#ifdef HAVE_FCNTL_H
#include <fcntl.h>
#endif

#ifdef HAVE_STRING_H
#include <string.h>		/* strcmp() */
#endif

#ifdef HAVE_ERRNO_H
#include <errno.h>
#endif

#include "fio_prot.h"
#include "debug.h"
#include "getbuf.h"
#include "filetype.h"
#include "function.h"	/* prototype for add_sizable() */
#include "img_file.h"
#include "fiojpeg.h"
#include "items.h"
#include "raw.h"
#include "uio.h"
#include "fileck.h"
#include "pathnm.h"

#ifdef HAVE_LIBAVCODEC
#include "my_avi.h"
#endif /* HAVE_LIBAVCODEC */

#ifdef HAVE_IOPEN
#include "glimage.h"			/* prototype for iopen() */
#endif /* HAVE_IOPEN */

#ifdef DEBUG
debug_flag_t debug_fileio=0;
#endif /* DEBUG */

static filetype_code filetype=IFT_HIPS1;
static int no_clobber=1;


static const char *iofile_directory=NULL;


/* local prototypes */

static off_t fio_seek_setup(Image_File *ifp, index_t n);

static double get_if_size(Item *ifp,int index);
static double get_if_il_flg(Item *ifp);

static void update_pathname(Image_File *ifp);
static int check_clobber(Image_File *ifp);
static int infer_filetype_from_name(const char *name);

ITEM_INTERFACE_DECLARATIONS(Image_File,img_file)

static Size_Functions imgfile_sf={
	/*(double (*)(Item *,int))*/		get_if_size,
	/*(Item * (*)(Item *,index_t))*/	NULL,
	/*(Item * (*)(Item *,index_t))*/	NULL,
	/*(double (*)(Item *))*/		get_if_il_flg
};


static int direct_flag=1;

#ifdef HAVE_GETTIMEOFDAY
static double get_if_seconds(Item *,dimension_t);
static double get_if_milliseconds(Item *,dimension_t);
static double get_if_microseconds(Item *,dimension_t);

static Timestamp_Functions if_tsf={
	{	get_if_seconds,
		get_if_milliseconds,
		get_if_microseconds
	}
};
#endif /* HAVE_GETTIMEOFDAY */


extern int force_matio_load, force_avi_load, force_avi_info_load;

void force_load_func(void)
{
	int i;

	/* This is a total kludge, which is here so that the file version info appears even for files
	 * that are ifdef'd out because of missing dependencies.  The reason we do this is so that buildall.shf
	 * will work and not report version mismatches.
	 */
	i=force_matio_load;
	i=force_avi_load;
	i=force_avi_info_load;
}

void set_direct_io(int flag)
{
	direct_flag=flag;
}

List *image_file_list(SINGLE_QSP_ARG_DECL)
{
	if( img_file_itp==NO_ITEM_TYPE ) return(NO_LIST);

	return( item_list(QSP_ARG  img_file_itp) );
}

static void update_pathname(Image_File *ifp)
{
	if( ifp->if_pathname != ifp->if_name ){
		rls_str((char *)ifp->if_pathname);
	}

	/* BUG? don't require UNIX delimiters... */

	if( iofile_directory != NULL && *ifp->if_name != '/' ){
		char str[LLEN];
		sprintf(str,"%s/%s",iofile_directory,ifp->if_name);
		ifp->if_pathname = savestr(str);
	} else {
		ifp->if_pathname = ifp->if_name;
	}
}

void set_iofile_directory(QSP_ARG_DECL  const char *dirname)
{
	if( !directory_exists(dirname) ){
		sprintf(error_string,
	"Directory %s does not exist or is not a directory", dirname);
		WARN(error_string);
		return;
	}

	if( iofile_directory != NULL ){
		rls_str(iofile_directory);
	}

	iofile_directory = savestr(dirname);
}

void image_file_init(SINGLE_QSP_ARG_DECL)
{
	static int inited=0;

	if( inited ) return;

#ifdef CAUTIOUS
	ft_tbl_check(SINGLE_QSP_ARG);
#endif /* CAUTIOUS */

#ifdef DEBUG
	debug_fileio = add_debug_module(QSP_ARG  "fileio");
#endif /* DEBUG */

	/* This may have already been called - e.g. mmvi */
	if( img_file_itp == NO_ITEM_TYPE )
		img_file_init(SINGLE_QSP_ARG);

#ifdef HAVE_GETTIMEOFDAY
	add_tsable( QSP_ARG   img_file_itp, &if_tsf, (Item * (*)(QSP_ARG_DECL  const char *))img_file_of);
#endif

	add_sizable(QSP_ARG  img_file_itp,&imgfile_sf,NULL);
	setstrfunc("iof_exists",iof_exists);

	define_port_data_type(QSP_ARG  P_FILE,"file","name of image file",
		(const char *(*)(QSP_ARG_DECL  Port *)) recv_file,
		null_proc,
		(const char *(*)(QSP_ARG_DECL  const char *))pick_img_file,
		(void (*)(QSP_ARG_DECL  Port *,const void *,int)) xmit_file);

	inited=1;
}

static double get_if_size(Item *ip,int index)
{
	Image_File *ifp;
	ifp = (Image_File *)ip;

#ifdef CAUTIOUS
	if( ifp->if_dp == NO_OBJ ){
		sprintf(DEFAULT_ERROR_STRING,"CAUTIOUS:  image file %s (type %s) has no associated data object!?",
				ifp->if_name,ft_tbl[ifp->if_type].ft_name);
		NERROR1(DEFAULT_ERROR_STRING);
	}
#endif /* CAUTIOUS */

	return( get_dobj_size((Item *)ifp->if_dp,index) );
}

static double get_if_il_flg(Item *ip)
{
	Image_File *ifp;
	ifp = (Image_File *)ip;

#ifdef CAUTIOUS
	if( ifp->if_dp == NO_OBJ ){
		sprintf(DEFAULT_ERROR_STRING,"CAUTIOUS:  get_if_il_flg:  image file %s (type %s) has no associated data object!?",
				ifp->if_name,ft_tbl[ifp->if_type].ft_name);
		NERROR1(DEFAULT_ERROR_STRING);
	}
#endif /* CAUTIOUS */

	return( get_dobj_il_flg((Item *)ifp->if_dp) );
}

/*
 * Delete the file struct, assume the file itself has already
 * been closed, if necessary.
 */

void delete_image_file(QSP_ARG_DECL  Image_File *ifp)
{
	if( ifp->if_dp != NO_OBJ ){
		/* Do we know that the other resources are already freed? */
		givbuf(ifp->if_dp);
		ifp->if_dp = NULL;
	}
	if( ifp->if_pathname != ifp->if_name ){
		rls_str((char *)ifp->if_pathname);
	}
	DEL_IMG_FILE(ifp->if_name);
	rls_str((char *)ifp->if_name);

	/* don't free the struct pointer, it's marked available
	 * for reuse by del_item (called from del_img_file)...
	 */
}

/*
 * Close the file associated with this image file structure.
 * Also delete the image file structure.  May appear
 * as tabled close routine for simple filetypes, also
 * may be called from filetype-specific close routine.
 */

void generic_imgfile_close(QSP_ARG_DECL  Image_File *ifp)
{
	if( USES_STDIO(ifp) ){
		if( ifp->if_fp != NULL ) {
			fclose(ifp->if_fp);
		}
	} else if( USES_UNIX_IO(ifp) ){
		if( ifp->if_fd != -1 ){
			close(ifp->if_fd);
		}
	}

	if( HAD_ERROR(ifp) && IS_WRITABLE(ifp) ){
		/* BUG this should only apply to file-system files */
		if( USES_STDIO(ifp) || USES_UNIX_IO(ifp) )
			unlink(ifp->if_pathname);	/* remove file */
	}
	delete_image_file(QSP_ARG  ifp);
}

void setup_dummy(Image_File *ifp)
{
#ifdef CAUTIOUS
	if( ifp->if_dp != NO_OBJ ){
		sprintf(DEFAULT_ERROR_STRING,
	"CAUTIOUS:  image file %s has already been set up with object %s",
			ifp->if_name,ifp->if_dp->dt_name);
		NWARN(DEFAULT_ERROR_STRING);
		return;
	}
#endif /* CAUTIOUS */

	/* these are not in the database... */

	ifp->if_dp = (Data_Obj *) getbuf(sizeof(*ifp->if_dp));

	ifp->if_dp->dt_name = ifp->if_name;
	ifp->if_dp->dt_ap = NO_AREA;
	ifp->if_dp->dt_parent = NO_OBJ;
	ifp->if_dp->dt_children = NO_LIST;
	ifp->if_dp->dt_data = NULL;
	ifp->if_dp->dt_flags = 0;

	ifp->if_dp->dt_prec = PREC_NONE;
}

int open_fd(Image_File *ifp)
{
	int o_direct=0;

#ifdef HAVE_RGB
	if( ifp->if_type == IFT_RGB )
		return(0);
#endif /* HAVE_RGB */

#ifdef HAVE_DIRECT_IO

	/* for SGI video disk files,
	 * use sgi extension "direct i/o"
	 * which causes the disk driver to dma directly to user space.
	 *
	 * This only works for a local disk, however,
	 * so we need to stat() the file first and find if it is local.
	 *
	 * BUG - this works ok for read files,
	 * but for writing a file, you can't stat()
	 * a file that doesn't exist!?
	 *
	 * We should get the directory and stat it...
	 */

	if( ifp->if_type == IFT_DISK ){
		if( IS_READABLE(ifp) ){
			struct statvfs vfsbuf;

			if( statvfs(ifp->if_pathname,&vfsbuf)< 0 ){
				sprintf(error_string,"statvfs (%s):",
					ifp->if_pathname);
				tell_sys_error(error_string);
				NWARN("Couldn't determine fs type, not using O_DIRECT");
			} else {
				if( vfsbuf.f_flag & ST_LOCAL ){
					o_direct = O_DIRECT;
				} else {
					o_direct = 0;
				}
			}
		} else {
			/* BUG - should stat the directory... */
			if( direct_flag ){
advise("writing file using DIRECT_IO based on flag (NEED TO FIX!)");
advise(ifp->if_pathname);
				o_direct = O_DIRECT;
			}
		}
	}
		
retry:

#endif /* HAVE_DIRECT_IO */


	if( IS_READABLE(ifp) )
		ifp->if_fd = open(ifp->if_pathname,O_RDONLY|o_direct);
	else

		/* open read-write so can rewrite header nframes */

		ifp->if_fd = open(ifp->if_pathname,
					O_RDWR|O_CREAT|O_TRUNC|o_direct,0644);

	if( ifp->if_fd < 0 ){

#ifdef HAVE_DIRECT_IO
		/* can't do O_DIRECT across NFS */
		if( o_direct && errno==EINVAL ){
			o_direct = 0;
sprintf(error_string,"Couldn't open file \"%s\" with direct i/o.",
ifp->if_pathname);
NWARN(error_string);
advise("retrying to open write file w/o DIRECT_IO");
			goto retry;
		}
#endif /* HAVE_DIRECT_IO */

		tell_sys_error("open");
		sprintf(DEFAULT_ERROR_STRING,
			"open_fd:  error getting descriptor for %s file %s",
			IS_READABLE(ifp)?"read":"write",ifp->if_pathname);
		NWARN(DEFAULT_ERROR_STRING);
		return(-1);
	}
	return(0);
}

#ifdef HAVE_IOPEN
int ifp_iopen(Image_File *ifp)
{
sprintf(error_string,"name %s  rows %d cols %d",
ifp->if_name,ifp->if_dp->dt_rows,ifp->if_dp->dt_cols);
advise(error_string);

	/* BUG maybe this 3 can be set to 2 if tdim==1 */
	ifp->if_hd.rgb_ip = iopen(ifp->if_pathname,"w",VERBATIM(1),3,
		ifp->if_dp->dt_rows,ifp->if_dp->dt_cols,ifp->if_dp->dt_comps);

	if( ifp->if_hd.rgb_ip == NULL ){
		sprintf(error_string,"Error iopening file %s",ifp->if_pathname);
		NWARN(error_string);
		return(-1);
	} else return(0);
}
#endif /* HAVE_IOPEN */

int open_fp(Image_File *ifp)
{
#ifndef HAVE_RGB

	if( IS_READABLE(ifp) ){
		ifp->if_fp = try_open(DEFAULT_QSP_ARG  ifp->if_pathname,"r");
	} else {
		/* open read-write so we can read back
		 * the header if necessary...  (see hips2.c)
		 */
		ifp->if_fp = try_open(DEFAULT_QSP_ARG  ifp->if_pathname,"w+");
	}
	if( ! ifp->if_fp ) return(-1);
	return(0);

#else /* HAVE_RGB */

	if( IS_READABLE(ifp) ){
		if( ifp->if_type == IFT_RGB ){
			ifp->if_hd.rgb_ip = iopen(ifp->if_pathname,"r");
			if( ifp->if_hd.rgb_ip == NULL ){
				sprintf(error_string,
			"open_fp:  error getting RGB descriptor for file %s",
					ifp->if_pathname);
				NWARN(error_string);
				return(-1);
			} else return(0);
		} else {
			ifp->if_fp = TRY_OPEN(ifp->if_pathname,"r");
		}
	} else {
		/* if .rgb defer the iopen until we know the image size */
		if( ifp->if_type != IFT_RGB ){
			/* open read-write so we can read back
			 * the header if necessary...  (see hips2.c)
			 */
			ifp->if_fp = TRY_OPEN(ifp->if_pathname,"w+");
		}
	}
	if( ifp->if_type != IFT_RGB ){
		if( ! ifp->if_fp ) return(-1);
	}
	return(0);

#endif /* HAVE_RGB */
}

static int check_clobber(Image_File *ifp)
{
	const char *dir;

	dir=parent_directory_of(ifp->if_pathname);


	/* now see if the file already exists, and if
	 * our application is permitting clobbers, then
	 * check the file system permissions
	 */

	if( file_exists(ifp->if_pathname) ){
		if( no_clobber ){
			sprintf(DEFAULT_ERROR_STRING,
				"Not clobbering existing file \"%s\"",
				ifp->if_pathname);
			NWARN(DEFAULT_ERROR_STRING);
			return(-1);
		} else if( !can_write_to(ifp->if_pathname) )
			return(-1);
	} else {
		/* We may have write permissions to the file even if we don't have
		 * write permissions on the directory, e.g. /dev/null
		 */
		if( !can_write_to(dir) ){
			sprintf(DEFAULT_ERROR_STRING, "Can't write to directory \"%s\"", dir);
			NWARN(DEFAULT_ERROR_STRING);
			return(-1);
		}
	}

	return(0);
}

void image_file_clobber(int flag)
{
	if( flag ) no_clobber=0;
	else no_clobber=1;
}

/*
 * This routine creates and initializes an image file struct,
 * and then opens the file using a method appropriate to the file type.
 *
 * Typically called from module open routines.
 */

Image_File *image_file_open(QSP_ARG_DECL  const char *name,int rw,filetype_code type)
{
	Image_File *ifp;
	int had_error=0;

	if( rw == FILE_READ && CANNOT_READ(type) ){
		sprintf(error_string,"Sorry, don't know how to read %s files",
			ft_tbl[type].ft_name);
		NWARN(error_string);
		return(NO_IMAGE_FILE);
	} else if( rw == FILE_WRITE && CANNOT_WRITE(type) ){
		sprintf(error_string,"Sorry, don't know how to write %s files",
			ft_tbl[type].ft_name);
		NWARN(error_string);
		return(NO_IMAGE_FILE);
	}

	ifp = new_img_file(QSP_ARG  name);
	if( ifp==NO_IMAGE_FILE ) return(ifp);

	ifp->if_flags = rw;
	ifp->if_nfrms = 0;

	ifp->if_pathname = ifp->if_name;	/* default */
	update_pathname(ifp);

	ifp->if_dp = NO_OBJ;
	if( IS_READABLE(ifp) )
		setup_dummy(ifp);
	if( IS_WRITABLE(ifp) ){
		if( check_clobber(ifp) < 0 ){
			had_error=1;
			goto dun;
		}
	}

	ifp->if_type = type;

	if( USES_STDIO(ifp) ){
		if( open_fp(ifp) < 0 ) had_error=1;
	} else if( USES_UNIX_IO(ifp) ){
		if( open_fd(ifp) < 0 ) had_error=1;
	}

dun:
	if( had_error ){
		if( IS_READABLE(ifp) ){
			givbuf(ifp->if_dp);
		}
		DEL_IMG_FILE(name);
		/* BUG should also rls_str(name) here??? */
		return(NO_IMAGE_FILE);
	}

	return(ifp);
}

int same_size(QSP_ARG_DECL  Data_Obj *dp,Image_File *ifp)
{
	if( ifp->if_dp == NO_OBJ ){
		sprintf(ERROR_STRING,"No size/prec info about image file %s",
			ifp->if_name);
		WARN(ERROR_STRING);
		return(0);
	}

	if(
		( ifp->if_dp->dt_rows != 0 && ifp->if_dp->dt_rows != dp->dt_rows )	||
		( ifp->if_dp->dt_cols != 0 && ifp->if_dp->dt_cols != dp->dt_cols )
		){

		sprintf(ERROR_STRING,"size mismatch, object %s and file %s",
			dp->dt_name,ifp->if_name);
		WARN(ERROR_STRING);
		sprintf(ERROR_STRING,"      %-24s %-24s",
			dp->dt_name,ifp->if_name);
		advise(ERROR_STRING);
		sprintf(ERROR_STRING,"rows: %-24d %-24d",dp->dt_rows,
			ifp->if_dp->dt_rows);
		advise(ERROR_STRING);
		sprintf(ERROR_STRING,"cols: %-24d %-24d",dp->dt_cols,
			ifp->if_dp->dt_cols);
		advise(ERROR_STRING);
		return(0);
	}
	return(1);
}

int same_type(QSP_ARG_DECL  Data_Obj *dp,Image_File *ifp)
{
	int retval=1;

	if( ifp->if_dp == NO_OBJ ){
		sprintf(ERROR_STRING,"No size/prec info about image file %s",
			ifp->if_name);
		WARN(ERROR_STRING);
		return(0);
	}

	if( dp->dt_prec != ifp->if_dp->dt_prec ){
		/* special case for unsigned (hips doesn't record this) */
		if(
		    (dp->dt_prec == PREC_UDI && ifp->if_dp->dt_prec == PREC_DI) ||
		    (dp->dt_prec == PREC_UIN && ifp->if_dp->dt_prec == PREC_IN) ||
		    (dp->dt_prec == PREC_UBY && ifp->if_dp->dt_prec == PREC_BY)
		){
			/* it's ok */
		} else {
			sprintf(ERROR_STRING,
	"Pixel format (%s) for file %s\n\tdoes not match object %s precision (%s)",
	prec_name[ifp->if_dp->dt_prec],ifp->if_name,
	dp->dt_name,prec_name[dp->dt_prec]);
			WARN(ERROR_STRING);
			retval=0;
		}
	}

	if( dp->dt_comps != ifp->if_dp->dt_comps ){
		sprintf(ERROR_STRING,
	"Pixel dimension (%d) for file %s\n\tdoes not match pixel dimension (%d) for object %s",
	ifp->if_dp->dt_comps,ifp->if_name,dp->dt_comps,dp->dt_name);
		WARN(ERROR_STRING);
		retval=0;
	}
	return(retval);
}

void copy_dimensions(Data_Obj *dpto,Data_Obj *dpfr)	/* used by write routines... */
{
	int i;

	for(i=0;i<N_DIMENSIONS;i++){
		dpto->dt_type_dim[i] = dpfr->dt_type_dim[i];
		dpto->dt_type_inc[i] = dpfr->dt_type_inc[i];
	}
	dpto->dt_prec = dpfr->dt_prec;
	dpto->dt_flags = dpfr->dt_flags;
	dpto->dt_maxdim = dpfr->dt_maxdim;
	dpto->dt_mindim = dpfr->dt_mindim;
	dpto->dt_n_type_elts = dpfr->dt_n_type_elts;
}

void if_info(QSP_ARG_DECL  Image_File *ifp)
{
	sprintf(msg_str,"File %s:",ifp->if_name);
	prt_msg(msg_str);
	sprintf(msg_str,"\tpathname:  %s:",ifp->if_pathname);
	prt_msg(msg_str);
	sprintf(msg_str,"\t%s format (%d)",ft_tbl[ifp->if_type].ft_name,ifp->if_type);
	prt_msg(msg_str);
	if( ifp->if_dp != NO_OBJ ){
		sprintf(msg_str,"\t%s pixels",prec_name[ifp->if_dp->dt_prec]);
		prt_msg(msg_str);
		if( ifp->if_dp->dt_seqs > 1 ){
			sprintf(msg_str,
			"\t%d sequences, ",ifp->if_dp->dt_seqs);
			prt_msg_frag(msg_str);
		} else {
			sprintf(msg_str,"\t");
			prt_msg_frag(msg_str);
		}
		sprintf(msg_str,
		"%d frames, %d rows, %d columns, %d components",
			ifp->if_dp->dt_frames,
			ifp->if_dp->dt_rows,
			ifp->if_dp->dt_cols,
			ifp->if_dp->dt_comps);
		prt_msg(msg_str);
	}
	if( IS_READABLE(ifp) ){
		prt_msg("\topen for reading");
		sprintf(msg_str,"\t%d frames already read",ifp->if_nfrms);
		prt_msg(msg_str);
	} else if( IS_WRITABLE(ifp) ){
		prt_msg("\topen for writing");
		sprintf(msg_str,"\t%d frames already written",ifp->if_nfrms);
		prt_msg(msg_str);
	}
#ifdef CAUTIOUS
	else
		prt_msg("Wacky RW mode!?");
#endif

	/* print format-specific info */
	(*ft_tbl[ifp->if_type].info_func)(QSP_ARG  ifp);
}

/* typical usage:
 *	dump_image_file("foo.viff",IFT_VIFF,buffer,width,height,PREC_BY);
 */

void dump_image_file(QSP_ARG_DECL  const char *filename,filetype_code filetype,void *data,dimension_t width,dimension_t height,prec_t prec)
{
	Data_Obj dobj;
	Image_File *ifp;

	sizinit();		/* just in case called by a standalone app */

	dobj.dt_data = data;
	dobj.dt_name = "dump_image";

	dobj.dt_comps=1;	dobj.dt_cinc=1;
	dobj.dt_cols=width;	dobj.dt_pinc=1;
	dobj.dt_rows=height;	dobj.dt_rinc=(incr_t)width;
	dobj.dt_frames=1;	dobj.dt_finc=(incr_t)(width*height);
	dobj.dt_seqs=1;		dobj.dt_sinc=(incr_t)(width*height);

	dobj.dt_n_type_elts = width*height;

	dobj.dt_prec=prec;

#ifdef CAUTIOUS
	if( filetype < 0 || filetype >= N_FILETYPE ){
		NWARN("CAUTIOUS:  bad filetype spec");
		return;
	}
#endif /* CAUTIOUS */

	ifp=(*ft_tbl[filetype].op_func)(QSP_ARG  filename,FILE_WRITE);
	if( ifp == NO_IMAGE_FILE ) return;
	write_image_to_file(QSP_ARG  ifp,&dobj);
}

#ifdef FOOBAR

/*
 * this one should allocate it's own storage
 */

void *load_image_file(char *name,filetype_code input_file_type,filetype_code desired_hdr_type)
{
	Image_File *ifp;
	Data_Obj *dp;
	void *new_hdp;

	ifp=open_image_file(name,"r");
	if( ifp == NO_IMAGE_FILE ) return(NULL);

	/*
	dp = make_dobj(name,ifp->if_dp->dt_type_dim,ifp->if_dp->dt_prec);
	*/
	/* originally we gave this the filename, but use localname() to
	 * work with Carlo's hack
	 */
	dp = make_dobj(localname(),
		ifp->if_dp->dt_type_dim,ifp->if_dp->dt_prec);

	if( dp == NO_OBJ ) return(NULL);

	/* read the data */
	(*ft_tbl[desired_hdr_type].rd_func)(dp,ifp,0,0,0);

	/* convert to desired type */
	if((*ft_tbl[desired_hdr_type].unconv_func)(&new_hdp,dp) < 0 )
		return(NULL);

	return(new_hdp);
}
#endif /* FOOBAR */

/* filetype independent stuff lifted from fiomenu.c */

filetype_code get_filetype(VOID)
{
	return(filetype);
}

void set_filetype(QSP_ARG_DECL  filetype_code n)
{
	filetype=n;

	if( ft_tbl[filetype].ft_type != filetype ){
		sprintf(ERROR_STRING,"set_filetype %d:  File type table out of order",n);
		ERROR1(ERROR_STRING);
	}
}

typedef struct known_suffix {
	const char *	sfx_name;
	int		sfx_code;
} Known_Suffix;

Known_Suffix sfx_tbl[]={
	{	"hips2",	IFT_HIPS2	},
	{	"vst",		IFT_VISTA	},
	{	"raw",		IFT_RAW		},
#ifdef HAVE_KHOROS
	{	"viff",		IFT_VIFF	},
#endif /* HAVE_KHOROS */
#ifdef HAVE_TIFF
	{	"tiff",		IFT_TIFF	},
	{	"tif",		IFT_TIFF	},
#endif /* HAVE_TIFF */
	{	"dsk",		IFT_DISK	},
	{	"dsk",		IFT_DISK	},
#ifdef HAVE_RGB
	{	"rgb",		IFT_RGB		},
#endif /* HAVE_RGB */
	{	"vl",		IFT_VL		},
	{	"ras",		IFT_SUNRAS	},
	{	"ppm",		IFT_PPM		},
#ifdef HAVE_JPEG_SUPPORT
	{	"wav",		IFT_WAV		},
	{	"WAV",		IFT_WAV		},
	{	"jpg",		IFT_JPEG	},
	{	"JPG",		IFT_JPEG	},
	{	"bmp",		IFT_BMP		},
	{	"BMP",		IFT_BMP		},
	{	"lml",		IFT_LML		},
	{	"asc",		IFT_ASC		},
	{	"jpeg",		IFT_JPEG	},
	{	"mjpg",		IFT_JPEG	},
#endif /* HAVE_JPEG_SUPPORT */

#ifdef HAVE_PNG
	{	"png",		IFT_PNG		},
#endif /* HAVE_PNG */

#ifdef HAVE_MPEG
	{	"mpeg",		IFT_MPEG	},
	{	"mpg",		IFT_MPEG	},
#endif /* HAVE_MPEG */

#ifdef HAVE_QUICKTIME
	{	"mov",		IFT_QT		},
#endif
	{	"hips1",	IFT_HIPS1	},
	{	"bdf",		IFT_BDF		},
#ifdef HAVE_MATIO
	{	"mat",		IFT_MATLAB	},
#endif /* HAVE_MATIO */

#ifdef HAVE_LIBAVCODEC
	{	"avi",		IFT_AVI		},
#endif /* HAVE_LIBAVCODEC */

	/* the last entry has no comma, so should not be ifdef-able ... */
};

static int n_suffices=sizeof(sfx_tbl)/sizeof(Known_Suffix);

static int infer_filetype_from_name(const char *name)
{
	const char *suffix=NULL;
	const char *s;
	int i;

	/* set the suffix to the string following the last '.' */

	s=name;
	while(*s!=0){
		if( *s == '.' ) suffix = s+1;
		s++;
	}
	if( suffix == NULL || *suffix == 0 ) return(-1);

	for(i=0;i<n_suffices;i++)
		if( !strcmp(suffix,sfx_tbl[i].sfx_name) ){
			int ft;
			ft=sfx_tbl[i].sfx_code;
			if( verbose ){
				sprintf(msg_str,
					"Inferring filetype %s (%d) from suffix .%s",
					ft_tbl[ft].ft_name,ft,suffix);
				advise(msg_str);
			}
			return(ft);
		}
	return(-1);
}

/*
 * Call type-specific function to open the file
 */

Image_File *read_image_file(QSP_ARG_DECL  const char *name)
{
	Image_File *ifp;
	int ft;

	ft = infer_filetype_from_name(name);
	if( ft < 0 ) ft=filetype;	/* use default if can't figure it out */
	else if( verbose && ft!=filetype ){
		sprintf(error_string,"Inferring filetype %s from filename %s, overriding default %s",
			ft_tbl[ft].ft_name,name,ft_tbl[filetype].ft_name);
		advise(error_string);
	}

	if( CANNOT_READ(ft) ){
		sprintf(error_string,"Sorry, can't read files of type %s",
			ft_tbl[ft].ft_name);
		NWARN(error_string);
		return(NO_IMAGE_FILE);
	}

	/* pathname hasn't been set yet... */
	ifp=(*ft_tbl[ft].op_func)( QSP_ARG  name, FILE_READ );

	if( ifp == NO_IMAGE_FILE ) {
		sprintf(error_string,
			"error reading %s file \"%s\"",ft_tbl[ft].ft_name,name);
		NWARN(error_string);
	}
	return(ifp);
}

/* Open a file for writing */

Image_File *write_image_file(QSP_ARG_DECL  const char *filename,dimension_t n)
{
	Image_File *ifp;
	int ft;

	ft = infer_filetype_from_name(filename);
	if( ft < 0 ) ft=filetype;	/* use default if can't figure it out */
	else if( ft != filetype ){
		sprintf(error_string,"Inferring filetype %s from filename %s, overriding default %s",
			ft_tbl[ft].ft_name,filename,ft_tbl[filetype].ft_name);
		advise(error_string);
	}

	if( CANNOT_WRITE(ft) ){
		sprintf(error_string,"Sorry, can't write files of type %s",
			ft_tbl[ft].ft_name);
		NWARN(error_string);
		return(NO_IMAGE_FILE);
	}

	ifp = (*ft_tbl[ft].op_func)( QSP_ARG  filename, FILE_WRITE ) ;
	if( ifp != NO_IMAGE_FILE )
		ifp->if_frms_to_wt = n;

	return(ifp);
}

/* Should we impose that the objects have the same size?? */

void read_object_from_file(QSP_ARG_DECL  Data_Obj *dp,Image_File *ifp)
{
	if( dp == NO_OBJ ) return;
	if( ifp == NO_IMAGE_FILE ) return;

	if( !IS_READABLE(ifp) ){
		sprintf(error_string,"File %s is not readable",ifp->if_name);
		WARN(error_string);
		return;
	}

	if( ifp->if_dp->dt_rows != 0 && dp->dt_rows != ifp->if_dp->dt_rows ){
		sprintf(error_string,"Row count mismatch, object %s (%d) and file %s (%d)",
				dp->dt_name,dp->dt_rows,ifp->if_name,ifp->if_dp->dt_rows);
		WARN(error_string);
	}

	if( ifp->if_dp->dt_cols != 0 && dp->dt_cols != ifp->if_dp->dt_cols ){
		sprintf(error_string,"Column count mismatch, object %s (%d) and file %s (%d)",
				dp->dt_name,dp->dt_cols,ifp->if_name,ifp->if_dp->dt_cols);
		WARN(error_string);
	}

	/* was this nfrms a BUG, or was there a reason??? */
	
	(*ft_tbl[ifp->if_type].rd_func)(QSP_ARG  dp,ifp,0,0,/* ifp->hf_nfrms */ 0 );
}

/*
 * Filetype independent way to close an image file.
 * Calls routine from table.
 */

void close_image_file(QSP_ARG_DECL  Image_File *ifp)
{
	if( ifp == NO_IMAGE_FILE ) return;
	(*ft_tbl[ifp->if_type].close_func)(QSP_ARG  ifp);
}

/*
 * Open for reading or writing
 *
 * High level routine, calls r/w specific routine, which may
 * call vectored module-specific routine...
 */

Image_File * open_image_file(QSP_ARG_DECL  const char *filename,const char *rw)
{
	Image_File *ifp;

sprintf(error_string,"open_image_file %s",filename);
advise(error_string);
	if( *rw == 'r' )
		ifp = read_image_file(QSP_ARG  filename);

	/* BUG 4096 is an arbitrary big number.  Originally we
	 * passed the number of frames to write to the open routine
	 * so we could write the header; Now for hips1 we can go
	 * back and edit it later...  need to support this feature
	 * for hips2, viff and ???
	 */

	else if( *rw == 'w' )
		ifp = write_image_file(QSP_ARG  filename,4096);

#ifdef CAUTIOUS
	else {
		WARN("CAUTIOUS:  bad r/w string passed to open_image_file");
		ifp = NO_IMAGE_FILE;
	}
#endif /* CAUTIOUS */

	return(ifp);
}

/* put an image out to a writable file */

void write_image_to_file(QSP_ARG_DECL  Image_File *ifp,Data_Obj *dp)
{
	/* take filetype from image file */
	if( dp == NO_OBJ ) return;
	if( ifp == NO_IMAGE_FILE ) return;

	if( ! IS_WRITABLE(ifp) ){
		sprintf(error_string,"File %s is not writable",ifp->if_name);
		WARN(error_string);
		return;
	}

	(*ft_tbl[ifp->if_type].wt_func)(QSP_ARG  dp,ifp);
}

static off_t fio_seek_setup(Image_File *ifp, index_t n)
{
	long frms_to_seek;
	index_t frmsiz;

	frms_to_seek = (long)(n - ifp->if_nfrms);

	/* figure out frame size */

	frmsiz = ifp->if_dp->dt_cols * ifp->if_dp->dt_rows
		* ifp->if_dp->dt_comps * siztbl[ifp->if_dp->dt_prec];

	if( ifp->if_type == IFT_DISK ){
		/* round up to block size */
		frmsiz += 511;
		frmsiz &= ~511;
		/* BUG? add a 1?? */
	}

	/* although IFT_RV frames are also rounded up to blocksize,
	 * we pass the frame number to rv_seek_frame() and let it worry
	 * about it...
	 */
	
	return( frms_to_seek * frmsiz );
}

int uio_seek(QSP_ARG_DECL  Image_File *ifp, index_t n)
{
	off_t offset;

	offset = fio_seek_setup(ifp,n);

	if( lseek(ifp->if_fd,offset,SEEK_CUR) < 0 ){
		WARN("lseek error");
		return(-1);
	}
	return(0);
}

int std_seek(QSP_ARG_DECL  Image_File *ifp, index_t n)
{
	off_t offset;

	offset = fio_seek_setup(ifp,n);

	if( offset == 0 ){
		if( verbose ) advise("Seek to current location requested!?");
		return(0);	/* nothing to do */
	}
	if( fseek(ifp->if_fp,offset,/*1*/ SEEK_CUR) != 0 ){
		WARN("fseek error");
		return(-1);
	}
	return(0);
}

int image_file_seek(QSP_ARG_DECL  Image_File *ifp,dimension_t n)
{
	/* BUG?  off_t is long long on new sgi!? */

#ifdef DEBUG
if( debug & debug_fileio ){
sprintf(error_string,"image_file_seek %s %d",
		ifp->if_name,n);
advise(error_string);
}
#endif /* DEBUG */
	if( ! IS_READABLE(ifp) ){
		sprintf(error_string,"File %s is not readable, can't seek",
			ifp->if_name);
		WARN(error_string);
		return(-1);
	}
	if( n >= ifp->if_dp->dt_frames ){
		sprintf(error_string,
	"Frame index %d is out of range for file %s (%d frames)",
			n,ifp->if_name,ifp->if_dp->dt_frames);
		WARN(error_string);
		return(-1);
	}

	/* how do we figure out what frame we are at currently?
	 * It looks like if_nfrms holds the index of the next frame
	 * we will read...
	 *
	 * For most of our formats, the frame size is fixed so we can
	 * just calculate the offset.  For jpeg, we have to use a seek
	 * table that we construct when we first scan the file for frames.
	 *
	 * This would be cleaner if we vectored the seek routine...
	 * rv is a special case, since we have multiple disks to worry about.
	 */

	if( (*ft_tbl[ ifp->if_type ].seek_func)(QSP_ARG  ifp,n) < 0 ){
		sprintf(error_string,"Error seeking frame %d on file %s",n,ifp->if_name);
		WARN(error_string);
		return(-1);
	}
	ifp->if_nfrms = n;
	return(0);
}

void check_auto_close(QSP_ARG_DECL  Image_File *ifp)
{
	if( ifp->if_nfrms >= ifp->if_frms_to_wt ){
		if( verbose ){
	sprintf(error_string, "closing file \"%s\" after writing %d frames",
			ifp->if_name,ifp->if_nfrms);
			advise(error_string);
		}
		close_image_file(QSP_ARG  ifp);
	}
}

double iof_exists(QSP_ARG_DECL  const char *s)
{
	Image_File *ifp;

	ifp=img_file_of(QSP_ARG  s);
	if( ifp==NO_IMAGE_FILE ) return(0.0);
	else return(1.0);
}

double get_if_seconds(Item *ip,dimension_t frame)
{
	Image_File *ifp;

	ifp = (Image_File *) ip;

	switch(ifp->if_type){
#ifdef HAVE_JPEG_SUPPORT
		case IFT_LML: return( get_lml_seconds(ifp,frame) ); break;
#endif /* HAVE_JPEG_SUPPORT */
		case IFT_RV: return( get_rv_seconds(ifp,frame)); break;
		default:
			NWARN("Timestamp functions are only supported for file types LML and RV");
			sprintf(DEFAULT_ERROR_STRING,"(file %s is type %s)",ifp->if_name,
					ft_tbl[ ifp->if_type ].ft_name);
			advise(DEFAULT_ERROR_STRING);
			return(-1);
	}
}

double get_if_milliseconds(Item *ip,dimension_t frame)
{
	Image_File *ifp;

	ifp = (Image_File *) ip;

	switch(ifp->if_type){
#ifdef HAVE_JPEG_SUPPORT
		case IFT_LML: return( get_lml_milliseconds(ifp,frame) ); break;
#endif /* HAVE_JPEG_SUPPORT */
		case IFT_RV: return( get_rv_milliseconds(ifp,frame) ); break;
		default:
			NWARN("Timestamp functions are only supported for file types LML and RV");
			sprintf(DEFAULT_ERROR_STRING,"(file %s is type %s)",ifp->if_name,
					ft_tbl[ ifp->if_type ].ft_name);
			advise(DEFAULT_ERROR_STRING);
			return(-1);
	}
}

double get_if_microseconds(Item *ip,dimension_t frame)
{
	Image_File *ifp;

	ifp = (Image_File *) ip;

	switch(ifp->if_type){
#ifdef HAVE_JPEG_SUPPORT
		case IFT_LML: return( get_lml_microseconds(ifp,frame) ); break;
#endif /* HAVE_JPEG_SUPPORT */
		case IFT_RV: return( get_rv_microseconds(ifp,frame) ); break;
		default:
			NWARN("Timestamp functions are only supported for file types LML and RV");
			sprintf(DEFAULT_ERROR_STRING,"(file %s is type %s)",ifp->if_name,
					ft_tbl[ ifp->if_type ].ft_name);
			advise(DEFAULT_ERROR_STRING);
			return(-1);
	}
}




