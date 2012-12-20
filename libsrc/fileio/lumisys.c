
#include "quip_config.h"

char VersionId_fio_lumisys[] = QUIP_VERSION_STRING;

#include <stdio.h>

#ifdef HAVE_UNISTD_H
#include <unistd.h>
#endif

#include "fio_prot.h"
#include "lumisys.h"


/* local prototypes */
//static int ls_to_dp(Data_Obj *dp,Lumisys_Hdr *inp);
static int rd_ls_hdr(int fd,Lumisys_Hdr *hd_p,const char *filename);

#define HDR_P(ifp)	((Image_File_Hdr *)ifp->if_hd)->ifh_u.ls_hd_p

int ls_to_dp(Data_Obj *dp,Lumisys_Hdr *hd_p)
{
	dp->dt_prec = PREC_IN;	/* short */

	dp->dt_comps=1;
	dp->dt_cols=hd_p->ls_width;
	dp->dt_rows=hd_p->ls_height;
	dp->dt_frames=1;
	dp->dt_seqs=1;

	dp->dt_cinc=1;
	dp->dt_pinc=1;
	dp->dt_rinc=dp->dt_pinc*(incr_t)dp->dt_cols;
	dp->dt_finc=dp->dt_rinc*(incr_t)dp->dt_rows;
	dp->dt_sinc=dp->dt_finc*(incr_t)dp->dt_frames;

	dp->dt_parent = NO_OBJ;
	dp->dt_children = NO_LIST;

	dp->dt_ap = ram_area;		/* the default */
	dp->dt_data = NULL;
	dp->dt_n_type_elts = dp->dt_comps * dp->dt_cols * dp->dt_rows
			* dp->dt_frames * dp->dt_seqs;

	set_shape_flags(&dp->dt_shape,dp,AUTO_SHAPE);

	return(0);
}



/* the unconvert routine creates a disk header */

int ls_unconv(void *hdr_pp,Data_Obj *dp)
{
	NWARN("ls_unconv() not implemented!?");
	return(-1);
}

int ls_conv(Data_Obj *dp,void *hd_pp)
{
	NWARN("ls_conv not implemented");
	return(-1);
}

static int rd_ls_hdr(int fd,Lumisys_Hdr *hd_p,const char *filename)
{
	int n;

	if( (n=read(fd,hd_p,sizeof(*hd_p))) != sizeof(*hd_p) ){
		if( n<0 ) perror("read");
#ifdef LONG_64_BIT
		sprintf(DEFAULT_ERROR_STRING,"rd_ls_hdr (file %s):  wanted %ld bytes, read %d",
			filename,sizeof(*hd_p),n);
#else
		sprintf(DEFAULT_ERROR_STRING,"rd_ls_hdr (file %s):  wanted %d bytes, read %d",
			filename,sizeof(*hd_p),n);
#endif
		NWARN(DEFAULT_ERROR_STRING);
		return(-1);
	}
	return(0);
}

FIO_OPEN_FUNC( ls_open )
{
	Image_File *ifp;

	ifp = IMAGE_FILE_OPEN(name,rw,IFT_LUM);
	if( ifp==NO_IMAGE_FILE ) return(ifp);

	ifp->if_hd = getbuf( sizeof(Lumisys_Hdr) );

	if( IS_READABLE(ifp) ){
		/* BUG: should check for error here */

		/* An error can occur here if the file exists,
		 * but is empty...  we therefore initialize
		 * the header strings (above) with nulls, so
		 * that the program won't try to free nonsense
		 * addresses
		 */

		if( rd_ls_hdr( ifp->if_fd, ifp->if_hd,
			ifp->if_name ) < 0 ){
			ls_close(QSP_ARG  ifp);
			return(NO_IMAGE_FILE);
		}
		ls_to_dp(ifp->if_dp,ifp->if_hd);
	}
#ifdef CAUTIOUS
	else {
		NERROR1("CAUTIOUS:  shouldn't be able to write lumisys format!?");
	}
#endif /* CAUTIOUS */
	return(ifp);
}

FIO_CLOSE_FUNC( ls_close )
{
	if( ifp->if_hd != NULL )
		givbuf(ifp->if_hd);
	GENERIC_IMGFILE_CLOSE(ifp);
}

