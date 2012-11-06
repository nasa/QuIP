#include "quip_config.h"

char VersionId_fio_fioasc[] = QUIP_VERSION_STRING;

#include <stdio.h>

#ifdef HAVE_STRING_H
#include <string.h>
#endif

#include "fio_prot.h"
#include "filetype.h"
#include "getbuf.h"
#include "data_obj.h"
#include "debug.h"
#include "savestr.h"
#include "fioasc.h"

#define hdr	if_hd.asc_hd_p

FIO_OPEN_FUNC( asc_open )
{
	Image_File *ifp;

	ifp = IMAGE_FILE_OPEN(name,rw,IFT_ASC);
	if( ifp==NO_IMAGE_FILE ) return(ifp);

	if( IS_READABLE(ifp) ){
		ifp->if_dp->dt_comps = 0;
		ifp->if_dp->dt_cols = 0;
		ifp->if_dp->dt_rows = 0;
		ifp->if_dp->dt_frames = 0;
		ifp->if_dp->dt_seqs = 0;

		ifp->if_dp->dt_prec = PREC_ANY;
	}

	return(ifp);
}


FIO_CLOSE_FUNC( asc_close )
{
	GENERIC_IMGFILE_CLOSE(ifp);
}

FIO_WT_FUNC( asc_wt )
{
	if( ifp->if_dp == NO_OBJ ){	/* first time? */

		/* set the rows & columns in our file struct */
		setup_dummy(ifp);
		copy_dimensions(ifp->if_dp, dp);

	} else if( !same_type(QSP_ARG  dp,ifp) ) return(-1);

	/* Now write the data! */

	pntvec(QSP_ARG  dp,ifp->if_fp);

	return(0);
}

FIO_RD_FUNC( asc_rd )
{
	/* BUG how do we handle the offsets?? */
	read_ascii_data(QSP_ARG  dp, ifp->if_fp, ifp->if_name, 1 /* expect_exact_count */ );
}

int asc_unconv(void *hdr_pp,Data_Obj *dp)
{
	NWARN("asc_unconv not implemented");
	return(-1);
}

int asc_conv(Data_Obj *dp,void *hd_pp)
{
	NWARN("asc_conv not implemented");
	return(-1);
}

