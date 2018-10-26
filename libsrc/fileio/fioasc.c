#include "quip_config.h"

#include <stdio.h>

#ifdef HAVE_STRING_H
#include <string.h>
#endif

#include "quip_prot.h"
#include "fio_prot.h"
#include "data_obj.h"
//#include "img_file/fioasc.h"

//#define hdr	if_hdr_p.asc_hd_p

FIO_OPEN_FUNC( ascii )
{
	Image_File *ifp;

	ifp = img_file_creat(name,rw,FILETYPE_FOR_CODE(IFT_ASC));
	if( ifp==NULL ) return(ifp);

	if( IS_READABLE(ifp) ){
		SET_OBJ_COMPS(ifp->if_dp, 0);
		SET_OBJ_COLS(ifp->if_dp, 0);
		SET_OBJ_ROWS(ifp->if_dp, 0);
		SET_OBJ_FRAMES(ifp->if_dp, 0);
		SET_OBJ_SEQS(ifp->if_dp, 0);

		SET_OBJ_PREC_PTR(ifp->if_dp, PREC_FOR_CODE(PREC_ANY));
	}

	return(ifp);
}


FIO_CLOSE_FUNC( ascii )
{
	generic_imgfile_close(ifp);
}

FIO_WT_FUNC( ascii )
{
	if( ifp->if_dp == NULL ){	/* first time? */

		/* set the rows & columns in our file struct */
		setup_dummy(ifp);
		copy_dimensions(ifp->if_dp, dp);

	} else if( !same_type(dp,ifp) ) return(-1);

	/* Now write the data! */

	pntvec(dp,ifp->if_fp);

	return(0);
}

FIO_RD_FUNC( ascii )
{
	/* BUG how do we handle the offsets?? */
	read_ascii_data_from_file(dp, ifp->if_fp, ifp->if_name, 1 /* expect_exact_count */ );
}

FIO_SEEK_FUNC( ascii )
{
	return std_seek_frame( ifp, n );
}

FIO_INFO_FUNC( ascii )
{
	// nop
}

FIO_UNCONV_FUNC( ascii )
{
	warn("asc_unconv not implemented");
	return(-1);
}

FIO_CONV_FUNC( ascii )
{
	warn("asc_conv not implemented");
	return(-1);
}

