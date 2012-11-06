
#include "quip_config.h"

char VersionId_fio_vl[] = QUIP_VERSION_STRING;

#include <stdio.h>

#ifdef HAVE_STRING_H
#include <string.h>
#endif

#ifdef HAVE_STDLIB_H
#include <stdlib.h>
#endif

#include "fio_prot.h"
#include "fio_api.h"
#include "filetype.h"
#include "getbuf.h"
#include "data_obj.h"
#include "debug.h"
#include "savestr.h"
#include "raw.h"
#include "uio.h"
#include "vl.h"

void vl_close(QSP_ARG_DECL  Image_File *ifp)
{
	GENERIC_IMGFILE_CLOSE(ifp);
}

Image_File *		/**/
vl_open(QSP_ARG_DECL  const char *name,int rw)		/**/
{
	Image_File *ifp;

#ifdef DEBUG
if( debug ) advise("opening image file");
#endif /* DEBUG */

	ifp = IMAGE_FILE_OPEN(name,rw,IFT_VL);

	/* image_file_open creates dummy if_dp only if readable */

	if( ifp==NO_IMAGE_FILE ) return(ifp);

#ifdef DEBUG
if( debug ) advise("allocating hips header");
#endif /* DEBUG */

	if( rw == FILE_READ ){
		char string[LLEN];
		char word1[LLEN];
		char word2[LLEN];
		int at_end=0;

		ifp->if_dp->dt_seqs=
		ifp->if_dp->dt_frames=
		ifp->if_dp->dt_rows=
		ifp->if_dp->dt_cols=
		ifp->if_dp->dt_comps=1;
		ifp->if_dp->dt_prec = PREC_BY;

		/* read strings until line with . encountered */
		while( !at_end ){
			if( fgets(string,LLEN,ifp->if_fp) == NULL ){
				WARN("bad VL header line");
				goto dun;
			}
			if( string[0]=='.' ) at_end=1;
			else {
				if( sscanf(string,"%s %s",word1,word2) != 2 ){
					WARN("bad VL header line format");
					goto dun;
				}
				if( !strcmp(word1,"UDIM:") ){
					ifp->if_dp->dt_cols = atoi(word2);
				} else if( ! strcmp(word1,"VDIM:") ){
					ifp->if_dp->dt_rows = atoi(word2);
				} else if( verbose ){
					sprintf(error_string,
				"Ignoring header field %s",word1);
					advise(error_string);
				}
			}
		}
		if( ifp->if_dp->dt_rows == 1 || ifp->if_dp->dt_cols == 1 )
			WARN("Image dimension(s) may not have been set from header");
	} else {
		WARN("Sorry, can't write VL files");
dun:
		vl_close(QSP_ARG  ifp);
		return(NO_IMAGE_FILE);
	}
	return(ifp);
}

int vl_unconv(void *hdr_pp,Data_Obj *dp)
{
	NWARN("vl_unconv not implemented");
	return(-1);
}

int vl_conv(Data_Obj *dp,void *hd_pp)
{
	NWARN("vl_conv not implemented");
	return(-1);
}

