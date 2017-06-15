
#include "quip_config.h"

#include <stdio.h>

#ifdef HAVE_STRING_H
#include <string.h>
#endif

#ifdef HAVE_STDLIB_H
#include <stdlib.h>
#endif

#include "quip_prot.h"
#include "fio_prot.h"
#include "fio_api.h"
#include "data_obj.h"
#include "img_file/raw.h"
//#include "img_file/uio.h"
//#include "img_file/vl.h"		// BUG need to add this for prototypes??

void vl_close(QSP_ARG_DECL  Image_File *ifp)
{
	GENERIC_IMGFILE_CLOSE(ifp);
}

Image_File *		/**/
vl_open(QSP_ARG_DECL  const char *name,int rw)		/**/
{
	Image_File *ifp;

#ifdef QUIP_DEBUG
if( debug ) advise("opening image file");
#endif /* QUIP_DEBUG */

	ifp = IMG_FILE_CREAT(name,rw,FILETYPE_FOR_CODE(IFT_VL) );

	/* img_file_creat creates dummy if_dp only if readable */

	if( ifp==NULL ) return(ifp);

#ifdef QUIP_DEBUG
if( debug ) advise("allocating hips header");
#endif /* QUIP_DEBUG */

	if( rw == FILE_READ ){
		char string[LLEN];
		char word1[LLEN];
		char word2[LLEN];
		int at_end=0;

		SET_OBJ_SEQS(ifp->if_dp,1);
		SET_OBJ_FRAMES(ifp->if_dp,1);
		SET_OBJ_ROWS(ifp->if_dp,1);
		SET_OBJ_COLS(ifp->if_dp,1);
		SET_OBJ_COMPS(ifp->if_dp,1);
		SET_OBJ_PREC_PTR(ifp->if_dp, PREC_FOR_CODE(PREC_BY) );

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
					SET_OBJ_COLS(ifp->if_dp, atoi(word2) );
				} else if( ! strcmp(word1,"VDIM:") ){
					SET_OBJ_ROWS(ifp->if_dp, atoi(word2) );
				} else if( verbose ){
					sprintf(ERROR_STRING,
				"Ignoring header field %s",word1);
					advise(ERROR_STRING);
				}
			}
		}
		if( OBJ_ROWS(ifp->if_dp) == 1 || OBJ_COLS(ifp->if_dp) == 1 )
			WARN("Image dimension(s) may not have been set from header");
	} else {
		WARN("Sorry, can't write VL files");
dun:
		vl_close(QSP_ARG  ifp);
		return(NULL);
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

