
#include "quip_config.h"

char VersionId_fio_matio[] = QUIP_VERSION_STRING;

int force_matio_load;		/* a hack so that this file (and its version string)
				 * are loaded even when HAVE_MATIO is not defined,
				 * and we don't need it.  This is so that the version
				 * strings don't cause a mismatch (and confusion
				 * over the need to rebuild).
				 */


#ifdef HAVE_MATIO

#include <stdio.h>
#ifdef HAVE_STRING_H
#include <string.h>
#endif

#include "fio_prot.h"
#include "filetype.h"
#include "matio_api.h"
#include "getbuf.h"
#include "data_obj.h"
#include "debug.h"
#include "savestr.h"
#include "matio.h"
#include "img_file.h"

#define info_p	if_hd.matvar_p
#define HDR_P	((Image_File_Hdr *)ifp->if_hd)->ifh_u.matvar_p

/* static int num_color=1; */
/* static void rewrite_mat_nf(TIFF* mat,dimension_t n); */
static prec_t prec_of_matlab_object(matvar_t *matp);

void mat_info(QSP_ARG_DECL  Image_File *ifp)
{
	WARN("Sorry, mat_info not implemented yet");
}

static prec_t prec_of_matlab_object(matvar_t *matp)
{
	switch(matp->data_type){
		case MAT_T_DOUBLE: return(PREC_DP);
		case MAT_T_SINGLE: return(PREC_SP);
		case MAT_T_INT8: return(PREC_BY);
		case MAT_T_INT16: return(PREC_IN);
		case MAT_T_INT32: return(PREC_DI);
		case MAT_T_UINT8: return(PREC_UBY);
		case MAT_T_UINT16: return(PREC_UIN);
		case MAT_T_UINT32: return(PREC_UDI);
		case MAT_T_MATRIX: return(BAD_PREC);
		case MAT_T_COMPRESSED: return(BAD_PREC);
		case MAT_T_STRING: return(BAD_PREC);
		case MAT_T_CELL: return(BAD_PREC);
		case MAT_T_STRUCT: return(BAD_PREC);
		case MAT_T_ARRAY: return(BAD_PREC);
		case MAT_T_FUNCTION: return(BAD_PREC);
		case MAT_T_UNKNOWN: return(BAD_PREC);
		default: return(BAD_PREC);
	}
	/* NOTREACHED */
	return(BAD_PREC);
}

int describe_matlab_object(matvar_t *matp)
{
	int i;

	sprintf(msg_str,"Matlab object %s:\trank %d, ",matp->name,matp->rank);
	prt_msg_frag(msg_str);

	for(i=0;i<matp->rank;i++){
		if( i == 0 ) sprintf(msg_str,"\t%d ",matp->dims[0]);
		else {
			sprintf(msg_str,"x %d ",matp->dims[i]);
		}
		prt_msg_frag(msg_str);
	}

	strcpy(msg_str,"\n\tdata_type:  ");
	switch(matp->data_type){
		case MAT_T_DOUBLE: strcat(msg_str,"double "); break;
		case MAT_T_SINGLE: strcat(msg_str,"float"); break;
		case MAT_T_INT8: strcat(msg_str,"byte"); break;
		case MAT_T_INT16: strcat(msg_str,"short"); break;
		case MAT_T_INT32: strcat(msg_str,"long"); break;
		case MAT_T_UINT8: strcat(msg_str,"u_byte"); break;
		case MAT_T_UINT16: strcat(msg_str,"u_short"); break;
		case MAT_T_UINT32: strcat(msg_str,"u_long"); break;
		case MAT_T_MATRIX: strcat(msg_str,"matrix"); break;
		case MAT_T_COMPRESSED: strcat(msg_str,"compressed"); break;
		case MAT_T_STRING: strcat(msg_str,"string"); break;
		case MAT_T_CELL: strcat(msg_str,"cell"); break;
		case MAT_T_STRUCT: strcat(msg_str,"struct"); break;
		case MAT_T_ARRAY: strcat(msg_str,"array"); break;
		case MAT_T_FUNCTION: strcat(msg_str,"function"); break;
		case MAT_T_UNKNOWN: strcat(msg_str,"unknown"); break;
		default:
			sprintf(DEFAULT_ERROR_STRING,"(strange matlab type code %d)",matp->data_type);
			strcat(msg_str,DEFAULT_ERROR_STRING);
			break;
	}
	prt_msg_frag(msg_str);

	strcpy(msg_str,"\n\tclass type:  ");
	switch(matp->class_type){
		case MAT_C_CELL: strcat(msg_str," cell array"); break;
		case MAT_C_STRUCT: strcat(msg_str," structure"); break;
		case MAT_C_OBJECT: strcat(msg_str," object"); break;
		case MAT_C_CHAR: strcat(msg_str," char"); break;
		case MAT_C_SPARSE: strcat(msg_str," sparse"); break;
		case MAT_C_DOUBLE: strcat(msg_str," double"); break;
		case MAT_C_SINGLE: strcat(msg_str," float"); break;
		case MAT_C_INT8: strcat(msg_str," byte"); break;
		case MAT_C_UINT8: strcat(msg_str," u_byte"); break;
		case MAT_C_INT16: strcat(msg_str," short"); break;
		case MAT_C_UINT16: strcat(msg_str," u_short"); break;
		case MAT_C_INT32: strcat(msg_str," long"); break;
		case MAT_C_UINT32: strcat(msg_str," u_long"); break;
		case MAT_C_INT64: strcat(msg_str," int64"); break;
		case MAT_C_UINT64: strcat(msg_str," uint64"); break;
		case MAT_C_FUNCTION: strcat(msg_str," function"); break;
	}
	prt_msg_frag(msg_str);

	sprintf(msg_str,"\n\t%d bytes/elt, %d bytes total",matp->data_size,matp->nbytes);
	prt_msg_frag(msg_str);

	if( matp->isComplex )
		prt_msg_frag(" (complex)");
	else
		prt_msg_frag(" (real)");

	prt_msg("");

	//sprintf(DEFAULT_ERROR_STRING,"\tdata addr = 0x%lx",(u_long)matp->data);
	//advise(DEFAULT_ERROR_STRING);

	return(0);
}

/* shaping objects:
 * Hilda's data are 1x360x2, the 360 index runs faster than the two...
 */

FIO_OPEN_FUNC( mat_open )
{
	Image_File *ifp;

	ifp = IMAGE_FILE_OPEN(name,rw,IFT_MATLAB);
	if( ifp==NO_IMAGE_FILE ) return(ifp);

	if( IS_READABLE(ifp) ){
		mat_t *mat;
		matvar_t *matvar;

		mat = Mat_Open(ifp->if_pathname,MAT_ACC_RDONLY);

		matvar = Mat_VarReadNext(mat);		/* read next object */
		/* BUG - resets ifp->if_dp for every object!? */
		while( matvar != NULL ){
			Data_Obj *dp;
			prec_t prec;
			Dimension_Set ds;
			int j;

			ifp->if_hd = matvar;
			if( verbose )
				describe_matlab_object(ifp->if_hd);

			/* does ifp already own an object? */
			ds.ds_dimension[0]=1;
			ds.ds_dimension[1]=1;
			ds.ds_dimension[2]=1;
			ds.ds_dimension[3]=1;
			ds.ds_dimension[4]=1;

			for(j=0;j<matvar->rank;j++){
				ds.ds_dimension[j]=matvar->dims[j];
			}

			prec = prec_of_matlab_object(matvar);
			if( prec != BAD_PREC ){
				//dp = make_dobj(matvar->name,&ds,prec);
				/* Do we need to allocate the memory, or does matio lib do this for us? */
				dp = _make_dp(QSP_ARG  matvar->name,&ds,prec);
				if( dp != NO_OBJ ){
					/* has the data already been read at this point? */
					dp->dt_data = matvar->data;
					ifp->if_dp=dp;
				}
			} else {
				sprintf(error_string,"Not making object %s, bad precision",matvar->name);
				WARN(error_string);
			}

			matvar = Mat_VarReadNext(mat);		/* read next object */
		}
	}
#ifdef FOO
	else {
		advise("Sorry no matlab writes (yet)");
	}
#endif
	return(ifp);
}


FIO_CLOSE_FUNC( mat_close )
{
	/* can we write multiple frames to mat??? */

	GENERIC_IMGFILE_CLOSE(ifp);
}

int dp_to_mat(matvar_t *matp,Data_Obj *dp)
{

	/* COMPRESSION_LZW not available due to Unisys patent enforcement */
	/* uint16 comp=COMPRESSION_LZW; */


	/* num_frame set when when write request given */

	return(0);
}

FIO_SETHDR_FUNC( set_mat_hdr )
{
	if( dp_to_mat(ifp->if_hd,ifp->if_dp) < 0 ){
		mat_close(QSP_ARG  ifp);
		return(-1);
	}
	return(0);
}

FIO_WT_FUNC( mat_wt )
{
	if( ifp->if_dp == NO_OBJ ){	/* first time? */

		/* set the rows & columns in our file struct */
		setup_dummy(ifp);
		copy_dimensions(ifp->if_dp, dp);

		ifp->if_dp->dt_frames = ifp->if_frms_to_wt;
		ifp->if_dp->dt_seqs = 1;

		if( set_mat_hdr(QSP_ARG  ifp) < 0 ) return(-1);

	} else if( !same_type(QSP_ARG  dp,ifp) ) return(-1);

	/* now write the data */
#ifdef FOOBAR
// We get an error from TIFFWriteScanline about setting PlanarConfig...
	datap = dp->dt_data;
	for(row=0;row<dp->dt_rows;row++){
		if( TIFFWriteScanline(ifp->if_mat,datap,row,0) != 1 )
			WARN("error writing TIFF scanline");
		datap += siztbl[MACHINE_PREC(dp)] * dp->dt_rowinc;
	}
#endif /* FOOBAR */

	ifp->if_nfrms ++ ;
	check_auto_close(QSP_ARG  ifp);
	return(0);
}

FIO_RD_FUNC(  mat_rd )
{
	if( x_offset != 0 || y_offset != 0  || t_offset != 0 ){
		sprintf(error_string,"mat_rd %s:  Sorry, don't know how to handle non-zero offsets",
			ifp->if_name);
		WARN(error_string);
		return;
	}

	/*
sprintf(error_string,"mat_rd:  reading %ld elements of size %d",
dp->dt_nelts,siztbl[MACHINE_PREC(dp)]);
advise(error_string);
	if( fread(dp->dt_data,siztbl[MACHINE_PREC(dp)],dp->dt_nelts,ifp->if_fp) != dp->dt_nelts ){
		sprintf(error_string,"mat_rd %s:  error reading data",ifp->if_name);
		WARN(error_string);
	}
	*/
	/* BUG check sizes... */
	dp_copy(QSP_ARG  dp,ifp->if_dp);
}

int mat_unconv(void *hdr_pp,Data_Obj *dp)
{
	NWARN("mat_unconv not implemented");
	return(-1);
}

int mat_conv(Data_Obj *dp,void *hd_pp)
{
	NWARN("mat_conv not implemented");
	return(-1);
}

#endif /* HAVE_MATIO */

