
#include "quip_config.h"

#define COMPRESSION_NONE	foobar

#ifdef HAVE_MATIO

#include <stdio.h>
#ifdef HAVE_STRING_H
#include <string.h>
#endif

#include "quip_prot.h"
#include "query_bits.h"	// LLEN - BUG
#include "fio_prot.h"
#include "img_file/matio_api.h"
#include "data_obj.h"
#include "matio_private.h"
#include "img_file.h"

#define info_p	((matvar_t *) if_hdr_p )
#define HDR_P	((matvar_t *)ifp->if_hdr_p)

/* static int num_color=1; */
/* static void rewrite_mat_nf(TIFF* mat,dimension_t n); */

FIO_INFO_FUNC(mat)
{
	// This function is called after the generic information
	// about the file has been printed.  It is OK if there
	// is no format-specific info to display...

	//advise("Sorry, mat_info not implemented yet");
}

static Precision * prec_of_matlab_object(matvar_t *matp)
{
	switch(matp->data_type){
		case MAT_T_DOUBLE: return(PREC_FOR_CODE(PREC_DP));
		case MAT_T_SINGLE: return(PREC_FOR_CODE(PREC_SP));
		case MAT_T_INT8: return(PREC_FOR_CODE(PREC_BY));
		case MAT_T_INT16: return(PREC_FOR_CODE(PREC_IN));
		case MAT_T_INT32: return(PREC_FOR_CODE(PREC_DI));
		case MAT_T_INT64: return(PREC_FOR_CODE(PREC_LI));
		case MAT_T_UINT8: return(PREC_FOR_CODE(PREC_UBY));
		case MAT_T_UINT16: return(PREC_FOR_CODE(PREC_UIN));
		case MAT_T_UINT32: return(PREC_FOR_CODE(PREC_UDI));
		case MAT_T_UINT64: return(PREC_FOR_CODE(PREC_ULI));
		case MAT_T_MATRIX:
			NWARN("prec_of_matlab_object:  Need to handle matlab matrix type!?");
			return(NULL);
		case MAT_T_COMPRESSED:
			NWARN("prec_of_matlab_object:  Need to handle matlab compressed type!?");
			return(NULL);
		case MAT_T_STRING:
			NWARN("prec_of_matlab_object:  Need to handle matlab string type!?");
			return(NULL);
		case MAT_T_CELL:
			//NWARN("prec_of_matlab_object:  Need to handle matlab cell type!?");
			sprintf(DEFAULT_ERROR_STRING,
	"NOT handling matlab type CELL, variable %s",matp->name);
			NADVISE(DEFAULT_ERROR_STRING);
			return(NULL);
		case MAT_T_STRUCT:
			//NWARN("prec_of_matlab_object:  Need to handle matlab struct type!?");
			return(NULL);
		case MAT_T_ARRAY:
			NWARN("prec_of_matlab_object:  Need to handle matlab array type!?");
			return(NULL);
		case MAT_T_FUNCTION:
			NWARN("prec_of_matlab_object:  Need to handle matlab function type!?");
			return(NULL);
		case MAT_T_UNKNOWN:
			NWARN("prec_of_matlab_object:  Need to handle matlab unknown type!?");
			return(NULL);
		case MAT_T_UTF8:
		case MAT_T_UTF16:
		case MAT_T_UTF32:
			NWARN("Not sure what to do with matlab UTF data type!?");
			return(NULL);

		// Comment out the default case to get compiler warnings
		// about un-handled cases...
		default:
			sprintf(DEFAULT_ERROR_STRING,
		"prec_of_matlab_object:  unexpected data_type %d (0x%x)!?",
				matp->data_type,matp->data_type);
			NADVISE(DEFAULT_ERROR_STRING);
			return(NULL);
	}
	/* NOTREACHED */
	return(NULL);
}

static int describe_matlab_object(QSP_ARG_DECL  matvar_t *matp)
{
	int i;

	sprintf(msg_str,"Matlab object %s:\trank %d, ",matp->name,matp->rank);
	prt_msg_frag(msg_str);

	for(i=0;i<matp->rank;i++){
		if( i == 0 ) sprintf(msg_str,"\t%ld ",(long)matp->dims[0]);
		else {
			sprintf(msg_str,"x %ld ",(long)matp->dims[i]);
		}
		prt_msg_frag(msg_str);
	}

	strcpy(msg_str,"\n\tdata_type:  ");
	switch(matp->data_type){
		case MAT_T_DOUBLE: strcat(msg_str,"double "); break;
		case MAT_T_SINGLE: strcat(msg_str,"float"); break;
		case MAT_T_INT8: strcat(msg_str,"byte"); break;
		case MAT_T_INT16: strcat(msg_str,"short"); break;
		case MAT_T_INT32: strcat(msg_str,"int32"); break;
		case MAT_T_INT64: strcat(msg_str,"int64"); break;
		case MAT_T_UINT8: strcat(msg_str,"u_byte"); break;
		case MAT_T_UINT16: strcat(msg_str,"u_short"); break;
		case MAT_T_UINT32: strcat(msg_str,"uint32"); break;
		case MAT_T_UINT64: strcat(msg_str,"uint64"); break;
		case MAT_T_MATRIX: strcat(msg_str,"matrix"); break;
		case MAT_T_COMPRESSED: strcat(msg_str,"compressed"); break;
		case MAT_T_STRING: strcat(msg_str,"string"); break;
		case MAT_T_CELL: strcat(msg_str,"cell"); break;
		case MAT_T_STRUCT: strcat(msg_str,"struct"); break;
		case MAT_T_ARRAY: strcat(msg_str,"array"); break;
		case MAT_T_FUNCTION: strcat(msg_str,"function"); break;
		case MAT_T_UNKNOWN: strcat(msg_str,"unknown"); break;
		case MAT_T_UTF8: strcat(msg_str,"utf8"); break;
		case MAT_T_UTF16: strcat(msg_str,"utf16"); break;
		case MAT_T_UTF32: strcat(msg_str,"utf32"); break;

		// comment out the default case to get compiler
		// warnings about un-handled cases
		default:
			sprintf(DEFAULT_ERROR_STRING,
				"(unexpected matlab type code %d)",matp->data_type);
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
#ifdef MAT_C_EMPTY
		case MAT_C_EMPTY: strcat(msg_str," empty"); break;
#endif // MAT_C_EMPTY
	}
	prt_msg_frag(msg_str);

	sprintf(msg_str,"\n\t%d bytes/elt, %ld bytes total",
		matp->data_size,(long)matp->nbytes);
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

static Data_Obj *make_obj_for_matvar(QSP_ARG_DECL  matvar_t * matvar, matvar_t *parent, Precision *prec_p )
{
	Dimension_Set ds;
	Data_Obj *dp;
	dimension_t j,nelts;
	char name[LLEN];

	if( prec_p == NULL ) return NULL;

	/* We assume that the memory is already allocated
	 * when the variable is read in by libmatio.
	 */

	ds.ds_dimension[0]=1;
	ds.ds_dimension[1]=1;
	ds.ds_dimension[2]=1;
	ds.ds_dimension[3]=1;
	ds.ds_dimension[4]=1;

	if( parent == NULL ){
		if( strlen(matvar->name) >= LLEN )
			ERROR1("matvar name is too long!?");
		strcpy(name,matvar->name);
	} else {
		if( strlen(matvar->name)+strlen(parent->name)+1 >= LLEN )
			ERROR1("matvar structure element name is too long!?");
		sprintf(name,"%s.%s",parent->name,matvar->name);
	}
//fprintf(stderr,"%s:  rank is %d\n",matvar->name,matvar->rank);
	nelts=1;
	for(j=0;j<matvar->rank;j++){
		if( j>=(N_DIMENSIONS-1) ) ERROR1("Matlab rank is too high!?");
		// We boost the dimension in our object...
		ds.ds_dimension[j+1]=matvar->dims[j];
		nelts *= matvar->dims[j];
//fprintf(stderr,"dim[%d] = %ld\n",j,matvar->dims[j]);
	}
//fprintf(stderr,"total element count is %d\n",nelts);

	dp = _make_dp(QSP_ARG  name,&ds,prec_p);
	if( dp != NULL ){
		/* has the data already been read at this point? */
		SET_OBJ_DATA_PTR(dp, matvar->data);
	}
//longlist(QSP_ARG  dp);
	return dp;
}

static void handle_struct(QSP_ARG_DECL  matvar_t * matvar )
{
	// We should do something sensible
	// with structs...
	//
	// For now we make an assumption that
	// all of the sub-variables have
	// the same dimensions; we put
	// the fields into different components.

//fprintf(stderr,"Matlab variable %s is a structure...\n",matvar->name);
	char * const * names;
	unsigned int nfields;

	nfields = Mat_VarGetNumberOfFields(matvar);
	if( nfields == 0 ){
		WARN("structure has 0 fields!?");
	}

	// BUG?  undefined on ubuntu???
#ifdef FOO	// BUG should switch on libmatio version
	names = Mat_VarGetStructFieldnames(matvar);
#else
	ERROR1("Need to find correct implementation of Mat_VarGetStructFieldnames!?");
#endif
	

	if( names == NULL ){
WARN("Mat_VarGetStructFieldnames failed!?");
	} else {
		int i;
		matvar_t **v_array;
		v_array = (matvar_t **) matvar->data;
		for(i=0;i<nfields;i++){
			Precision * prec_p;
			Data_Obj *dp;

//fprintf(stderr,"\tfield %d:  %s\n",i,names[i]);
//fprintf(stderr,"\t\t\tvarname = %s\n",v_array[i]->name);
//fprintf(stderr,"\t\t\trank = %d\n",v_array[i]->rank);
//for(k=0;k<v_array[i]->rank;k++)
//fprintf(stderr,"\t\t\tdim[%d] = %ld\n",k,v_array[i]->dims[k]);
			prec_p = prec_of_matlab_object(v_array[i]);
			dp=make_obj_for_matvar(QSP_ARG  v_array[i],
							matvar,prec_p);
			if( dp == NULL ){
		//WARN("error making structure element!?");
				sprintf(ERROR_STRING,
					"NOT making data object for %s.%s\n",
					matvar->name,v_array[i]->name);
				advise(ERROR_STRING);
			}

		}
	}
}

/* shaping objects:
 * Hilda's data are 1x360x2, the 360 index runs faster than the two...
 */

FIO_OPEN_FUNC( mat )
{
	Image_File *ifp;

	// Make sure the file exists here...

	ifp = IMG_FILE_CREAT(name,rw,FILETYPE_FOR_CODE(IFT_MATLAB));
	if( ifp==NULL ) return(ifp);

	if( IS_READABLE(ifp) ){
		mat_t *mat;
		matvar_t *matvar;

		mat = Mat_Open(ifp->if_pathname,MAT_ACC_RDONLY);
		// BUG if the file doesn't exist?
		if( mat == NULL ){
			sprintf(ERROR_STRING,"Error opening file %s!?",ifp->if_pathname);
			WARN(ERROR_STRING);
			// need to deallocate ifp...
			mat_close(QSP_ARG  ifp);
			return NULL;
		}
		matvar = Mat_VarReadNext(mat);		/* read next object */
		/* BUG - resets ifp->if_dp for every object!? */
		while( matvar != NULL ){
			Data_Obj *dp;
			Precision * prec_p;

			ifp->if_hdr_p = matvar;
			if( verbose )
				describe_matlab_object(QSP_ARG  ifp->if_hdr_p);

			/* does ifp already own an object? */

			prec_p = prec_of_matlab_object(matvar);
			if( prec_p != NULL ){
				dp = make_obj_for_matvar(QSP_ARG  matvar,
								NULL,prec_p);
				ifp->if_dp=dp;
			} else {
				ifp->if_dp=NULL;
				if( matvar->data_type == MAT_T_STRUCT ){
					handle_struct(QSP_ARG  matvar);
				} else {
					sprintf(ERROR_STRING,
			"mat_open:  Not making object %s, unhandled type",
						matvar->name);
					advise(ERROR_STRING);
				}
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


FIO_CLOSE_FUNC( mat )
{
	/* can we write multiple frames to mat??? */

	GENERIC_IMGFILE_CLOSE(ifp);
}

FIO_DP_TO_FT_FUNC(mat,/*matvar_t*/ Matio_Hdr )
{

	/* COMPRESSION_LZW not available due to Unisys patent enforcement */
	/* uint16 comp=COMPRESSION_LZW; */


	/* num_frame set when when write request given */

	return(0);
}

FIO_SETHDR_FUNC( mat )
{
	if( FIO_DP_TO_FT_FUNC_NAME(mat)(ifp->if_hdr_p,ifp->if_dp) < 0 ){
		mat_close(QSP_ARG  ifp);
		return(-1);
	}
	return(0);
}

FIO_WT_FUNC( mat )
{
	if( ifp->if_dp == NULL ){	/* first time? */

		/* set the rows & columns in our file struct */
		setup_dummy(ifp);
		copy_dimensions(ifp->if_dp, dp);

		SET_OBJ_FRAMES(ifp->if_dp, ifp->if_frms_to_wt);
		SET_OBJ_SEQS(ifp->if_dp, 1);

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

FIO_RD_FUNC(  mat )
{
	if( x_offset != 0 || y_offset != 0  || t_offset != 0 ){
		sprintf(ERROR_STRING,"mat_rd %s:  Sorry, don't know how to handle non-zero offsets",
			ifp->if_name);
		WARN(ERROR_STRING);
		return;
	}

	/*
sprintf(ERROR_STRING,"mat_rd:  reading %ld elements of size %d",
dp->dt_nelts,siztbl[MACHINE_PREC(dp)]);
advise(ERROR_STRING);
	if( fread(dp->dt_data,siztbl[MACHINE_PREC(dp)],dp->dt_nelts,ifp->if_fp) != dp->dt_nelts ){
		sprintf(ERROR_STRING,"mat_rd %s:  error reading data",ifp->if_name);
		WARN(ERROR_STRING);
	}
	*/
	/* BUG check sizes... */
	dp_copy(QSP_ARG  dp,ifp->if_dp);
}

FIO_UNCONV_FUNC( mat )
{
	NWARN("mat_unconv not implemented");
	return(-1);
}

FIO_CONV_FUNC( mat )
{
	NWARN("mat_conv not implemented");
	return(-1);
}

FIO_SEEK_FUNC( mat )
{
	WARN("mat_seek not implemented");
	return(-1);
}

#endif /* HAVE_MATIO */

