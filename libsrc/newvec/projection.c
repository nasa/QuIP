#include "quip_config.h"

char VersionId_newvec_projection[] = QUIP_VERSION_STRING;

#include "nvf.h"

/* Here we use the term "projection" to describe applying an operation
 * that has a single scalar return value to the rows, or columns, (or
 * whatever) of the source object.  Originally we only allowed a single
 * scalar for the destination object, and ran the function over the whole
 * source object, but now we infer the dimensions over which to search
 * from the shapes of the input and output.  The logic is similar to
 * that in the generalized outer product, the output shape must be legal
 * for an outer product op with the source object...
 */

#define ADVANCE_DATA_PTR(sub_dp,parent_dp)					\
										\
	cp = sub_dp->dt_data;							\
	cp += parent_dp->dt_increment[i] * siztbl[ MACHINE_PREC(parent_dp) ];	\
	sub_dp->dt_data = cp;

#define SUB_ITERATE( dst_dp, src_dp )							\
											\
	Dimension_Set dst_sizes;							\
	Dimension_Set src_sizes;							\
	doff_t src_offsets[N_DIMENSIONS]={0,0,0,0,0};					\
	doff_t dst_offsets[N_DIMENSIONS]={0,0,0,0,0};					\
	incr_t src_incrs[N_DIMENSIONS]={1,1,1,1,1};					\
	incr_t dst_incrs[N_DIMENSIONS]={1,1,1,1,1};					\
	Data_Obj *sub_dst_dp, *sub_src_dp;						\
	char tmp_dst_name[LLEN];							\
	char tmp_src_name[LLEN];							\
											\
	/* iterate over src_dp */							\
	dst_sizes=dst_dp->dt_dimset;							\
	dst_sizes.ds_dimension[i]=1;							\
	sprintf(tmp_dst_name,"dsb.%s",dst_dp->dt_name);					\
	sub_dst_dp=make_subsamp(tmp_dst_name,dst_dp,&dst_sizes,dst_offsets,dst_incrs);	\
											\
	src_sizes=src_dp->dt_dimset;							\
	src_sizes.ds_dimension[i]=1;							\
	sprintf(tmp_src_name,"ssb.%s",src_dp->dt_name);					\
	sub_src_dp=make_subsamp(tmp_src_name,src_dp,&src_sizes,src_offsets,src_incrs);	\
											\
	for(j=0;j<src_dp->dt_dimension[i];j++){						\
		char *cp;								\
											\
		/* now the objects all have matching sizes in THIS dimension! */	\
		project_op(sub_dst_dp,sub_src_dp,code);					\
											\
		/* now step the offset... */						\
		ADVANCE_DATA_PTR(sub_dst_dp,dst_dp)					\
		ADVANCE_DATA_PTR(sub_src_dp,src_dp)					\
	}										\
	delvec(sub_dst_dp);								\
	delvec(sub_src_dp);


int project_op(QSP_ARG_DECL  Data_Obj *dst_dp, Data_Obj *src_dp, Vec_Func_Code code)
{
	int i,j;
	Vec_Obj_Args oargs;

	/* find the first dimension we have to iterate over */

	for(i=N_DIMENSIONS-1;i>=0;i--){
		if( dst_dp->dt_dimension[i] != 1 ){
			if( src_dp->dt_dimension[i] == dst_dp->dt_dimension[i] ){
				SUB_ITERATE(dst_dp,src_dp)
				return(0);	/* all the work done in recursion */
			} else {
				sprintf(error_string,
	"project_op:  %s size mismatch between source %s (%ld) and destination %s (%ld)",
					dimension_name[i],
					src_dp->dt_name,src_dp->dt_dimension[i],
					dst_dp->dt_name,dst_dp->dt_dimension[i]);
				NWARN(error_string);
				return(-1);
			}
		}
	}
	/* now we know ALL dimensions match.
	 * Call the appropriate vector-vector op.
	 */

	oargs.oa_4 = oargs.oa_3 = oargs.oa_2 = oargs.oa_bmap =
		oargs.oa_s2 = NO_OBJ;
	oargs.oa_1 = src_dp;
	oargs.oa_dest = oargs.oa_s1 = dst_dp;

	perf_vfunc(QSP_ARG  code,&oargs);

	return(0);
	/* BUG?  we should be able to bail out if there is an error in perf_vfunc! */
}



void do_projection(Vec_Func_Code code)
{
	Data_Obj *src_dp, *dst_dp;

	src_dp = pick_obj("source object");
	dst_dp = pick_obj("destination object");

	if( src_dp == NO_OBJ || dst_dp == NO_OBJ )
		return;

	/* here is where we should make sure the precisions match... */
	if( src_dp->dt_prec != dst_dp->dt_prec ){
		if( ((vec_func_tbl[code].vf_flags & LONG_RESULT_SCALAR) == 0 )
			|| ((MACHINE_PREC(dst_dp)!=PREC_DI)&&(MACHINE_PREC(dst_dp)!=PREC_UDI)) ){

			/* this is not ok */
			sprintf(error_string,"do_projection %s:  precision mismatch between destination %s (%s) and source %s (%s)",
				vec_func_tbl[code].vf_name,
				dst_dp->dt_name,name_for_prec(dst_dp->dt_prec),
				src_dp->dt_name,name_for_prec(src_dp->dt_prec) );
			NWARN(error_string);
			return;
		}
	}

	project_op(dst_dp,src_dp,code);
}

