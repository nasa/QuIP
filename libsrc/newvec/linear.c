/* This file contains the stuff for linear algebra that we want to implement for
 * both single and double precision.
 */

/* Because this file is included, it can't have a version string */

int OBJ_METHOD_NAME(arg_chk)(Data_Obj *dpto, Data_Obj *dpfr, Data_Obj *xform, const char *func_name)
{
	if( MACHINE_PREC(dpto) != REQUIRED_DST_PREC ){
		sprintf(DEFAULT_ERROR_STRING,"target object %s (%s) must have %s precision for %s",
			dpto->dt_name,name_for_prec(dpto->dt_prec), name_for_prec(REQUIRED_DST_PREC),func_name);
		NWARN(DEFAULT_ERROR_STRING);
		return(-1);
	}
	if( MACHINE_PREC(dpfr) != REQUIRED_SRC_PREC ){
		sprintf(DEFAULT_ERROR_STRING,"source object %s (%s) must have %s precision for %s",
			dpfr->dt_name,name_for_prec(dpfr->dt_prec), name_for_prec(REQUIRED_SRC_PREC),func_name);
		NWARN(DEFAULT_ERROR_STRING);
		return(-1);
	}
	if( MACHINE_PREC(xform) != REQUIRED_SRC_PREC ){
		sprintf(DEFAULT_ERROR_STRING,"matrix object %s (%s) must have %s precision for %s",
			xform->dt_name,name_for_prec(xform->dt_prec),name_for_prec(REQUIRED_SRC_PREC),func_name);
		NWARN(DEFAULT_ERROR_STRING);
		return(-1);
	}
	if( dpto == dpfr ){
		/* BUG this test is not sufficiently rigorous...  it could be that the destination is
		 * an alias of some sort for the source, having a different header structure but pointing
		 * to the same data!?
		 */
		sprintf(DEFAULT_ERROR_STRING,"arg_chk:  destination (%s) must be distinct from source (%s)",
			dpto->dt_name,dpfr->dt_name);
		NWARN(DEFAULT_ERROR_STRING);
		return(-1);
	}
	return(0);
}


/*
 * apply a matrix to a list of elements
 *
 * The transformation consists of a series of dot products, but depending on the shapes
 * of the objects we want to vectorize differently.  One case is a big matrix transforming
 * a (single?) big vector - in this case we want to use vdot to compute each element of
 * the output (xform_list).  But this is very inefficient when we are transforming a big list of coordinates
 * with a small 3x3 or 4x4 matrix.  In this case we use vector operations to compute
 * all the dot products in parallel (vec_xform).
 *
 * Originally we assumed that the transformation was a matrix (rows and colums only
 * non-1 dimensions), and that the objects to be transformed were column vectors (xform_list).
 * But for vec_xform, the input vectors run along tdim (not row vectors!).
 *
 * We ought to have a general purpose way to indicate which dimensions should be dotted...
 * But for now we just try to reproduce the old system.
 *
 */
/* this routine vectorizes the dot products;
	good for big matrices or short lists */

/* there should be a better routine for long lists of short elts. */

void OBJ_METHOD_NAME(xform_list)(QSP_ARG_DECL  Data_Obj *dpto,Data_Obj *dpfr,Data_Obj *xform_dp)
{
	u_long i,j,k;
	Vec_Obj_Args oargs;
	Data_Obj *sub_dst_dp, *sub_xf_dp, *xf_row_dp, *sub_src_dp;
	Dimension_Set sizes, row_dimset={{1,1,1,1,1}};
	index_t offsets[N_DIMENSIONS]={0,0,0,0,0};
	index_t dst_indices[N_DIMENSIONS]={0,0,0,0,0};
	index_t src_indices[N_DIMENSIONS]={0,0,0,0,0};
	index_t xf_indices[N_DIMENSIONS]={0,0,0,0,0};

	if( xform_chk(dpto,dpfr,xform_dp) == -1 )
		return;

	if( OBJ_METHOD_NAME(arg_chk)(dpto,dpfr,xform_dp,TYPED_STRING(xform_list)) < 0 )
		return;

#ifdef CAUTIOUS
	clear_obj_args(&oargs);
#endif /* CAUTIOUS */

	if( IS_COMPLEX(dpto) )	oargs.oa_argstype = COMPLEX_ARGS;
	else			oargs.oa_argstype = REAL_ARGS;

	sizes = dpto->dt_type_dimset;
	sizes.ds_dimension[2]=1;				/* select one component */
	sizes.ds_dimension[1]=1;
	sizes.ds_dimension[0]=1;
	sub_dst_dp = mk_subseq(QSP_ARG  "_sub_dst",dpto,offsets,&sizes);	/* xform_list */

	sizes = xform_dp->dt_type_dimset;
	sizes.ds_dimension[2] = 1;					/* select one row */
	sub_xf_dp = mk_subseq(QSP_ARG  "_sub_xf",xform_dp,offsets,&sizes);	/* one row */
	row_dimset.ds_dimension[0] = sub_xf_dp->dt_cols;
	xf_row_dp = make_equivalence(QSP_ARG  "_xf_row",sub_xf_dp,&row_dimset,sub_xf_dp->dt_prec);

	sizes = dpfr->dt_type_dimset;
	sizes.ds_dimension[2]=1;				/* select one pixel */
	sizes.ds_dimension[1] = 1;
	sub_src_dp = mk_subseq(QSP_ARG  "_sub_src",dpfr,offsets,&sizes);

	oargs.oa_dest = sub_dst_dp;
	oargs.oa_1 = sub_src_dp;
	oargs.oa_2 = xf_row_dp;
	oargs.oa_argsprec = ARGSET_PREC( MACHINE_PREC(dpto) );

	for(i=0;i<dpto->dt_rows;i++){
		dst_indices[2]=i;		/* i_th row */
		src_indices[2]=i;		/* i_th row */
		for(j=0;j<dpto->dt_cols;j++){
			dst_indices[1]=j;		/* j_th column */
			src_indices[1]=j;		/* j_th column */
			sub_src_dp->dt_data = multiply_indexed_data(dpfr,src_indices);
			for(k=0;k<dpto->dt_comps;k++){
				dst_indices[0]=k;		/* k_th component */
				xf_indices[2]=k;		/* k_th row of xform matrix */
				sub_dst_dp->dt_data = multiply_indexed_data(dpto,dst_indices);
				xf_row_dp->dt_data = multiply_indexed_data(xform_dp,xf_indices);
				vdot(&oargs);
			}
		}
	}

	delvec(QSP_ARG  sub_dst_dp);
	delvec(QSP_ARG  sub_src_dp);
	delvec(QSP_ARG  sub_xf_dp);
}

/* like xform_list(), but vectorizes over list instead of matrix row.
 * good for long lists of short vectors, prototypical examples
 * are using a matrix to transform color images between color spaces,
 * or geometric transformations of arrays of points or vectors.
 */

void OBJ_METHOD_NAME(vec_xform)(QSP_ARG_DECL  Data_Obj *dpto,Data_Obj *dpfr,Data_Obj *xform_dp)
{
	Data_Obj *coeff_dp, *sub_src_dp, *sub_dst_dp, *tmp_dst_dp;
	dimension_t i,j;
	Vec_Obj_Args oargs1,oargs2,oargs3;
	Dimension_Set sizes;
	index_t offsets[N_DIMENSIONS]={0,0,0,0,0};
	index_t xform_indices[N_DIMENSIONS]={0,0,0,0,0};
	index_t src_indices[N_DIMENSIONS]={0,0,0,0,0};
	index_t dst_indices[N_DIMENSIONS]={0,0,0,0,0};

	if( xform_chk(dpto,dpfr,xform_dp) == -1 )
		return;

	if( OBJ_METHOD_NAME(arg_chk)(dpto,dpfr,xform_dp,TYPED_STRING(vec_xform)) < 0 )
		return;

	/* make a temporary output component */
	/* like the destination except for tdim */

	coeff_dp = mk_subimg(QSP_ARG  xform_dp, 0, 0, "_xform_coeff",1,1);	/* 1x1 subimage at 0 0 */

	sizes = dpto->dt_type_dimset;
	sizes.ds_dimension[0] = 1;			/* single component */
	sub_dst_dp = mk_subseq(QSP_ARG  "_sub_dst",dpto,offsets,&sizes);	/* vec_xform - RxC image w/ one component */
	sub_src_dp = mk_subseq(QSP_ARG  "_sub_src",dpfr,offsets,&sizes);
	tmp_dst_dp = make_dobj(QSP_ARG  "_tmp_dst",&sub_dst_dp->dt_type_dimset,sub_dst_dp->dt_prec);

	if( sub_dst_dp == NO_OBJ || sub_src_dp == NO_OBJ || tmp_dst_dp == NO_OBJ ){
		NWARN("error creating temporary object for vec_xform");
		return;
	}

	/* BUG should we check for mixed types here???  */
	if( IS_COMPLEX(dpto) )	oargs1.oa_argstype = COMPLEX_ARGS;
	else			oargs1.oa_argstype = REAL_ARGS;

	oargs2.oa_argstype = oargs1.oa_argstype;
	oargs3.oa_argstype = oargs1.oa_argstype;

	/* do the first multiply right to the target.
	 *
	 * oargs1: dst = src * coeff
	 * oargs2: tmp = src * coeff
	 * oargs3: dst = dst + tmp
	 */

	oargs1.oa_dest=sub_dst_dp;
	oargs1.oa_1 = sub_src_dp;
	oargs1.oa_sdp[0]= coeff_dp;

	oargs2.oa_dest = tmp_dst_dp;
	oargs2.oa_1 = sub_src_dp;
	oargs2.oa_sdp[0] = coeff_dp;

	oargs3.oa_dest = sub_dst_dp;
	oargs3.oa_1 = sub_dst_dp;
	oargs3.oa_2 = tmp_dst_dp;

	for(i=0;i<xform_dp->dt_rows;i++){
		/* Each row of the transform generates one component of each
		 * of the output vectors.
		 */


		/* choose the matrix coefficient with xform_indices */
		xform_indices[2] = i;
		xform_indices[1] = 0;
		coeff_dp->dt_data = (std_type *) multiply_indexed_data(xform_dp,xform_indices);
		oargs1.oa_svp[0]= (Scalar_Value *) coeff_dp->dt_data;
		oargs2.oa_svp[0]= (Scalar_Value *) coeff_dp->dt_data;

		dst_indices[0]=i;	/* select i_th component */
		src_indices[0]=0;	/* select i_th component */
		sub_src_dp->dt_data = multiply_indexed_data(dpfr,src_indices);
		sub_dst_dp->dt_data = multiply_indexed_data(dpto,dst_indices);

		vsmul(&oargs1);

		for(j=1;j<xform_dp->dt_cols;j++){
			/* choose the matrix coefficient with xform_indices */
			/* row is the same as above */
			xform_indices[1] = j;
			coeff_dp->dt_data = (std_type *) multiply_indexed_data(xform_dp,xform_indices);
			oargs1.oa_svp[0]= (Scalar_Value *) coeff_dp->dt_data;
			oargs2.oa_svp[0]= (Scalar_Value *) coeff_dp->dt_data;

			src_indices[0]=j;	/* select j_th component */
			sub_src_dp->dt_data = multiply_indexed_data(dpfr,src_indices);

			vsmul(&oargs2);
			vadd(&oargs3);
		}
	}
	delvec(QSP_ARG  sub_src_dp);
	delvec(QSP_ARG  sub_dst_dp);
	delvec(QSP_ARG  tmp_dst_dp);
	delvec(QSP_ARG  coeff_dp);

} /* end vec_xform() */

/*
 * like vec_xform, but does the division for homgenous coords
 * But the code doesn't seem to reflect this description??
 */

void OBJ_METHOD_NAME(homog_xform)(QSP_ARG_DECL  Data_Obj *dpto,Data_Obj *dpfr,Data_Obj *xform)
{
	Data_Obj *tmp_obj;
	dimension_t i,j;
	Vec_Obj_Args oargs1,oargs2,oargs3;

	if( xform_chk(dpto,dpfr,xform) == -1 )
		return;

	/* make a temporary output component */
	/* like the destination except for tdim */

	tmp_obj=make_obj(QSP_ARG  "_tmptmp",1,dpto->dt_rows,dpto->dt_cols,
		1,REQUIRED_DST_PREC);
	if( tmp_obj == NO_OBJ ){
		NWARN("homog_xform:  couldn't create temporary data object");
		return;
	}

	if( IS_COMPLEX(dpto) )	oargs1.oa_argstype = COMPLEX_ARGS;
	else			oargs1.oa_argstype = REAL_ARGS;

	for(i=0;i<xform->dt_rows;i++){
		/* compute a dot product */

		/* do the first multiply right to the target */
		vsmul(&oargs1);

		for(j=1;j<xform->dt_cols;j++){
			vsmul(&oargs2);
			vadd(&oargs3);
		}
	}
	delvec(QSP_ARG  tmp_obj);
} /* end homog_xorm */

#ifdef FOOBAR
void OBJ_METHOD_NAME(unity)(mp)				/**/
Data_Obj *mp;
{
	u_long i,j;
	std_type *f;

	f=(std_type *)(mp->dt_data);
	for(i=0;i<mp->dt_cols;i++){
		for(j=0;j<mp->dt_cols;j++){
			if(i==j) *f++ = 1.0;
			else *f++ = 0.0;
		}
	}
}
#endif /* FOOBAR */

/* Compute the determinant of a square matrix */

double OBJ_METHOD_NAME(determinant)(Data_Obj *dp)
{
	/* When the index vars were signed, this comment was here:
	 * test j<0 goes bad for unsigned j
	 * BUT it doesn't look like neg j is needed...
	 */
	dimension_t i,j;
	double det=0.0;

	/* BUG check for errors here */

	for(i=0;i<dp->dt_cols;i++){
		std_type *pp;
		std_type prod;

		/* do the positive term */
		pp = (std_type *)dp->dt_data;
		pp += i * dp->dt_pinc;
		prod = 1.0;
		for(j=i;j<dp->dt_cols;j++){
			prod *= *pp;
			pp += dp->dt_rinc + dp->dt_pinc;
		}
		/* wrap around to beginning of row */
		pp -= dp->dt_cols * dp->dt_pinc;
		for(j=0;j<i;j++){
			prod *= *pp;
			pp += dp->dt_rinc + dp->dt_pinc;
		}
		det += prod;

		/* now do the negative term */
		pp = (std_type *)dp->dt_data;
		pp += i * dp->dt_pinc;
		prod = 1.0;
		for(j=i;j>=0;j--){
			prod *= *pp;
			pp += dp->dt_rinc - dp->dt_pinc;
		}
		pp += dp->dt_cols * dp->dt_pinc;
		for(j=dp->dt_cols-1;j>i;j--){
			prod *= *pp;
			pp += dp->dt_rinc - dp->dt_pinc;
		}
		det -= prod;
	}
	return(det);
}
			

