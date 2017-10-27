/* This file contains the stuff for linear algebra that we want to implement for
 * both single and double precision.
 */

/* Because this file is included, it can't have a version string */

static int HOST_TYPED_CALL_NAME(arg_chk,type_code)(Data_Obj *dpto, Data_Obj *dpfr, Data_Obj *xform, const char *func_name)
{
	if( OBJ_MACH_PREC(dpto) != REQUIRED_DST_PREC ){
		sprintf(DEFAULT_ERROR_STRING,"target object %s (%s) must have %s precision for %s",
			OBJ_NAME(dpto),OBJ_PREC_NAME(dpto), NAME_FOR_PREC_CODE(REQUIRED_DST_PREC),func_name);
		NWARN(DEFAULT_ERROR_STRING);
		return(-1);
	}
	if( OBJ_MACH_PREC(dpfr) != REQUIRED_SRC_PREC ){
		sprintf(DEFAULT_ERROR_STRING,"source object %s (%s) must have %s precision for %s",
			OBJ_NAME(dpfr),OBJ_PREC_NAME(dpfr), NAME_FOR_PREC_CODE(REQUIRED_SRC_PREC),func_name);
		NWARN(DEFAULT_ERROR_STRING);
		return(-1);
	}
	if( OBJ_MACH_PREC(xform) != REQUIRED_SRC_PREC ){
		sprintf(DEFAULT_ERROR_STRING,"matrix object %s (%s) must have %s precision for %s",
			OBJ_NAME(xform),OBJ_PREC_NAME(xform),NAME_FOR_PREC_CODE(REQUIRED_SRC_PREC),func_name);
		NWARN(DEFAULT_ERROR_STRING);
		return(-1);
	}
	if( dpto == dpfr ){
		/* BUG this test is not sufficiently rigorous...  it could be that the destination is
		 * an alias of some sort for the source, having a different header structure but pointing
		 * to the same data!?
		 */
		sprintf(DEFAULT_ERROR_STRING,"arg_chk:  destination (%s) must be distinct from source (%s)",
			OBJ_NAME(dpto),OBJ_NAME(dpfr));
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

#ifdef HAVE_FVDOT
//forwar declaration of typed rvdot
static void HOST_TYPED_CALL_NAME(rvdot,type_code)(HOST_CALL_ARG_DECLS);
#endif // HAVE_FVDOT


//static void HOST_TYPED_CALL_NAME(rxform_list,type_code)(QSP_ARG_DECL  Data_Obj *dpto,Data_Obj *dpfr,Data_Obj *xform_dp)
static void HOST_TYPED_CALL_NAME(rxform_list,type_code)(HOST_CALL_ARG_DECLS)
{
	u_long i,j,k;
	//Vec_Obj_Args oa1;
	Data_Obj *sub_dst_dp, *sub_xf_dp, *xf_row_dp, *sub_src_dp;
	Dimension_Set sz1, *sizes=(&sz1);
	Dimension_Set rs1, *row_dimset=(&rs1);
	index_t offsets[N_DIMENSIONS]={0,0,0,0,0};
	index_t dst_indices[N_DIMENSIONS]={0,0,0,0,0};
	index_t src_indices[N_DIMENSIONS]={0,0,0,0,0};
	index_t xf_indices[N_DIMENSIONS]={0,0,0,0,0};
	Data_Obj *dpto;
	Data_Obj *dpfr;
	Data_Obj *xform_dp;
	Vec_Obj_Args oa2, *oap2=(&oa2);

	dpto = OA_DEST(oap);
	dpfr = OA_SRC1(oap);
	xform_dp = OA_SRC2(oap);

	set_dimension(row_dimset,0,1);
	set_dimension(row_dimset,1,1);
	set_dimension(row_dimset,2,1);
	set_dimension(row_dimset,3,1);
	set_dimension(row_dimset,4,1);

	if( xform_chk(dpto,dpfr,xform_dp) == -1 )
		return;

	if( HOST_TYPED_CALL_NAME(arg_chk,type_code)(dpto,dpfr,xform_dp,TYPED_STRING(xform_list)) < 0 )
		return;

	//oap = (&oa1);
#ifdef CAUTIOUS
	/*clear_obj_args(oap);*/	/* now const */
#endif /* CAUTIOUS */

#ifdef FOOBAR
	/* BUG need to do this elsewhere!? */
	// Is ARGSTYPE set by the caller???

if( OA_ARGSTYPE(oap) == COMPLEX_ARGS )
fprintf(stderr,"complex args\n");
else if( OA_ARGSTYPE(oap) == REAL_ARGS )
fprintf(stderr,"real args\n");
else
NWARN("rxform_list:  Need to make set oa_argstype before calling!?");
	/*
	if( IS_COMPLEX(dpto) )	SET_OA_ARGSTYPE(oap, COMPLEX_ARGS);
	else			SET_OA_ARGSTYPE(oap, REAL_ARGS);
	*/
#endif // FOOBAR

	//sizes = dpto->dt_type_dimset;
	// Copy values, don't assign pointer!
	* sizes = * OBJ_TYPE_DIMS(dpto);

	//sizes.ds_dimension[2]=1;				/* select one component */
	//sizes.ds_dimension[1]=1;
	//sizes.ds_dimension[0]=1;
	ASSIGN_IDX_COUNT(sizes,2,1);
	ASSIGN_IDX_COUNT(sizes,1,1);
	ASSIGN_IDX_COUNT(sizes,0,1);
	sub_dst_dp = _mk_subseq(DEFAULT_QSP_ARG  "_sub_dst",dpto,offsets,sizes);	/* xform_list */

	* sizes = * OBJ_TYPE_DIMS(xform_dp);
	ASSIGN_IDX_COUNT(sizes,2,1);					/* select one row */
	sub_xf_dp = _mk_subseq(DEFAULT_QSP_ARG  "_sub_xf",xform_dp,offsets,sizes);	/* one row */
	ASSIGN_IDX_COUNT(row_dimset,0,OBJ_COLS(sub_xf_dp));
	xf_row_dp = _make_equivalence(DEFAULT_QSP_ARG  "_xf_row",sub_xf_dp,row_dimset,OBJ_PREC_PTR(sub_xf_dp));

	* sizes = * OBJ_TYPE_DIMS(dpfr);
	ASSIGN_IDX_COUNT(sizes,2,1);				/* select one pixel */
	ASSIGN_IDX_COUNT(sizes,1,1);
	sub_src_dp = _mk_subseq(DEFAULT_QSP_ARG  "_sub_src",dpfr,offsets,sizes);

	*oap2 = *oap;
	SET_OA_DEST(oap2,sub_dst_dp);
	SET_OA_SRC1(oap2,sub_src_dp);
	SET_OA_SRC2(oap2,xf_row_dp);
	//SET_OA_ARGSPREC(oap2,ARGSET_PREC( OBJ_MACH_PREC(dpto) ) );
	set_obj_arg_flags(oap2);

	for(i=0;i<OBJ_ROWS(dpto);i++){
		dst_indices[2]=(index_t)i;		/* i_th row */
		src_indices[2]=(index_t)i;		/* i_th row */
		for(j=0;j<OBJ_COLS(dpto);j++){
			dst_indices[1]=(index_t)j;		/* j_th column */
			src_indices[1]=(index_t)j;		/* j_th column */
			SET_OBJ_DATA_PTR(sub_src_dp,
				multiply_indexed_data(dpfr,src_indices));
			for(k=0;k<OBJ_COMPS(dpto);k++){
				dst_indices[0]=(index_t)k;		/* k_th component */
				xf_indices[2]=(index_t)k;		/* k_th row of xform matrix */
				SET_OBJ_DATA_PTR(sub_dst_dp,multiply_indexed_data(dpto,dst_indices));
				SET_OBJ_DATA_PTR(xf_row_dp,multiply_indexed_data(xform_dp,xf_indices));
				//vdot(oap);
#ifdef HAVE_FVDOT
				HOST_TYPED_CALL_NAME(rvdot,type_code)(FVDOT,oap2);
#else // ! HAVE_FVDOT
				NWARN("rxform_list:  Sorry, FVDOT not defined!?");
#endif // ! HAVE_FVDOT
			} // comps of dpto
		} // cols of dpto
	} // rows of dpto

	_delvec(DEFAULT_QSP_ARG  sub_dst_dp);
	_delvec(DEFAULT_QSP_ARG  sub_src_dp);
	_delvec(DEFAULT_QSP_ARG  sub_xf_dp);
}


/* like xform_list(), but vectorizes over list instead of matrix row.
 * good for long lists of short vectors, prototypical examples
 * are using a matrix to transform color images between color spaces,
 * or geometric transformations of arrays of points or vectors.
 */

static void HOST_TYPED_CALL_NAME(rvsmul,type_code)(HOST_CALL_ARG_DECLS);
static void HOST_TYPED_CALL_NAME(rvadd,type_code)(HOST_CALL_ARG_DECLS);

static void HOST_TYPED_CALL_NAME(rvec_xform,type_code)(HOST_CALL_ARG_DECLS)
/*(QSP_ARG_DECL  Data_Obj *dpto,Data_Obj *dpfr,Data_Obj *xform_dp)*/
{
	Data_Obj *sub_src_dp, *sub_dst_dp, *tmp_dst_dp;
	Data_Obj *dpto;
	Data_Obj *dpfr;
	Data_Obj *xform_dp;
	// Data_Obj *coeff_dp;
	void *coeff_ptr;
	dimension_t i,j;
	Vec_Obj_Args oa1, *oap1=(&oa1);
	Vec_Obj_Args oa2, *oap2=(&oa2);
	Vec_Obj_Args oa3, *oap3=(&oa3);
	Dimension_Set tmp_sizes;
	index_t offsets[N_DIMENSIONS]={0,0,0,0,0};
	index_t xform_indices[N_DIMENSIONS]={0,0,0,0,0};
	index_t src_indices[N_DIMENSIONS]={0,0,0,0,0};
	index_t dst_indices[N_DIMENSIONS]={0,0,0,0,0};

	dpto = OA_DEST(oap);
	dpfr = OA_SRC1(oap);
	xform_dp = OA_SRC2(oap);

	if( xform_chk(dpto,dpfr,xform_dp) == -1 )
		return;

	if( HOST_TYPED_CALL_NAME(arg_chk,type_code)(dpto,dpfr,xform_dp,TYPED_STRING(vec_xform)) < 0 )
		return;

	/* make a temporary output component */
	/* like the destination except for tdim */

	//coeff_dp = _mk_subimg(DEFAULT_QSP_ARG  xform_dp, 0, 0, "_xform_coeff",1,1);	/* 1x1 subimage at 0 0 */

	tmp_sizes = *( OBJ_TYPE_DIMS(dpto) );
	ASSIGN_IDX_COUNT(&tmp_sizes,0,1);			/* single component */
	sub_dst_dp = _mk_subseq(DEFAULT_QSP_ARG  "_sub_dst",dpto,offsets,&tmp_sizes);	/* vec_xform - RxC image w/ one component */

	sub_src_dp = _mk_subseq(DEFAULT_QSP_ARG  "_sub_src",dpfr,offsets,&tmp_sizes);
	tmp_dst_dp = _make_dobj(DEFAULT_QSP_ARG  "_tmp_dst",OBJ_TYPE_DIMS(sub_dst_dp),OBJ_PREC_PTR(sub_dst_dp));

	if( sub_dst_dp == NULL || sub_src_dp == NULL || tmp_dst_dp == NULL ){
		NWARN("error creating temporary object for vec_xform");
		return;
	}

#ifdef FOOBAR
	// BUG now this is rvec_xform, so we don't need complex!?
	/* BUG should we check for mixed types here???  */

	if( IS_COMPLEX(dpto) )	SET_OA_ARGSTYPE(oap1,COMPLEX_ARGS);
	else if( IS_REAL(dpto) )SET_OA_ARGSTYPE(oap1,REAL_ARGS);
	else {
		NWARN("Sorry, args must be real or complex");
	}
#endif // FOOBAR

	assert( IS_REAL(dpto) );

	SET_OA_ARGSTYPE(oap1,REAL_ARGS);
	SET_OA_ARGSTYPE(oap2,OA_ARGSTYPE(oap1) );
	SET_OA_ARGSTYPE(oap3,OA_ARGSTYPE(oap1) );

	/* do the first multiply right to the target.
	 *
	 * oargs1: dst = src * coeff
	 * oargs2: tmp = src * coeff
	 * oargs3: dst = dst + tmp
	 */

	SET_OA_DEST(oap1,sub_dst_dp);
	SET_OA_SRC1(oap1,sub_src_dp);
	// BUG OA_SCLR for return scalars only?
	// SET_OA_SCLR1(oap1,coeff_dp);

	SET_OA_DEST(oap2,tmp_dst_dp);
	SET_OA_SRC1(oap2,sub_src_dp);
	// BUG OA_SCLR for return scalars only?
	// SET_OA_SCLR1(oap2,coeff_dp);

	SET_OA_DEST(oap3,sub_dst_dp);
	SET_OA_SRC1(oap3,sub_dst_dp);
	SET_OA_SRC2(oap3,tmp_dst_dp);

	for(i=0;i<OBJ_ROWS(xform_dp);i++){
		/* Each row of the transform generates one component of each
		 * of the output vectors.
		 */


		/* choose the matrix coefficient with xform_indices */
		xform_indices[2] = i;
		xform_indices[1] = 0;
		//SET_OBJ_DATA_PTR(coeff_dp,multiply_indexed_data(xform_dp,xform_indices));
		coeff_ptr=multiply_indexed_data(xform_dp,xform_indices);

		//SET_OA_SVAL1(oap1,(Scalar_Value *) OBJ_DATA_PTR(coeff_dp) );
		//SET_OA_SVAL1(oap2,(Scalar_Value *) OBJ_DATA_PTR(coeff_dp) );
		SET_OA_SVAL1(oap1,(Scalar_Value *) coeff_ptr );
		SET_OA_SVAL1(oap2,(Scalar_Value *) coeff_ptr );

		dst_indices[0]=i;	/* select i_th component */
		src_indices[0]=0;	/* select i_th component */
		SET_OBJ_DATA_PTR(sub_src_dp,multiply_indexed_data(dpfr,src_indices));
		SET_OBJ_DATA_PTR(sub_dst_dp,multiply_indexed_data(dpto,dst_indices));

		//vsmul(oap1);
		HOST_TYPED_CALL_NAME(rvsmul,type_code)(FVSMUL,oap1);

		for(j=1;j<OBJ_COLS(xform_dp);j++){
			/* choose the matrix coefficient with xform_indices */
			/* row is the same as above */
			xform_indices[1] = j;
			coeff_ptr=multiply_indexed_data(xform_dp,xform_indices);
			SET_OA_SVAL1(oap1,(Scalar_Value *) coeff_ptr );
			SET_OA_SVAL1(oap2,(Scalar_Value *) coeff_ptr );

			src_indices[0]=j;	/* select j_th component */
			SET_OBJ_DATA_PTR(sub_src_dp,multiply_indexed_data(dpfr,src_indices));

			HOST_TYPED_CALL_NAME(rvsmul,type_code)(FVSMUL,oap2);
			HOST_TYPED_CALL_NAME(rvadd,type_code)(FVADD,oap3);
		}
	}
	_delvec(DEFAULT_QSP_ARG  sub_src_dp);
	_delvec(DEFAULT_QSP_ARG  sub_dst_dp);
	_delvec(DEFAULT_QSP_ARG  tmp_dst_dp);
} /* end rvec_xform() */

#ifdef FOOBAR

/*
 * like vec_xform, but does the division for homgenous coords
 * But the code doesn't seem to reflect this description??
 */

static void HOST_TYPED_CALL_NAME(homog_xform,type_code)(QSP_ARG_DECL  Data_Obj *dpto,Data_Obj *dpfr,Data_Obj *xform)
{
	Data_Obj *tmp_obj;
	dimension_t i,j;
	Vec_Obj_Args oa1, *oap1=(&oa1);
	Vec_Obj_Args oa2, *oap2=(&oa2);
	Vec_Obj_Args oa3, *oap3=(&oa3);

	if( xform_chk(dpto,dpfr,xform) == -1 )
		return;

	/* make a temporary output component */
	/* like the destination except for tdim */

	tmp_obj=make_obj(QSP_ARG  "_tmptmp",1,OBJ_ROWS(dpto),OBJ_COLS(dpto),
		1,prec_for_code(REQUIRED_DST_PREC));
	if( tmp_obj == NULL ){
		NWARN("homog_xform:  couldn't create temporary data object");
		return;
	}

	// BUG - we need to set the objects???
	//SET_OA_DEST(oap1,sub_dst_dp);
	//SET_OA_SRC1(oap1,sub_src_dp);
	//SET_OA_SVAL1(oap1,(Scalar_Value *) OBJ_DATA_PTR(coeff_dp) );

	if( IS_COMPLEX(dpto) )	SET_OA_ARGSTYPE(oap1,COMPLEX_ARGS);
	else			SET_OA_ARGSTYPE(oap1,REAL_ARGS);

	for(i=0;i<OBJ_ROWS(xform);i++){
		/* compute a dot product */

		/* do the first multiply right to the target */
		vsmul(oap1);

		for(j=1;j<OBJ_COLS(xform);j++){
			h_vl2_vsmul(oap2);
			h_vl2_vadd(oap3);
		}
	}
	delvec(tmp_obj);
} /* end homog_xorm */
#endif // FOOBAR

#ifdef FOOBAR
void HOST_TYPED_CALL_NAME(unity,type_code)(mp)				/**/
Data_Obj *mp;
{
	u_long i,j;
	std_type *f;

	f=(std_type *)(OBJ_DATA_PTR(mp));
	for(i=0;i<OBJ_COLS(mp);i++){
		for(j=0;j<OBJ_COLS(mp);j++){
			if(i==j) *f++ = 1.0;
			else *f++ = 0.0;
		}
	}
}
#endif /* FOOBAR */

#ifdef FOOBAR
/* Compute the determinant of a square matrix */

static double HOST_TYPED_CALL_NAME(determinant,type_code)(Data_Obj *dp)
{
	/* When the index vars were signed, this comment was here:
	 * test j<0 goes bad for unsigned j
	 * BUT it doesn't look like neg j is needed...
     * HOWEVER, below we test for j>=0, that could never be false!?
	 */
	incr_t i,j;
	double det=0.0;

	/* BUG check for errors here */

	for(i=0;i<OBJ_COLS(dp);i++){
		std_type *pp;
		std_type prod;

		/* do the positive term */
		pp = (std_type *)OBJ_DATA_PTR(dp);
		pp += i * OBJ_PXL_INC(dp);
		prod = 1.0;
		for(j=i;j<OBJ_COLS(dp);j++){
			prod *= *pp;
			pp += OBJ_ROW_INC(dp) + OBJ_PXL_INC(dp);
		}
		/* wrap around to beginning of row */
		pp -= OBJ_COLS(dp) * OBJ_PXL_INC(dp);
		for(j=0;j<i;j++){
			prod *= *pp;
			pp += OBJ_ROW_INC(dp) + OBJ_PXL_INC(dp);
		}
		det += prod;

		/* now do the negative term */
		pp = (std_type *)OBJ_DATA_PTR(dp);
		pp += i * OBJ_PXL_INC(dp);
		prod = 1.0;
		for(j=i;j>=0;j--){
			prod *= *pp;
			pp += OBJ_ROW_INC(dp) - OBJ_PXL_INC(dp);
		}
		pp += OBJ_COLS(dp) * OBJ_PXL_INC(dp);
		for(j=OBJ_COLS(dp)-1;j>i;j--){
			prod *= *pp;
			pp += OBJ_ROW_INC(dp) - OBJ_PXL_INC(dp);
		}
		det -= prod;
	}
	return(det);
}
#endif /* FOOBAR */

			

