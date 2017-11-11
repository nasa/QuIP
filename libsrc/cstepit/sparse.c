/* Optimize using sparse Levenberg-Marquardt
 *
 * Needs 3 libraries:  
 *	sparselm-1.3
 *	SuiteSparse
 *	metis-5.1.0
 *
 * This type of optimization is used for computer vision problems
 * such as solving for camera poses and object locations, where
 * the positions of features in a frame are only affected by
 * the camera pose parameters for that frame.
 *
 * I am integrating it now in order to apply to the problem of
 * joint estimation of camera and display gamma nonlinearities.
 * 
 */

#include "quip_config.h"
#include "quip_prot.h"
#include "data_obj.h"
#include "sparse.h"
#ifdef HAVE_SPARSELM
#include "splm.h"
#endif // HAVE_SPARSELM

typedef struct splm_info {
	Query_Stack *	_qsp;
	Data_Obj *	_trial_dp;
	Data_Obj *	_param_dp;
	Data_Obj *	_jac_dp;
	const char *	_cmd;
	int		_nnz;
} SpLM_Info;

#ifdef HAVE_SPARSELM
// function to calculate the prediction (hat_x) from the current guess
// of the parameters

#ifdef THREAD_SAFE_QUERY
static Query_Stack *sparse_qsp=NULL;
#endif // THREAD_SAFE_QUERY

static void predictor_func( double *parameters, double *hat_x, int nvars,
				int nobs, void *adata )
{
	SpLM_Info *splmi_p;
	Query_Stack *qsp;
	double *p,*q;
#ifdef THREAD_SAFE_QUERY
	Query_Stack *qsp;

	assert(sparse_qsp!=NULL);
	qsp = sparse_qsp;
#endif // THREAD_SAFE_QUERY

	// pass these things to a script function???
	splmi_p = (SpLM_Info *) adata;

	// copy the parameters
	p=parameters;
	q=(double *)OBJ_DATA_PTR(splmi_p->_param_dp);
	while( nvars-- )
		*q++ = *p++;

//fprintf(stderr,"Pushing command \"%s\"\n",splmi_p->_cmd);
	qsp = splmi_p->_qsp;

	chew_text( splmi_p->_cmd, "Sparse LM optimization" );
	// now need to execute...

	// after execution, store the results in hat_x
	p=hat_x;
	q=(double *)OBJ_DATA_PTR(splmi_p->_trial_dp);
	while( nobs-- ){
		*p++ = *q++;
	}
}

// function to initialize the structure indicating the non-zero entries
// of the Jacobian
//
// Compressed row storage (CRS) is efficient when there are more non-zero
// entries in a row than in a column...


static void jac_func_crs( double *parameters, struct splm_crsm *jac, int nvars,
				int nobs, void *adata )
{
	// This should be pre-specified by the user?
	// Should it be specified as a non-compressed matrix of 0's and 1's,
	// or a list of index pairs?

	// The entries here shouldn't really depend on the value
	// of the parameters?

	Data_Obj *mat_dp;	// full matrix with 1's indicating
				// non-zero entries
	int i,j;
	int current_idx;
	float *data_p;
	SpLM_Info *splmi_p;
	struct splm_stm jac_st;

	splmi_p = (SpLM_Info *) adata;
	mat_dp = splmi_p->_jac_dp;

	if( OBJ_PREC(mat_dp) != PREC_SP ){
		sprintf(ERROR_STRING,
"Jacobian structure specification matrix %s (%s) should have %s precision!?",
			OBJ_NAME(mat_dp),PREC_NAME(OBJ_PREC_PTR(mat_dp)),
			PREC_NAME(PREC_FOR_CODE(PREC_SP)) );
		warn(ERROR_STRING);
		return;
	}
	if( OBJ_COLS(mat_dp) != nvars ){
		sprintf(ERROR_STRING,
"Number of columns of %s (%d) does not match number of variables (%d)!?",
			OBJ_NAME(mat_dp),OBJ_COLS(mat_dp),nvars);
		warn(ERROR_STRING);
		return;
	}
	if( OBJ_ROWS(mat_dp) != nobs ){
		sprintf(ERROR_STRING,
"Number of rows of %s (%d) does not match number of observations (%d)!?",
			OBJ_NAME(mat_dp),OBJ_COLS(mat_dp),nvars);
		warn(ERROR_STRING);
		return;
	}

	splm_stm_alloc(&jac_st, OBJ_ROWS(mat_dp), OBJ_COLS(mat_dp), splmi_p->_nnz);

	current_idx=0;
	for(i=0;i<OBJ_ROWS(mat_dp);i++){
		jac->rowptr[i] = current_idx;
		for(j=0;j<OBJ_COLS(mat_dp);j++){
			data_p = ((float *)OBJ_DATA_PTR(mat_dp)) +
				j * OBJ_PXL_INC(mat_dp) +
				i * OBJ_ROW_INC(mat_dp);
			if( *data_p != 0 ){
				//jac->colidx[current_idx++]=j;
				// add this element
				// what is the correct order of the args???
				splm_stm_nonzero(&jac_st, i, j);
			}
		}
	}
	/* ...convert to CCS */
	splm_stm2crsm(&jac_st, jac);
	splm_stm_free(&jac_st);

}
#endif // HAVE_SPARSELM

#define INSIST_DP( dp )							\
	if( OBJ_PREC(param_dp) != PREC_DP ){				\
		sprintf(ERROR_STRING,					\
	"Parameter object %s has %s precision, should be %s",		\
			OBJ_NAME(dp),PREC_NAME(OBJ_PREC_PTR(dp)),	\
				PREC_NAME(PREC_FOR_CODE(PREC_DP)) );	\
		WARN(ERROR_STRING);					\
		return;							\
	}

#define INSIST_CONTIG( dp )						\
	if( ! IS_CONTIGUOUS(dp) ){					\
		sprintf(ERROR_STRING,					\
	"Parameter vector %s must be contiguous!?",OBJ_NAME(dp));	\
		WARN(ERROR_STRING);					\
		return;							\
	}


static COMMAND_FUNC( do_difcrs )
{
	double *param_array;
	double *measurement_array;	// the data to fit
	int nvars;
	int nobs;
	int Jnnz;			// garbage value!?
					// number of non-zero Jacobian entries

#ifdef HAVE_SPARSELM
	int nconvars=0;			// number of params not to modify (?)
	int JtJnnz=(-1);		// number of non-zeros in J^t*J,
					// -1 if unknown
					// (does "J^t" mean J transpose?)
	int max_iterations=100000;
	double opts[SPLM_OPTS_SZ];
	double info[SPLM_INFO_SZ];	// output data, set arg to NULL
					// if don't care
	int stop_code;
#endif // HAVE_SPARSELM

	SpLM_Info splm_i1;

	Data_Obj *param_dp, *true_dp, *trial_dp, *jac_dp;

	param_dp = pick_obj("data object for parameter values (will be overwritten)");
	true_dp = pick_obj("data object for target values");
	trial_dp = pick_obj("data object for trial values");
	jac_dp = pick_obj("data object holding Jacobian info");
	Jnnz = HOW_MANY("number of non-zero Jacobian entries");
	splm_i1._cmd = NAMEOF("script to evaluate parameters");

	if( param_dp == NULL || true_dp == NULL || trial_dp == NULL || jac_dp == NULL )
		return;

fprintf(stderr,"do_difcrs:  app data at 0x%lx\n",
(long)(&splm_i1));

	INSIST_DP(param_dp)
	INSIST_DP(true_dp)
	INSIST_DP(trial_dp)
	INSIST_CONTIG( param_dp )
	INSIST_CONTIG( true_dp )
	INSIST_CONTIG( trial_dp )

#ifdef HAVE_SPARSELM
	opts[0] = 1;	// scale factor for initial mu
	opts[1] = 0.001;	// stopping threshold for ||J^T e||_inf,
	opts[2] = 0.001;	// stopping threshold for ||dp||_2,
	opts[3] = 0.001;	// stopping threshold for ||e||_2,
	opts[4] = 0.00001;	// delta for Jacobian estimation
				// negative delta means use central
				// difference method (more accurate but
				// more expensive to compute).
	// Set opts arg to NULL to use default values
#endif // HAVE_SPARSELM

	measurement_array=(double *)OBJ_DATA_PTR(true_dp);
	nobs = (int) OBJ_N_MACH_ELTS(true_dp);
	param_array=(double *) OBJ_DATA_PTR(param_dp);
	nvars = (int) OBJ_N_MACH_ELTS(param_dp);

	if( OBJ_COLS(jac_dp) != nvars ){
		sprintf(ERROR_STRING,
"Number of columns (%d) of Jacobian (%s) does not match number of elements (%d) of param vector %s!?",
			OBJ_COLS(jac_dp),OBJ_NAME(jac_dp),
			nvars,OBJ_NAME(param_dp) );
		warn(ERROR_STRING);
		return;
	}

	if( OBJ_ROWS(jac_dp) != nobs ){
		sprintf(ERROR_STRING,
"Number of rows (%d) of Jacobian (%s) does not match number of elements (%d) of observation vector %s!?",
			OBJ_ROWS(jac_dp),OBJ_NAME(jac_dp),
			nobs,OBJ_NAME(true_dp) );
		warn(ERROR_STRING);
		return;
	}

fprintf(stderr,"%d observations, %d parameters\n",nobs,nvars);
fprintf(stderr,"%d non-zero Jacobian entries\n",Jnnz);

	splm_i1._jac_dp = jac_dp;
	splm_i1._qsp = THIS_QSP;
	splm_i1._trial_dp = trial_dp;
	splm_i1._param_dp = param_dp;
	splm_i1._nnz = Jnnz;

#ifdef HAVE_SPARSELM
	sparselm_difcrs( &predictor_func, &jac_func_crs, param_array,
		measurement_array, nvars, nconvars, nobs, Jnnz, JtJnnz,
		max_iterations, /* opts */ NULL, info, &splm_i1 /* adata */ );

#define PRINT_INFO(desc_str,index)				\
								\
	sprintf(MSG_STR,"%s:  %g",#desc_str,info[index]);	\
	prt_msg(MSG_STR);

	PRINT_INFO(Initial error,0);
	PRINT_INFO(Final error,1);
	PRINT_INFO(Final J^t e,2);
	PRINT_INFO(Final dp,3);
	PRINT_INFO(Final mu/max(J^T J),4);
	PRINT_INFO(number of iterations,5);
	stop_code = (int) info[6];
	switch(stop_code){
		case 1: prt_msg("Stopped by small gradient J^T e"); break;
		case 2: prt_msg("Stopped by small dp"); break;
		case 3: prt_msg("Stopped by max iterations"); break;
		case 4: prt_msg("Singular matrix - restart from current p with increased mu"); break;
		case 5: prt_msg("Too many attempts to increase damping.  Restart from current p with increased mu"); break;
		case 6: prt_msg("Stopped by small error"); break;
		case 7: prt_msg("Stopped by invalid (NaN or Inf) func values - user error"); break;
		default:
			prt_msg("Unexpected info code!?"); break;
	}
	PRINT_INFO(Number of function evaluations,7);
	PRINT_INFO(Number of Jacobian evaluations,8);
	PRINT_INFO(Number of linear systems solved,9);

#else // ! HAVE_SPARSELM

	WARN("No support for Sparse L-M in this build!?");

#endif // ! HAVE_SPARSELM

}

static COMMAND_FUNC( do_difccs )
{
	WARN("do_difccs:  not implemented!?");
}

static COMMAND_FUNC( do_derccs )
{
	WARN("do_derccs:  not implemented!?");
}

static COMMAND_FUNC( do_dercrs )
{
	WARN("do_dercrs:  not implemented!?");
}

#define ADD_CMD(s,f,h)	ADD_COMMAND(sparse_menu,s,f,h)

MENU_BEGIN(sparse)
ADD_CMD(derccs,	do_derccs,	provide analytic Jacobian (compressed column storage) )
ADD_CMD(difccs,	do_difccs,	finite difference Jacobian (compressed column storage) )
ADD_CMD(dercrs,	do_dercrs,	provide analytic Jacobian (compressed row storage) )
ADD_CMD(difcrs,	do_difcrs,	finite difference Jacobian (compressed row storage) )
MENU_END(sparse)

COMMAND_FUNC( do_sparse_menu )
{
	CHECK_AND_PUSH_MENU(sparse);
}

