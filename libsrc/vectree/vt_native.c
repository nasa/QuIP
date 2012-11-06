#include "quip_config.h"

char VersionId_vectree_vt_native[] = QUIP_VERSION_STRING;

#include "vectree.h"
#include "vt_native.h"
#include "vec_util.h"
#include "nvf_api.h"			/* xform_list() */
#ifdef HAVE_NUMREC
#include "nrm_api.h"
#endif /* HAVE_NUMREC */
#include "debug.h"		/* verbose */

#define EVAL_VT_NATIVE_WORK(enp)		eval_vt_native_work(QSP_ARG enp)

Keyword vt_native_func_tbl[N_VT_NATIVE_FUNCS+1]={
#ifdef HAVE_NUMREC
{	"svdcmp",	NATIVE_SVDCMP	},
{	"svbksb",	NATIVE_SVBKSB	},
{	"jacobi",	NATIVE_JACOBI	},
{	"eigsrt",	NATIVE_EIGSRT	},
{	"choldc",	NATIVE_CHOLDC	},
#endif /* HAVE_NUMREC */
{	"system",	NATIVE_SYSTEM	},
{	"render",	NATIVE_RENDER	},
{	"xform_list",	NATIVE_XFORM_LIST	},
{	"invert",	NATIVE_INVERT	},
{	"foobar",	-1		}		/* must be last */
};

const char *eval_vt_native_string(Vec_Expr_Node *enp)
{
	NWARN("eval_vt_native_string:  not implemented for vt!?");
	return("");
}

float eval_vt_native_flt(Vec_Expr_Node *enp)
{
	NWARN("eval_vt_native_flt:  not implemented for vt!?");
	return(0.0);
}

void eval_vt_native_assignment(Data_Obj *dp, Vec_Expr_Node *enp )
{
	switch(enp->en_intval){
#ifdef CAUTIOUS
		default:
			sprintf(DEFAULT_ERROR_STRING,"CAUTIOUS:  eval_vt_native_assignment (vt):  unhandled keyword %s (%ld)",vt_native_func_tbl[enp->en_intval].kw_token,enp->en_intval);
			NWARN(DEFAULT_ERROR_STRING);
			break;
#endif /* CAUTIOUS */
	}
}

#define CHECK_ARGLIST(enp,name)							\
										\
	if( enp == NO_VEXPR_NODE ){						\
		NWARN(error_string);						\
		sprintf(error_string,"missing arg list for native function %s",name);	\
		return;								\
	}									\
	if( enp->en_code != T_ARGLIST ){					\
		sprintf(error_string,"Oops, missing arglist for native function %s!?",name);	\
		DUMP_TREE(enp);							\
		return;								\
	}

void eval_vt_native_work(QSP_ARG_DECL  Vec_Expr_Node *enp )
{
	Vec_Expr_Node *arg_enp;

	/* BUG? we need to define a template for the args somewhere,
	 * so that we can make sure that the args are correct during tree-building...
	 */

	switch(enp->en_intval){
		case NATIVE_RENDER:
			{
			Data_Obj *dst_dp,*coord_dp,*src_dp;

			if( arg_count(enp->en_child[0]) != 3 ){
				NODE_ERROR(enp);
				NWARN("render function requires 3 args");
				return;
			}

			arg_enp = NTH_ARG(enp->en_child[0],0);
			dst_dp = EVAL_OBJ_REF(arg_enp);

			arg_enp = NTH_ARG(enp->en_child[0],1);
			coord_dp = EVAL_OBJ_EXP(arg_enp,NO_OBJ);

			arg_enp = NTH_ARG(enp->en_child[0],2);
			src_dp = EVAL_OBJ_EXP(arg_enp,NO_OBJ);

			if( dst_dp == NO_OBJ || coord_dp == NO_OBJ || src_dp == NO_OBJ ){
				NODE_ERROR(enp);
				NWARN("problem with render args");
				break;
			}
			if( FLOATING_PREC(coord_dp->dt_prec) )
				render_samples2(QSP_ARG  dst_dp,coord_dp,src_dp);
			else
				render_samples(QSP_ARG  dst_dp,coord_dp,src_dp);
			}
			break;

		case NATIVE_INVERT:
			{
			Data_Obj *dst_dp, *src_dp;

			if( arg_count(enp->en_child[0]) != 2 ){
				NODE_ERROR(enp);
				NWARN("invert function requires 2 args");
				return;
			}

			arg_enp = NTH_ARG(enp->en_child[0],0);
			dst_dp = EVAL_OBJ_REF(arg_enp);

			arg_enp = NTH_ARG(enp->en_child[0],1);
			src_dp = EVAL_OBJ_EXP(arg_enp,NO_OBJ);

			/* BUG I use convert() here because I am lazy */
			/* should just use vmov... */
			convert(QSP_ARG  dst_dp,src_dp);
			/* dtinvert operates inplace */
			dt_invert(QSP_ARG  dst_dp);
			/* set value set flag??? */
			}
			break;


		case NATIVE_SYSTEM:
			{
			const char *s;

			s=EVAL_STRING(enp->en_child[0]);
			system(s);
			}
			break;

#ifdef HAVE_NUMREC
		case NATIVE_CHOLDC:
			{
			Data_Obj *inmat_dp, *diag_dp;

advise("evaluating choldc...");
			if( arg_count(enp->en_child[0]) != 2) {
				NODE_ERROR(enp);
				NWARN("choldc requires 2 args");
				return;
			}

			/* first arg is the input matrix, second arg is for the diagonal elements... */
			arg_enp  = NTH_ARG(enp->en_child[0],0);
			inmat_dp = EVAL_OBJ_REF(arg_enp);
		
			arg_enp = NTH_ARG(enp->en_child[0],1);
			diag_dp = EVAL_OBJ_REF(arg_enp);

			if ( inmat_dp == NO_OBJ || diag_dp == NO_OBJ )
				return;

#ifdef HAVE_NUMREC	
			dp_choldc(inmat_dp,diag_dp);
#else
			NWARN("No numerical recipes library, can't compute CHOLESKY");
#endif
			}
			break;
		case NATIVE_SVDCMP:
			{
			Data_Obj *umat_dp, *vmat_dp, *ev_dp;
			/* We need to get three args... */
			if( arg_count(enp->en_child[0]) != 3 ){
				NODE_ERROR(enp);
				NWARN("svdcmp requires 3 args");
				return;
			}
			/* v matrix on the second branch */
			arg_enp = NTH_ARG(enp->en_child[0],0);
			umat_dp = EVAL_OBJ_REF(arg_enp);

			arg_enp = NTH_ARG(enp->en_child[0],1);
			ev_dp = EVAL_OBJ_REF(arg_enp);

			arg_enp = NTH_ARG(enp->en_child[0],2);
			vmat_dp = EVAL_OBJ_REF(arg_enp);

			if( ev_dp == NO_OBJ || umat_dp == NO_OBJ || vmat_dp == NO_OBJ )
				return;
				
			/*
			sprintf(error_string,"Ready to compute SVD, umat = %s, vmat= %s, ev = %s",
				umat_dp->dt_name,vmat_dp->dt_name,ev_dp->dt_name);
			advise(error_string);
			*/

#ifdef HAVE_NUMREC
			dp_svd(umat_dp,ev_dp,vmat_dp);
#else
			NWARN("No numerical recipes library, can't compute SVD");
#endif
			}
			break;
		case NATIVE_SVBKSB:
			{
			/* svbksb(x,u,w,v,b); */
			Data_Obj *umat_dp, *vmat_dp, *ev_dp;
			Data_Obj *x_dp, *b_dp;

			enp=enp->en_child[0];		/* the arg list */
			CHECK_ARGLIST(enp,"svbksb")
			b_dp = EVAL_OBJ_REF(enp->en_child[1]);
			enp=enp->en_child[0];
			CHECK_ARGLIST(enp,"svbksb")
			vmat_dp = EVAL_OBJ_REF(enp->en_child[1]);
			enp=enp->en_child[0];
			CHECK_ARGLIST(enp,"svbksb")
			ev_dp = EVAL_OBJ_REF(enp->en_child[1]);
			enp=enp->en_child[0];
			CHECK_ARGLIST(enp,"svbksb")
			umat_dp = EVAL_OBJ_REF(enp->en_child[1]);
			x_dp = EVAL_OBJ_REF(enp->en_child[0]);

			if( x_dp == NO_OBJ || umat_dp == NO_OBJ || ev_dp == NO_OBJ ||
				vmat_dp == NO_OBJ || b_dp == NO_OBJ )
				return;

#ifdef HAVE_NUMREC
			dp_svbksb(x_dp,umat_dp,ev_dp,vmat_dp,b_dp);
#else
			NWARN("No numerical recipes library, can't compute SVBKSB");
#endif
			}
			break;
		case NATIVE_JACOBI:			/* eval_vt_native_work */
			{
			/* jacobi(eigenvectors,eigenvalues,input_matrix,nrot) */
			/* do we use nrot?? */
			Data_Obj *a_dp,*d_dp,*v_dp;
			int nrot;

			enp=enp->en_child[0];		/* the arg list */
			CHECK_ARGLIST(enp,"jacobi")
			a_dp = EVAL_OBJ_REF(enp->en_child[1]);

			enp=enp->en_child[0];
			CHECK_ARGLIST(enp,"jacobi")
			d_dp = EVAL_OBJ_REF(enp->en_child[1]);
			v_dp = EVAL_OBJ_REF(enp->en_child[0]);

			if( a_dp == NO_OBJ || d_dp == NO_OBJ || v_dp == NO_OBJ )
				return;

#ifdef HAVE_NUMREC
			dp_jacobi(v_dp,d_dp,a_dp,&nrot);
			if( verbose ){
				sprintf(error_string,"jacobi(%s,%s,%s) done after %d rotations",
					v_dp->dt_name,d_dp->dt_name,a_dp->dt_name,nrot);
				advise(error_string);
			}
			v_dp->dt_flags |= DT_ASSIGNED;
#else
			NWARN("No numerical recipes library, can't compute JACOBI");
#endif
			}
			break;
		case NATIVE_EIGSRT:
			{
			Data_Obj *d_dp,*v_dp;
			/* eigsrt(eigenvectors,eigenvalues) */
			enp=enp->en_child[0];
			CHECK_ARGLIST(enp,"eigsrt")
			d_dp = EVAL_OBJ_REF(enp->en_child[1]);
			v_dp = EVAL_OBJ_REF(enp->en_child[0]);

			if( d_dp == NO_OBJ || v_dp == NO_OBJ )
				return;

#ifdef HAVE_NUMREC
			dp_eigsrt(v_dp,d_dp);
#else
			NWARN("No numerical recipes library, can't compute EIGSRT");
#endif
			}
			break;
#endif /* HAVE_NUMREC */

		case NATIVE_XFORM_LIST:
			{
			Data_Obj *dst_dp, *src_dp, *mat_dp;
			/* xform_list(&dst,&src,&mat); */
			enp=enp->en_child[0];
			CHECK_ARGLIST(enp,"xform_list")
			/* left child is an arglist */
#ifdef CAUTIOUS
			if( enp->en_child[0]->en_code != T_ARGLIST ){
				NODE_ERROR(enp);
				sprintf(error_string,
	"CAUTIOUS:  NATIVE_XFORM_LIST arglist left child should be T_ARGLIST");
				NWARN(error_string);
				return;
			}
#endif /* CAUTIOUS */
				
			dst_dp = EVAL_OBJ_REF(enp->en_child[0]->en_child[0]);
			src_dp = EVAL_OBJ_REF(enp->en_child[0]->en_child[1]);
			mat_dp = EVAL_OBJ_REF(enp->en_child[1]);

			if( dst_dp == NO_OBJ || src_dp == NO_OBJ || mat_dp == NO_OBJ )
				return;

			xform_list(QSP_ARG  dst_dp,src_dp,mat_dp);
			dst_dp->dt_flags |= DT_ASSIGNED;
			}
			break;

#ifdef CAUTIOUS
		default:
			sprintf(error_string,"CAUTIOUS:  eval_vt_native_work (vt):  unhandled keyword %s (%ld)",vt_native_func_tbl[enp->en_intval].kw_token,enp->en_intval);
			NWARN(error_string);
			break;
#endif /* CAUTIOUS */
	}
}

void update_vt_native_shape(Vec_Expr_Node *enp)
{
	switch(enp->en_intval){
#ifdef CAUTIOUS
		default:
			sprintf(DEFAULT_ERROR_STRING,"CAUTIOUS:  update_native_shape (vt):  unhandled keyword %s (%ld)",vt_native_func_tbl[enp->en_intval].kw_token,enp->en_intval);
			NWARN(DEFAULT_ERROR_STRING);
			break;
#endif /* CAUTIOUS */
	}
}

void prelim_vt_native_shape(QSP_ARG_DECL  Vec_Expr_Node *enp)
{
	switch(enp->en_intval){
		case  NATIVE_SYSTEM:
		case  NATIVE_RENDER:
		case  NATIVE_XFORM_LIST:
		case  NATIVE_INVERT:
#ifdef HAVE_NUMREC
		case  NATIVE_CHOLDC:
		case  NATIVE_SVDCMP:
		case  NATIVE_SVBKSB:
		case  NATIVE_JACOBI:
		case  NATIVE_EIGSRT:
			/* no shape, do nothing */
#ifdef CAUTIOUS
			verify_null_shape(QSP_ARG  enp);
#endif /* CAUTIOUS */
			break;
#endif /* HAVE_NUMREC */

#ifdef CAUTIOUS
		default:
			sprintf(error_string,"CAUTIOUS:  prelim_vt_native_shape (vt):  unhandled keyword %s (%ld)",vt_native_func_tbl[enp->en_intval].kw_token,enp->en_intval);
			NWARN(error_string);
			break;
#endif /* CAUTIOUS */
	}
}

