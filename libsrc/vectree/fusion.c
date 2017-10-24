// Take a subroutine tree and create the code for a gpu kernel?

#include "quip_config.h"
#include "quip_prot.h"
#include "veclib_api.h"
#include "vectree.h"
#include "platform.h"
#include "subrt.h"
#include <string.h>

// BUG - use of these global vars not thread-safe,
// but let's just get it working first...

static int indentation_level=0;
static List *global_var_list=NULL;
static int indices_inited;	// flag var

static void add_to_global_var_list(const char *varname)
{
	Node *np;

	if( global_var_list == NULL )
		global_var_list = new_list();

	// BUG?  should we make sure that this var is not already on the list?
	np = mk_node( (void *) varname);
	addTail( global_var_list, np );
}

static void emit_indentation(QSP_ARG_DECL  String_Buf *sbp)
{
	int i=indentation_level;

	while(i--)
		cat_string(sbp,"\t");
}

static void emit_index_declarations(QSP_ARG_DECL  String_Buf *sbp)
{
	Node *np;

	if( global_var_list == NULL ) return;
	np = QLIST_HEAD(global_var_list);
	while(np!=NULL){
		const char *varname;
		varname = NODE_DATA(np);
		emit_indentation(QSP_ARG  sbp);
		cat_string(sbp,"int ");
		cat_string(sbp,varname);
		cat_string(sbp,"_index;\n");
		np = NODE_NEXT(np);
	}
}

static void check_index_initialization(QSP_ARG_DECL  String_Buf *sbp)
{
	Node *np;

	if( indices_inited ) return;
	if( global_var_list == NULL ) return;

	np = QLIST_HEAD(global_var_list);
	while(np!=NULL){
		const char *varname;
		varname = NODE_DATA(np);
		emit_indentation(QSP_ARG  sbp);
		cat_string(sbp,varname);
		cat_string(sbp,"_index = ");
		cat_string(sbp,"get_global_id(0)");	// BUG - platform dependent!
		cat_string(sbp,";\n");
		np = NODE_NEXT(np);
	}

	indices_inited=1;
}

static void rls_global_var_list()
{
	Node *np;

	if( global_var_list == NULL ) return;

	while( (np=remHead(global_var_list)) != NULL ){
		rls_node(np);
	}
}

static void emit_symbol_for_func(QSP_ARG_DECL  String_Buf *sbp, Vec_Func_Code code)
{
	const char *s;
	switch(code){
		case FVADD:
		case FVSADD:
			s=" + "; break;
		case FVMUL:
		case FVSMUL:
			s=" * "; break;
		case FVDIV:
		case FVSDIV:
			s=" / "; break;
		case FVSUB:
		case FVSSUB:
			s=" - "; break;
		default:
			sprintf(ERROR_STRING,
	"emit_symbol_for_func:  unhandled function %s!?",
	VF_NAME(FIND_VEC_FUNC(code)));
			WARN(ERROR_STRING);
			return;
	}
	cat_string(sbp,s);

}

static void emit_ptr_index(QSP_ARG_DECL  String_Buf *sbp, Vec_Expr_Node *enp)
{
	// We need to keep track of which arg is which!
	cat_string(sbp,VN_STRING(enp));
	cat_string(sbp,"_index");
	// offsets are used only for OpenCL??? - platform-dependent!
	cat_string(sbp,"+");
	cat_string(sbp,VN_STRING(enp));
	cat_string(sbp,"_offset");
}

static void emit_kern_arg_decl(QSP_ARG_DECL  String_Buf *sbp, Vec_Expr_Node *enp)
{
	assert(enp!=NULL);
	if( enp == NULL ) return;

	switch( VN_CODE(enp) ){
		case T_DECL_STAT_LIST:
			emit_kern_arg_decl(QSP_ARG  sbp, VN_CHILD(enp,0) );
			cat_string(sbp,", ");
			emit_kern_arg_decl(QSP_ARG  sbp, VN_CHILD(enp,1) );
			break;
		case T_DECL_STAT:
			assert( VN_CHILD(enp,0) != NULL );

			if( VN_CODE(VN_CHILD(enp,0)) == T_PTR_DECL ){
				const char *s;

				s = (*(PF_KERNEL_STRING_FN( PFDEV_PLATFORM(curr_pdp) ) ))
					(QSP_ARG  PKS_ARG_QUALIFIER );
				cat_string(sbp,s);
				if( strlen(s) > 0 )
					cat_string(sbp," ");
				add_to_global_var_list(VN_DECL_NAME(VN_CHILD(enp,0)));
			}

			cat_string(sbp,PREC_NAME(VN_DECL_PREC(enp)));
			cat_string(sbp," ");
			emit_kern_arg_decl(QSP_ARG  sbp, VN_CHILD(enp,0) );
			break;
		case T_PTR_DECL:
			cat_string(sbp,"*");
			cat_string(sbp,VN_DECL_NAME(enp));

			// For opencl, need an offset arg
			cat_string(sbp,", int ");
			cat_string(sbp,VN_DECL_NAME(enp));
			cat_string(sbp,"_offset");
			break;
		case T_SCAL_DECL:
			cat_string(sbp,VN_DECL_NAME(enp));
			break;
		default:
			MISSING_CASE(enp,"emit_kern_arg_decl");
			break;
	}
}

static void emit_bool_op(QSP_ARG_DECL  String_Buf *sbp, Vec_Expr_Node *enp)
{
	switch( VN_BM_CODE(enp) ){
		case FVV_VV_GT:
		case FVS_VV_GT:
		case FVV_VS_GT:
		case FVS_VS_GT:
		case FSS_VV_GT:
		case FSS_VS_GT:
			cat_string(sbp," > "); break;
		case FVV_VV_LT:
		case FVS_VV_LT:
		case FVV_VS_LT:
		case FVS_VS_LT:
		case FSS_VV_LT:
		case FSS_VS_LT:
			cat_string(sbp," < "); break;
		case FVV_VV_GE:
		case FVS_VV_GE:
		case FVV_VS_GE:
		case FVS_VS_GE:
		case FSS_VV_GE:
		case FSS_VS_GE:
			cat_string(sbp," >= "); break;
		case FVV_VV_LE:
		case FVS_VV_LE:
		case FVV_VS_LE:
		case FVS_VS_LE:
		case FSS_VV_LE:
		case FSS_VS_LE:
			cat_string(sbp," <= "); break;
		case FVV_VV_EQ:
		case FVS_VV_EQ:
		case FVV_VS_EQ:
		case FVS_VS_EQ:
		case FSS_VV_EQ:
		case FSS_VS_EQ:
			cat_string(sbp," == "); break;
		case FVV_VV_NE:
		case FVS_VV_NE:
		case FVV_VS_NE:
		case FVS_VS_NE:
		case FSS_VV_NE:
		case FSS_VS_NE:
			cat_string(sbp," != "); break;
		default:
			sprintf(ERROR_STRING,
				"emit_bool_op:  unhandled bitmap code %d!?",VN_BM_CODE(enp));
			WARN(ERROR_STRING);
			break;
	}
}

// We need to initialize the indices after any local declarations, but before the
// first action statement!?  So we insert the check function in the case for any
// action nodes...

static void emit_kern_body_node(QSP_ARG_DECL  String_Buf *sbp, Vec_Expr_Node *enp)
{
	switch( VN_CODE(enp) ){
		case T_TYPECAST:
			cat_string(sbp, "(");
			cat_string(sbp, PREC_NAME(VN_CAST_PREC_PTR(enp)));
			cat_string(sbp, ")(");
			emit_kern_body_node(QSP_ARG  sbp, VN_CHILD(enp,0));
			cat_string(sbp, ")");
			break;
		case T_VS_VS_CONDASS:
		//ALL_CONDASS_CASES
			emit_kern_body_node(QSP_ARG  sbp, VN_CHILD(enp,2));
			emit_bool_op(QSP_ARG  sbp, enp);
			emit_kern_body_node(QSP_ARG  sbp, VN_CHILD(enp,3));
			cat_string(sbp, " ? ");
			emit_kern_body_node(QSP_ARG  sbp, VN_CHILD(enp,0));
			cat_string(sbp, " : ");
			emit_kern_body_node(QSP_ARG  sbp, VN_CHILD(enp,1));
			break;
		case T_BITRSHIFT:
			emit_kern_body_node(QSP_ARG  sbp, VN_CHILD(enp,0));
			cat_string(sbp, " >> ");
			emit_kern_body_node(QSP_ARG  sbp, VN_CHILD(enp,1));
			break;
		case T_BITLSHIFT:
			emit_kern_body_node(QSP_ARG  sbp, VN_CHILD(enp,0));
			cat_string(sbp, " << ");
			emit_kern_body_node(QSP_ARG  sbp, VN_CHILD(enp,1));
			break;
		case T_MATH2_FN:
			cat_string(sbp, FUNC_NAME(VN_FUNC_PTR(enp) ) );
			cat_string(sbp, "(");
			emit_kern_body_node(QSP_ARG  sbp, VN_CHILD(enp,0));
			cat_string(sbp, ",");
			emit_kern_body_node(QSP_ARG  sbp, VN_CHILD(enp,1));
			cat_string(sbp, ")");
			break;
		case T_MATH1_FN:
			cat_string(sbp, FUNC_NAME(VN_FUNC_PTR(enp) ) );
			cat_string(sbp, "(");
			emit_kern_body_node(QSP_ARG  sbp, VN_CHILD(enp,0));
			cat_string(sbp, ")");
			break;
		case T_LIT_INT:
			sprintf(msg_str,"%ld",VN_INTVAL(enp));
			cat_string(sbp,msg_str);
			break;
		case T_LIT_DBL:
			sprintf(msg_str," %g",VN_DBLVAL(enp));
			cat_string(sbp,msg_str);
			break;
		case T_DECL_STAT:
			emit_indentation(QSP_ARG  sbp);
			// problem with int32?
			cat_string(sbp,PREC_NAME(VN_DECL_PREC(enp)));
			cat_string(sbp," ");
			emit_kern_arg_decl(QSP_ARG  sbp, VN_CHILD(enp,0) );
			cat_string(sbp,";\n");
			break;
		case T_DYN_OBJ: // assume this is a var holding a scalar!?
		case T_SCALAR_VAR:
			cat_string(sbp,VN_STRING(enp));
			break;

		case T_DEREFERENCE:
			enp = VN_CHILD(enp,0);
			assert(enp!=NULL);
			assert(VN_CODE(enp)==T_POINTER);

			cat_string(sbp,VN_STRING(enp));
			cat_string(sbp,"[");
			emit_ptr_index(QSP_ARG  sbp,enp);
			cat_string(sbp,"]");

			break;

		case T_STAT_LIST:
		case T_DECL_STAT_LIST:
			emit_kern_body_node(QSP_ARG  sbp, VN_CHILD(enp,0));
			emit_kern_body_node(QSP_ARG  sbp, VN_CHILD(enp,1));
			break;
		case T_ASSIGN:
			// Can this occur in a declaration???
			check_index_initialization(QSP_ARG  sbp);

			emit_indentation(QSP_ARG  sbp);
			emit_kern_body_node(QSP_ARG  sbp, VN_CHILD(enp,0));
			cat_string(sbp," = ");
			emit_kern_body_node(QSP_ARG  sbp, VN_CHILD(enp,1));
			cat_string(sbp," ;\n");
			break;
		case T_VV_FUNC:
			cat_string(sbp,"( ");
			emit_kern_body_node(QSP_ARG  sbp, VN_CHILD(enp,0));
			emit_symbol_for_func(QSP_ARG  sbp,VN_VFUNC_CODE(enp));
			emit_kern_body_node(QSP_ARG  sbp, VN_CHILD(enp,1));
			cat_string(sbp," )");
			break;

		case T_VS_FUNC:
			cat_string(sbp,"( ");
			// scalar should be emitted first...
			emit_kern_body_node(QSP_ARG  sbp, VN_CHILD(enp,1));
			emit_symbol_for_func(QSP_ARG  sbp,VN_VFUNC_CODE(enp));
			emit_kern_body_node(QSP_ARG  sbp, VN_CHILD(enp,0));
			cat_string(sbp," )");
			break;
		default:
			MISSING_CASE(enp,"emit_kern_body_node");
			break;
	}
}

static void emit_kern_body(QSP_ARG_DECL  String_Buf *sbp, Subrt *srp)
{
	cat_string(sbp,"{\n");
	indentation_level++;
	emit_index_declarations(QSP_ARG  sbp);
	emit_kern_body_node(QSP_ARG  sbp, SR_BODY(srp));
	cat_string(sbp,"}\n");
}

static void emit_kern_decl(QSP_ARG_DECL  String_Buf *sbp, const char *kname, Subrt *srp)
{
	Vec_Expr_Node *arg_decl_enp;
	const char *s;

	// Make up a name for the kernel

	s = (*(PF_KERNEL_STRING_FN( PFDEV_PLATFORM(curr_pdp) ) ))
			(QSP_ARG  PKS_KERNEL_QUALIFIER );
	cat_string(sbp,s);	/*"__kernel " (ocl) or "global" (cuda) */
	if( strlen(s) > 0 )
		cat_string(sbp," ");
	cat_string(sbp,"void ");
	cat_string(sbp,kname);
	cat_string(sbp,"(");

	arg_decl_enp = SR_ARG_DECLS(srp);
	emit_kern_arg_decl(QSP_ARG  sbp, arg_decl_enp);

	cat_string(sbp,")\n");
}

static void * make_platform_kernel(QSP_ARG_DECL  const char *src, const char *name)
{
	void *kp;

	assert( curr_pdp != NULL );
	kp = (*(PF_MAKE_KERNEL_FN( PFDEV_PLATFORM(curr_pdp) ) ))
		(QSP_ARG  src, name, curr_pdp );
	if( kp == NULL ){ 
		NERROR1("kernel creation failure!?");
	}
	return kp;
}

void * find_fused_kernel(QSP_ARG_DECL  Subrt *srp, Platform_Device *pdp )
{
	Kernel_Info_Ptr kip;

	kip = SR_KERNEL_INFO_PTR(srp,PF_TYPE(PFDEV_PLATFORM(pdp)));
	return (*(PF_FETCH_KERNEL_FN(PFDEV_PLATFORM(pdp))))(QSP_ARG  kip, pdp );
}

void run_fused_kernel(QSP_ARG_DECL  Subrt_Call *scp, void * kp, Platform_Device *pdp)
{
	(*(PF_RUN_KERNEL_FN(PFDEV_PLATFORM(pdp))))(QSP_ARG  kp, SC_ARG_VALS(scp), pdp);
	//fprintf(stderr,"Sorry, run_fused_kernel not implemented yet...\n");
}

static void store_platform_kernel(QSP_ARG_DECL  Subrt *srp, void *kp, Platform_Device *pdp)
{
	Kernel_Info_Ptr *kip_p;
	kip_p = SR_KERNEL_INFO_PTR_ADDR(srp,PF_TYPE(PFDEV_PLATFORM(pdp)));
	(*(PF_STORE_KERNEL_FN(PFDEV_PLATFORM(pdp))))(QSP_ARG  kip_p, kp, pdp );
}

static void make_kernel_name(String_Buf *sbp, Subrt *srp, const char *speed)
{
	cat_string(sbp,PF_PREFIX_STR(PFDEV_PLATFORM(curr_pdp)));
	cat_string(sbp,"_");
	cat_string(sbp,speed);
	cat_string(sbp,"_");
	cat_string(sbp,SR_NAME(srp));
}

// Should we assume the current platform?

void fuse_subrt(QSP_ARG_DECL  Subrt *srp)
{
	String_Buf *sbp;
	String_Buf *kname;
	void *kp;

	assert( ! IS_SCRIPT(srp) );

	if( curr_pdp == NULL ){
		WARN("fuse_subrt:  no platform selected!?");
		return;
	}

	// Make sure that this one hasn't already been fused...
	kp = find_fused_kernel(QSP_ARG  srp, curr_pdp);
	if( kp != NULL ){
		sprintf(ERROR_STRING,"fuse_subrt:  Subroutine %s has already been fused!?",SR_NAME(srp));
		WARN(ERROR_STRING);
		return;
	}

	// The subrt args determine the kernel args...
	sbp = new_stringbuf();
	assert(sbp!=NULL);

	kname = new_stringbuf();
	make_kernel_name(kname, srp,"fast");
	emit_kern_decl(QSP_ARG  sbp, sb_buffer(kname), srp );
	indices_inited=0;
	emit_kern_body(QSP_ARG  sbp, srp );
	rls_global_var_list();

	if( verbose )
		fprintf(stderr,"Kernel source:\n\n%s\n\n",sb_buffer(sbp));

	// BUG OpenCL specific code!?!?
	kp = make_platform_kernel(QSP_ARG  sb_buffer(sbp), sb_buffer(kname) );

	// BUG release stringbufs here!

	store_platform_kernel(QSP_ARG  srp,kp,curr_pdp);
}

void fuse_kernel(QSP_ARG_DECL  Vec_Expr_Node *enp)
{
	Subrt *srp;

	switch(VN_CODE(enp)){
		case T_SUBRT:
			srp = VN_SUBRT(enp);
			if( IS_SCRIPT(srp) ){
				WARN("Sorry, can't fuse script subroutines");
			} else {
				fuse_subrt(QSP_ARG  srp);
			}
			break;
		default:
			MISSING_CASE(enp,"fuse_kernel");
			break;
	}
}

// returns number of elements to process
// BUG - this is good for fast kernels, but for slow kernels we need the shapes too!?

long set_fused_kernel_args(QSP_ARG_DECL  void *kp, int *idx_p, Vec_Expr_Node *enp, Compute_Platform *cpp)
{
	Data_Obj *dp;
	long l1, l2, ret_val;

	switch( VN_CODE(enp) ){
		case T_ARGLIST:
			l1=set_fused_kernel_args(QSP_ARG  kp,idx_p,VN_CHILD(enp,0),cpp);
			l2=set_fused_kernel_args(QSP_ARG  kp,idx_p,VN_CHILD(enp,1),cpp);
			ret_val = l1 > l2 ? l1 : l2;	// max
//fprintf(stderr,"set_fused_kernel_args will return %ld (max of %ld and %ld)\n",ret_val,l1,l2);
			break;

		case T_REFERENCE:
			enp = VN_CHILD(enp,0);
			assert(enp!=NULL);
			assert(VN_CODE(enp)==T_STATIC_OBJ);
			dp = VN_OBJ(enp);
			assert(dp!=NULL);
			ret_val = OBJ_N_MACH_ELTS(dp);

			(*(PF_SET_KERNEL_ARG_FN(cpp)))(QSP_ARG  kp,idx_p, &OBJ_DATA_PTR(dp), KERNEL_ARG_VECTOR );
			break;

		case T_LIT_DBL:
			ret_val = 1;
			(*(PF_SET_KERNEL_ARG_FN(cpp)))(QSP_ARG  kp,idx_p, &VN_DBLVAL(enp), KERNEL_ARG_DBL );
			break;

		case T_LIT_INT:
			ret_val = 1;
			(*(PF_SET_KERNEL_ARG_FN(cpp)))(QSP_ARG  kp,idx_p, &VN_INTVAL(enp), KERNEL_ARG_INT );
			break;

		default:
			ret_val = -1;
			sprintf(ERROR_STRING,"set_fused_kernel_args:  unhandled case %s !?",node_desc(enp));
			ERROR1(ERROR_STRING);
	}
//fprintf(stderr,"set_fused_kernel_args will return %ld\n",ret_val);
	return ret_val;
}

