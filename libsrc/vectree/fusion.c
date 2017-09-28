// Take a subroutine tree and create the code for a gpu kernel?

#include "quip_config.h"
#include "quip_prot.h"
#include "veclib_api.h"
#include "vectree.h"
#include "platform.h"
#include "subrt.h"

static int indentation_level=0;	// BUG not thread-safe

static void emit_indentation(QSP_ARG_DECL  String_Buf *sbp)
{
	int i=indentation_level;

	while(i--)
		cat_string(sbp,"\t");
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
	cat_string(sbp,"idx");
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
//fprintf(stderr,"emit_kern_arg_decl:  declaration statement:\n");
//dump_tree(QSP_ARG  enp);
			cat_string(sbp,PREC_NAME(VN_DECL_PREC(enp)));
			cat_string(sbp," ");
			emit_kern_arg_decl(QSP_ARG  sbp, VN_CHILD(enp,0) );
			break;
		case T_PTR_DECL:
			cat_string(sbp,"*");
			cat_string(sbp,VN_STRING(enp));
			break;
		case T_SCAL_DECL:
			cat_string(sbp,VN_STRING(enp));
			break;
		default:
			MISSING_CASE(enp,"emit_kern_arg_decl");
			break;
	}
}

static void emit_kern_body_node(QSP_ARG_DECL  String_Buf *sbp, Vec_Expr_Node *enp)
{
	switch( VN_CODE(enp) ){
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
			cat_string(sbp,PREC_NAME(VN_DECL_PREC(enp)));
			cat_string(sbp," ");
			emit_kern_arg_decl(QSP_ARG  sbp, VN_CHILD(enp,0) );
			cat_string(sbp,";\n");
			break;
		case T_DYN_OBJ:
			// assume this is a var holding a scalar!?
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
			emit_kern_body_node(QSP_ARG  sbp, VN_CHILD(enp,0));
			emit_kern_body_node(QSP_ARG  sbp, VN_CHILD(enp,1));
			break;
		case T_ASSIGN:
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
	emit_kern_body_node(QSP_ARG  sbp, SR_BODY(srp));
	cat_string(sbp,"}\n");
}

static void emit_kern_decl(QSP_ARG_DECL  String_Buf *sbp, Subrt *srp)
{
	Vec_Expr_Node *arg_decl_enp;

	// Make up a name for the kernel

	cat_string(sbp,"pf_kern_");

	cat_string(sbp,SR_NAME(srp));
	cat_string(sbp,"(");

	arg_decl_enp = SR_ARG_DECLS(srp);
	emit_kern_arg_decl(QSP_ARG  sbp, arg_decl_enp);

	cat_string(sbp,")\n");
}

String_Buf *fuse_subrt(QSP_ARG_DECL  Subrt *srp)
{
	String_Buf *sbp;

	assert( ! IS_SCRIPT(srp) );

	// The subrt args determine the kernel args...
	sbp = new_stringbuf();
	assert(sbp!=NULL);

	emit_kern_decl(QSP_ARG  sbp, srp );
	emit_kern_body(QSP_ARG  sbp, srp );

	fprintf(stderr,"Kernel source:\n\n%s\n\n",sb_buffer(sbp));
	return sbp;
}

String_Buf *fuse_kernel(QSP_ARG_DECL  Vec_Expr_Node *enp)
{
	Subrt *srp;
	String_Buf *sbp;

	switch(VN_CODE(enp)){
		case T_SUBRT:
			srp = VN_SUBRT(enp);
			if( IS_SCRIPT(srp) ){
				WARN("Sorry, can't fuse script subroutines");
				return NULL;
			}
			sbp = fuse_subrt(QSP_ARG  srp);
			break;
		default:
			MISSING_CASE(enp,"fuse_kernel");
			return NULL;
			break;
	}
	return sbp;
}


