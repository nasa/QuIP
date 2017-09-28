// Take a subroutine tree and create the code for a gpu kernel?

#include "quip_config.h"
#include "quip_prot.h"
#include "veclib_api.h"
#include "vectree.h"
#include "platform.h"
#include "subrt.h"

static void emit_kern_arg_decl(QSP_ARG_DECL  String_Buf *sbp, Vec_Expr_Node *enp)
{
	assert(enp!=NULL);
	if( enp == NULL ) return;

fprintf(stderr,"from emit_kern_arg_decl BEGIN\n");
	switch( VN_CODE(enp) ){
		case T_DECL_STAT_LIST:
			emit_kern_arg_decl(QSP_ARG  sbp, VN_CHILD(enp,0) );
			emit_kern_arg_decl(QSP_ARG  sbp, VN_CHILD(enp,1) );
			break;
		case T_DECL_STAT:
fprintf(stderr,"emit_kern_arg_decl:  declaration statement:\n");
dump_tree(QSP_ARG  enp);
			break;
		default:
			MISSING_CASE(enp,"emit_kern_arg_decl");
			break;
	}
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


