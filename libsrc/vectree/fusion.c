// Take a subroutine tree and create the code for a gpu kernel?

#include "quip_config.h"
#include "quip_prot.h"
#include "veclib_api.h"
#include "vectree.h"
#include "platform.h"
#include "subrt.h"

static void emit_kern_decl(QSP_ARG_DECL  String_Buf *sbp, Vec_Expr_Node *enp)
{
	assert(enp!=NULL);
	if( enp == NULL ) return;

	switch( VN_CODE(enp) ){
		case T_DECL_STAT_LIST:
			emit_kern_decl(QSP_ARG  sbp, VN_CHILD(enp,0) );
			emit_kern_decl(QSP_ARG  sbp, VN_CHILD(enp,1) );
			break;
		case T_DECL_STAT:
fprintf(stderr,"emit_kern_decl:  declaration statement:\n");
dump_tree(QSP_ARG  enp);
			break;
		default:
			MISSING_CASE(enp,"emit_kern_decl");
			break;
	}
}

static String_Buf *fuse_subrt(QSP_ARG_DECL  Vec_Expr_Node *enp)
{
	String_Buf *sbp;
	Subrt *srp;
	Vec_Expr_Node *arg_decl_enp;

	assert(VN_CODE(enp)==T_SUBRT);
	//srp = runnable_subrt(QSP_ARG  enp);
	srp = VN_SUBRT(enp);
	assert( ! IS_SCRIPT(srp) );

	// The subrt args determine the kernel args...
	sbp = new_stringbuf();
	assert(sbp!=NULL);

	arg_decl_enp = SR_ARG_DECLS(srp);

	emit_kern_decl(QSP_ARG  sbp, arg_decl_enp);
}

String_Buf *fuse_kernel(QSP_ARG_DECL  Vec_Expr_Node *enp)
{
	switch(VN_CODE(enp)){
		case T_SUBRT:
			return fuse_subrt(QSP_ARG  enp);
			break;
		default:
			MISSING_CASE(enp,"fuse_kernel");
			break;
	}
	return NULL;
}


