
/* BUG?  the order in which these are listed matters, because it has to match
 * the order in the table in vt_native.c...  We might fix this by having the program
 * sort the table on startup...
 */
typedef enum {
	NATIVE_SVDCMP,
	NATIVE_SVBKSB,
	NATIVE_JACOBI,
	NATIVE_EIGSRT,
	NATIVE_CHOLDC,
	NATIVE_SYSTEM,
	NATIVE_RENDER,
	NATIVE_XFORM_LIST,
	NATIVE_INVERT,
	NATIVE_CUMSUM,
	N_VT_NATIVE_FUNCS
} VT_Native;

extern const char *eval_vt_native_string(Vec_Expr_Node *enp);
extern float eval_vt_native_flt(Vec_Expr_Node *enp);
extern void eval_vt_native_assignment(Data_Obj *dp, Vec_Expr_Node *enp );
extern void eval_vt_native_work(QSP_ARG_DECL  Vec_Expr_Node *enp );
extern void update_vt_native_shape(Vec_Expr_Node *enp);
extern void prelim_vt_native_shape(QSP_ARG_DECL  Vec_Expr_Node *enp);

