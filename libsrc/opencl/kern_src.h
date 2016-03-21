// We have declared the strings that implement all of the kernels,
// but we need to be able to find them...
// So we put them all in a table.

typedef struct typed_kern_func_source {
	const char *		tkfs_fast_s;
	const char *		tkfs_eqsp_s;
	const char *		tkfs_flen;
	const char *		tkfs_elen;
	const char *		tkfs_slen_s;
} Typed_Kern_Func_Source;

typedef struct kern_func_source {
	Typed_Kern_Func_Source	kfs_by_s;
	Typed_Kern_Func_Source	kfs_in_s;
	Typed_Kern_Func_Source	kfs_di_s;
	Typed_Kern_Func_Source	kfs_li_s;
	Typed_Kern_Func_Source	kfs_uby_s;
	Typed_Kern_Func_Source	kfs_uin_s;
	Typed_Kern_Func_Source	kfs_udi_s;
	Typed_Kern_Func_Source	kfs_uli_s;
	Typed_Kern_Func_Source	kfs_sp_s;
	Typed_Kern_Func_Source	kfs_dp_s;
} Kern_Func_Source;

