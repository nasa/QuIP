
/* cu2_host_call_defs.m4 BEGIN */
include(`cu2_host_call_defs.m4')

// First the typed functions...

dnl	dumpdef
dnl	traceon

include(`../../include/veclib/host_typed_call_defs.m4')
include(`../../include/veclib/gen_host_calls.m4')


dnl	UNIMP_MSG(f)
define(`UNIMP_MSG',`
	sprintf(DEFAULT_ERROR_STRING,"%s is unimplemented!?","$1");\
	NWARN(DEFAULT_ERROR_STRING);
')

void h_cu2_sp_rvfft(HOST_CALL_ARG_DECLS)
{
	UNIMP_MSG(h_cu2_sp_rvfft)
}

void h_cu2_sp_rvift(HOST_CALL_ARG_DECLS)
{
	UNIMP_MSG(h_cu2_sp_rvift)
}

void h_cu2_dp_rvfft(HOST_CALL_ARG_DECLS)
{
	UNIMP_MSG(h_cu2_dp_rvfft)
}

void h_cu2_dp_rvift(HOST_CALL_ARG_DECLS)
{
	UNIMP_MSG(h_cu2_dp_rvift)
}

void h_cu2_sp_cvfft(HOST_CALL_ARG_DECLS)
{
	UNIMP_MSG(h_cu2_sp_cvfft)
}

void h_cu2_sp_cvift(HOST_CALL_ARG_DECLS)
{
	UNIMP_MSG(h_cu2_sp_cvift)
}

void h_cu2_dp_cvfft(HOST_CALL_ARG_DECLS)
{
	UNIMP_MSG(h_cu2_dp_cvfft)
}

void h_cu2_dp_cvift(HOST_CALL_ARG_DECLS)
{
	UNIMP_MSG(h_cu2_dp_cvift)
}


