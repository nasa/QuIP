
#ifdef HAVE_X11
#define CHECK_DPYP(funcname)						\
									\
	if( current_dpyp == NULL ){				\
		sprintf(ERROR_STRING,"%s:  no display set",funcname);	\
		WARN(ERROR_STRING);					\
		return;							\
	}
#else
#define CHECK_DPYP(funcname)
#endif

