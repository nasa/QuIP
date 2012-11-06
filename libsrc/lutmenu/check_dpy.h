
#ifdef HAVE_X11
#define CHECK_DPYP(funcname)						\
									\
	if( current_dpyp == NO_DISPLAY ){				\
		sprintf(error_string,"%s:  no display set",funcname);	\
		WARN(error_string);					\
		return;							\
	}
#else
#define CHECK_DPYP(funcname)
#endif

