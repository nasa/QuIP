
#define GET_VIEWER( funcname )						\
									\
	vp=PICK_VWR("");						\
	if( vp == NO_VIEWER ) {						\
		sprintf(ERROR_STRING,"%s:  invalid viewer selection",	\
						funcname);		\
		WARN(ERROR_STRING);					\
	}

