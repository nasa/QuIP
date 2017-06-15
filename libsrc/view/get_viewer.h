
#define GET_VIEWER( funcname )						\
									\
	vp=PICK_VWR("");						\
	if( vp == NULL ) {						\
		sprintf(ERROR_STRING,"%s:  invalid viewer selection",	\
						funcname);		\
		WARN(ERROR_STRING);					\
	}

