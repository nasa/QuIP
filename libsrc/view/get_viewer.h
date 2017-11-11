
#define GET_VIEWER( funcname )						\
									\
	vp=pick_vwr("");						\
	if( vp == NULL ) {						\
		sprintf(ERROR_STRING,"%s:  invalid viewer selection",	\
						funcname);		\
		WARN(ERROR_STRING);					\
	}

