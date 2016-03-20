/** user interface routines for interactive progs */

#include "quip_config.h"

#include <stdio.h>
#include "quip_prot.h"
#include "data_obj.h"


/* Look for an object described by a possibly indexed string
 * no warning issued if the object does not exist
 */

Data_Obj * hunt_obj(QSP_ARG_DECL  const char *name)
{
	Data_Obj *dp;
	char stem[LLEN], *cp;
	const char *s;

#ifdef QUIP_DEBUG
/*
if( debug & debug_data ){
sprintf(ERROR_STRING,"hunt_obj:  passed name \"%s\"",name);
advise(ERROR_STRING);
list_dobjs(SINGLE_QSP_ARG);
}
*/
#endif /* QUIP_DEBUG */

	dp=dobj_of(QSP_ARG  name);
	if( dp != NO_OBJ ){
		return(dp);
	}

	/* maybe this string has an index tacked onto the end? */
	/* copy the stem name into the buffer */

	cp=stem;
	s=name;

	/* use symbolic constants for delimiters here? */
	while( *s && *s != '[' && *s != '{' )
		*cp++ = *s++;
	*cp=0;

	dp = dobj_of(QSP_ARG  stem);
	if( dp == NO_OBJ ) return(dp);

	return( index_data(QSP_ARG  dp,s) );
}

Data_Obj *get_obj(QSP_ARG_DECL  const char *name)
{
	Data_Obj *dp;

	dp = hunt_obj(QSP_ARG  name);
	if( dp == NO_OBJ ){
		sprintf(ERROR_STRING,"No data object \"%s\"",name);
		WARN(ERROR_STRING);
	}
	return(dp);
}

Data_Obj *get_vec(QSP_ARG_DECL  const char *s)
{
	Data_Obj *dp;

	dp=get_obj(QSP_ARG  s);
	if( dp==NO_OBJ ) return(dp);
	if( !(OBJ_FLAGS(dp) & DT_VECTOR) ){
		sprintf(ERROR_STRING,"object \"%s\" is not a vector",s);
		WARN(ERROR_STRING);
		return(NO_OBJ);
	}
	return(dp);
}

Data_Obj *				/**/
img_of(QSP_ARG_DECL  const char *s)				/**/
{
	Data_Obj *dp;

	dp=get_obj(QSP_ARG  s);
	if( dp==NO_OBJ ) return(dp);
	if( !(OBJ_FLAGS(dp) & DT_IMAGE) ){
		sprintf(ERROR_STRING,"object \"%s\" is not an image",s);
		WARN(ERROR_STRING);
		return(NO_OBJ);
	}
	return(dp);
}

Data_Obj *get_seq(QSP_ARG_DECL  const char *s)
{
	Data_Obj *dp;

	dp=get_obj(QSP_ARG  s);
	if( dp==NO_OBJ ) return(dp);
	if( !(OBJ_FLAGS(dp) & DT_SEQUENCE) ){
		sprintf(ERROR_STRING,"object \"%s\" is not an sequence",s);
		WARN(ERROR_STRING);
		return(NO_OBJ);
	}
	return(dp);
}

Data_Obj * get_img( QSP_ARG_DECL  const char *s )
{
	Data_Obj *dp;

	dp=get_obj(QSP_ARG  s);
	if( dp==NO_OBJ ) return(dp);
	if( !IS_IMAGE(dp) ){
		sprintf(ERROR_STRING,"data object %s is not an image",s);
		WARN(ERROR_STRING);
		return(NO_OBJ);
	}
	return(dp);
}

