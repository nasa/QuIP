#include "quip_config.h"

#include <stdio.h>

#ifdef HAVE_STRING_H
#include <string.h>
#endif

#include "quip_prot.h"
#include "query_stack.h"

#define parent_name	QS_PATHNAME(THIS_QSP)

const char *_parent_directory_of(QSP_ARG_DECL  const char *pathname)
{
	char *s;

	/* if the string contains a /, then we
	 * just search backwards from the end of the string
	 */

	strcpy(parent_name,pathname);

	s=parent_name;
	s+=strlen(parent_name)-1;		/* now points to last char */

	if( *s == '/' ){
		sprintf(ERROR_STRING,"Pathname \"%s\" ends with a slash!?",
			pathname);
		warn(ERROR_STRING);
	}
	while( s!=parent_name ){
		if( *s == '/' ){
			*s=0;
			return(parent_name);
		}
		s--;
	}

	/* if we get here, s points to the first char,
	 * and we have not encountered a slash at any
	 * other positions.  Check for /filename
	 */

	if( *s == '/' ){
		*(s+1) = 0;
		return(parent_name);
	}

	/* must be a relative pathname.
	 * We may want to use getcwd(), but for now
	 * we'll just return '.'
	 */

	strcpy(parent_name,".");
	return(parent_name);
}


/*
 * change the pointed to string to point to the first char after the last /
 *
 * This seems like a useful function which perhaps should go elsewhere...
 */

void strip_fullpath(char **strp)
{
	int result;
	char *str;

#ifdef PC
	/* Get rid of extension if there */
	if ( (str = strchr(*strp,'.')) != NULL) {
		result = (int)(str - *strp);
		*(*strp+result) = 0;
	}

	/* Get rid of path */
	if ( (str = strrchr(*strp,'\\')) != NULL) {
		result = (int)(str - *strp + 1);
		*strp += result;
	}
#else /* PC */
	/* Get rid of leading components of the path */
	if ( (str = strrchr(*strp,'/')) != NULL) {
		result = (int)(str - *strp + 1);
		*strp += result;
	}
#endif /* PC */
}

