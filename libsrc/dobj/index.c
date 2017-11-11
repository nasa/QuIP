#include "quip_config.h"

#ifdef HAVE_STDLIB_H
#include <stdlib.h>
#endif

#ifdef HAVE_CTYPE_H
#include <ctype.h>
#endif

#include "quip_prot.h"
#include "data_obj.h"
#include "nexpr.h"	// pexpr

/*
 * Return a data object indexed by the index string.
 * The index string can be a series of expressions
 * delimited by square or curly braces, and the expressions
 * can be indices (numbers) or the star character '*',
 * which means skip this dimension and go on the the next.
 * In the future it would be nice to also allow ranges
 * (e.g., x[3:6]) as a way of doing subimages.
 */

Data_Obj * _index_data( QSP_ARG_DECL  Data_Obj *dp, const char *index_str )
{
	const char *cp;
	int i;
	index_t index;
	char str[64];
	Data_Obj *newdp;
	int left_delim,right_delim;
	int right_need;
	int maxd, mind;
	int is_star;

	if( dp == NULL ) return(dp);

	cp=index_str;
	while( *cp && isspace(*cp) ) cp++;

	maxd = OBJ_MAXDIM(dp);
	mind = OBJ_MINDIM(dp);

next_index:

	if( *cp != '[' && *cp != '{' ){
		sprintf(DEFAULT_ERROR_STRING,"bad index delimiter \"%s\"",index_str);
		NWARN(DEFAULT_ERROR_STRING);
		return(NULL);
	}
	left_delim = *cp;
	right_delim = left_delim+2;	/* takes '[' -> ']' and '{' -> '}' */
	right_need=1;
	cp++;
	i=0;
	/*
	 * Scan characters until we find the matching right delimiter.
	 * Originally we just scanned until we found a right delimiter,
	 * but that failed for cases where the index was an expression
	 * which itself contained the same delimiter, e.g.:
	 * x{ncols(rgb{0})-1}
	 *
	 * The fix was to count left and right delimiters to make sure
	 * that we find the match.  Note that this will still fail
	 * if there is a quoted string with an unmatched delimiter...BUG
	 */

	while( *cp && ((*cp!=right_delim) || (right_need>1)) ){
		if( *cp == left_delim ) right_need++;
		else if( *cp == right_delim ) right_need--;
		str[i++]=(*cp++);
	}
	if( *cp != right_delim ){
		sprintf(DEFAULT_ERROR_STRING,"missing index delimiter '%c'",right_delim);
		NWARN(DEFAULT_ERROR_STRING);
		return(NULL);
	}
	cp++;
	str[i]=0;

	/* first check to see if the index string is just a star */
	i=0;
	is_star=1;
	while( str[i] ){
		if( isspace(str[i]) ) i++;
		else if( str[i] == '*' ) i++;
		else {
			is_star=0;
			i++;
		}
	}

	if( is_star ) {
		if( right_delim == ']' ) maxd--;
		else               mind++;
		goto next_index;
	} else {
		Typed_Scalar *tsp;
		tsp = pexpr( QSP_ARG  str );
		index = index_for_scalar(tsp);
		RELEASE_SCALAR(tsp)
		if( right_delim == ']' )
			newdp=gen_subscript(dp,maxd,index,SQUARE);
		else
			newdp=gen_subscript(dp,mind,index,CURLY);
	}

	if( *cp ) return(index_data(newdp,cp));
	else return(newdp);
} // index_data



