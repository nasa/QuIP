#ifndef _DOBJ_PRIVATE_H_
#define _DOBJ_PRIVATE_H_

/* BUG - dimension_t is unsigned, because it naturally is, but then we
 * have a problem because how_many can return a negative number,
 * which gets converted to a big unsigned number.  Should we be able to
 * check for these bad values?  Should dimension_t just be signed?
 *
 * For now we leave dimension_t unsigned, but prompt for values using a signed long.
 * This means we can't enter the largest values, but that is better than having
 * the program blow up if we accidentally input a negative number...
 */

#define SET_DIMENSION(dsp,idx,value)		(dsp)->ds_dimension[idx] = value

// ascii.c
extern long next_input_int_with_format(QSP_ARG_DECL   const char *pmpt);
extern double next_input_flt_with_format(QSP_ARG_DECL  const char *pmpt);

#endif // ! _DOBJ_PRIVATE_H_

