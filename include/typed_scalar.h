#ifndef _TYPED_SCALAR_H
#define _TYPED_SCALAR_H

#include "quip_config.h"
#include "shape_info.h"
#include "scalar_value.h"

// This was introduced to that pexpr can return long long's
// in addition to doubles...
//
// At some point we allowed these to be strings as well.  Not quite
// sure why at the moment, but perhaps to help with vector versions
// of functions like is_upper, toupper, etc?
// Strings aren't really scalars, so it isn't clear that this
// makes sense at all.

/* typedef */ struct typed_scalar {
	Scalar_Value	ts_value;
	prec_t		ts_prec_code;
	int		ts_flags;
} /* Typed_Scalar */;

#define TS_FREE		1
#define TS_STATIC	2

#define STATIC_SCALAR(tsp)	((tsp)->ts_flags & TS_STATIC)
#define FREE_SCALAR(tsp)	((tsp)->ts_flags & TS_FREE)

#define SCALAR_PREC_CODE(tsp)		((tsp)->ts_prec_code)
#define SCALAR_MACH_PREC_CODE(tsp)	(SCALAR_PREC_CODE(tsp)&MACH_PREC_MASK)

//#define RELEASE_SCALAR(tsp)	(tsp)->ts_prec_code |= FREE_PREC_BIT;
//#define RELEASE_SCALAR(tsp)	{ if( ! STATIC_SCALAR(tsp) ){ TELL_REL(tsp) (tsp)->ts_prec_code |= FREE_PREC_BIT; } }

#define TELL_NOREL(tsp)	fprintf(stderr,"NOT Releasing static scalar at 0x%lx\n",(long)tsp); CHKIT(tsp)
#define TELL_REL(tsp)	fprintf(stderr,"Releasing scalar at 0x%lx\n",(long)tsp); CHKIT(tsp)
#define CHKIT(tsp)	if( (((long)tsp)&0xfff0000) == 0x1cc0000 ) abort();

#define RELEASE_SCALAR(tsp)			\
	{					\
	if( ! STATIC_SCALAR(tsp) )		\
		(tsp)->ts_flags |= TS_FREE;	\
	}

extern int has_zero_value(Typed_Scalar *tsp);
extern int scalars_are_equal(Typed_Scalar *tsp1, Typed_Scalar *tsp2);
extern double dbl_scalar_value(Typed_Scalar *tsp);
extern Typed_Scalar *free_typed_scalar(SINGLE_QSP_ARG_DECL);
extern Typed_Scalar *scalar_for_long(long l);
extern Typed_Scalar *scalar_for_llong(int64_t l);
extern Typed_Scalar *scalar_for_double(double d);
extern Typed_Scalar *scalar_for_string(const char *s);
extern double double_for_scalar(Typed_Scalar *tsp);
extern int64_t llong_for_scalar(Typed_Scalar *tsp);
extern int32_t long_for_scalar(Typed_Scalar *tsp);
extern index_t index_for_scalar(Typed_Scalar *tsp);
extern void show_typed_scalar(Typed_Scalar *tsp);
extern void string_for_typed_scalar(char *buf, int buflen, Typed_Scalar *tsp);
extern char *scalar_string(Typed_Scalar *tsp);

#endif // _TYPED_SCALAR_H
