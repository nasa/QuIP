
// redo this with macros?
#ifdef CONVERSION_FUNC
#undef CONVERSION_FUNC
#endif // CONVERSION_FUNC

#define CONVERSION_FUNC(p1,p2)		\
					\
extern void CONVERSION_NAME(p1,p2) ( HOST_CALL_ARG_DECLS);

#define CONVERSION_NAME(p1,p2)	_CONVERSION_NAME(v##p1##2##p2)
#define _CONVERSION_NAME(func)	HOST_CALL_NAME(func)

#include "conversions.h"

