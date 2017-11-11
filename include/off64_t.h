#ifndef _OFF64_T_H_
#define _OFF64_T_H_

#include "typedefs.h"

#ifndef HAVE_OFF64_T
#ifndef off64_t

#if __WORDSIZE == 8

#define off64_t off_t

#else /* __WORDSIZE != 8 */

#define off64_t uint64_t

#endif /* __WORDSIZE != 8 */

#endif /* ! off64_t */
#endif /* ! HAVE_OFF64_t */

#endif /* _OFF64_T_H_ */
