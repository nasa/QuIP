
#ifndef TYPEDEFS_H
#define TYPEDEFS_H

#include "quip_config.h"

#if HAVE_STDINT_H
#include <stdint.h>
#elif HAVE_INTTYPES_H
#include <inttypes.h>
#endif

#if HAVE_SYS_TYPES_H
#include <sys/types.h>
#else
typedef uint32_t u_long;
typedef unsigned int u_int;
#endif

#ifdef BUILD_FOR_WINDOWS
typedef uint32_t u_long;
typedef unsigned int u_int;
typedef unsigned short u_short;
typedef unsigned char u_char;
typedef unsigned char uint8_t;
#endif /* BUILD_FOR_WINDOWS */

//typedef uint32_t u_int32;
//typedef u_long	int_for_addr;

#define USHORT_ARG unsigned short

typedef unsigned int count_t;

#ifdef USE_SSE
typedef int v4sf __attribute__ ((vector_size(sizeof(float)*4))); // vector of four single floats
#endif // USE_SSE

#endif /* ! TYPEDEFS_H */

