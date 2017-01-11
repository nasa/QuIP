#ifndef _HOST_CALLS_CPU_H_
#define _HOST_CALLS_CPU_H_

//#define H_CALL_PROJ_2V(name,type)	SLOW_HOST_CALL(name,,,,2)
#define H_CALL_PROJ_2V(name,type)	SLOW_HOST_CALL(name,,,,2)
#define H_CALL_PROJ_3V(name,type)	SLOW_HOST_CALL(name,,,,3)
#define H_CALL_PROJ_2V_IDX(name)	SLOW_HOST_CALL(name,,,,2)

#define H_CALL_MM_NOCC(name)				\
							\
GENERIC_HOST_FAST_CALL(name,/*bitmap*/,/*typ*/,/*scalars*/,/*vectors*/)			\
GENERIC_HOST_EQSP_CALL(name,,,,)			\
GENERIC_HOST_SLOW_CALL(name,,,,)			\
							\
GENERIC_HOST_FAST_SWITCH(name,,,,NOCC)

#endif // ! _HOST_CALLS_CPU_H_
