#ifndef _WARN_H_
#define _WARN_H_


#ifdef __cplusplus
extern "C" {
#endif

#include "quip_fwd.h"

//extern void warn(QSP_ARG_DECL  const char *msg);

#ifdef BUILD_FOR_IOS
extern void _error1(QSP_ARG_DECL  const char *msg);
#else // ! BUILD_FOR_IOS
__attribute__ ((__noreturn__)) extern void _error1(QSP_ARG_DECL  const char *msg);
#endif // ! BUILD_FOR_IOS

// Phasing out all-caps WARN...
#define WARN(msg)	warn(msg)
#define warn(msg)	script_warn(QSP_ARG msg)

#define NWARN(msg)	script_warn(DEFAULT_QSP_ARG msg)

// when we run this on the iOS simulator, the bell char prints
// as a lower case A!?
#ifdef BUILD_FOR_IOS
#define WARNING_PREFIX	"WARNING:  "
#define ERROR_PREFIX	"ERROR:  "
#else // ! BUILD_FOR_IOS
#define WARNING_PREFIX	"WARNING:  "
#define ERROR_PREFIX	"ERROR:  "
#endif // ! BUILD_FOR_IOS

#define EXPECTED_PREFIX	"Expected warning received:  "

#define WARN_ONCE(s)					\
							\
	{						\
		static int warned=0;			\
		if( ! warned ){				\
			warn(s);			\
			warned=1;			\
		}					\
	}

#define NWARN_ONCE(s)					\
							\
	{						\
		static int warned=0;			\
		if( ! warned ){				\
			NWARN(s);			\
			warned=1;			\
		}					\
	}


#define error1(msg)	q_error1(QSP_ARG  msg)
#define NERROR1(msg)	q_error1(DEFAULT_QSP_ARG  msg)

// in iOS, error1 needs to return, so routines that call it have to return
// themselves...

#ifdef BUILD_FOR_IOS
#define IOS_RETURN		return;
#define IOS_RETURN_VAL(v)	return(v);
#else // ! BUILD_FOR_IOS
#define IOS_RETURN
#define IOS_RETURN_VAL(v)
#endif // ! BUILD_FOR_IOS

extern void _advise(QSP_ARG_DECL  const char *msg);
#define advise(s)	_advise(QSP_ARG  s)

#define NADVISE(s)	_advise(DEFAULT_QSP_ARG  s)


#define prt_msg(s)	_prt_msg(QSP_ARG  s)
#define prt_msg_frag(s)	_prt_msg_frag(QSP_ARG  s)

extern void _prt_msg_frag(QSP_ARG_DECL  const char *msg);
extern void _prt_msg(QSP_ARG_DECL  const char *msg);

extern void tty_advise(QSP_ARG_DECL  const char *s);

#define tell_sys_error(s)	_tell_sys_error(QSP_ARG  s)

extern void _tell_sys_error(QSP_ARG_DECL  const char* s);
extern void set_warn_func(void (*func)(QSP_ARG_DECL  const char *));


#ifdef __cplusplus
}
#endif


#endif /* ! _WARN_H_ */

