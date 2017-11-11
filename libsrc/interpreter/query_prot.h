/* This file contains prototypes for functions that are internal
 * to the interpreter module...
 */

#ifndef _QUERY_PROT_H_
#define _QUERY_PROT_H_

#include "quip_prot.h"
#include "getbuf.h"

/* moved to pipe_support.h
extern void creat_pipe(QSP_ARG_DECL  const char *name, const char* command, const char* rw);
extern void sendto_pipe(QSP_ARG_DECL  Pipe *pp,const char* text);
extern void readfr_pipe(QSP_ARG_DECL  Pipe *pp,const char* varname);
*/

extern void call_funcs_from_list(QSP_ARG_DECL  List *lp );

extern void show_macro(QSP_ARG_DECL  Macro *mp);
extern void dump_macro(QSP_ARG_DECL  Macro *mp);
extern void macro_info(QSP_ARG_DECL  Macro *mp);
extern int macro_is_invoked(Macro *);
extern void allow_recursion_for_macro(Macro *);
extern void rls_macro(QSP_ARG_DECL  Macro *);
extern const char * macro_filename(Macro *);
extern int macro_lineno(Macro *);
extern void rls_macro_arg(Macro_Arg *);


extern void _zap_fore(QSP_ARG_DECL  Foreach_Loop *frp);
#define zap_fore(frp) _zap_fore(QSP_ARG  frp)

extern void init_query_stack( Query_Stack *qsp );
extern void	set_progname(const char *program_name);

extern const char * _nextline(QSP_ARG_DECL const char *pline);
extern const char * _next_query_word(QSP_ARG_DECL const char *pline);
extern const char * _qline(QSP_ARG_DECL const char *pline);
extern void _dup_word(QSP_ARG_DECL const char *);
extern void _whileloop(QSP_ARG_DECL  int value);
extern void _foreach_loop(QSP_ARG_DECL Foreach_Loop *frp);
extern void _push_if(QSP_ARG_DECL const char *text);
extern void _reset_return_strings(SINGLE_QSP_ARG_DECL);
extern int _dupout(QSP_ARG_DECL  FILE *fp);
extern int _check_adequate_return_strings(QSP_ARG_DECL  int n);

#define nextline(pline)				_nextline(QSP_ARG  pline)
#define next_query_word(pline)			_next_query_word(QSP_ARG  pline)
#define qline(pline)				_qline(QSP_ARG  pline)
#define dup_word(s)				_dup_word(QSP_ARG  s)
#define whileloop(value)			_whileloop(QSP_ARG  value)
#define foreach_loop(frp)			_foreach_loop(QSP_ARG  frp)
#define push_if(text)				_push_if(QSP_ARG  text)
#define reset_return_strings()			_reset_return_strings(SINGLE_QSP_ARG)
#define dupout(fp)				_dupout(QSP_ARG  fp)
#define check_adequate_return_strings(n)	_check_adequate_return_strings(QSP_ARG  n)

extern COMMAND_FUNC( suspend_chewing );
extern COMMAND_FUNC( unsuspend_chewing );


/* global variable(s) */
#ifdef THREAD_SAFE_QUERY
extern int n_active_threads;	// Number of qsp's

#ifdef HAVE_PTHREADS
extern void report_mutex_error(QSP_ARG_DECL  int status,const char *whence);
#endif /* HAVE_PTHREADS */

#endif /* THREAD_SAFE_QUERY */


#endif /* _QUERY_PROT_H_ */



