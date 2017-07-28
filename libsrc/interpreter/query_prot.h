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
extern void rls_macro_arg(Macro_Arg *map);
extern int macro_is_invoked(Macro *);
extern void allow_recursion_for_macro(Macro *);
extern void rls_macro(QSP_ARG_DECL  Macro *);
extern const char * macro_filename(Macro *);
extern int macro_lineno(Macro *);


extern const char * nextline(QSP_ARG_DECL const char *pline);
//extern const char * gword(QSP_ARG_DECL const char *pline);
extern const char * next_query_word(QSP_ARG_DECL const char *pline);
extern const char * qline(QSP_ARG_DECL const char *pline);
extern void init_query_stack( Query_Stack *qsp );
extern void dup_word(QSP_ARG_DECL const char *);
extern int dupout(QSP_ARG_DECL  FILE *fp);
extern FILE *tfile(SINGLE_QSP_ARG_DECL);
extern void	set_progname(const char *program_name);
extern Item *	new_item(QSP_ARG_DECL  Item_Type * item_type,const char *name,size_t size);
extern void inhibit_next_prompt_format(SINGLE_QSP_ARG_DECL);
extern void enable_prompt_format(SINGLE_QSP_ARG_DECL);
extern void _whileloop(QSP_ARG_DECL  int value);
extern void foreach_loop(QSP_ARG_DECL Foreach_Loop *frp);
extern void zap_fore(Foreach_Loop *frp);
extern void push_if(QSP_ARG_DECL const char *text);
//extern void swallow(QSP_ARG_DECL  const char *text);
extern int check_adequate_return_strings(QSP_ARG_DECL  int n);
extern void reset_return_strings(SINGLE_QSP_ARG_DECL);

extern COMMAND_FUNC( suspend_chewing );
extern COMMAND_FUNC( unsuspend_chewing );

#define ASSIGN_VAR( s1 , s2 )	assign_var(QSP_ARG  s1 , s2 )



/* global variable(s) */
#ifdef THREAD_SAFE_QUERY
extern int n_active_threads;	// Number of qsp's

#ifdef HAVE_PTHREADS
extern void report_mutex_error(QSP_ARG_DECL  int status,const char *whence);
#endif /* HAVE_PTHREADS */

#endif /* THREAD_SAFE_QUERY */


#endif /* _QUERY_PROT_H_ */



