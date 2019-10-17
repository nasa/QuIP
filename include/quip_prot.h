#ifndef _QUIP_PROT_H_
#define _QUIP_PROT_H_

#ifdef __cplusplus
extern "C" {
#endif

#include "quip_config.h"

#ifndef CAUTIOUS
#define NDEBUG		// make assertions no-ops
#endif // ! CAUTIOUS

#ifdef HAVE_ASSERT_H
#include <assert.h>
#else // ! HAVE_ASSERT_H
#define assert(c)
#endif // ! HAVE_ASSERT_H

#ifdef __cplusplus
extern "C" {
#endif

#ifndef TRUE
#define TRUE 1
#endif // ! TRUE
#ifndef FALSE
#define FALSE 0
#endif // ! FALSE

#include <stdio.h>
#include "quip_fwd.h"	// forward definitions of structs and typedefs

#include "debug.h"	// to pick up AERROR...

// BUG - this really should be eliminated?
#include "llen.h"

// This used to be a macro - do we still need it?
extern Query * query_at_level(QSP_ARG_DECL  int l);
extern Query *_new_query(SINGLE_QSP_ARG_DECL);
#define new_query() _new_query(SINGLE_QSP_ARG)

extern void _rls_query(QSP_ARG_DECL  Query *);
#define rls_query(qp) _rls_query(QSP_ARG  qp)

extern int query_has_text(Query *);
extern void exit_current_file(SINGLE_QSP_ARG_DECL);
extern void exit_current_macro(SINGLE_QSP_ARG_DECL);
extern void set_query_arg_at_index(Query *qp,int index,const char *s);
extern void _rls_macro(QSP_ARG_DECL  Macro *mp);
#define rls_macro(mp) _rls_macro(QSP_ARG  mp)

extern int qs_level(SINGLE_QSP_ARG_DECL);
extern FILE * _qs_msg_file(SINGLE_QSP_ARG_DECL);
#define qs_msg_file() _qs_msg_file(SINGLE_QSP_ARG)

extern void push_vector_parser_data(SINGLE_QSP_ARG_DECL);
extern void pop_vector_parser_data(SINGLE_QSP_ARG_DECL);

extern Vector_Parser_Data	* new_vector_parser_data(void);

extern Vector_Parser_Data	* _qs_vector_parser_data(SINGLE_QSP_ARG_DECL);
#define qs_vector_parser_data() _qs_vector_parser_data(SINGLE_QSP_ARG)

extern List	* qs_vector_parser_data_freelist(SINGLE_QSP_ARG_DECL);
extern List	* qs_vector_parser_data_stack(SINGLE_QSP_ARG_DECL);

extern void set_vector_parser_data_stack(QSP_ARG_DECL  List *);
extern void set_vector_parser_data_freelist(QSP_ARG_DECL  List *);
extern void set_vector_parser_data(QSP_ARG_DECL  Vector_Parser_Data *);


// quip_main.c
extern void push_quip_menu(Query_Stack *qsp);
extern void exec_quip(SINGLE_QSP_ARG_DECL);
extern void exec_this_level(SINGLE_QSP_ARG_DECL);
extern void exec_at_level(QSP_ARG_DECL  int level);
// query_funcs.c
extern void finish_swallowing(SINGLE_QSP_ARG_DECL);
extern void swallow(QSP_ARG_DECL const char *text, const char *filename);
extern void chew_mouthful(Mouthful *mfp);
extern int current_line_number(SINGLE_QSP_ARG_DECL);
extern const char * current_filename(SINGLE_QSP_ARG_DECL);

#ifdef BUILD_FOR_OBJC
extern void ios_exit_program(void);
#endif /* BUILD_FOR_OBJC */

extern COMMAND_FUNC( do_encrypt_string );
extern COMMAND_FUNC( do_decrypt_string );
extern COMMAND_FUNC( do_encrypt_file );
extern COMMAND_FUNC( do_decrypt_file );
extern COMMAND_FUNC( do_read_encrypted_file );

//extern void exec_pending_commands(SINGLE_QSP_ARG_DECL);

// substr.c
extern  int is_a_substring(const char *s, const char *w);

// unix_menu.c
extern COMMAND_FUNC( do_unix_menu );

// BUG these are really local to unix module
// fork.c
extern COMMAND_FUNC( do_fork );
extern COMMAND_FUNC( do_wait_child );
// threads.c
extern COMMAND_FUNC( do_thread_menu );
// timex.c
extern COMMAND_FUNC( do_timex_menu );


// pipes.c		// BUG should be local to module
#ifdef HAVE_POPEN
extern COMMAND_FUNC( do_pipe_menu );
extern void close_pipe(QSP_ARG_DECL  Pipe *pp);
ITEM_INTERFACE_PROTOTYPES( Pipe, pipe )

#define pick_pipe(pmpt)		_pick_pipe(QSP_ARG  pmpt)
#define init_pipes()		_init_pipes(SINGLE_QSP_ARG)
#define pipe_of(name)		_pipe_of(QSP_ARG  name)
#define new_pipe(name)		_new_pipe(QSP_ARG  name)
#define del_pipe(name)		_del_pipe(QSP_ARG  name)
#define list_pipes(fp)		_list_pipes(QSP_ARG  fp)

#endif /* HAVE_POPEN */


// try_hard.c
extern FILE *_try_hard( QSP_ARG_DECL  const char *filename, const char *mode );
extern FILE *_try_nice(QSP_ARG_DECL  const char *fnam, const char *mode);
extern QUIP_BOOL _confirm(QSP_ARG_DECL  const char *pmpt);
extern FILE *_try_open(QSP_ARG_DECL  const char *filename, const char *mode);

#define try_open(s,m)		_try_open(QSP_ARG  s,m)
#define try_hard(s,m)		_try_hard(QSP_ARG  s,m)
#define try_nice(s,m)		_try_nice(QSP_ARG  s,m)
#define confirm(p)		_confirm(QSP_ARG  p)

extern COMMAND_FUNC( togclobber );


// my_stty.c		// BUG should be local to module
extern COMMAND_FUNC( do_stty_menu );

// stamps.c		// BUG should be local to module
extern COMMAND_FUNC( do_stamp_menu );

// mouse.c		// BUG should be local to module
extern COMMAND_FUNC( do_mouse_menu );

// serial.c		// BUG should be local to module
extern COMMAND_FUNC( do_ser_menu );

// history.c
extern int history_flag;

// bi_menu.c
extern void set_discard_func(void (*func)(SINGLE_QSP_ARG_DECL) );
extern void init_aux_menus(Query_Stack *qsp);

// complete.c
extern void _simulate_typing(QSP_ARG_DECL  const char *str);
#define simulate_typing(str) _simulate_typing(QSP_ARG  str)

// query_stack.h
extern Query_Stack *init_first_query_stack(void);
extern const char *_nameof( QSP_ARG_DECL  const char *pmpt);
#define howmany_type	int64_t	// previously long...
extern howmany_type _how_many(QSP_ARG_DECL  const char *);
extern double _how_much(QSP_ARG_DECL  const char *);
// gradually phase these all-caps versions out...
#define NAMEOF(s)		_nameof(QSP_ARG  s)
#define HOW_MANY(pmpt)		_how_many(QSP_ARG  pmpt)
#define HOW_MUCH(pmpt)		_how_much(QSP_ARG  pmpt)
#define ASKIF(p)		_askif(QSP_ARG  p )
#define WHICH_ONE(p,n,ch)	_which_one(QSP_ARG  p, n, ch )

#define nameof(s)		_nameof(QSP_ARG  s)
#define how_many(pmpt)		_how_many(QSP_ARG  pmpt)
#define how_much(pmpt)		_how_much(QSP_ARG  pmpt)
#define askif(p)		_askif(QSP_ARG  p )
#define which_one(p,n,ch)	_which_one(QSP_ARG  p, n, ch )
#define format_prompt(fmt,prompt) _format_prompt(QSP_ARG  fmt,prompt)


extern const char *_format_prompt(QSP_ARG_DECL  const char *fmt, const char *prompt);
extern void inhibit_next_prompt_format(SINGLE_QSP_ARG_DECL);
extern void enable_prompt_format(SINGLE_QSP_ARG_DECL);
extern int _askif( QSP_ARG_DECL  const char *pmpt);


extern int _which_one( QSP_ARG_DECL  const char *pmpt, int n, const char **choices );

extern Query_Stack *new_qstk(QSP_ARG_DECL  const char *name);
ITEM_INIT_PROT(Query_Stack,query_stack)
ITEM_LIST_PROT(Query_Stack,query_stack)
ITEM_NEW_PROT(Query_Stack,query_stack)
ITEM_PICK_PROT(Query_Stack,query_stack)

#define init_query_stacks()	_init_query_stacks(SINGLE_QSP_ARG)
#define query_stack_list()	_query_stack_list(SINGLE_QSP_ARG)
#define new_query_stack(name)	_new_query_stack(QSP_ARG  name)
#define pick_query_stack(p)	_pick_query_stack(QSP_ARG  p)
#define list_query_stacks(fp)	_list_query_stacks(QSP_ARG  fp)

extern Mouthful *new_mouthful(const char * text, const char *filename);

// item_type.c
ITEM_LIST_PROT(Item_Type,ittyp)
ITEM_PICK_PROT(Item_Type,ittyp)
#define pick_ittyp(pmpt)	_pick_ittyp(QSP_ARG  pmpt)

#ifdef HAVE_LIBCURL
//extern String_Buf *curl_stringbuf(SINGLE_QSP_ARG_DECL);
//#define CURL_STRINGBUF	curl_stringbuf(SINGLE_QSP_ARG)
extern Curl_Info *		qs_curl_info(SINGLE_QSP_ARG_DECL);
#define QS_CURL_INFO		qs_curl_info(SINGLE_QSP_ARG)

#endif // HAVE_LIBCURL


extern String_Buf *		qs_scratch_buffer(SINGLE_QSP_ARG_DECL);
#define QS_SCRATCH		qs_scratch_buffer(SINGLE_QSP_ARG)

extern Input_Format_Spec *	qs_ascii_input_format(SINGLE_QSP_ARG_DECL);
#define	ascii_input_fmt_tbl	qs_ascii_input_format(SINGLE_QSP_ARG)

extern int			qs_serial_func(SINGLE_QSP_ARG_DECL);
#define QS_SERIAL		qs_serial_func(SINGLE_QSP_ARG)

#define CURRENT_FILENAME	current_filename(SINGLE_QSP_ARG)

extern const char *		qs_curr_string(SINGLE_QSP_ARG_DECL);
extern void			set_curr_string(QSP_ARG_DECL  const char *);
#define CURR_STRING		qs_curr_string(SINGLE_QSP_ARG)
#define SET_CURR_STRING(s)	set_curr_string(QSP_ARG  s)

extern Vec_Expr_Node *		qs_top_node(SINGLE_QSP_ARG_DECL);
#define TOP_NODE		qs_top_node(SINGLE_QSP_ARG)
extern void			set_top_node(QSP_ARG_DECL  Vec_Expr_Node *);
#define SET_TOP_NODE(enp)	set_top_node(QSP_ARG  enp)

#define VEXP_STR		QS_EXPR_STRING
extern String_Buf *		qs_expr_string(SINGLE_QSP_ARG_DECL);
#define QS_EXPR_STRING		qs_expr_string(SINGLE_QSP_ARG)

//#define	ascii_input_fmt		THIS_QSP->qs_dai_p->dai_input_fmt


extern char *error_string(SINGLE_QSP_ARG_DECL);
extern char *vector_parser_error_string(SINGLE_QSP_ARG_DECL);
extern char *message_string(SINGLE_QSP_ARG_DECL);
#define ERROR_STRING		error_string(SINGLE_QSP_ARG)
#define VECTOR_PARSER_ERROR_STRING		vector_parser_error_string(SINGLE_QSP_ARG)
#define DEFAULT_ERROR_STRING	error_string(SGL_DEFAULT_QSP_ARG)
#define MSG_STR			message_string(SINGLE_QSP_ARG)
#define DEFAULT_MSG_STR		message_string(SGL_DEFAULT_QSP_ARG)

#define msg_str 		MSG_STR

// quip_menu.h

extern COMMAND_FUNC( do_pop_menu );
extern COMMAND_FUNC( do_exit_prog );
extern COMMAND_FUNC( do_dobj_menu );

// features.c
extern COMMAND_FUNC( do_list_features );

extern void _init_builtins(SINGLE_QSP_ARG_DECL);
#define init_builtins() _init_builtins(SINGLE_QSP_ARG)
extern void _push_menu(QSP_ARG_DECL  Menu *mp);
#define push_menu(mp) _push_menu(QSP_ARG  mp)

/* chewtext.c */
extern void _chew_text(QSP_ARG_DECL const char *, const char *filename );
#define chew_text(str,filename)		_chew_text(QSP_ARG  str, filename)

/* dict.c */

extern Dictionary *	create_dictionary(const char *);
#define dictionary_list(dict_p)	_dictionary_list(QSP_ARG  dict_p)
extern List *		_dictionary_list(QSP_ARG_DECL  Dictionary *dict_p);
extern void		delete_dictionary(Dictionary *dict_p);
extern Item *		fetch_name(const char *name,Dictionary *dict_p);
extern int		insert_name(Item *ip,Node *np,Dictionary *dict_p);
#define cat_dict_items(lp,dict_p)	_cat_dict_items(QSP_ARG  lp,dict_p)
extern void		_cat_dict_items(QSP_ARG_DECL  List *lp,Dictionary *dict_p);
extern void		tell_name_stats(QSP_ARG_DECL  Dictionary *dict_p);
extern int		remove_name(Item *ip,Dictionary *dict_p);
#define dump_dict_info(dict_p)	_dump_dict_info(QSP_ARG  dict_p)
extern void		_dump_dict_info(QSP_ARG_DECL  Dictionary *dict_p);




// query_funcs.c

extern char *qpfgets(QSP_ARG_DECL  void *buf, int size, void *fp);
extern void input_on_stdin(void);
extern void _warn(QSP_ARG_DECL  const char *);
extern void _expect_warning(QSP_ARG_DECL  const char *);
#define expect_warning(s) _expect_warning(QSP_ARG  s)

extern void _check_expected_warnings(QSP_ARG_DECL  int clear_flag);
#define check_expected_warnings(clear_flag) _check_expected_warnings(QSP_ARG  clear_flag)

#ifdef BUILD_FOR_IOS
extern void q_error1(QSP_ARG_DECL  const char *);
#else // ! BUILD_FOR_IOS
__attribute__ ((__noreturn__)) extern void q_error1(QSP_ARG_DECL  const char *);
#endif // ! BUILD_FOR_IOS
extern const char *savestr(const char *);
extern const char *save_possibly_empty_str(const char *);
extern void rls_str(const char *);
extern COMMAND_FUNC( tog_pmpt );
extern void qdump( SINGLE_QSP_ARG_DECL );
extern FILE * _tfile(SINGLE_QSP_ARG_DECL);
#define tfile()					_tfile(SINGLE_QSP_ARG)
extern int _intractive(SINGLE_QSP_ARG_DECL);
#define intractive() _intractive(SINGLE_QSP_ARG)

extern void _set_args(QSP_ARG_DECL  int ac,char** av);
extern String_Buf * _read_macro_body(SINGLE_QSP_ARG_DECL);
extern Macro_Arg ** _setup_macro_args(QSP_ARG_DECL int n);
extern Macro_Arg ** _create_generic_macro_args(QSP_ARG_DECL  int n);

#define set_args(ac,av) _set_args(QSP_ARG  ac,av)
#define read_macro_body() _read_macro_body(SINGLE_QSP_ARG)
#define setup_macro_args(n) _setup_macro_args(QSP_ARG n)
#define create_generic_macro_args(n) _create_generic_macro_args(QSP_ARG  n)

extern Macro * _create_macro(QSP_ARG_DECL  const char *name, int n, Macro_Arg **ma_tbl, String_Buf *sbp, int lineno);
#define create_macro(name, n, ma_tbl, sbp, lineno) _create_macro(QSP_ARG  name, n, ma_tbl, sbp, lineno)

extern void set_query_readfunc( QSP_ARG_DECL
	char * (*func)(QSP_ARG_DECL  void *buf, int size, void *fp ) );
extern void resume_execution(SINGLE_QSP_ARG_DECL);
extern void resume_quip(SINGLE_QSP_ARG_DECL);
extern void _suspend_quip(SINGLE_QSP_ARG_DECL);
#define suspend_quip() _suspend_quip(SINGLE_QSP_ARG)
extern const char *query_filename(SINGLE_QSP_ARG_DECL);
extern void set_query_filename(Query *, const char *);
extern void set_query_macro(Query *,Macro *);
extern void set_query_args(Query *,const char **);
// BUG should be in local header file, libsrc/interpreter...
extern void report_qs_line_numbers(QSP_ARG_DECL  int *level_to_print, int n_levels_to_print);
extern int *get_levels_to_print(QSP_ARG_DECL  int *n_ptr);

extern void _add_event_func(QSP_ARG_DECL  void (*func)(SINGLE_QSP_ARG_DECL) );
extern int _rem_event_func(QSP_ARG_DECL  void (*func)(SINGLE_QSP_ARG_DECL) );

#define add_event_func(func)	_add_event_func(QSP_ARG  func)
#define rem_event_func(func)	_rem_event_func(QSP_ARG  func)

#ifdef HAVE_HISTORY
#ifdef TTY_CTL
extern COMMAND_FUNC( set_completion );
#endif /* TTY_CTL */
#endif /* HAVE_HISTORY */

//extern void pushtext(QSP_ARG_DECL const char *text);

extern void show_menu_stack(SINGLE_QSP_ARG_DECL);

// for stdC only
extern Item_Type *ittyp_itp;
extern Item_Type *macro_itp;

extern FILE *_tell_msgfile(SINGLE_QSP_ARG_DECL);
extern FILE *_tell_errfile(SINGLE_QSP_ARG_DECL);

#define tell_errfile()	_tell_errfile(SINGLE_QSP_ARG)
#define tell_msgfile()	_tell_msgfile(SINGLE_QSP_ARG)

// BUG?  why not use ITEM macro here?
extern Quip_Function *_function_of(QSP_ARG_DECL  const char *name);
extern void _list_vars(SINGLE_QSP_ARG_DECL);
extern Variable *_get_var(QSP_ARG_DECL  const char *name);
#define list_vars()	_list_vars(SINGLE_QSP_ARG);
#define get_var(s)	_get_var(QSP_ARG  s);


// error.c
extern const char *tell_version(void);
extern void set_error_func(void (*func)(QSP_ARG_DECL  const char *));
extern void set_advise_func(void (*func)(QSP_ARG_DECL  const char *));
extern void set_prt_msg_frag_func(void (*func)(QSP_ARG_DECL  const char *));
extern int string_is_printable(const char *s);

extern void _tty_warn(QSP_ARG_DECL  const char *s);
extern void _set_max_warnings(QSP_ARG_DECL  int n);
extern char *_show_printable(QSP_ARG_DECL  const char* s);

#define tty_warn(s) _tty_warn(QSP_ARG  s)
#define set_max_warnings(n) _set_max_warnings(QSP_ARG  n)
#define show_printable(s) _show_printable(QSP_ARG  s)

extern void _error_redir(QSP_ARG_DECL  FILE *fp);
extern void _output_redir(QSP_ARG_DECL  FILE *fp);

#define error_redir(fp) _error_redir(QSP_ARG  fp)
#define output_redir(fp) _output_redir(QSP_ARG  fp)

extern const char *get_date_string(SINGLE_QSP_ARG_DECL);

extern void _log_message(QSP_ARG_DECL  const char *msg);
extern void _nice_exit(QSP_ARG_DECL  int status);
extern const char *_tell_progname(SINGLE_QSP_ARG_DECL);
extern int _do_on_exit(QSP_ARG_DECL  void (*func)(SINGLE_QSP_ARG_DECL));

#define log_message(s)	_log_message(QSP_ARG  s);
#define nice_exit(s)	_nice_exit(QSP_ARG  s);
#define tell_progname() _tell_progname(SINGLE_QSP_ARG)
#define do_on_exit(func) _do_on_exit(QSP_ARG  func)


// debug.c
extern void debug_module(QSP_ARG_DECL  const char *s);
extern debug_flag_t _add_debug_module(QSP_ARG_DECL  const char *name);
extern void _set_debug(QSP_ARG_DECL  Debug_Module *dbmp);

#define add_debug_module(n)	_add_debug_module(QSP_ARG  n)

ITEM_INIT_PROT(Debug_Module,debug)
ITEM_NEW_PROT(Debug_Module,debug)
ITEM_CHECK_PROT(Debug_Module,debug)
ITEM_GET_PROT(Debug_Module,debug)
ITEM_PICK_PROT(Debug_Module,debug)
ITEM_LIST_PROT(Debug_Module,debug)

#define set_debug(dbmp)	_set_debug(QSP_ARG  dbmp)
#define init_debugs()	_init_debugs(SINGLE_QSP_ARG)
#define new_debug(name)	_new_debug(QSP_ARG  name)
#define debug_of(name)	_debug_of(QSP_ARG  name)
#define get_debug(name)	_get_debug(QSP_ARG  name)
#define pick_debug(prompt)	_pick_debug(QSP_ARG  prompt)
#define list_debugs(fp)	_list_debugs(QSP_ARG  fp)


// rcfile.c
extern void rcfile( Query_Stack *qsp, char* progname );

// pathnm.c
extern void strip_fullpath(char **strp);
extern const char *_parent_directory_of(QSP_ARG_DECL  const char *pathname);
#define parent_directory_of(pathname) _parent_directory_of(QSP_ARG  pathname)

// function.c ?
extern void declare_functions(SINGLE_QSP_ARG_DECL);

// callback.c
extern COMMAND_FUNC( call_event_funcs );

#ifdef HAVE_OPENGL
// glmenu module
extern COMMAND_FUNC( do_gl_menu );
#endif /* HAVE_OPENGL */
extern COMMAND_FUNC( do_stereo_menu );

// BUG - should this go somewhere else???
extern COMMAND_FUNC( do_port_menu );
extern COMMAND_FUNC( do_rv_menu );
extern COMMAND_FUNC( do_movie_menu );
extern COMMAND_FUNC( do_mseq_menu );
//extern COMMAND_FUNC( do_staircase_menu );
extern COMMAND_FUNC( do_exp_menu );
extern COMMAND_FUNC( do_sound_menu );
extern COMMAND_FUNC( do_requant );
extern COMMAND_FUNC( do_knox_menu );
extern COMMAND_FUNC( do_step_menu );
extern COMMAND_FUNC( do_nr_menu );
extern COMMAND_FUNC( do_ocv_menu );
extern COMMAND_FUNC( do_gsl_menu );
extern COMMAND_FUNC( do_atc_menu );
extern COMMAND_FUNC( do_pgr_menu );
extern COMMAND_FUNC( do_fly_menu );
extern COMMAND_FUNC( do_spink_menu );
extern COMMAND_FUNC( do_seq_menu );
extern COMMAND_FUNC( do_meteor_menu );
extern COMMAND_FUNC( do_pic_menu );
extern COMMAND_FUNC( do_visca_menu );
extern COMMAND_FUNC( do_v4l2_menu );
extern COMMAND_FUNC( do_dv_menu );
extern COMMAND_FUNC( do_aio_menu );
extern COMMAND_FUNC( do_parport_menu );

extern COMMAND_FUNC( do_cuda_menu );

extern COMMAND_FUNC( do_platform_menu );

extern COMMAND_FUNC( do_fann_menu );

// freel.c
extern COMMAND_FUNC( heap_report );

// these are part of OS support - where should they really go?
extern void _set_alarm_script(QSP_ARG_DECL  const char *s);
extern void _set_alarm_time(QSP_ARG_DECL  float f);
#define set_alarm_script(s) _set_alarm_script(QSP_ARG  s)
#define set_alarm_time(f) _set_alarm_time(QSP_ARG  f)

// macro.c
extern const char *macro_text(Macro *);
#define MACRO_TEXT(mp)	macro_text(mp)
ITEM_INIT_PROT(Macro,macro)
ITEM_NEW_PROT(Macro,macro)
ITEM_CHECK_PROT(Macro,macro)
ITEM_PICK_PROT(Macro,macro)
ITEM_DEL_PROT(Macro,macro)

#define init_macros()		_init_macros(SINGLE_QSP_ARG)
#define new_macro(name)		_new_macro(QSP_ARG  name)
#define macro_of(name)		_macro_of(QSP_ARG  name)
#define pick_macro(pmpt)	_pick_macro(QSP_ARG  pmpt)
#define del_macro(name)		_del_macro(QSP_ARG  name)


// from query_stack.h

extern void start_quip(int argc, char **argv);
extern void start_quip_with_menu(int argc, char **argv, Menu *mp);
extern void push_first_menu(Query_Stack *qsp);

/* BUG - these prototypes should go somewhere else and be shared w/ ObjC!?!? */
extern int _lookahead_til(QSP_ARG_DECL  int level);
#define lookahead_til(level) _lookahead_til(QSP_ARG  level)

extern Query * _pop_file( SINGLE_QSP_ARG_DECL );
#define pop_file() _pop_file(SINGLE_QSP_ARG)

extern void redir_with_flags( QSP_ARG_DECL  FILE *fp, const char *filename, uint32_t flags );
extern void _redir( QSP_ARG_DECL  FILE *fp, const char *filename );
extern void _redir_from_pipe( QSP_ARG_DECL  Pipe *pp, const char *cmd );

#define redir(fp,s)	_redir(QSP_ARG  fp,s)
#define redir_from_pipe( pp, cmd ) _redir_from_pipe( QSP_ARG  pp, cmd )

extern void add_cmd_callback(QSP_ARG_DECL  void (*f)(SINGLE_QSP_ARG_DECL) );

extern const char *current_input_file(SINGLE_QSP_ARG_DECL);

extern int _max_vectorizable(SINGLE_QSP_ARG_DECL);
extern void _set_max_vectorizable(QSP_ARG_DECL  int v);

#define max_vectorizable() _max_vectorizable(SINGLE_QSP_ARG)
#define set_max_vectorizable(v) _set_max_vectorizable(QSP_ARG  v)

extern void _push_text(QSP_ARG_DECL const char *text, const char *filename );
extern void _digest(QSP_ARG_DECL const char *text, const char *filename );
#define digest(text,filename) _digest(QSP_ARG text,filename)
#define push_text(t,f)	_push_text(QSP_ARG  t,f)

extern void push_top_menu(SINGLE_QSP_ARG_DECL);
extern void _pop_to_top_menu(SINGLE_QSP_ARG_DECL);
#define pop_to_top_menu() _pop_to_top_menu(SINGLE_QSP_ARG)

extern void do_cmd(SINGLE_QSP_ARG_DECL);
extern void enable_stripping_quotes(SINGLE_QSP_ARG_DECL);
extern void disable_stripping_quotes(SINGLE_QSP_ARG_DECL);
extern void lookahead(SINGLE_QSP_ARG_DECL);

extern Menu * _pop_menu(SINGLE_QSP_ARG_DECL);

#define pop_menu() _pop_menu(SINGLE_QSP_ARG)

extern Query_Stack *new_qstack(QSP_ARG_DECL  const char *name);

extern void qs_do_cmd(Query_Stack *qsp);

// BUG move to local include in libsrc/interpreter
extern void _open_loop(QSP_ARG_DECL  int n);
extern void _end_loop(SINGLE_QSP_ARG_DECL);
extern void _open_while_loop(SINGLE_QSP_ARG_DECL);
#define open_loop(n)	_open_loop(QSP_ARG  n)
#define end_loop()	_end_loop(SINGLE_QSP_ARG)
#define open_while_loop() _open_while_loop(SINGLE_QSP_ARG)

extern void _list_current_menu(SINGLE_QSP_ARG_DECL);
extern void _list_builtin_menu(SINGLE_QSP_ARG_DECL);
#define list_current_menu()	_list_current_menu(SINGLE_QSP_ARG)
#define list_builtin_menu()	_list_builtin_menu(SINGLE_QSP_ARG)


// variable.c
extern void init_variables(SINGLE_QSP_ARG_DECL);
extern void _set_script_var_from_int(QSP_ARG_DECL  const char *varname, long val );
#define set_script_var_from_int(varname, val ) _set_script_var_from_int(QSP_ARG  varname, val )

extern Variable *_assign_var(QSP_ARG_DECL  const char *name, const char *value);
extern Variable *_assign_reserved_var(QSP_ARG_DECL  const char *name, const char *value);
#define assign_var( s1 , s2 )	_assign_var(QSP_ARG  s1 , s2 )
#define assign_reserved_var( s1 , s2 )	_assign_reserved_var(QSP_ARG  s1 , s2 )
ITEM_PICK_PROT(Variable,var_)
#define var_of(s)	_var_of(QSP_ARG  s)
extern Variable *_var_of(QSP_ARG_DECL const char *name);
#define pick_var(s)	_pick_var_(QSP_ARG  s)


// hash.c
extern void		_zap_hash_tbl(QSP_ARG_DECL  Hash_Tbl *);
extern List *		_ht_list(QSP_ARG_DECL  Hash_Tbl *);
#define zap_hash_tbl(htp) _zap_hash_tbl(QSP_ARG  htp)
#define ht_list(htp) _ht_list(QSP_ARG  htp)

extern Hash_Tbl *	_enlarge_ht(QSP_ARG_DECL  Hash_Tbl *);
#define enlarge_ht(htp) _enlarge_ht(QSP_ARG  htp)

extern Hash_Tbl *	_ht_init(QSP_ARG_DECL  const char *name);
#define ht_init(name) _ht_init(QSP_ARG  name)

extern int		_insert_hash(QSP_ARG_DECL  void *ptr,Hash_Tbl *table);
#define insert_hash(ptr,table) _insert_hash(QSP_ARG  ptr,table)

extern void		_show_ht(QSP_ARG_DECL  Hash_Tbl *table);
#define show_ht(table) _show_ht(QSP_ARG  table)

extern void *		_fetch_hash(QSP_ARG_DECL  const char *name,Hash_Tbl *table);
#define fetch_hash(name,table) _fetch_hash(QSP_ARG  name,table)

//extern int		remove_hash(void *ptr,Hash_Tbl *table);
extern int		_remove_name_from_hash(QSP_ARG_DECL  const char *name,Hash_Tbl *table);
#define remove_name_from_hash(name,table) _remove_name_from_hash(QSP_ARG  name,table)

extern int		_remove_item_from_hash(QSP_ARG_DECL  const Item *ip,Hash_Tbl *table);
#define remove_item_from_hash(ip,table) _remove_item_from_hash(QSP_ARG  ip,table)

extern void		tell_hash_stats(QSP_ARG_DECL  Hash_Tbl *table);


// strbuf.c
extern void enlarge_buffer(String_Buf *sbp,size_t size);
extern void copy_string(String_Buf *sbp,const char *str);
extern void copy_strbuf(String_Buf *dst_sbp,String_Buf *src_sbp);
extern void cat_string(String_Buf *sbp,const char *str);
extern void copy_string_n(String_Buf *sbp,const char *str,int n);
extern void cat_string_n(String_Buf *sbp,const char *str, int n);
extern char *sb_buffer(String_Buf *sbp);
extern void rls_sb_buffer(String_Buf *sbp);
extern size_t sb_size(String_Buf *sbp);
//#define SB_BUF(sbp)	sb_buffer(sbp)

extern String_Buf *new_stringbuf(void);
extern String_Buf *create_stringbuf(const char *s);
extern void rls_stringbuf(String_Buf *);


extern COMMAND_FUNC(do_protomenu);

extern int is_portrait(void);

#ifdef BUILD_FOR_IOS
extern int ios_read_global_startup(SINGLE_QSP_ARG_DECL);
extern void sync_with_ios(void);
extern uint64_t my_absolute_to_nanoseconds( uint64_t *t );
#endif /* BUILD_FOR_IOS */

#ifdef BUILD_FOR_MACOS
extern int macos_read_global_startup(SINGLE_QSP_ARG_DECL);
#endif /* BUILD_FOR_MACOS */

#ifdef __cplusplus
}
#endif

extern COMMAND_FUNC(do_show_prompt);
#define SHOWP do_show_prompt(SINGLE_QSP_ARG);

#ifdef THREAD_SAFE_QUERY
extern void report_mutex_error(QSP_ARG_DECL  int status, const char *whence);
#endif // THREAD_SAFE_QUERY

extern Data_Obj *	_pick_obj(QSP_ARG_DECL const char *pmpt);
#define pick_obj(pmpt)		_pick_obj(QSP_ARG   pmpt)

#define vwr_of( s )	_vwr_of(QSP_ARG  s )
#define get_vwr( s )	_get_vwr(QSP_ARG  s )
#define pick_vwr( s )	_pick_vwr(QSP_ARG  s )
#define list_vwrs( fp )	_list_vwrs(QSP_ARG  fp )
#define init_vwrs()	_init_vwrs(SINGLE_QSP_ARG)
#define del_vwr( s )	_del_vwr(QSP_ARG  s )
#define new_vwr( s )	_new_vwr(QSP_ARG  s )

// BUG this should be per-qsp
extern int quip_verbose;
// We need to un-define this when we declare the menu...
#define verbose quip_verbose

#include "warn.h"


#ifdef __cplusplus
}
#endif

#endif /* ! _QUIP_PROT_H_ */


