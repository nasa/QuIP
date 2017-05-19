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

// BUG - this really should be eliminated?
#include "llen.h"

// This used to be a macro - do we still need it?
extern Query * query_at_level(QSP_ARG_DECL  int l);
extern Query *new_query(void);
extern void rls_query(Query *);
extern int query_has_text(Query *);
extern void exit_current_file(SINGLE_QSP_ARG_DECL);
extern void exit_current_macro(SINGLE_QSP_ARG_DECL);
extern void set_query_arg_at_index(Query *qp,int index,const char *s);

extern int qs_level(SINGLE_QSP_ARG_DECL);
extern FILE * qs_msg_file(SINGLE_QSP_ARG_DECL);

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
#define PICK_PIPE(pmpt)		pick_pipe(QSP_ARG  pmpt)
#endif /* HAVE_POPEN */


// try_hard.c
extern FILE *try_hard( QSP_ARG_DECL  const char *filename, const char *mode );
extern FILE *trynice(QSP_ARG_DECL  const char *fnam, const char *mode);
extern QUIP_BOOL confirm(QSP_ARG_DECL  const char *pmpt);
extern FILE *try_open(QSP_ARG_DECL  const char *filename, const char *mode);

#define TRY_OPEN(s,m)		try_open(QSP_ARG  s,m)
#define TRY_HARD(s,m)		try_hard(QSP_ARG  s,m)
#define TRYNICE(s,m)		trynice(QSP_ARG  s,m)
#define CONFIRM(p)		confirm(QSP_ARG  p)

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

// list.c
extern void report_node_data(SINGLE_QSP_ARG_DECL);
extern count_t eltcount( List * lp );
#define NEW_LIST		new_list()
extern List *new_list(void);
extern void zap_list(List *lp);
extern Node *mk_node( void * ip );
extern Node *remHead(List *lp);
extern Node *remTail(List *lp);
extern Node * remNode(List *lp, Node *np);
extern Node *remData(List *lp, void * data);
extern void rls_list(List *lp);
extern void rls_nodes_from_list(List *lp);
extern void addTail(List *lp, Node *np);
extern void addHead( List *lp, Node* np );
extern void dellist(List *lp);
extern Node *nodeOf( List *lp, void * ip );
extern Node * list_find_named_item(List *lp, const char *name);

extern void p_sort(List *lp);
extern Node *nth_elt(List *lp, count_t k);
extern Node *nth_elt_from_tail(List *lp, count_t k);


// bi_menu.c
extern void set_discard_func(void (*func)(SINGLE_QSP_ARG_DECL) );
extern void init_aux_menus(Query_Stack *qsp);

// complete.c
extern void simulate_typing(const char *str);

// query_stack.h
extern Query_Stack *init_first_query_stack(void);
extern const char *nameof( QSP_ARG_DECL  const char *pmpt);
extern long how_many(QSP_ARG_DECL  const char *);
extern double how_much(QSP_ARG_DECL  const char *);
#define NAMEOF(s)			nameof(QSP_ARG  s)
#define HOW_MANY(pmpt)			how_many(QSP_ARG  pmpt)
#define HOW_MUCH(pmpt)			how_much(QSP_ARG  pmpt)


extern const char * nameof2(QSP_ARG_DECL  const char *prompt);
extern int askif( QSP_ARG_DECL  const char *pmpt);
#define ASKIF(p)		askif(QSP_ARG  p )


extern int which_one( QSP_ARG_DECL  const char *pmpt, int n, const char **choices );
extern int which_one2(QSP_ARG_DECL  const char* s,int n,const char** choices);
#define WHICH_ONE(p,n,ch)	which_one(QSP_ARG  p, n, ch )
#define WHICH_ONE2(p,n,ch)	which_one2(QSP_ARG  p, n, ch )

extern Query_Stack *new_qstk(QSP_ARG_DECL  const char *name);
ITEM_INIT_PROT(Query_Stack,query_stack)
ITEM_LIST_PROT(Query_Stack,query_stack)
ITEM_NEW_PROT(Query_Stack,query_stack)
ITEM_PICK_PROT(Query_Stack,query_stack)

extern Mouthful *new_mouthful(const char * text, const char *filename);

// item_type.c
ITEM_LIST_PROT(Item_Type,ittyp)
ITEM_PICK_PROT(Item_Type,ittyp)
#define PICK_ITTYP(pmpt)	pick_ittyp(QSP_ARG  pmpt)

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
extern char *			qs_expr_string(SINGLE_QSP_ARG_DECL);
#define QS_EXPR_STRING		qs_expr_string(SINGLE_QSP_ARG)

//#define	ascii_input_fmt		THIS_QSP->qs_dai_p->dai_input_fmt


extern char *error_string(SINGLE_QSP_ARG_DECL);
extern char *message_string(SINGLE_QSP_ARG_DECL);
#define ERROR_STRING		error_string(SINGLE_QSP_ARG)
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

extern void init_builtins(void);
extern void push_menu(QSP_ARG_DECL  Menu *mp);

/* chewtext.c */
extern void chew_text(QSP_ARG_DECL const char *, const char *filename );
//extern void digest(QSP_ARG_DECL  const char *text);
#define CHEW_TEXT(str,filename)		chew_text(QSP_ARG  str, filename)
//#define DIGEST(t)		digest(QSP_ARG t)

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
extern void script_warn(QSP_ARG_DECL  const char *);
extern void expect_warning(QSP_ARG_DECL  const char *);
#ifdef BUILD_FOR_IOS
extern void q_error1(QSP_ARG_DECL  const char *);
#else // ! BUILD_FOR_IOS
__attribute__ ((__noreturn__)) extern void q_error1(QSP_ARG_DECL  const char *);
#endif // ! BUILD_FOR_IOS
extern const char *savestr(const char *);
extern void rls_str(const char *);
extern COMMAND_FUNC( tog_pmpt );
extern void qdump( SINGLE_QSP_ARG_DECL );
extern FILE *tfile(SINGLE_QSP_ARG_DECL);
extern int intractive(SINGLE_QSP_ARG_DECL);
extern void set_args(QSP_ARG_DECL  int ac,char** av);
extern String_Buf * rdmtext(SINGLE_QSP_ARG_DECL);
extern Macro_Arg ** read_macro_arg_table(QSP_ARG_DECL int n);
extern Macro_Arg ** create_generic_macro_args(int n);
extern void create_macro(QSP_ARG_DECL  const char *name, int n, Macro_Arg **ma_tbl, String_Buf *sbp, int lineno);
extern void set_query_readfunc( QSP_ARG_DECL
	char * (*func)(QSP_ARG_DECL  void *buf, int size, void *fp ) );
extern void add_event_func(QSP_ARG_DECL  void (*func)(SINGLE_QSP_ARG_DECL) );
extern int rem_event_func(QSP_ARG_DECL  void (*func)(SINGLE_QSP_ARG_DECL) );
//extern void resume_chewing(SINGLE_QSP_ARG_DECL);
extern void resume_execution(SINGLE_QSP_ARG_DECL);
extern void resume_quip(SINGLE_QSP_ARG_DECL);
extern const char *query_filename(SINGLE_QSP_ARG_DECL);
extern void set_query_filename(Query *, const char *);
extern void print_qs_levels(QSP_ARG_DECL  int *level_to_print, int n_levels_to_print);
extern int *get_levels_to_print(QSP_ARG_DECL  int *n_ptr);

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

extern FILE *tell_msgfile(SINGLE_QSP_ARG_DECL);
extern FILE *tell_errfile(SINGLE_QSP_ARG_DECL);

extern Quip_Function *function_of(QSP_ARG_DECL  const char *name);
extern void list_vars(SINGLE_QSP_ARG_DECL);
extern Variable *get_var(QSP_ARG_DECL  const char *name);
#define GET_VAR(s)	get_var(QSP_ARG  s);


// error.c
extern int do_on_exit(void (*func)(SINGLE_QSP_ARG_DECL));
extern void nice_exit(QSP_ARG_DECL  int status);
extern const char *tell_progname(void);
extern const char *tell_version(void);
extern void tty_warn(QSP_ARG_DECL  const char *s);
extern void set_error_func(void (*func)(QSP_ARG_DECL  const char *));
extern void set_advise_func(void (*func)(QSP_ARG_DECL  const char *));
extern void set_prt_msg_frag_func(void (*func)(QSP_ARG_DECL  const char *));
extern void set_max_warnings(QSP_ARG_DECL  int n);
extern char *show_printable(QSP_ARG_DECL  const char* s);
extern void error_redir(QSP_ARG_DECL  FILE *fp);
extern void output_redir(QSP_ARG_DECL  FILE *fp);
extern void _log_message(QSP_ARG_DECL  const char *msg);
extern const char *get_date_string(SINGLE_QSP_ARG_DECL);

#define log_message(s)	_log_message(QSP_ARG  s);


// debug.c
extern void debug_module(QSP_ARG_DECL  const char *s);
extern debug_flag_t add_debug_module(QSP_ARG_DECL  const char *name);
extern void set_debug(QSP_ARG_DECL  Debug_Module *dbmp);
ITEM_INIT_PROT(Debug_Module,debug)
ITEM_NEW_PROT(Debug_Module,debug)
ITEM_CHECK_PROT(Debug_Module,debug)
ITEM_GET_PROT(Debug_Module,debug)
ITEM_PICK_PROT(Debug_Module,debug)
ITEM_LIST_PROT(Debug_Module,debug)


// rcfile.c
extern void rcfile( Query_Stack *qsp, char* progname );

// pathnm.c
extern void strip_fullpath(char **strp);
extern const char *parent_directory_of(const char *pathname);

// function.c ?
extern void declare_functions(SINGLE_QSP_ARG_DECL);

// callback.c
extern COMMAND_FUNC( call_event_funcs );

#ifdef HAVE_OPENGL
// glmenu module
extern COMMAND_FUNC( do_gl_menu );
extern COMMAND_FUNC( do_stereo_menu );
#endif /* HAVE_OPENGL */

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
extern void set_alarm_script(QSP_ARG_DECL  const char *s);
extern void set_alarm_time(QSP_ARG_DECL  float f);

// macro.c
extern const char *macro_text(Macro *);
#define MACRO_TEXT(mp)	macro_text(mp)
ITEM_INIT_PROT(Macro,macro)
ITEM_NEW_PROT(Macro,macro)
ITEM_CHECK_PROT(Macro,macro)
ITEM_PICK_PROT(Macro,macro)
ITEM_DEL_PROT(Macro,macro)
#define PICK_MACRO(pmpt)	pick_macro(QSP_ARG  pmpt)


// from query_stack.h

extern void start_quip(int argc, char **argv);
extern void start_quip_with_menu(int argc, char **argv, Menu *mp);
extern void push_first_menu(Query_Stack *qsp);

/* BUG - these prototypes should go somewhere else and be shared w/ ObjC!?!? */
extern int lookahead_til(QSP_ARG_DECL  int level);
extern int tell_qlevel(SINGLE_QSP_ARG_DECL);
extern Query * pop_file( SINGLE_QSP_ARG_DECL );

extern void redir_with_flags( QSP_ARG_DECL  FILE *fp, const char *filename, uint32_t flags );
extern void redir( QSP_ARG_DECL  FILE *fp, const char *filename );
extern void redir_from_pipe( QSP_ARG_DECL  Pipe *pp, const char *cmd );

extern void add_cmd_callback(QSP_ARG_DECL  void (*f)(SINGLE_QSP_ARG_DECL) );

extern const char *current_input_file(SINGLE_QSP_ARG_DECL);

extern void push_text(QSP_ARG_DECL const char *text, const char *filename );
extern void digest(QSP_ARG_DECL const char *text, const char *filename );
#define PUSH_TEXT(t,f)	push_text(QSP_ARG  t,f)

extern void push_top_menu(SINGLE_QSP_ARG_DECL);

extern void do_cmd(SINGLE_QSP_ARG_DECL);
extern Menu * pop_menu(SINGLE_QSP_ARG_DECL);
extern void enable_stripping_quotes(SINGLE_QSP_ARG_DECL);
extern void disable_stripping_quotes(SINGLE_QSP_ARG_DECL);
extern void lookahead(SINGLE_QSP_ARG_DECL);

extern Query_Stack *new_qstack(QSP_ARG_DECL  const char *name);

extern void qs_do_cmd(Query_Stack *qsp);

extern void open_loop(QSP_ARG_DECL  int n);
#define OPEN_LOOP(n)	open_loop(QSP_ARG  n)
extern void close_loop(SINGLE_QSP_ARG_DECL);
#define CLOSE_LOOP	close_loop(SINGLE_QSP_ARG)

extern void list_current_menu(SINGLE_QSP_ARG_DECL);
extern void list_builtin_menu(SINGLE_QSP_ARG_DECL);
#define LIST_CURRENT_MENU	list_current_menu(SINGLE_QSP_ARG)
#define LIST_BUILTIN_MENU	list_builtin_menu(SINGLE_QSP_ARG)


// variable.c
extern void init_variables(SINGLE_QSP_ARG_DECL);
extern void set_script_var_from_int(QSP_ARG_DECL  const char *varname, long val );
extern Variable *assign_var(QSP_ARG_DECL  const char *name, const char *value);
extern Variable *assign_reserved_var(QSP_ARG_DECL  const char *name, const char *value);
#define ASSIGN_VAR( s1 , s2 )	assign_var(QSP_ARG  s1 , s2 )
#define ASSIGN_RESERVED_VAR( s1 , s2 )	assign_reserved_var(QSP_ARG  s1 , s2 )
ITEM_PICK_PROT(Variable,var_)
#define VAR_OF(s)	var_of(QSP_ARG  s)
extern Variable *var_of(QSP_ARG_DECL const char *name);
#define PICK_VAR(s)	pick_var_(QSP_ARG  s)

// hash.c
extern void		zap_hash_tbl(Hash_Tbl *);
extern List *		ht_list(Hash_Tbl *);
extern Hash_Tbl *	enlarge_ht(Hash_Tbl *);
extern Hash_Tbl *	ht_init(const char *name);
extern int		insert_hash(void *ptr,Hash_Tbl *table);
extern void		show_ht(Hash_Tbl *table);
extern void *		fetch_hash(const char *name,Hash_Tbl *table);
//extern int		remove_hash(void *ptr,Hash_Tbl *table);
extern int		remove_name_from_hash(const char *name,Hash_Tbl *table);
extern int		remove_item_from_hash(const Item *ip,Hash_Tbl *table);
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
extern int sb_size(String_Buf *sbp);
//#define SB_BUF(sbp)	sb_buffer(sbp)

extern String_Buf *new_stringbuf(void);
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

extern Data_Obj *	pick_obj(QSP_ARG_DECL const char *pmpt);
#define PICK_OBJ(pmpt)		pick_obj(QSP_ARG   pmpt)

#define VWR_OF( s )	vwr_of(QSP_ARG  s )
#define GET_VWR( s )	get_vwr(QSP_ARG  s )
#define PICK_VWR( s )	pick_vwr(QSP_ARG  s )

// BUG this should be per-qsp
extern int quip_verbose;
// We need to un-define this when we declare the menu...
#define verbose quip_verbose

#include "warn.h"


#ifdef __cplusplus
}
#endif

#endif /* ! _QUIP_PROT_H_ */


