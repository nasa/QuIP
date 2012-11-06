

/* callback.c */
extern void add_event_func( void (*func)(SINGLE_QSP_ARG_DECL) );

#ifdef FOOBAR
extern int rem_event_func( void (*func)(SINGLE_QSP_ARG_DECL) );
extern List *new_callback_list(void);
extern void add_callback_func(List *,void (*func)(void) );
extern void call_callback_list(List *);
#endif /* FOOBAR */

/* builtin.c */
extern void set_discard_func( void (*func)(SINGLE_QSP_ARG_DECL) );


