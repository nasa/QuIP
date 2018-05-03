
#ifndef _HISTORY_H_
#define _HISTORY_H_

#define CYC_FORWARD	1
#define CYC_BACKWARD	2

#ifdef HAVE_HISTORY

#include "query_stack.h"


/* We used to have separate item types for each history list (prompt)
 * but now the lists are contexts...
 */

typedef struct hist_choice {
	Item		hc_item;
} Hist_Choice ;

#define hc_text	hc_item.item_name

#define MAX_MENU_RSPS	64		/* this is too many for most */
					/* might be nice to have dynamic
						sizing */

#ifdef QUIP_DEBUG
extern debug_flag_t hist_debug;		/* used in items.c */
#endif /* QUIP_DEBUG */

extern int history;		/* global flag */

extern FILE *_tty_out;	/* used to be static in complete.c ... */


/* history.c */

extern Item_Context *	_find_hist(QSP_ARG_DECL  const char *);
extern void		_preload_history_list(QSP_ARG_DECL  const char *prompt,unsigned int n,const char **choices);
extern void		_rem_def(QSP_ARG_DECL  const char *,const char *);
extern void		_new_defs(QSP_ARG_DECL  const char *);
extern void		_add_def(QSP_ARG_DECL  const char *,const char *);
extern void		_rem_phist(QSP_ARG_DECL  const char *,const char *);
extern void		_add_phist(QSP_ARG_DECL  const char *,const char *);
extern const char *	_get_match(QSP_ARG_DECL  const char *prompt, const char *so_far);
extern const char *	_cyc_match(QSP_ARG_DECL  const char *string,int direction);
extern void		_init_hist_from_item_list(QSP_ARG_DECL  const char *pmpt, List *lp);
extern void		_init_hist_from_list(QSP_ARG_DECL  const char *pmpt, List *lp);
extern void		_init_hist_from_class(QSP_ARG_DECL  const char *pmpt, Item_Class *icp);

#define find_hist(s)	_find_hist(QSP_ARG  s)
#define preload_history_list(prompt,n,choices)	_preload_history_list(QSP_ARG  prompt,n,choices)
#define rem_def(p,s)	_rem_def(QSP_ARG  p,s)
#define new_defs(p)	_new_defs(QSP_ARG  p)
#define add_def(p,s)	_add_def(QSP_ARG  p,s)
#define rem_phist(p,s)	_rem_phist(QSP_ARG  p,s)
#define add_phist(p,s)	_add_phist(QSP_ARG  p,s)
#define get_match(p,f)	_get_match(QSP_ARG  p, f)
#define cyc_match(str,dir)	_cyc_match(QSP_ARG  str,dir)
#define init_hist_from_item_list(pmpt,lp)	_init_hist_from_item_list(QSP_ARG  pmpt, lp)
#define init_hist_from_list(pmpt,lp)	_init_hist_from_list(QSP_ARG  pmpt, lp)
#define init_hist_from_class(pmpt,icp)	_init_hist_from_class(QSP_ARG  const char *pmpt, Item_Class *icp)

/* complete.c */

#ifdef TTY_CTL
extern void tty_reset(FILE *tty);
extern void save_keystroke(int c);
extern int get_keystroke(void);
extern void hist_bis(const char *pmpt);
extern const char *get_response_from_user( QSP_ARG_DECL  const char *prompt, FILE *tty_in, FILE *tty_out );
extern void sane_tty(SINGLE_QSP_ARG_DECL);
extern void check_events(QSP_ARG_DECL  FILE *);
extern int _keyboard_hit(QSP_ARG_DECL  FILE *);
#define keyboard_hit(fp)	_keyboard_hit(QSP_ARG  fp)
#endif /* TTY_CTL */

#endif /* HAVE_HISTORY */

#endif /* _HISTORY_H_ */

