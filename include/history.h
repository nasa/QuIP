
#ifndef _HISTORY_H_
#define _HISTORY_H_

#ifdef HAVE_HISTORY

#include "query_stack.h"


/* We used to have separate item types for each history list (prompt)
 * but now the lists are contexts...
 */

typedef struct hist_choice {
	Item		hc_item;
} Hist_Choice ;

#define hc_text	hc_item.item_name
#define NO_CHOICE	((Hist_Choice *) NULL)


#define MAX_MENU_RSPS	64		/* this is too many for most */
					/* might be nice to have dynamic
						sizing */

#ifdef QUIP_DEBUG
extern debug_flag_t hist_debug;		/* used in items.c */
#endif /* QUIP_DEBUG */

extern int history;		/* global flag */

#define CYC_FORWARD	1
#define CYC_BACKWARD	2

extern FILE *_tty_out;	/* used to be static in complete.c ... */


/* history.c */

extern Item_Context *	find_hist(QSP_ARG_DECL  const char *);
extern void		set_defs(QSP_ARG_DECL  const char *prompt,unsigned int n,const char **choices);
extern void		rem_def(QSP_ARG_DECL  const char *,const char *);
extern void		new_defs(QSP_ARG_DECL  const char *);
extern void		add_def(QSP_ARG_DECL  const char *,const char *);
extern void		rem_phist(QSP_ARG_DECL  const char *,const char *);
extern void		add_phist(QSP_ARG_DECL  const char *,const char *);
extern const char *	get_match(QSP_ARG_DECL  const char *prompt, const char *so_far);
extern const char *	cyc_match(QSP_ARG_DECL  const char *string,int direction);
extern void		init_hist_from_item_list(QSP_ARG_DECL  const char *pmpt, List *lp);
extern void		init_hist_from_list(QSP_ARG_DECL  const char *pmpt, List *lp);
extern void		init_hist_from_class(QSP_ARG_DECL  const char *pmpt, Item_Class *icp);

/* complete.c */

#ifdef TTY_CTL
extern void tty_reset(FILE *tty);
extern void save_keystroke(int c);
extern int get_keystroke(void);
extern void hist_bis(const char *pmpt);
extern const char *get_sel( QSP_ARG_DECL  const char *prompt, FILE *tty_in, FILE *tty_out );
extern void sane_tty(SINGLE_QSP_ARG_DECL);
extern void check_events(QSP_ARG_DECL  FILE *);
extern int keyboard_hit(QSP_ARG_DECL  FILE *);
#endif /* TTY_CTL */

#endif /* HAVE_HISTORY */

#endif /* _HISTORY_H_ */

