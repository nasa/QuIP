#ifndef _QUIP_MENU_H_
#define _QUIP_MENU_H_

#include "command.h"
#include "getbuf.h"
//#include "dictionary.h"
//#include "dict.h"


#include "container_fwd.h"


struct menu {
//	Dictionary *		mn_dict;
	Container *		mn_cnt_p;
	const char *		mn_prompt;

#ifdef HAVE_HISTORY

	int			mn_flags;

// flag bits
#define MENU_FLAG_CMDS_STASHED	1
#define MENU_COMMANDS_STASHED(mp)	((mp)->mn_flags & MENU_FLAG_CMDS_STASHED)
#define SET_MENU_FLAG_BITS(mp,b)	(mp)->mn_flags |= b
#define CLEAR_MENU_FLAG_BITS(mp,b)	(mp)->mn_flags &= ~(b)

#endif /* HAVE_HISTORY */

} ;

#ifdef HAVE_HISTORY
#define CLEAR_MENU_FLAGS(prompt)	prompt##_menu->mn_flags = 0;
#else
#define CLEAR_MENU_FLAGS(prompt)
#endif


#define MENU_BEGIN(prompt)						\
static Menu *prompt##_menu=NULL;					\
static void init_##prompt##_menu(SINGLE_QSP_ARG_DECL)			\
{									\
	Command *cp;							\
									\
	prompt##_menu = (Menu *)getbuf(sizeof(Menu));			\
	prompt##_menu->mn_prompt = savestr(#prompt);			\
	/*prompt##_menu->mn_dict = create_dictionary(#prompt);*/	\
	prompt##_menu->mn_cnt_p = create_container(#prompt,LIST_CONTAINER);		\
	CLEAR_MENU_FLAGS(prompt)

#define ADD_COMMAND(mp,selector,function,help_string)			\
	cp = (Command *)getbuf(sizeof(Command));			\
	cp->cmd_selector = #selector;					\
	cp->cmd_action = function;					\
	cp->cmd_help = #help_string;					\
	add_command_to_menu(mp,cp);

#define MENU_END(prompt)						\
	ADD_COMMAND(prompt##_menu,quit,do_pop_menu,exit submenu)	\
	MENU_SIMPLE_END(prompt)


#define MENU_SIMPLE_END(prompt)						\
}

//#define PUSHCMD(menu,prompt)	pushcmd(THIS_QSP menu)

// COMMAND_FUNC macro moved to quip_fwd.h

#define CHECK_MENU(prompt)					\
								\
	if( prompt##_menu == NULL ){				\
		init_##prompt##_menu(SINGLE_QSP_ARG);		\
	}

extern void _add_command_to_menu(QSP_ARG_DECL  Menu *mp, Command *cp );
#define add_command_to_menu(mp,cp) _add_command_to_menu(QSP_ARG  mp,cp)

extern void _list_menu( QSP_ARG_DECL  const Menu *mp );
#define list_menu( mp ) _list_menu( QSP_ARG  mp )

#define CHECK_AND_PUSH_MENU(prompt)		\
						\
		CHECK_MENU(prompt)		\
		push_menu(prompt##_menu);

#define MENU_CONTAINER(mp)	mp->mn_cnt_p
#define MENU_LIST(mp)		container_list(MENU_CONTAINER(mp))
#define MENU_PROMPT(mp)		mp->mn_prompt


#endif /* ! _QUIP_MENU_H_ */

