
#ifdef HELPFUL

#ifndef HELP_INDEX

#define HELP_PREVIOUS	0
#define HELP_INDEX	1

/* default help directory */
#define CMD_HELP_DIR	"/u/jbm/cmd_help"
#define GEN_HELP_DIR	"/u/jbm/libsrc/helpfiles"
#define MAXHDEPTH	4
#define MAX_H_CHOICES	10

extern void help_index(void);
extern void help_debug(void);
extern void builtin_help(char *);
extern void command_help(char *);
extern void help_menu(void);
extern void deliver_help(char *,char *);
extern void give_help(void);
extern void set_help(void);

#endif /* ! HELP_INDEX */

#endif /* HELPFUL */
