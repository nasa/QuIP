#include "quip_config.h"

char VersionId_interpreter_help[] = QUIP_VERSION_STRING;

/* What is the general strategy for help files?
 *
 * There is a help directory: CMD_HELP_DIR.
 *
 * CMD_HELP_DIR defaults to /u/jbm/cmd_help.
 * It is reset from the environment if defined.
 *
 * In ncmds.c, command_help() is called before each command
 * word is read in, with the prompt as argument.
 * Help files for each command in
 * the menu would then be looked for in CMD_HELP_DIR/prompt.
 * If the prompt ends in the string "> " (the convention for
 * UNIX interpreters), then these characters will be stripped.
 *
 * We would like to automate the generation of help files from
 * comments embedded in the source files, to allow easy updating.
 * (along the lines of c2man).  Some thought should be given to this...
 * 
 *
 */

#ifdef HELPFUL

#include <stdio.h>
#include <stdlib.h>

#include "debug.h"
#include "savestr.h"
#include "menuname.h"
#include "history.h"

#include <errno.h>
extern int errno;
#include <sys/types.h>
#include <dirent.h>
extern struct dirent *readdir();

#include <sys/stat.h>

#include <sys/param.h>		/* MAXPATHLEN */
#ifndef MAXPATHLEN
#define MAXPATHLEN	1024
#endif

/* default help directory */
#define DEFAULT_HELP_DIR	"/u/jbm/helpfiles"
/* #define DEFAULT_HELP_DIR	"/u/jbm/cmd_help" */

char *cmd_hlp_dir=NULL;
char *bi_hlp_dir;	/* builtin commands */
char *pmpt_hlp_dir=NULL;	/* current menu commands */


static char *topic_pmpt="command";

#define MAX_H_CHOICES	10
static char *h_choice[MAX_H_CHOICES];
static int n_h_choices=0;

static int expect_help=0;

/* local prototypes */
static void help_index(void);
static char *help_dir(char *prompt,char buffer[]);
static void get_dir_entries(char *strlist[],char *dirname);
static void set_help(void);
static void help_init(void);

static void help_index()
{
	char cmd[MAXPATHLEN];
	int cmd_stat;

#ifdef DEBUG
if( debug ){
if( cmd_hlp_dir != NULL ){
sprintf(ERROR_STRING,"cmd_hlp_dir = %s",cmd_hlp_dir);
advise(ERROR_STRING);
}
if( pmpt_hlp_dir != NULL ){
sprintf(ERROR_STRING,"pmpt_hlp_dir = %s",pmpt_hlp_dir);
advise(ERROR_STRING);
}
if( bi_hlp_dir != NULL ){
sprintf(ERROR_STRING,"bi_hlp_dir = %s",bi_hlp_dir);
advise(ERROR_STRING);
}
}
#endif /* DEBUG */
	if( pmpt_hlp_dir != NULL ){
		printf(
"\nHelp is avalailable on the following commands from the current menu:\n\n");
		sprintf(cmd,"ls %s",pmpt_hlp_dir);
		cmd_stat=system(cmd);
		printf("\n");
	}
#ifdef CAUTIOUS
	else {
		WARN("CAUTIOUS:  pmpt_hlp_dir not set");
	}
#endif /* CAUTIOUS */
	if( bi_hlp_dir != NULL ){
		printf(
"\nHelp is also avalailable on the following builtin commands:\n\n");
		sprintf(cmd,"ls %s",bi_hlp_dir);
		cmd_stat=system(cmd);
		printf("\n");
	}
#ifdef CAUTIOUS
	else {
		WARN("CAUTIOUS:  bi_hlp_dir not set");
	}
#endif /* CAUTIOUS */
}

void help_debug()
{
	if( pmpt_hlp_dir != NULL ){
		sprintf(ERROR_STRING, "pmpt_hlp_dir = %s",pmpt_hlp_dir);
		advise(ERROR_STRING);
	} else
		WARN("no command help dir");

	if( bi_hlp_dir != NULL ){
		sprintf(ERROR_STRING, "bi_hlp_dir = %s",bi_hlp_dir);
		advise(ERROR_STRING);
	} else
		WARN("no builtin help dir");
}

/*
 * We will create subdirectories for the commonly used menus (prompts),
 * but because a lot of menu commands appear in many test programs,
 * we do not want to create a directory for every test program prompt
 * in order to get the generic help.  Therefore if we can't find
 * a prompt-specific help directory, we revert to the grab-bag.
 */

static char *help_dir(prompt,buffer)
char *prompt, buffer[];
{
	char pmpt_copy[32];
	int i;
	DIR *dirp;

	if( cmd_hlp_dir == NULL ) help_init();

	i=strlen(prompt);
	if( i-- >= 2 && prompt[i--]==' ' && prompt[i]=='>' ){
		strcpy(pmpt_copy,prompt);
		pmpt_copy[i++]=0;
		pmpt_copy[i++]=0;
		sprintf(buffer,"%s/%s", cmd_hlp_dir,pmpt_copy);
	} else
		sprintf(buffer,"%s/%s", cmd_hlp_dir,prompt);

#ifdef DEBUG
if( debug ){
sprintf(ERROR_STRING,"pmpt_hlp_dir:  \"%s\"",buffer);
advise(ERROR_STRING);
}
#endif /* DEBUG */
	/* now see if this directory really exists? */

	dirp=opendir(buffer);
	if( dirp == NULL ){
		if( expect_help || verbose ){
			sprintf(ERROR_STRING,"No help directory %s.",buffer);
			advise(ERROR_STRING);
		}
		sprintf(buffer,"%s/commands", cmd_hlp_dir);
		dirp=opendir(buffer);
	}
	if( dirp == NULL ){
		if( expect_help || verbose )
			advise("Default command help directory not found.");
		return(NULL);
	} else {
		closedir(dirp);
		return(buffer);
	}
}

void builtin_help(prompt)
char *prompt;
{
	static char bi_hlp_dirname[MAXPATHLEN];

	bi_hlp_dir = help_dir(prompt,bi_hlp_dirname);
}

void command_help(prompt)
char *prompt;
{
	static char cmd_hlp_dirname[MAXPATHLEN];
	static char *last_prompt="";

	if( cmd_hlp_dir == NULL ) help_init();

	if( !strcmp(prompt,last_prompt) ) return;
	last_prompt=prompt;

#ifdef DEBUG
if( debug ){
sprintf(ERROR_STRING,"setting pmpt help dir, root = %s",cmd_hlp_dir);
advise(ERROR_STRING);
}
#endif /* DEBUG */
	pmpt_hlp_dir = help_dir(prompt,cmd_hlp_dirname);
}

static void help_init()
{
	char *s;

	s=getenv("CMD_HELP_DIR");
	if( s!=NULL )
		cmd_hlp_dir = s;
	else
		cmd_hlp_dir = DEFAULT_HELP_DIR;
}

static int deliver_help(dir,file)
char *dir,*file;
{
	char filename[MAXPATHLEN];
	struct stat statb;

	sprintf(filename,"%s/%s",dir,file);
	if( stat(filename,&statb) != -1 ){	/* file exists ? */
		if( S_ISREG(statb.st_mode) ){		/* plain file ?? */
			int status;
			char cmd[MAXPATHLEN];

			sprintf(cmd,"more %s",filename);
			status=system(cmd);
			return(1);
		}
		else WARN("sorry, help file is not a plain file");
	}
	return(0);
}

void give_help()
{
	char *s;
	int index;

	/*
	 * really shouldn't do this
	 * if the help directory hasn't changed...
	 */

	set_help();
	index=which_one(topic_pmpt,n_h_choices,h_choice);
	if( index == -1 ){
		help_index();
		return;
	}
	s=h_choice[index];

	/* see if there is a help file with this name */

	if( pmpt_hlp_dir != NULL )
		if( deliver_help(pmpt_hlp_dir,s) ) return;

	if( bi_hlp_dir != NULL )
		if( deliver_help(bi_hlp_dir,s) ) return;
#ifdef CAUTIOUS
	ERROR1("CAUTIOUS:  no help given!?");
#endif
}

static void get_dir_entries(strlist,dirname)
char *strlist[], *dirname;
{
	DIR *dirp;
	struct dirent *dp;
	int warned_once=0;

	if( dirname == NULL ) return;
	dirp=opendir(dirname);
	if( dirp == NULL ) return;
	while( (dp=readdir(dirp)) != ((struct dirent *)NULL) ){
		if( n_h_choices >= MAX_H_CHOICES && !warned_once ){
			WARN("get_dir_entries:  Too many directory entries");
			advise("Increase MAX_H_CHOICES");
			warned_once=1;
		} else if ( *dp->d_name != '.' ){	/* don't add . & .. */
			strlist[n_h_choices++]=savestr(dp->d_name);
		}
	}
	closedir(dirp);
}

static void set_help()
{
	int i;

#ifdef HAVE_HISTORY
	/* first clear out any old defaults */

	new_defs(topic_pmpt);
#endif /* HAVE_HISTORY */

	/* remove old choice, if any */

	for(i=0;i<n_h_choices;i++)
		rls_str(h_choice[i]);

	n_h_choices=0;

	/* now get the list of topics... */

	get_dir_entries(h_choice, pmpt_hlp_dir);
	get_dir_entries(h_choice, bi_hlp_dir);
}

#endif /* HELPFUL */
