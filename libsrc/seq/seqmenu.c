#include "quip_config.h"

char VersionId_seq_seqmenu[] = QUIP_VERSION_STRING;

#ifdef HAVE_STRING_H
#include <string.h>
#endif

#include "data_obj.h"
#include "seq.h"
#include "menuname.h"

static COMMAND_FUNC( do_show_seq )
{
	Seq *sp;

	sp = PICK_SEQ("");
	if( sp == NO_SEQ ) return;

	init_show_seq(sp);
	evalseq(sp);
	wait_show_seq();
}

static COMMAND_FUNC( do_def_seq )
{
	const char *s;
	char nmbuf[LLEN];

	s = NAMEOF("sequence name");
	strcpy(nmbuf,s);
	s=NAMEOF("sequence (quote if includes spaces)");

	if( defseq(QSP_ARG  nmbuf,s) == NO_SEQ )
		WARN("couldn't create new sequence");
}

static COMMAND_FUNC( do_redef_seq )
{
	Seq *sp;
	char nmbuf[LLEN];
	const char *s;

	sp = PICK_SEQ("");
	s=NAMEOF("sequence (quote if includes spaces)");

	if( sp == NO_SEQ ){
		WARN("sequence did not already exist, can't redefine");
		return;
	}

	strcpy(nmbuf,sp->seq_name);
	delseq(QSP_ARG  sp);

	if( (sp=defseq(QSP_ARG  nmbuf,s)) == NO_SEQ )
		WARN("couldn't create new sequence");
}

static COMMAND_FUNC( do_prt_seq )
{
	Seq *seqptr;

	seqptr=PICK_SEQ("");
	if( seqptr != NO_SEQ ) pseq(seqptr);
}

static COMMAND_FUNC( do_del_seq )
{
	Seq *seqptr;

	seqptr=PICK_SEQ("");
	if( seqptr != NO_SEQ ) delseq(QSP_ARG  seqptr);
}

static COMMAND_FUNC( do_list_Seqs ){ list_mviseqs(SINGLE_QSP_ARG); }

Command seqctbl[]={
{ "define",	do_def_seq,	"define movie sequence"			},
{ "redefine",	do_redef_seq,	"redefine existing movie sequence"	},
{ "list",	do_list_Seqs,	"list all defined sequences"		},
{ "show",	do_show_seq,	"show a sequence"			},
{ "print",	do_prt_seq,	"print sequence definition"		},
{ "delete",	do_del_seq,	"delete sequence"			},
{ "quit",	popcmd,		"exit submenu"				},
{ NULL,		NULL,		NULL					}
};

COMMAND_FUNC( seq_menu )
{
	static int vinited=0;

	if( !vinited ){
		verseq(SINGLE_QSP_ARG);
		vinited++;
	}
	PUSHCMD(seqctbl,SEQ_MENU_NAME);
}

