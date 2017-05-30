#include "quip_config.h"

#ifdef HAVE_STRING_H
#include <string.h>
#endif

#include "quip_prot.h"
#include "data_obj.h"
#include "seq.h"
#include "query_bits.h"	// LLEN - get rid of this!  BUG

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
	if( seqptr != NO_SEQ ) pseq(QSP_ARG  seqptr);
}

static COMMAND_FUNC( do_del_seq )
{
	Seq *seqptr;

	seqptr=PICK_SEQ("");
	if( seqptr != NO_SEQ ) delseq(QSP_ARG  seqptr);
}

static COMMAND_FUNC( do_list_Seqs ){ list_mviseqs(QSP_ARG  tell_msgfile(SINGLE_QSP_ARG)); }

#define ADD_CMD(s,f,h)	ADD_COMMAND(sequence_menu,s,f,h)

MENU_BEGIN(sequence)
ADD_CMD( define,	do_def_seq,	define movie sequence )
ADD_CMD( redefine,	do_redef_seq,	redefine existing movie sequence )
ADD_CMD( list,		do_list_Seqs,	list all defined sequences )
ADD_CMD( show,		do_show_seq,	show a sequence )
ADD_CMD( print,		do_prt_seq,	print sequence definition )
ADD_CMD( delete,	do_del_seq,	delete sequence )
MENU_END(sequence)

COMMAND_FUNC( do_seq_menu )
{
	PUSH_MENU(sequence);
}

