#include "quip_config.h"

char VersionId_seq_verseq[] = QUIP_VERSION_STRING;

#include "seq.h"
#include "version.h"

#ifdef FOOBAR

/* This code was for dos */

extern char	VersionId_seq_seqparse[];
extern char	VersionId_seq_seqprint[];
extern char	VersionId_seq_seqmenu[];
extern char	VersionId_seq_verseq[];

#define N_SEQ_FILES 4

FileVersion seq_files[N_SEQ_FILES] = {
	{	VersionId_seq_seqparse,	"seqparse.y"	},
	{	VersionId_seq_seqprint,	"seqprint.c"	},
	{	VersionId_seq_seqmenu,	"seqmenu.c"	},
	{	VersionId_seq_verseq,	"verseq.c"	}
};


void verseq()
{
	mkver ("SEQ", seq_files, N_SEQ_FILES);
}

#else /* ! FOOBAR */

void verseq(SINGLE_QSP_ARG_DECL)
{
	auto_version(QSP_ARG  "SEQ","VersionId_seq");
}

#endif /* ! FOOBAR */

