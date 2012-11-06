#ifndef _SEQ_H_
#define _SEQ_H_

#ifdef INC_VERSION
char VersionId_inc_seq[] = QUIP_VERSION_STRING;
#endif /* INC_VERSION */

#include "quip_config.h"

/* sequencing software */

#include "query.h"
#include "seq_api.h"

#define NO_SEQ	((Seq *) NULL)

#define SEQFREE		(-7)
#define SEQ_MOVIE	(-2)
#define SUPSEQ		(-4)

#define NO_SEQ_MODULE	((Seq_Module *)NULL)

extern Item_Type * mviseq_itp;	/* global so mvimenu can check it */


/* prototypes */


/* seqparse.y */
ITEM_INTERFACE_PROTOTYPES(Seq,mviseq)

extern void	load_seq_module(Seq_Module *smp);
extern int	init_show_seq(Seq *);
extern void	wait_show_seq(void);

extern int	yylex(void);
extern int	yyerror(const char *s);
extern void	seqinit(void);
extern Seq *	defseq(QSP_ARG_DECL  const char *name,const char *seqstr);
//extern void	delseq(QSP_ARG_DECL  Seq *sp);
extern void	evalseq(Seq *seqptr);
extern void	reverse_eval(Seq *seqptr);
extern void	setrefn(int n);

/* seqprint.c */
extern void	pseq(Seq *seqptr);
extern void	pframe(Seq *seqptr);
extern void	pfull(Seq *seqptr);
extern void	pone(Seq *seqptr);

/* seqmenu.c */
extern COMMAND_FUNC( seq_menu );

/* verseq.c */
extern void verseq(SINGLE_QSP_ARG_DECL);


#endif /* ! _SEQ_H_ */

