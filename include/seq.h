#ifndef _SEQ_H_
#define _SEQ_H_

#ifdef INC_VERSION
char VersionId_inc_seq[] = QUIP_VERSION_STRING;
#endif /* INC_VERSION */

#include "quip_config.h"

/* sequencing software */

#include "seq_api.h"

#define SEQFREE		(-7)
#define SEQ_MOVIE	(-2)
#define SUPSEQ		(-4)

/* prototypes */


/* seqparse.y */
ITEM_INTERFACE_PROTOTYPES(Seq,mviseq)
#define PICK_SEQ(p)	pick_mviseq(QSP_ARG  p)

extern void	load_seq_module(Seq_Module *smp);
extern int	init_show_seq(Seq *);
extern void	wait_show_seq(void);
extern void	init_movie_sequences(SINGLE_QSP_ARG_DECL);

//extern int	yylex(void);
//extern int	yyerror(const char *s);
extern void	seqinit(void);
extern Seq *	defseq(QSP_ARG_DECL  const char *name,const char *seqstr);
//extern void	delseq(QSP_ARG_DECL  Seq *sp);
extern void	evalseq(Seq *seqptr);
extern void	reverse_eval(Seq *seqptr);
extern void	setrefn(int n);

/* seqprint.c */
extern void	pseq(QSP_ARG_DECL  Seq *seqptr);
extern void	pframe(QSP_ARG_DECL  Seq *seqptr);
extern void	pfull(QSP_ARG_DECL  Seq *seqptr);
extern void	pone(QSP_ARG_DECL  Seq *seqptr);

#endif /* ! _SEQ_H_ */

