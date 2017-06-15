#ifndef _SEQ_API_H_
#define _SEQ_API_H_

#include "quip_config.h"

#include "quip_prot.h"
#include "item_obj.h"

/* Public structures */

typedef struct sequence {
	Item			seq_item;
#define seq_name		seq_item.item_name

	struct sequence *	seq_first;
	struct sequence *	seq_next;
	void *			seq_data;
	short			seq_count;
	short			seq_flags;
	short			seq_refcnt;
} Seq ;


typedef struct seq_module {
	void *	(*get_func)(const char *);	/* lookup function for leaf's */
	int	(*init_func)(void *);	/* setup device for playback */
	void	(*show_func)(void *);
	void	(*rev_func)(void *);
	void	(*wait_func)(void);
	void	(*ref_func)(void *);	/* note reference to a leaf */
} Seq_Module;


/* Public data */

/* Public prototypes */

/* seqmenu.c */
extern void show_sequence(QSP_ARG_DECL  const char *);
extern List *	seqs_referring(QSP_ARG_DECL  void *);
extern void	delseq(QSP_ARG_DECL  Seq *sp);
extern void	load_seq_module(Seq_Module *smp);

/* seqparse.y */
ITEM_INTERFACE_PROTOTYPES(Seq,mviseq)



#endif /* ! _SEQ_API_H_ */

