#ifndef _CONTAINER_H_
#define _CONTAINER_H_

// implementation-agnostic interface

#include "list.h"
#include "hash.h"
#include "rbtree.h"
#include "item_type.h"
#include "container_fwd.h"

struct container {
	const char *	name;
	int		types;		// mask with bit set
					// if container type exists
	int		primary_type;	// operate on this one first?
	int		is_current;	// mask with bit set if container
					// type is up-to-date
	List *		cnt_lp;
	Hash_Tbl *	cnt_htp;
	rb_tree *	cnt_tree_p;
};

typedef struct enumerator {
	int type;	// should only have 1 bit set
	Container *e_cnt_p;
	union {
		List_Enumerator *lep;
		Hash_Tbl_Enumerator *htep;
		RB_Tree_Enumerator *rbtep;
		void *vp;
	} e_p;
} Enumerator;

extern Enumerator *new_enumerator(Container *cnt_p, int type);
extern Enumerator *advance_enumerator(Enumerator *ep );
extern Enumerator *backup_enumerator(Enumerator *ep );
extern void *enumerator_item(Enumerator *ep);
extern List_Enumerator *new_list_enumerator(List *lp);
extern Item *current_frag_item(Frag_Match_Info *fmi_p);

extern void container_find_substring_matches(Frag_Match_Info *fmi_p, Container *cnt_p, const char *frag);


#endif // ! _CONTAINER_H_
