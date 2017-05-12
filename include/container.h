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
//	int		types;		// mask with bit set
					// if container type exists
	int		primary_type;	// operate on this one first?
//	int		is_current;	// mask with bit set if container
					// type is up-to-date
	union {
		List *		u_lp;
		Hash_Tbl *	u_htp;
		qrb_tree *	u_tree_p;
	} cnt_u;

	// methods
	void (* substring_find_func)(Container *,Frag_Match_Info *, const char *);
	void *(* insert_item_func)(Container *,Item *);
};

#define cnt_lp		cnt_u.u_lp
#define cnt_htp		cnt_u.u_htp
#define cnt_tree_p	cnt_u.u_tree_p

#define CONTAINER_TYPE(cnt_p)	(cnt_p)->primary_type

typedef struct enumerator {
	Container *e_cnt_p;
	union {
		List_Enumerator *lep;
		Hash_Tbl_Enumerator *htep;
		RB_Tree_Enumerator *rbtep;
		void *vp;
	} e_p;
} Enumerator;

#define ENUMERATOR_CONTAINER(ep)	(ep)->e_cnt_p
#define ENUMERATOR_TYPE(ep)		CONTAINER_TYPE(ENUMERATOR_CONTAINER(ep))

extern Enumerator *new_enumerator(Container *cnt_p );
extern Enumerator *advance_enumerator(Enumerator *ep );
extern Enumerator *backup_enumerator(Enumerator *ep );
extern void *enumerator_item(Enumerator *ep);
extern Item *current_frag_item(Frag_Match_Info *fmi_p);

extern void container_find_substring_matches(Frag_Match_Info *fmi_p, Container *cnt_p, const char *frag);


#endif // ! _CONTAINER_H_

