#ifndef _CONTAINER_H_
#define _CONTAINER_H_

// implementation-agnostic interface

#include "list.h"
#include "hash.h"
#include "rbtree.h"
#include "item_type.h"
#include "container_fwd.h"

typedef int container_type_code;

struct container_type {
	// methods
	void *(* insert_item)(Container *,Item *);
	int (*remove_name)(Container *, const char *);
	Item * (*find_match)(Container *, const char *);
	long (*eltcount)(Container *);
	void (*delete)(Container *);
	void (*init_data)(Container *);
	List * (*list_of_items)(Container *);
	void (*dump_info)(QSP_ARG_DECL  Container *);
	Enumerator *(* new_enumerator)(Container *);

	// should these be frag match methods?
	void (* substring_find)(Frag_Match_Info *, const char *);
	Item *(* frag_item)(Frag_Match_Info *);
	Item *(* current_frag_match_item)(Frag_Match_Info *);
	const char *(* advance_frag_match)(Frag_Match_Info *,int direction);
	void (* reset_frag_match)(Frag_Match_Info *, int direction);
};

struct container {
	const char *		name;
	Container_Type *	cnt_typ_p;
//	int		primary_type;	// operate on this one first?
	union {
		List *		u_lp;
		Hash_Tbl *	u_htp;
		qrb_tree *	u_tree_p;
	} cnt_u;
};

#define cnt_lp		cnt_u.u_lp
#define cnt_htp		cnt_u.u_htp
#define cnt_tree_p	cnt_u.u_tree_p

//#define CONTAINER_TYPE(cnt_p)	(cnt_p)->primary_type

struct enumerator_type {
	Enumerator *	(*advance_enum)(Enumerator *);
	void		(*release_enum)(Enumerator *);
	void *		(*current_enum_item)(Enumerator *);
};

struct enumerator {
	Container *e_cnt_p;
	union {
		List_Enumerator *lep;
		Hash_Tbl_Enumerator *htep;
		RB_Tree_Enumerator *rbtep;
		void *vp;
	} e_p;
	Enumerator_Type *	e_typ_p;
};

#define ENUMERATOR_CONTAINER(ep)	(ep)->e_cnt_p
#define ENUMERATOR_TYPE(ep)		CONTAINER_TYPE(ENUMERATOR_CONTAINER(ep))

//extern Enumerator *new_enumerator(Container *cnt_p );
//extern Enumerator *advance_enumerator(Enumerator *ep );
//extern Enumerator *backup_enumerator(Enumerator *ep );
//extern void *enumerator_item(Enumerator *ep);
extern Item *current_frag_item(Frag_Match_Info *fmi_p);

extern void container_find_substring_matches(Frag_Match_Info *fmi_p, const char *frag);


#endif // ! _CONTAINER_H_

