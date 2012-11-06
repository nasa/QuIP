
#ifndef NO_NAMESPACE

#include "node.h"
#include "hash.h"

typedef struct item {
	const char *	item_name;
	/* usually more stuff */
} Item;

#define NO_ITEM		((Item *)NULL)

typedef struct namesp {
	const char *	ns_name;	/* not items, but the name is useful */
	List *		ns_lp;
	Hash_Tbl *	ns_htp;
	int		ns_flags;
	int		ns_fetches;	/* number of name lookups */
	int		ns_ncmps;	/* number of strcmps w/ list search */
} Name_Space;

#define NO_NAMESPACE	((Name_Space *)NULL)

/* flag bits */

#define NS_LIST_IS_CURRENT	1

#define IS_HASHING( nsp )	( (nsp)->ns_htp != NO_HASH_TBL )


/* namesp.c */

extern Name_Space *	create_namespace(const char *);
extern List *		namespace_list(Name_Space *nsp);
extern void		delete_namespace(Name_Space *nsp);
extern Item *		fetch_name(const char *name,Name_Space *nsp);
extern int		insert_name(Item *ip,Node *np,Name_Space *nsp);
extern void		cat_ns_items(List *lp,Name_Space *nsp);
extern void		tell_name_stats(Name_Space *nsp);
extern int		remove_name(Item *ip,Name_Space *nsp);
extern void		dump_ns_info(Name_Space *nsp);


#endif /* ! NO_NAMESPACE */
