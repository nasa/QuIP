
#ifndef _DICT_H_
#define _DICT_H_

#include "item_type.h"
//#include "node.h"
//#include "hash.h"

struct hash_tbl;	// forward defn.

typedef struct dictionary {
	const char *		dict_name;	/* not items, but the name is useful */
	List *			dict_lp;
	struct hash_tbl *	dict_htp;
	int			dict_flags;
	int			dict_fetches;	/* number of name lookups */
	int			dict_ncmps;	/* number of strcmps w/ list search */
} Dictionary;

#define NO_DICTIONARY	((Dictionary *)NULL)

/* flag bits */

#define NS_LIST_IS_CURRENT	1

#define IS_HASHING( dict_p )	( (dict_p)->dict_htp != NULL )

extern Dictionary *new_dictionary(void);
extern int insert_name(Item* ip, Node* np, Dictionary* dict_p);

#endif /* ! _DICT_H_ */

