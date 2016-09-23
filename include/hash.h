#ifndef _HASH_H_
#define _HASH_H_

#include "typedefs.h"
#include "node.h"
#include "list.h"
#include "item_obj.h"
#include "query_stack.h"

typedef struct hash_tbl {
	const char *	ht_name;
	u_long		ht_size;
	u_long		ht_n_entries;
	u_long		ht_removals;
	u_long		ht_moved;
	u_long		ht_checked;
	u_long		ht_warned;
	void **		ht_entries;
#ifdef MONITOR_COLLISIONS
	/* These used to be unsigned long, but we ran into a problem when we had more than
	 * a billion macro calls, this made the collision rate seem very high and the tables
	 * grew without bound...
	 *
	 * Changing these to doubles keeps us out of trouble for a while,
	 * but is not a permanent fix, since we can still have overflows!?
	 */

	/*
	u_long	ht_searches;
	u_long	ht_collisions;
	*/
	double		ht_searches;
	double		ht_collisions;
#endif /* MONITOR_COLLIISIONS */
} Hash_Tbl ;

#define NO_HASH_TBL		((Hash_Tbl *) NULL)


extern void		zap_hash_tbl(Hash_Tbl *);
extern List *		ht_list(QSP_ARG_DECL  Hash_Tbl *);
extern Hash_Tbl *	enlarge_ht(Hash_Tbl *);
extern Hash_Tbl *	ht_init(const char *name);
extern int		insert_hash(void *ptr,Hash_Tbl *table);
extern void		show_ht(Hash_Tbl *table);
extern void *		fetch_hash(const char *name,Hash_Tbl *table);
//extern int		remove_hash(void *ptr,Hash_Tbl *table);
extern int		remove_name_from_hash(const char *name,Hash_Tbl *table);
extern int		remove_item_from_hash(const Item *ip,Hash_Tbl *table);
extern void		tell_hash_stats(QSP_ARG_DECL  Hash_Tbl *table);

typedef struct {
	Hash_Tbl *htp;
	void **current_entry;
} Hash_Tbl_Enumerator;

extern Hash_Tbl_Enumerator *new_hash_tbl_enumerator(Hash_Tbl *htp);
extern void advance_ht_enumerator(Hash_Tbl_Enumerator *htep);
extern Item * ht_enumerator_item(Hash_Tbl_Enumerator *htep);

#endif // ! _HASH_H_

