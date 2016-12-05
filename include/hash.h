#ifndef _HASH_H_
#define _HASH_H_

#include "quip_fwd.h"

struct hash_tbl {
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
} ;

#define NO_HASH_TBL		((Hash_Tbl *) NULL)


typedef struct {
	Hash_Tbl *htp;
	void **current_entry;
} Hash_Tbl_Enumerator;

extern Hash_Tbl_Enumerator *new_hash_tbl_enumerator(Hash_Tbl *htp);
extern void advance_ht_enumerator(Hash_Tbl_Enumerator *htep);
extern void rls_hash_tbl_enumerator(Hash_Tbl_Enumerator *htep);
extern Item * ht_enumerator_item(Hash_Tbl_Enumerator *htep);

#endif // ! _HASH_H_

