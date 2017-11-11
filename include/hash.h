#ifndef _HASH_H_
#define _HASH_H_

#include "quip_fwd.h"

struct hash_tbl {
	const char *	ht_name;
	u_long		ht_size;
	u_long		ht_n_entries;
	u_long		ht_n_removals;
	u_long		ht_n_moved;
	u_long		ht_n_checked;
//	u_long		ht_warned;
	u_long		ht_flags;
	void **		ht_entries;
#ifdef MONITOR_COLLISIONS
	/* These used to be unsigned long, but we ran into a problem when we had more than
	 * a billion macro calls, this made the collision rate seem very high and the tables
	 * grew without bound...
	 *
	 * Changing these to doubles keeps us out of trouble for a while,
	 * but is not a permanent fix, since we can still have overflows!?
	 * Need to check...
	 */

	/*
	u_long	ht_searches;
	u_long	ht_collisions;
	*/
	uint64_t	ht_n_searches;
	uint64_t	ht_n_collisions;
#endif /* MONITOR_COLLIISIONS */
	
	List *		ht_item_list;
} ;

// Hash table flags
#define HT_WARNED		1
#define HT_CHEKED		2
#define HT_LIST_IS_CURRENT	4

#define HT_FLAGS(htp)	(htp)->ht_flags
#define SET_HT_FLAG_BITS(htp,bits)	(htp)->ht_flags |= bits
#define CLEAR_HT_FLAG_BITS(htp,bits)	(htp)->ht_flags &= ~(bits)

#define HASH_TBL_WARNED(htp)	HT_FLAGS(htp) & HT_WARNED

#define HASH_TBL_LIST_IS_CURRENT(htp)	(HT_FLAGS(htp) & HT_LIST_IS_CURRENT)

#define MARK_DIRTY(htp)		CLEAR_HT_FLAG_BITS(htp,HT_LIST_IS_CURRENT)
#define MARK_CURRENT(htp)	SET_HT_FLAG_BITS(htp,HT_LIST_IS_CURRENT)

#define HT_ITEM_LIST(htp)		(htp)->ht_item_list
#define SET_HT_ITEM_LIST(htp,lp)	(htp)->ht_item_list = lp

typedef struct {
	Hash_Tbl *htp;
	void **current_entry;
} Hash_Tbl_Enumerator;

extern Hash_Tbl_Enumerator *_new_hash_tbl_enumerator(QSP_ARG_DECL  Hash_Tbl *htp);
#define new_hash_tbl_enumerator(htp) _new_hash_tbl_enumerator(QSP_ARG  htp)

extern void advance_ht_enumerator(Hash_Tbl_Enumerator *htep);
extern void rls_hash_tbl_enumerator(Hash_Tbl_Enumerator *htep);
extern Item * ht_enumerator_item(Hash_Tbl_Enumerator *htep);

extern List *_hash_tbl_list(QSP_ARG_DECL  Hash_Tbl *htp);
#define hash_tbl_list(htp) _hash_tbl_list(QSP_ARG  htp)

#endif // ! _HASH_H_

