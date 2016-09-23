#include "quip_config.h"

#include <stdio.h>
#ifdef HAVE_STRING_H
#include <string.h>
#endif

#include "quip_prot.h"

#ifdef QUIP_DEBUG
static u_long hash_debug=HASH_DEBUG_MASK;
#endif

#define MAX_SILENT_COLL_RATE	2.5

#define N_PRIMES	16
static u_long prime[N_PRIMES]={
	31L,
	61L,
	121L,
	241L,
	479L,
	953L,
	1901L,
	3797L,
	7591L,
	15173L,
	30341L,
	60679L,
	121357L,
	242713L,
	485423L,
	970829L
};

/* local prototypes */
static void clear_entries(void **entry,u_long size);
static void setup_ht(Hash_Tbl *htp,u_long size);

static void clear_entries(void **entry, u_long size)
{
	u_long i;

	for(i=0;i<size;i++) entry[i] = NULL ;
}



static void setup_ht(Hash_Tbl *htp, u_long size)
{
	u_long i;

	for(i=0;i<N_PRIMES;i++)
		if( prime[i] > size ){
			size=prime[i];
			i=N_PRIMES+2;
		}
	if( i == N_PRIMES ){
		sprintf(DEFAULT_ERROR_STRING,"setup_ht( tbl = %s, size = %ld (0x%lx) )",htp->ht_name,size,size);
		NADVISE(DEFAULT_ERROR_STRING);

		NERROR1("need to enlarge prime number table in hash.c");
		IOS_RETURN
	}

	htp->ht_size=size;
	htp->ht_entries = (void**) getbuf(size * sizeof(void *));
	//if( htp->ht_entries==NULL ) mem_err("setup_ht");
	if( htp->ht_entries==NULL ) NWARN("setup_ht memory allocation error");

	clear_entries(htp->ht_entries,size);

	htp->ht_n_entries=0;
	htp->ht_removals=0;
	htp->ht_moved=0;
	htp->ht_checked=0;
	htp->ht_warned=0;

#ifdef MONITOR_COLLISIONS
	htp->ht_searches=0;
	htp->ht_collisions=0;
#endif
}

Hash_Tbl * enlarge_ht(Hash_Tbl *htp)
{
	u_long newsize;
	u_long i;
	void **oldents;
	void **oe;
	u_long oldsize;

if( verbose ){
sprintf(DEFAULT_ERROR_STRING,"doubling size of hash table \"%s\" (old size = %ld)",
htp->ht_name,htp->ht_size);
NADVISE(DEFAULT_ERROR_STRING);
}


	oldsize = htp->ht_size;
	newsize = 2*oldsize;
	oldents=htp->ht_entries;

	setup_ht(htp,newsize);

	/* now copy all of the entries */
	oe= oldents;
	for(i=0;i<oldsize;i++)
		if( oe[i] != NULL ){
			if( insert_hash(oe[i],htp) < 0 ){
				NERROR1("error growing hash table");
				IOS_RETURN_VAL(NULL)
			}
		}

	/* now free the old entries */
	givbuf(oldents);

	return(htp);
}

/*
 * Allocate and initialize a hash table with the given name and size entries
 */

Hash_Tbl *ht_init(const char *name)
		/* name for hash table */
{
	register Hash_Tbl *htp;

	htp= (Hash_Tbl*) getbuf( sizeof(*htp) );
	//if( htp == NO_HASH_TBL ) mem_err("ht_init");
	if( htp == NO_HASH_TBL ) {
		NERROR1("ht_init memory allocation failure");
		IOS_RETURN_VAL(NULL)
	}
	
	if( name != NULL )
		htp->ht_name=savestr(name);
	else
		htp->ht_name=NULL;

	setup_ht(htp,prime[0]);

	return(htp);
}

void zap_hash_tbl(Hash_Tbl *htp)
{
	givbuf(htp->ht_entries);
	rls_str(htp->ht_name);
}


#define KEY_MULT	127

#define compute_key(keyvar,name,table)					\
				keyvar = 0;				\
				while(*name){				\
					keyvar +=(*name++);		\
					keyvar *= KEY_MULT;		\
					keyvar %= table->ht_size;	\
				}

/*
 * Insert the item pointed to by ptr into hash table.
 * ptr must point to a structure whose first element is a
 * pointer to a string with the item's name.
 *
 * If the node pointer is not null, the node is added to the
 * table list;  This routine may be called with a null node
 * pointer because the list already exists and doesn't need
 * to be updated, or because we don't care about the list...
 *
 * Returns 0 on success, -1 on failure.
 */

int insert_hash(void *ptr,Hash_Tbl *htp)
			/* item to insert */
			/* hash table */
{
	u_long key;
	register char *name;
	u_long start;
	void **entry;

	/* check load factor */
	if( ((float)htp->ht_n_entries/(float)htp->ht_size) > 0.7 ){

if( verbose ){
sprintf(DEFAULT_ERROR_STRING,
"enlarging hash table %s: %ld of %ld entries filled",
htp->ht_name,htp->ht_n_entries,
htp->ht_size);
NADVISE(DEFAULT_ERROR_STRING);
}

//fprintf(stderr,"enlarging hash table at 0x%lx\n",(long)htp);
		htp=enlarge_ht(htp);
//fprintf(stderr,"enlarged hash table at 0x%lx\n",(long)htp);
	}

	/*
	 *	we assume that ptr points to a structure
	 *	whose first member is a char *
	 */

	name = * (char **) ptr;

#ifdef QUIP_DEBUG
if( debug & hash_debug ){
sprintf(DEFAULT_ERROR_STRING,"insert_hash: table %s at 0x%lx, item %s",
htp->ht_name,(long)htp,name);
NADVISE(DEFAULT_ERROR_STRING);
}
#endif /* QUIP_DEBUG */
	compute_key(key,name,htp);	/* a macro for setting key */

	start = key;

	entry = htp->ht_entries;

	while( entry[key] != NULL ){
		key++;
		if( key >= htp->ht_size ) key=0;
		if( key==start )	/* hash table full */
			return(-1);
	}

	/* now we've found an empty slot */

//fprintf(stderr,"insert_hash: setting entry at 0x%lx to 0x%lx\n", (long)&entry[key],(long)ptr);
	entry[key] = ptr;
	htp->ht_n_entries++;

//fprintf(stderr,"insert_hash: table at 0x%lx now has %d entries\n", (long)htp,htp->ht_n_entries);
	return(0);
}

void show_ht(Hash_Tbl *htp)	/* print out the contents of the given hash table */
{
	unsigned long i;
	register char *name;
	char *s;
	void **entry;
	u_long key;

	entry = htp->ht_entries;
	for(i=0;i<htp->ht_size;i++)
		if( entry[i] != NULL ){
			name = * ( char ** ) entry[i];
			s=name;
			compute_key(key,s,htp);	/* a macro for setting key */
			sprintf(DEFAULT_ERROR_STRING,"%ld\t%s, key = %ld",i,name,key);
			NADVISE(DEFAULT_ERROR_STRING);
		}
}

/*
 * Search the given table for an item with the given name.
 * Returns a pointer to the item if successful, or a null pointer.
 */

void *fetch_hash(const char *name,Hash_Tbl* htp)
		/* name = target name */
		/* htp = table to search */
{
	u_long key;
	const char *s;
	u_long start;
	void **entry;

	if( htp->ht_warned ){
		if( verbose ){
sprintf(DEFAULT_ERROR_STRING,
"Enlarging hash table %s due to collisions, load factor is %g",
htp->ht_name,
((float)htp->ht_n_entries/(float)htp->ht_size) );
NADVISE(DEFAULT_ERROR_STRING);
		}
		htp=enlarge_ht(htp);
		htp->ht_warned=0;
	}

	entry = htp->ht_entries;
		
	s=name;
	compute_key(key,s,htp);	/* a macro for setting key */

	start = key;
#ifdef MONITOR_COLLISIONS
	htp->ht_searches++;
#endif /* MONITOR_COLLISIONS */

	while( entry[key] != NULL ){
		/* test this one */

		/* the entries are typically structure pointers */
		/* we assume that the name is the first field */

		s = *((char **)entry[key]);
//#ifdef CAUTIOUS
//		if( s == NULL ){
//			sprintf(DEFAULT_ERROR_STRING,
//	"CAUTIOUS:  item with null name string, ip = 0x%lx!?", (u_long)entry[key]);
//			NWARN(DEFAULT_ERROR_STRING);
//			return(NULL);
//		}
//#endif /* CAUTIOUS */
		assert( s != NULL );

		if( !strcmp(s,name) ) return(entry[key]);

#ifdef MONITOR_COLLISIONS
		if( htp->ht_collisions >
			MAX_SILENT_COLL_RATE * htp->ht_searches
			&& !htp->ht_warned ){
			if( verbose ){
				sprintf(DEFAULT_ERROR_STRING,
"High collision rate (%f collisions/search), hash table \"%s\"",
					(float) htp->ht_collisions /
					(float) htp->ht_searches, htp->ht_name);
				NADVISE(DEFAULT_ERROR_STRING);
			}
			htp->ht_warned = 1;
		}

		htp->ht_collisions++;
#endif

		key++;
		if( key >= htp->ht_size ) key=0;
		if( key==start ) return(NULL);
	}
	return(NULL);
}

/*
 * was called remove_hash...
 *
 * Remove the pointed to item from the given hash table.
 * Returns 0 on success, -1 if the item is not found.
 */

int remove_item_from_hash(const Item *ptr,Hash_Tbl *htp)
			/* pointer to the item to remove */
		/* table to search */
{
	const char *name;

//	name = * (char **) ptr;
	name = ITEM_NAME(ptr);
	return remove_name_from_hash(name,htp);
}

int remove_name_from_hash(const char * name, Hash_Tbl *htp )
{
	u_long key;
	const char *s;
	u_long start;
	u_long k2;
	void **entry;

	entry = htp->ht_entries;
		
	s=name;
	compute_key(key,s,htp);	/* a macro for setting key */

	start = key;
#ifdef MONITOR_COLLISIONS
	htp->ht_searches++;
#endif /* MONITOR_COLLISIONS */

#ifdef QUIP_DEBUG
if( debug & hash_debug ){
sprintf(DEFAULT_ERROR_STRING,"key:  %ld",key);
NADVISE(DEFAULT_ERROR_STRING);
}
#endif

	while( entry[key] != NULL ){
		/* test this one */

		/* the entries are typically structure pointers */
		/* we assume that the name is the first field */

		s = *((char **)entry[key]);
		if( !strcmp(s,name) ){
#ifdef QUIP_DEBUG
if( debug & hash_debug ){
sprintf(DEFAULT_ERROR_STRING,"removing item at location %ld",key);
NADVISE(DEFAULT_ERROR_STRING);
}
#endif
			/* found the one to remove */
			entry[key] = NULL;
			start=key;
			/* now check for collisions */
			while(1){
				char **sp;

				key++;
				if( key >= htp->ht_size ) key=0;

				if( entry[key] == NULL ){
					htp->ht_removals++;
					htp->ht_n_entries--;
					return(0);
				}

				sp=(char **)entry[key];
				s=(*sp);
#ifdef QUIP_DEBUG
if( debug & hash_debug ){
sprintf(DEFAULT_ERROR_STRING,"considering shifting item %s",s);
NADVISE(DEFAULT_ERROR_STRING);
}
#endif
				compute_key(k2,s,htp);	/* a macro for setting key */

				/* move if k2 is outside the range */
				if( ( k2 <= start &&
					( key > start || k2 > key ) )
				|| ( k2 > key &&  key > start ) ){
#ifdef QUIP_DEBUG
if( debug & hash_debug ){
sprintf(DEFAULT_ERROR_STRING,"shifting at %ld",key);
NADVISE(DEFAULT_ERROR_STRING);
}
#endif
					entry[start]
					  = entry[key];
					htp->ht_moved++;
					entry[key] = NULL;
					start=key;
				}
				htp->ht_checked++;
			}
		}

#ifdef MONITOR_COLLISIONS
		htp->ht_collisions++;
#endif

		key++;
		if( key >= htp->ht_size ) key=0;
		if( key==start ) goto not_found;
	}
not_found:
	sprintf(DEFAULT_ERROR_STRING,"word \"%s\" not found in hash table \"%s\"",
		name,htp->ht_name);
	NADVISE(DEFAULT_ERROR_STRING);
	return(-1);
}

/*
 * Print the statistics for the given hash table to stdout
 */

void tell_hash_stats(QSP_ARG_DECL  Hash_Tbl *htp)
		/* hash table */
{
	sprintf(DEFAULT_MSG_STR,"\tStatistics for hash table \"%s\":",
		htp->ht_name);
	prt_msg(MSG_STR);
	sprintf(DEFAULT_MSG_STR,"\tsize %ld, %ld entries",
		htp->ht_size,htp->ht_n_entries);
	prt_msg(DEFAULT_MSG_STR);

#ifdef MONITOR_COLLISIONS
	/*
	 * This was the old version when the var was u_long
	sprintf(DEFAULT_MSG_STR,"\t%ld searches",htp->ht_searches);
	*/
	sprintf(DEFAULT_MSG_STR,"\t%g searches",htp->ht_searches);
	prt_msg(DEFAULT_MSG_STR);

	if( htp->ht_searches > 0 ){
		double d;

		sprintf(DEFAULT_MSG_STR,"\t%g collisions",htp->ht_collisions);
		prt_msg(DEFAULT_MSG_STR);
		d=(double)htp->ht_collisions;
		d /= (double) htp->ht_searches;
		sprintf(DEFAULT_MSG_STR,"\t%f collisions/search",d);
		prt_msg(DEFAULT_MSG_STR);
	}
#endif
	if( htp->ht_removals > 0 ){
		sprintf(DEFAULT_MSG_STR,"\t%ld removals",htp->ht_removals);
		prt_msg(DEFAULT_MSG_STR);
		sprintf(DEFAULT_MSG_STR,"\t%ld items checked",htp->ht_checked);
		prt_msg(DEFAULT_MSG_STR);
		sprintf(DEFAULT_MSG_STR,"\t%ld items moved",htp->ht_moved);
		prt_msg(DEFAULT_MSG_STR);
	}
	if( verbose ) show_ht(htp);
}


/* Build a list of all the items in this hash table */

List *ht_list(QSP_ARG_DECL  Hash_Tbl *htp)
{
	void **entry;
	unsigned int i;
	List *lp;

//advise("ht_list creating list for table");
	lp=new_list();
	entry = htp->ht_entries;
	for(i=0;i<htp->ht_size;i++){
		Item *ip;

		ip = (Item*) entry[i];
		if( ip != NO_ITEM ){
			Node *np;
			np=mk_node(ip);
			addTail(lp,np);
		}
	}
	return(lp);
}

void advance_ht_enumerator(Hash_Tbl_Enumerator *htep)
{
	void **entry;

	entry = htep->current_entry;
	if( entry == NULL ) return;

	while( ++entry < htep->htp->ht_entries+htep->htp->ht_size ){
		if( *entry != NULL ){
			htep->current_entry = entry;
			return;
		}
	}
	// end of table reached
	htep->current_entry = NULL;
}

Item *ht_enumerator_item(Hash_Tbl_Enumerator *htep)
{
	if( htep->current_entry == NULL ) return NULL;
	return (Item *) *(htep->current_entry);
}

Hash_Tbl_Enumerator *new_hash_tbl_enumerator(Hash_Tbl *htp)
{
	Hash_Tbl_Enumerator *htep;

	htep = getbuf( sizeof(*htep) );
	htep->htp = htp;	// needed!  we will need to know the size...
	htep->current_entry = htp->ht_entries;

	// advance to the first non-null entry
	while( *(htep->current_entry) == NULL && htep->current_entry < htep->htp->ht_entries+htep->htp->ht_size ){
//fprintf(stderr,"new_hash_tbl_enumerator:  * 0x%lx = 0x%lx\n", (long)htep->current_entry,(long)*htep->current_entry);
		htep->current_entry ++;
	}
	if( htep->current_entry == htep->htp->ht_entries+htep->htp->ht_size ){
		htep->current_entry = NULL;	// nothing there.
//fprintf(stderr,"new_hash_tbl_enumerator:  table at 0x%lx is empty!?\n",(long)htp);
	}
//else
//fprintf(stderr,"new_hash_tbl_enumerator:  first item is 0x%lx\n",(long)*(htep->current_entry));

	return htep;
}

