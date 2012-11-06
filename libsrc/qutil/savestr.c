#include "quip_config.h"

char VersionId_qutil_savestr[] = QUIP_VERSION_STRING;

#ifdef HAVE_STRING_H
#include <string.h>
#endif

#include "savestr.h"

/* #define NEW_SAVESTR */

/*
 * Allocate sufficient storage for the string s, copy it, and
 * return a pointer to the copy.
 *
 * We might like to have a separate string storage area,
 * and phase out the use of getbuf...  However, all modules
 * use givbuf() to release strings, so it will be difficult
 * to find all the places that would need to be switched
 * to rls_str().
 *
 * For now, we make rls_str() simply a wrapper for givbuf()...
 * In the fullness of time maybe we will make the switch.
 */

#ifndef NEW_SAVESTR

#include "getbuf.h"

const char *savestr(const char* astring)
{
	char *sp;

	sp=(char*) getbuf( strlen(astring) +1 );
	if( sp == NULL ) mem_err("savestr");
	strcpy(sp,astring);
	return(sp);
}

void rls_str(const char* astring)
{
	givbuf((void *)astring);
}

#else /* NEW_SAVESTR */

// This "new" method looks to be not thread-safe!?
// But we don't appear to be using it?

#include <stdlib.h>	/* malloc() */
#include "freel.h"
#include "savestr.h"

#ifdef DEBUG
static u_long string_debug=0;
#endif /* DEBUG */

/* The string space has the potential to become quite fragmented!? */
#define MAX_STRING_FRAGS	1024

static FreeBlk string_free_list[MAX_STRING_FRAGS];
static int inited=0;
#define MEM_CHUNK_SIZE		4096

#ifdef CAUTIOUS
#define MAX_STRING_CHUNKS	32
static char *chunk_base[MAX_STRING_CHUNKS];
static int n_string_chunks=0;
#endif /* CAUTIOUS */

char *savestr(char* astring)
{
	int nneed;
	long buf;
	char *newbuf;

	if( !inited ){
		int i;

		inited=1;
		freeinit(string_free_list,MAX_ASTRING_FRAGS,MEM_CHUNK_SIZE);

		newbuf = malloc(MEM_CHUNK_SIZE);

		if( newbuf == NULL )
			error1("unable to malloc initial string buffer");

		/* freeinit() makes the "block" addresses start at 0.
		 * We therefore go through and add the base addr to each.
		 */

		for(i=0;i<MAX_STRING_FRAGS;i++)
			string_free_list[i].blkno += (u_long) newbuf;
#ifdef DEBUG
		string_debug = add_debug_module("strings");
#endif /* DEBUG */

#ifdef CAUTIOUS
		chunk_base[n_string_chunks++] = newbuf;
#endif /* CAUTIOUS */
	}

	nneed = strlen(astring) + 1;

	buf = getspace(string_free_list,nneed);

	if( buf == -1 ){	/* no more space */
		newbuf = malloc(MEM_CHUNK_SIZE);
		if( newbuf == NULL )
			error1("couldn't get another page for strings");

		/* add this to the free list */

		addspace(string_free_list,MEM_CHUNK_SIZE,(u_long)newbuf);

#ifdef CAUTIOUS
		if( n_string_chunks >= MAX_ASTRING_CHUNKS )
			error1("rls_str:  out of string chunks");
		chunk_base[n_string_chunks++] = newbuf;
#endif /* CAUTIOUS */

		/* This is not exactly kosher, because getspace/givspace
		 * were not originally written with disjoint areas
		 * in mind...  The problem is that if a buggy program
		 * frees memory in the "hole", it will not be detected,
		 * and worse yet, could be given out later by a call
		 * to getspace()...  In this case, however, we
		 * are probably safe as long as we only release
		 * astrings obtained from getspace.
		 *
		 * Another potential bug is that as rls_str()
		 * uses strlen() to determine how much to give back,
		 * if the terminal NULL is moved the amount of
		 * memory returned will be wrong...
		 */

		buf = getspace(string_free_list,nneed);

		if( buf == -1 ){
			sprintf(error_astring,
		"couldn't allocate %d bytes for astring",nneed);
			error1(error_astring);
		}
	}

	newbuf = (char *) buf;

	strcpy(newbuf,astring);
#ifdef DEBUG
if( debug & astring_debug ){
sprintf(error_astring,"storing %d astring bytes at addr 0x%lx",nneed,buf);
advise(error_astring);
}
#endif /* DEBUG */
	return(newbuf);
}

void rls_str(char *astring)
{
#ifdef CAUTIOUS
	int i,ok=0;

	for(i=0;i<n_string_chunks;i++){
		if( astring >= chunk_base[i] &&
			astring <= (chunk_base[i]+MEM_CHUNK_SIZE) )
			ok=1;
	}
	if( !ok ){
		sprintf(error_astring,
	"rls_str:  address 0x%lx does not fall within any known astring chunks!?",
			(u_long)astring);
		error1(error_string);
	}
#endif /* CAUTIOUS */

#ifdef DEBUG
if( debug & astring_debug ){
sprintf(error_string,"freeing %d astring bytes at addr 0x%lx",
strlen(astring)+1,(u_long)astring);
advise(error_string);
}
#endif /* DEBUG */

	givspace(string_free_list, strlen(astring) + 1, (u_long)astring);
}





#endif /* NEW_SAVESTR */

