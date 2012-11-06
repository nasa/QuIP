char VersionId_sup_permute[] = "$RCSfile$ $Revision$ $Date$";

/* permute the lines of a file */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>		/* strcmp() */
#include "node.h"
#include "getbuf.h"
#include "rn.h"

#define N_CHAR_CHUNK	16000
static char charbuf[N_CHAR_CHUNK];

static u_long nlines;
static u_long nchrs;

#define LBSIZ	200
static char lbuf[LBSIZ];

static u_long *permbuf;
static List *line_list;
static char **line_tbl;


int main(ac,av)
char **av;
{
	register char *bptr;
	u_long i;
	int do_rev=0;
	int do_lace=0;
	Node *np;

	if( !strcmp(av[0],"reverse") ) do_rev=1;
	else if( !strcmp(av[0],"ilace") ) do_lace=1;

	bptr=(&charbuf[0]);
	nlines=nchrs=0;

	line_list = new_list();

	while( fgets(lbuf,LBSIZ,stdin) !=NULL ){

		if( nchrs >= (N_CHAR_CHUNK-LBSIZ) ){
			/* get another buffer */
			bptr = getbuf(N_CHAR_CHUNK);
			if( bptr == NULL ){
				error1("no more character storage space");
			}
			nchrs=0;
		}
		strcpy( bptr, lbuf );
		np = mk_node(bptr);
		nchrs += 1 + strlen(bptr);
		bptr += 1 + strlen(bptr);
		addHead(line_list,np);
		nlines++;
	}

	permbuf = (u_long *) getbuf( sizeof(u_long) * nlines );
	if( permbuf == NULL ) error1("error making permutation buffer");

	line_tbl = (char **)getbuf( sizeof(char *) * nlines );
	if( line_tbl == NULL ) error1("error making line table");

	/* now transfer the pointers from the list to the table */

	np=line_list->l_head;
	i=0;
	while(np!=NO_NODE){
		line_tbl[i] = np->n_data;
		np=np->n_next;
		i++;
	}

	if( do_rev ){
		for(i=0;i<nlines;i++)
			permbuf[i]=nlines-(1+i);
	} else if( do_lace ){
		for(i=0;i<nlines;i++)
			permbuf[i]= (nlines-1) - (i ^ 1);
	} else {
		//fprintf(stderr,"permuting %d lines\n",nlines);
		rninit();
		permute(permbuf,nlines);
	}

	for(i=0;i<nlines;i++)
		printf("%s",line_tbl[ permbuf[i] ]);

	return 0;
}

