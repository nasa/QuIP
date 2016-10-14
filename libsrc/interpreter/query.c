#include <string.h>
#include "quip_config.h"
#include "quip_prot.h"
#include "query.h"
#include "warn.h"
#include "getbuf.h"

char *end_text(QSP_ARG_DECL  void *buf,int size,void *stream);

// what goes here?

static List *query_free_list=NO_LIST;

Query *new_query(void)
{
	Query *qp;

	if( query_free_list == NO_LIST )
		query_free_list = new_list();

	if( QLIST_HEAD(query_free_list) != NO_NODE ){
		Node *np;
		np = remHead(query_free_list);
		qp = (Query *)NODE_DATA(np);
		// We don't re-initialize the fields,
		// so structs that need to be allocated are
		// already available...
		rls_node(np);
	} else {
		qp=(Query *)getbuf(sizeof(Query));
		// Set the whole thing to zero
		memset( (void *) qp, 0, sizeof(*qp) );
	}
	SET_QRY_RETSTR_IDX(qp,0);
	// If these are not null now, that means they have been allocated!?
	//for(i=0;i<N_QRY_RETSTRS;i++){
	//	SET_QRY_RETSTR_AT_IDX(qp,i,NULL);
	//}
	return(qp);
}

void rls_query(Query *qp)
{
	// release any resources held by the query?
	Node *np;

	np=mk_node(qp);
	addHead(query_free_list,np);
}

