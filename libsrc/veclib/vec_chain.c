#include "quip_config.h"
#include "quip_prot.h"
#include "vec_chain.h"

Chain *new_chain( QSP_ARG_DECL  const char *s )
{
	Chain *chp;
	chp = getbuf(sizeof(Chain));
	SET_CHAIN_NAME(chp,savestr(s));
	return chp;
}

void del_vec_chain(QSP_ARG_DECL  const char *s)
{
	warn("del_vec_chain not implemented!?");
}

