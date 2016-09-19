#include "quip_config.h"
#include "quip_prot.h"
#include "container.h"
#include <assert.h>

Container * new_container(QSP_ARG_DECL  container_type_code type)
{
	Container *cnt_p=NULL;

	assert( type >= 0 && type < N_CONTAINER_TYPES );

	cnt_p = getbuf( sizeof(Container) );
	cnt_p->type = type;
	switch(type){
		case LIST_CONTAINER:
			cnt_p->ptr.cnt_lp = new_list();
			break;
		case HASH_TBL_CONTAINER:
			cnt_p->ptr.cnt_htp = ht_init(NULL);
			break;
		case RB_TREE_CONTAINER:
			cnt_p->ptr.cnt_tree_p = create_rb_tree();
		default:
			// could be assertion?
			sprintf(ERROR_STRING,"Invalid container type code %d",type);
			ERROR1(ERROR_STRING);
			break;
	}
	return cnt_p;
}
			
//extern void add_to_container(QSP_ARG_DECL  Container *cnt_p, Item *ip);
//extern void remove_from_container(QSP_ARG_DECL  Container *cnt_p, const char *name);
//extern Item *container_find_match(QSP_ARG_DECL  Container *cnt_p, const char *name);
//extern Item *container_find_substring_match(QSP_ARG_DECL  Container *cnt_p, const char *frag);


