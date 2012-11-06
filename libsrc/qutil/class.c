
#include "quip_config.h"

char VersionId_qutil_class[] = QUIP_VERSION_STRING;

#include "data_obj.h"	/* hunt_obj() */

static Item_Type * icl_itp=NO_ITEM_TYPE;

static void icl_init(SINGLE_QSP_ARG_DECL);
static Item_Class *new_ic(QSP_ARG_DECL  const char *name);

static DECL_INIT_FUNC(icl_init,icl_itp,"ITEM_CLASS")
static DECL_NEW_FUNC(new_ic,Item_Class *,icl_itp,icl_init)


Item_Class * new_item_class(QSP_ARG_DECL  const char *name)
{
	Item_Class *icp;

	icp = new_ic(QSP_ARG  name);
	if( icp == NO_ITEM_CLASS )
		return(icp);

	icp->icl_lp = new_list();
	icp->icl_flags = NEED_CLASS_CHOICES;
	return(icp);
}

/* expand the membership of this class */

void add_items_to_class(Item_Class *icp,Item_Type * itp,void* data,
		Item * (*lookup)(QSP_ARG_DECL  const char *))
{
	Node *np;
	Member_Info *mip;

	mip = (Member_Info*) getbuf(sizeof(*mip));
	mip->mi_itp = itp;
	mip->mi_data = data;
	mip->mi_lookup = lookup;
	np = mk_node(mip);
	addTail(icp->icl_lp,np);

	/* now make the item type point to this class too */
	if( itp->it_classlist == NO_LIST )
		itp->it_classlist = new_list();
	np = mk_node(icp);
	addTail(itp->it_classlist,np);

	icp->icl_flags |= NEED_CLASS_CHOICES;
}

/* return the member info struct for this item */

Member_Info *get_member_info(QSP_ARG_DECL  Item_Class *icp,const char *name)
{
	Node *np;
	Member_Info *mip;

	np = icp->icl_lp->l_head;
	while(np!=NO_NODE){
		Item *ip;

		mip = (Member_Info*) np->n_data;

		if( mip->mi_lookup != NULL ){
			ip = (*mip->mi_lookup)(QSP_ARG  name);
		} else {
			ip = item_of(QSP_ARG  mip->mi_itp,name);
		}

		if( ip != NO_ITEM )
			return(mip);

		np = np->n_next;
	}

	sprintf(ERROR_STRING,"No member %s in item class %s",name,icp->icl_name);
	WARN(ERROR_STRING);

	return(NO_MEMBER_INFO);
}

/* return a ptr to the named member of this class */

Item * get_member(QSP_ARG_DECL  Item_Class *icp,const char *name)
{
	Node *np;
	Item *ip;
	Member_Info *mip;

	np = icp->icl_lp->l_head;
	while(np!=NO_NODE){
		mip = (Member_Info*) np->n_data;

		if( mip->mi_lookup != NULL )
			ip = (*mip->mi_lookup)(QSP_ARG  name);
		else
			ip = item_of(QSP_ARG  mip->mi_itp,name);

		if( ip != NO_ITEM ) return(ip);
		np=np->n_next;
	}
	sprintf(ERROR_STRING,"No member %s found in %s class",
		name,icp->icl_name);
	WARN(ERROR_STRING);

	return(NO_ITEM);
}


