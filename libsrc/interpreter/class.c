
#include "quip_config.h"
#include "quip_prot.h"

#include "data_obj.h"	/* hunt_obj() */

static Item_Type * icl_itp=NULL;

static ITEM_INIT_PROT(Item_Class,icl)
static ITEM_NEW_PROT(Item_Class,icl)
static ITEM_INIT_FUNC(Item_Class,icl,0)
static ITEM_NEW_FUNC(Item_Class,icl)

#define init_icls()	_init_icls(SINGLE_QSP_ARG)
#define new_icl(name)	_new_icl(QSP_ARG  name)

Item_Class * _new_item_class(QSP_ARG_DECL  const char *name)
{
	Item_Class *icp;

	icp = new_icl(name);
	if( icp == NULL )
		return(icp);

	icp->icl_lp = new_list();
	icp->icl_flags = NEED_CLASS_CHOICES;	// BUG item_class items are per-thread!
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
	if( IT_CLASS_LIST(itp) == NULL )
		SET_IT_CLASS_LIST(itp, new_list());
	np = mk_node(icp);
	addTail(IT_CLASS_LIST(itp),np);

	icp->icl_flags |= NEED_CLASS_CHOICES;	// BUG item_class items are per-thread!
}

/* return the member info struct for this item */

Member_Info *_check_member_info(QSP_ARG_DECL  Item_Class *icp,const char *name)
{
	Node *np;
	Member_Info *mip;

	np = QLIST_HEAD(icp->icl_lp);
	while(np!=NULL){
		Item *ip;

		mip = (Member_Info*) np->n_data;

		if( mip->mi_lookup != NULL ){
			ip = (*mip->mi_lookup)(QSP_ARG  name);
		} else {
			ip = item_of(mip->mi_itp,name);
		}

		if( ip != NULL ){
			return(mip);
		}

		np = np->n_next;
	}
	return(NULL);
}

Member_Info *_get_member_info(QSP_ARG_DECL  Item_Class *icp,const char *name)
{
	Member_Info *mip;
	mip = check_member_info(icp, name);
	if( mip == NULL ){
		sprintf(ERROR_STRING,
	"No member %s in item class %s",name,CL_NAME(icp));
		warn(ERROR_STRING);
	}
	return mip;
}

/* return a ptr to the named member of this class */

Item * _check_member(QSP_ARG_DECL  Item_Class *icp,const char *name)
{
	Node *np;
	Item *ip;
	Member_Info *mip;

	np = QLIST_HEAD(icp->icl_lp);
	while(np!=NULL){
		mip = (Member_Info*) np->n_data;

		if( mip->mi_lookup != NULL )
			ip = (*mip->mi_lookup)(QSP_ARG  name);
		else
			ip = item_of(mip->mi_itp,name);

		if( ip != NULL ) return(ip);
		np=np->n_next;
	}
	return(NULL);
}

Item * _get_member(QSP_ARG_DECL  Item_Class *icp,const char *name)
{
	Item *ip;

	ip = check_member(icp, name );
	if( ip == NULL ){
		sprintf(ERROR_STRING,"No member %s found in %s class",
			name,CL_NAME(icp));
		warn(ERROR_STRING);
	}
	return ip;
}


