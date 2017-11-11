#include "quip_config.h"

#include "quip_prot.h"
#include "identifier.h"

Item_Type *id_itp=NULL;

ITEM_INIT_FUNC(Identifier,id,0)
ITEM_GET_FUNC(Identifier,id)
ITEM_CHECK_FUNC(Identifier,id)
ITEM_NEW_FUNC(Identifier,id)
ITEM_DEL_FUNC(Identifier,id)

void _restrict_id_context(QSP_ARG_DECL  int flag)
{
	RESTRICT_ITEM_CONTEXT(id_itp,flag);
}

Item_Context *_create_id_context(QSP_ARG_DECL  const char *name)
{
	return create_item_context(id_itp, name);
}

Identifier * _new_identifier(QSP_ARG_DECL  const char *name)
{
	Identifier *idp;

	idp = new_id(name);
	if( idp == NULL ) return idp;

	// now clear the fields
	SET_ID_TYPE(idp,ID_INVALID);
	SET_ID_SHAPE(idp,NULL);
	SET_ID_DATA(idp,NULL);
	SET_ID_DOBJ_CTX(idp,NULL);

	return idp;
}

void set_id_shape(Identifier *idp, Shape_Info *shpp)
{
	if( shpp == NULL ){
		if( ID_SHAPE(idp) != NULL ){
			rls_shape(ID_SHAPE(idp));
		}
		SET_ID_SHAPE(idp,NULL);
		return;
	} else {
		if( ID_SHAPE(idp) == NULL ){
			SET_ID_SHAPE(idp,alloc_shape());
		}
		(*(ID_SHAPE(idp))) = *shpp;
	}
}

