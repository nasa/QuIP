
#include "quip_config.h"
#include "vector_parser_data.h"
#include "list.h"
#include "getbuf.h"

//#define init_vector_parser_data(vpd_p) _init_vector_parser_data(QSP_ARG  vpd_p)

static void init_vector_parser_data(Vector_Parser_Data *vpd_p)
{
	bzero(vpd_p,sizeof(*vpd_p));

	// // Now allocate the strings
	SET_VPD_YY_INPUT_LINE(vpd_p,new_stringbuf());
	SET_VPD_YY_LAST_LINE(vpd_p,new_stringbuf());
	SET_VPD_EXPR_STRING(vpd_p,new_stringbuf());
	SET_VPD_YY_WORD_BUF(vpd_p,new_stringbuf());
	SET_VPD_EDEPTH(vpd_p, -1);
	SET_VPD_CURR_STRING(vpd_p, sb_buffer(VPD_EXPR_STRING(vpd_p)) );
	SET_VPD_SUBRT_CTX_STACK(vpd_p,new_list());
}

// we don't think we will ever need to release these...

#ifdef FOOBAR
#define rls_vector_parser_data(vpd_p) _rls_vector_parser_data(QSP_ARG  vpd_p)

static void _rls_vector_parser_data(QSP_ARG_DECL  Vector_Parser_Data *vpd_p)
{
	rls_stringbuf(VPD_YY_INPUT_LINE(vpd_p));
	rls_stringbuf(VPD_YY_LAST_LINE(vpd_p));
	rls_stringbuf(VPD_EXPR_STRING(vpd_p));
	rls_stringbuf(VPD_YY_WORD_BUF(vpd_p));
	rls_list(VPD_SUBRT_CTX_STACK(vpd_p));
}
#endif // FOOBAR

Vec_Expr_Node * qs_top_node( SINGLE_QSP_ARG_DECL  )
{
	assert( THIS_VPD != NULL );
	//return THIS_QSP->qs_top_enp;
	return VPD_TOP_ENP( THIS_VPD );
}

void set_top_node( QSP_ARG_DECL  Vec_Expr_Node *enp )
{
	assert( THIS_VPD != NULL );
	SET_VPD_TOP_ENP( THIS_VPD, enp );
}

void set_curr_string(QSP_ARG_DECL  const char *s)
{
	assert( THIS_VPD != NULL );
	SET_VPD_CURR_STRING( THIS_VPD,s);
}

const char *qs_curr_string(SINGLE_QSP_ARG_DECL)
{
	assert( THIS_VPD != NULL );
	return VPD_CURR_STRING( THIS_VPD );
}

String_Buf *qs_expr_string(SINGLE_QSP_ARG_DECL)
{
	assert( THIS_VPD != NULL );
	return VPD_EXPR_STRING( THIS_VPD );
}

Vector_Parser_Data *new_vector_parser_data()
{
	Vector_Parser_Data *vpd_p;
	vpd_p = getbuf( sizeof(Vector_Parser_Data) );
	init_vector_parser_data(vpd_p);
	return vpd_p;
}
