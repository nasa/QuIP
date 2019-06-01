// support for the scalar expression parser

#include <string.h>
#include "quip_prot.h"
#include "nexpr.h"
#include "function.h"
#include "spink.h"

#define eval_cam_expr( enp ) _eval_cam_expr( QSP_ARG  enp )

static Item * _eval_cam_expr( QSP_ARG_DECL  Scalar_Expr_Node *enp )
{
	Item *ip=NULL;
	const char *s;

	switch(enp->sen_code){
		case N_LITSTR:
		case N_QUOT_STR:
			s = eval_scalexp_string(enp);
			ip = (Item *)spink_cam_of(s);		// BUG - support for other cameras??? 
			if( ip == NULL ){
				sprintf(ERROR_STRING,
					"No camera \"%s\"!?",s);
				warn(ERROR_STRING);
				return NULL;
			}
			break;
		default:
			sprintf(ERROR_STRING,
		"unexpected case in eval_cam_expr %d",enp->sen_code);
			warn(ERROR_STRING);
			assert(0);
			break;
	}
	return(ip);
}

void _init_cam_expr_funcs(SINGLE_QSP_ARG_DECL)
{
	set_eval_cam_func(QSP_ARG  _eval_cam_expr );
}

