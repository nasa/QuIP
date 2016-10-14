/* for the unix version, this file includes autoconf's config.h... */

#ifndef _STDC_DEFS_H_
#define _STDC_DEFS_H_

#include <stdlib.h>		/* malloc */
#include <string.h>		/* strcmp */


/* this is used in a cuda struct... */
/*#define error_string 		ERROR_STRING */


/* Dictionary */

#define SET_DICT_NAME(dict_p,s)		dict_p->dict_name = s
#define SET_DICT_LIST(dict_p,lp)	dict_p->dict_lp = lp
#define SET_DICT_HT(dict_p,htp)		dict_p->dict_htp = htp
#define SET_DICT_FLAGS(dict_p,f)	dict_p->dict_flags = f
#define SET_DICT_N_COMPS(dict_p,n)	dict_p->dict_ncmps = n
#define SET_DICT_N_FETCHES(dict_p,n)	dict_p->dict_fetches = n
#define SET_DICT_FLAG_BITS(dict_p,f)	dict_p->dict_flags |= f
#define CLEAR_DICT_FLAG_BITS(dict_p,f)	dict_p->dict_flags &= ~(f)

#define DICT_NAME(dict_p)		dict_p->dict_name
#define DICT_LIST(dict_p)		dict_p->dict_lp
#define DICT_HT(dict_p)			dict_p->dict_htp
#define DICT_FLAGS(dict_p)		dict_p->dict_flags
#define DICT_N_COMPS(dict_p)		dict_p->dict_ncmps
#define DICT_N_FETCHES(dict_p)		dict_p->dict_fetches


#define CURR_QRY(qsp)			((Query *)TOP_OF_STACK((qsp)->qs_query_stack))
#define PREV_QRY(qsp)			((Query *)NODE_DATA(nth_elt((qsp)->qs_query_stack,1)))
#define FIRST_MENU(qsp)			((Menu *)BOTTOM_OF_STACK((qsp)->qs_menu_stack))
#define FIRST_QRY(qsp)			((Query *)BOTTOM_OF_STACK((qsp)->qs_query_stack))
#define QS_RETSTR_IDX(qsp)		(qsp)->qs_which_retstr
#define SET_QS_RETSTR_IDX(qsp,n)	(qsp)->qs_which_retstr=n
#define QS_RETSTR_AT_IDX(qsp,idx)	(qsp)->qs_retstr[idx]
#define SET_QS_RETSTR_AT_IDX(qsp,idx,sbp)	(qsp)->qs_retstr[idx] = sbp
/*#define QS_WHICH_ESTR(qsp)		(qsp)->qs_which_estr */
/*#define SET_QS_WHICH_ESTR(qsp,idx)	(qsp)->qs_which_estr= idx */

#define SET_QS_EDEPTH(qsp,d)		(qsp)->qs_edepth=d
#define QS_EDEPTH(qsp)			(qsp)->qs_edepth
/*#define QS_ESTRING(qsp)			((qsp)->qs_estr)[QS_WHICH_ESTR(qsp)] */
/*#define SET_QS_ESTR_ARRAY(qsp,str_p)	(qsp)->qs_estr = str_p */
/*#define QS_ESTRING(qsp)			(qsp)->qs_expr_string */
/*#define SET_QS_ESTRING(qsp,s)		(qsp)->qs_expr_string = s */
#define QS_CURR_STRING(qsp)		(qsp)->qs_curr_string
#define SET_QS_CURR_STRING(qsp,s)	(qsp)->qs_curr_string=s

#define QS_MAX_WARNINGS(qsp)		(qsp)->qs_max_warnings
#define QS_N_WARNINGS(qsp)		(qsp)->qs_n_warnings
#define SET_QS_MAX_WARNINGS(qsp,n)	(qsp)->qs_max_warnings=n
#define SET_QS_N_WARNINGS(qsp,n)	(qsp)->qs_n_warnings = n
#define INC_QS_N_WARNINGS(qsp)	SET_QS_N_WARNINGS(qsp,1+QS_N_WARNINGS(qsp))

#define QS_CALLBACK_LIST(qsp)		(qsp)->qs_callback_lp
#define SET_QS_CALLBACK_LIST(qsp,lp)	(qsp)->qs_callback_lp = lp
#define QS_EVENT_LIST(qsp)		(qsp)->qs_event_lp
#define SET_QS_EVENT_LIST(qsp,lp)	(qsp)->qs_event_lp = lp

#ifdef BUILD_FOR_IOS
#define QS_QUEUE(qsp)			(qsp)->qs_queue
#define SET_QS_QUEUE(qsp,q)		(qsp)->qs_queue = q
#endif /* BUILD_FOR_IOS */

#ifdef FOOBAR
/* Are these for the scalar parser, the vector parser, or both? */

#define TOP_NODE		((Query_Stack *)THIS_QSP)->qs_top_enp
#define END_SEEN		THIS_QSP->qs_end_seen
#define YY_CP			THIS_QSP->qs_yy_cp
#define SET_YY_CP(s)		THIS_QSP->qs_yy_cp = s
#define EXPR_LEVEL		THIS_QSP->qs_expr_level
#define SET_EXPR_LEVEL(l)	THIS_QSP->qs_expr_level = l
#define LASTLINENO		THIS_QSP->qs_last_line_num
#define SET_LASTLINENO(n)	THIS_QSP->qs_last_line_num =  n
#define PARSER_LINENO		THIS_QSP->qs_parser_line_num
#define SET_PARSER_LINENO(n)	THIS_QSP->qs_parser_line_num = n
#define YY_LAST_LINE 		THIS_QSP->qs_yy_last_line
#define YY_INPUT_LINE 		THIS_QSP->qs_yy_input_line
#define SEMI_SEEN 		THIS_QSP->qs_semi_seen
#define SET_SEMI_SEEN(v) 	THIS_QSP->qs_semi_seen = v
/*#define VEXP_STR		((THIS_QSP->qs_estr)[THIS_QSP->qs_which_estr]) */
#define VEXP_STR		QS_EXPR_STRING(THIS_QSP)
#define FINAL			THIS_QSP->qs_final

#define SET_QS_YY_INPUT_LINE(qsp,l)	(qsp)->qs_yy_input_line=l
#define SET_QS_YY_LAST_LINE(qsp,l)	(qsp)->qs_yy_last_line=l
#define QS_EXPR_STRING(qsp)		(qsp)->qs_expr_string
#define SET_QS_EXPR_STRING(qsp,l)	(qsp)->qs_expr_string=l

#endif // FOOBAR

/* on unix this is how we set the flag: */
/*	dst_dp->dt_flags |= DT_ASSIGNED; */

/* for Objective C: */

/* These were getbuf calls in the old system... */
#define NEW_NODE(np)		{np=(Node *)getbuf(sizeof(Node));	\
				(np)->n_data=NULL; (np)->n_next=NO_NODE; (np)->n_prev=NO_NODE; }

#define NEW_QUERY_STACK		((Query_Stack *)getbuf(sizeof(Query_Stack)))

/* Need macros for push and pop, but these are functions? */

#define DV_VEC(dvp)			dvp->dv_vec
#define DV_INC(dvp)			dvp->dv_inc
#define DV_COUNT(dvp)			dvp->dv_count
#define DV_PREC(dvp)			dvp->dv_prec
#define DV_BIT0(dvp)			dvp->dv_bit0

#define SET_DV_VEC(dvp,v)		dvp->dv_vec=v
#define SET_DV_INC(dvp,v)		dvp->dv_inc=v
#define SET_DV_COUNT(dvp,v)		dvp->dv_count=v
#define SET_DV_PREC(dvp,v)		dvp->dv_prec=v
#define SET_DV_BIT0(dvp,v)		dvp->dv_bit0=v
#define SET_DV_FLAG_BITS(dvp,v)		dvp->dv_flags |= v



#define SUBRT_ITEM_TYPE			subrt_itp
#define ID_ITEM_TYPE			id_itp
#define DOBJ_ITEM_TYPE			dobj_itp

/* in the multi-thread environment, we have per-qsp context stacks!? */

#define INIT_INCSET_PTR(isp)		isp=((Increment_Set *)getbuf(sizeof(Increment_Set)));
#define INIT_OBJARG_PTR(oap)		oap=((Vec_Obj_Args *)getbuf(sizeof(Vec_Obj_Args)));
#define RELEASE_OBJARG_PTR(oap)		givbuf(oap)
#define INIT_OBJ_PTR(dp)		dp=((Data_Obj *)getbuf(sizeof(Data_Obj)));
/*#define INIT_SHAPE_PTR(shpp)		shpp=((Shape_Info *)getbuf(sizeof(Shape_Info))); */

#ifdef FOOBAR
#define DECLARE_INCSET(s)		/* nop for objective C... */
#define DECLARE_DIMSET(s)		/* nop */
#define DECLARE_OBJARGS(s)		/* nop */

// What is the difference - there seems to be no copy???
#define INIT_INCSET_PTR_FROM_OBJ(isp,incset)	\
	INIT_INCSET_PTR(isp)
#endif // FOOBAR

//#define _prt_msg_frag	c_prt_msg_frag

/* is p a code or a precision struct ptr??? */

/*#define ASSIGN_VAR(vname,val)	[[Variable insure : STRINGOBJ(vname) ] setValue : STRINGOBJ(val) ] */

//#define _prt_msg			c_prt_msg

#endif /* ! _STDC_DEFS_H_ */




