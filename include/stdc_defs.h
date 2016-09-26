/* for the unix version, this file includes autoconf's config.h... */

#ifndef _STDC_DEFS_H_
#define _STDC_DEFS_H_

#include <stdlib.h>		/* malloc */
#include <string.h>		/* strcmp */


#define msg_str 		MSG_STR
/* this is used in a cuda struct... */
/*#define error_string 		ERROR_STRING */

/*#define READFUNC_CAST		char *(*)(TMP_QSP_ARG_DECL  void *, int, void *) */
#define READFUNC_CAST		char *(*)(QSP_ARG_DECL  void *, int, void *)


/* Foreach_Loop */
#define FL_VARNAME(flp)		flp->f_varname
#define FL_LIST(flp)		flp->f_word_lp
#define FL_NODE(flp)		flp->f_word_np
#define FL_WORD(flp)		((const char *)NODE_DATA(FL_NODE(flp)))

#define SET_FL_VARNAME(flp,s)	flp->f_varname = s
#define SET_FL_LIST(flp,lp)	flp->f_word_lp = lp
#define SET_FL_NODE(flp,np)	flp->f_word_np = np
#define NEW_FOREACH_LOOP	((Foreach_Loop*) getbuf( sizeof(*frp) ))


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

#define NEW_VARIABLE(vp)	{vp=(Variable *)getbuf(sizeof(Variable));	\
				SET_VAR_NAME(vp,NULL); SET_VAR_VALUE(vp,NULL); }

/* BUG?  do init here??? */
#define NEW_DATA_OBJ(dp)	dp=(Data_Obj *)getbuf(sizeof(Data_Obj))

#define NEW_QUERY_STACK		((Query_Stack *)getbuf(sizeof(Query_Stack)))

#define NEW_FUNC_PTR		((Subrt *)getbuf(sizeof(Subrt)))
#define NEW_REFERENCE		((Reference *)getbuf(sizeof(Reference)))
#define NEW_POINTER		((Pointer *)getbuf(sizeof(Pointer)))
#define NEW_ITEM_CONTEXT(icp)	icp=((Item_Context *)getbuf(sizeof(Item_Context)))

/* Variable */
#define VAR_NAME(vp)		vp->var_item.item_name
#define VAR_VALUE(vp)		vp->var_u.u_value
#define VAR_TYPE(vp)		vp->var_type
#define VAR_FUNC(vp)		vp->var_u.u_func
#define SET_VAR_NAME(vp,s)	vp->var_item.item_name=s
/* this crashes if the value is NULL... */
/*#define SET_VAR_VALUE(vp,s)	{if (vp->var_u.u_value != NULL ) rls_str(vp->var_u.u_value); vp->var_u.u_value=savestr(s); } */
#define SET_VAR_VALUE(vp,s)	vp->var_u.u_value=s
#define SET_VAR_TYPE(vp,t)		vp->var_type=t
#define SET_VAR_FUNC(vp,f)		vp->var_u.u_func=f


/* Item_Class */
#define CL_NAME(icp)			(icp)->icl_item.item_name
#define SET_CL_FLAG_BITS(iclp,f)	(iclp)->icl_flags |= f

/* Need macros for push and pop, but these are functions? */

/* Query */
#define SET_QUERY_MACRO(qp,mp)		(qp)->q_mp = mp
#define SET_QUERY_ARGLIST(qp,lp)	(qp)->q_arg_lp = lp


/* Identifier */
#define ID_NAME(idp)			(idp)->id_item.item_name

#define ID_TYPE(idp)			(idp)->id_type
#define SET_ID_TYPE(idp,t)		(idp)->id_type = t
#define ID_REF(idp)			((Reference *)(idp)->id_data)
#define SET_ID_REF(idp,refp)		(idp)->id_data = refp
#define ID_FUNC(idp)			((Function_Ptr *)(idp)->id_data)
#define SET_ID_FUNC(idp,funcp)		(idp)->id_data = funcp
#define ID_DOBJ_CTX(idp)		(idp)->id_icp
#define SET_ID_DOBJ_CTX(idp,icp)	(idp)->id_icp = icp
#define ID_PTR(idp)			((Pointer *)(idp)->id_data)
#define ID_SHAPE(idp)			(idp)->id_shpp
#define SET_ID_SHAPE(idp,shpp)		(idp)->id_shpp = shpp
#define PUSH_ID_CONTEXT(icp)		PUSH_ITEM_CONTEXT(id_itp,icp)
#define POP_ID_CONTEXT			POP_ITEM_CONTEXT(id_itp)

/* ContextPair */
#define CP_ID_CTX(cpp)			(cpp)->id_icp
#define SET_CP_ID_CTX(cpp,icp)		(cpp)->id_icp = icp
#define CP_OBJ_CTX(cpp)			(cpp)->dobj_icp
#define SET_CP_OBJ_CTX(cpp,icp)		(cpp)->dobj_icp = icp

#define SET_ID_PTR(idp,p)		(idp)->id_data = p

/* Reference */
#define REF_OBJ(refp)		(refp)->ref_dp
#define SET_REF_OBJ(refp,dp)	(refp)->ref_dp = dp
#define REF_ID(refp)		(refp)->ref_idp
#define SET_REF_ID(refp,idp)	(refp)->ref_idp = idp
#define REF_TYPE(refp)		(refp)->ref_type
#define SET_REF_TYPE(refp,t)	(refp)->ref_type = t
#define REF_SBUF(refp)		(refp)->ref_sbp
#define SET_REF_SBUF(refp,sbp)	(refp)->ref_sbp = sbp
#define REF_DECL_VN(refp)	(refp)->ref_decl_enp
#define SET_REF_DECL_VN(refp,enp)	(refp)->ref_decl_enp = enp

/* Pointer */
#define PTR_DECL_VN(ptrp)		(ptrp)->ptr_decl_enp
#define SET_PTR_DECL_VN(ptrp,enp)	(ptrp)->ptr_decl_enp = enp
#define PTR_REF(ptrp)			(ptrp)->ptr_refp
#define SET_PTR_REF(ptrp,refp)		(ptrp)->ptr_refp = refp
#define PTR_FLAGS(ptrp)			(ptrp)->ptr_flags
#define SET_PTR_FLAGS(ptrp,f)		(ptrp)->ptr_flags = f
#define SET_PTR_FLAG_BITS(ptrp,f)	(ptrp)->ptr_flags |= f
#define CLEAR_PTR_FLAG_BITS(ptrp,f)	(ptrp)->ptr_flags &= ~(f)

/* Subrt */

#define SR_DEST_SHAPE(srp)		(srp)->sr_dest_shpp
#define SET_SR_DEST_SHAPE(srp,shpp)	(srp)->sr_dest_shpp = shpp
#define SR_ARG_DECLS(srp)		(srp)->sr_arg_decls
#define SET_SR_ARG_DECLS(srp,enp)	(srp)->sr_arg_decls = enp
#define SR_ARG_VALS(srp)		(srp)->sr_arg_vals
#define SET_SR_ARG_VALS(srp,lp)		(srp)->sr_arg_vals = lp
#define SR_SHAPE(srp)			(srp)->sr_shpp
#define SET_SR_SHAPE(srp,shpp)		(srp)->sr_shpp = shpp
#define SR_BODY(srp)			(srp)->sr_body
#define SET_SR_BODY(srp,enp)		(srp)->sr_body = enp
#define SR_ARG_DECLS(srp)		(srp)->sr_arg_decls
#define SR_FLAGS(srp)			(srp)->sr_flags
#define SET_SR_FLAG_BITS(srp,b)		(srp)->sr_flags |= b
#define CLEAR_SR_FLAG_BITS(srp,b)	(srp)->sr_flags &= ~(b)
#define SET_SR_FLAGS(srp,f)		(srp)->sr_flags = f
#define SR_NAME(srp)			(srp)->sr_item.item_name
#define SR_N_ARGS(srp)			(srp)->sr_n_args
#define SET_SR_N_ARGS(srp,n)		(srp)->sr_n_args = n
#define SR_RET_LIST(srp)		(srp)->sr_ret_lp
#define SET_SR_RET_LIST(srp,lp)		(srp)->sr_ret_lp = lp
#define SR_CALL_LIST(srp)		(srp)->sr_call_lp
#define SET_SR_CALL_LIST(srp,lp)	(srp)->sr_call_lp = lp
#define SR_CALL_VN(srp)			(srp)->sr_call_enp
#define SET_SR_CALL_VN(srp,enp)		(srp)->sr_call_enp = enp

#define SR_PREC_PTR(srp)		(srp)->sr_prec_p
#define SR_PREC_CODE(srp)		PREC_CODE(SR_PREC_PTR(srp))
#define SET_SR_PREC_PTR(srp,p)		(srp)->sr_prec_p = p

/*#define SET_SR_CALL_ENP(srp,enp)	[srp setCall_enp : enp] */


/* Dimension_Set macros */

//#define ALLOC_DIMSET			((Dimension_Set *)getbuf(sizeof(Dimension_Set)))
#define NEW_DIMSET			((Dimension_Set *)getbuf(sizeof(Dimension_Set)))
#define INIT_DIMSET_PTR(dsp)		dsp=NEW_DIMSET;
#define RELEASE_DIMSET(dsp)		givbuf(dsp);

#define NEW_INCSET			((Increment_Set *)getbuf(sizeof(Increment_Set)))

#define DIMENSION(dsp,idx)		(dsp)->ds_dimension[idx]
#define SET_DIMENSION(dsp,idx,v)	(dsp)->ds_dimension[idx]=(dimension_t)v

/* IncrementSet macros */
#define INCREMENT(isp,idx)		isp->is_increment[idx]
#define SET_INCREMENT(isp,idx,v)	isp->is_increment[idx]=v

/* Macro argument */
#define MA_PROMPT(map)			map->ma_prompt
#define MA_ITP(map)			map->ma_itp

/* Macro macros */
#define SET_MACRO_NAME(mp,s)		(mp)->m_item.item_name=s
#define SET_MACRO_N_ARGS(mp,n)		(mp)->m_n_args=n
#define SET_MACRO_TEXT(mp,t)		(mp)->m_text=t
#define SET_MACRO_FLAGS(mp,f)		(mp)->m_flags=f
#define SET_MACRO_ARG_TBL(mp,tbl)	(mp)->m_arg_tbl=tbl
#define SET_MACRO_FILENAME(mp,s)	(mp)->m_filename = s
#define SET_MACRO_LINENO(mp,n)		(mp)->m_lineno = n

#define MACRO_NAME(mp)			(mp)->m_item.item_name
#define MACRO_N_ARGS(mp)		(mp)->m_n_args
#define MACRO_TEXT(mp)			(mp)->m_text
#define MACRO_FLAGS(mp)			(mp)->m_flags
#define MACRO_ARG_TBL(mp)		(mp)->m_arg_tbl
#define MACRO_ARG(mp,idx)		MACRO_ARG_TBL(mp)[idx]
#define MACRO_PROMPT(mp,idx)		MA_PROMPT(MACRO_ARG(mp,idx))
#define MACRO_ITPS(mp)			(mp)->m_itps
#define MACRO_FILENAME(mp)		(mp)->m_filename
#define MACRO_LINENO(mp)		(mp)->m_lineno
#define SET_MACRO_FLAG_BITS(mp,f)	(mp)->m_flags |= f
#define CLEAR_MACRO_FLAG_BITS(mp,f)	(mp)->m_flags &= ~(f)

#define DIMENSION_NAME(idx)	dimension_name[idx]

/* Command */
#define CMD_SELECTOR(cp)	cp->cmd_selector

/* Vector_Function stuff - moved to obj_args.h */

#define VF_NAME(vfp)			(vfp)->vf_item.item_name
#define VF_FLAGS(vfp)			(vfp)->vf_flags
#define VF_CODE(vfp)			(vfp)->vf_code
#define VF_TYPEMASK(vfp)		(vfp)->vf_typemask
#define VF_PRECMASK(vfp)		(vfp)->vf_precmask

#define SET_VF_NAME(vfp,s)		(vfp)->vf_item.item_name = s
#define SET_VF_FLAGS(vfp,f)		(vfp)->vf_flags = f
#define SET_VF_CODE(vfp,c)		(vfp)->vf_code = c
#define SET_VF_TYPEMASK(vfp,m)		(vfp)->vf_typemask = m
#define SET_VF_PRECMASK(vfp,m)		(vfp)->vf_precmask = m

#define FIND_VEC_FUNC(code)		(&vec_func_tbl[code])

#define UNKNOWN_OBJ_SHAPE(dp)		UNKNOWN_SHAPE(OBJ_SHAPE(dp))

#define HOW_MANY(pmpt)			how_many(QSP_ARG  pmpt)
#define HOW_MUCH(pmpt)			how_much(QSP_ARG  pmpt)

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

#define NAMEOF(s)			nameof(QSP_ARG  s)



/* remove from the dictionary... */
#define DELETE_OBJ_ITEM(dp)		del_dobj(QSP_ARG  dp)

#define ADD_OBJ_ITEM(dp)		/* add to the dictionary... */

#define SUBRT_ITEM_TYPE			subrt_itp
#define ID_ITEM_TYPE			id_itp
#define DOBJ_ITEM_TYPE			dobj_itp
#define POP_SUBRT_ID_CTX(s)		pop_subrt_ctx(QSP_ARG  s, ID_ITEM_TYPE)
#define POP_SUBRT_DOBJ_CTX(s)		pop_subrt_ctx(QSP_ARG  s, DOBJ_ITEM_TYPE)

#define PUSH_DOBJ_CONTEXT(icp)		push_dobj_context(QSP_ARG  icp)
#define POP_DOBJ_CONTEXT		pop_dobj_context(SINGLE_QSP_ARG)
#define DOBJ_CONTEXT_LIST		CONTEXT_LIST(dobj_itp)
#define ID_CONTEXT_LIST			CONTEXT_LIST(id_itp)

/* in the multi-thread environment, we have per-qsp context stacks!? */

#define INIT_INCSET_PTR(isp)		isp=((Increment_Set *)getbuf(sizeof(Increment_Set)));
#define INIT_OBJARG_PTR(oap)		oap=((Vec_Obj_Args *)getbuf(sizeof(Vec_Obj_Args)));
#define RELEASE_OBJARG_PTR(oap)		givbuf(oap)
#define INIT_OBJ_PTR(dp)		dp=((Data_Obj *)getbuf(sizeof(Data_Obj)));
#define INIT_ENODE_PTR(enp)		enp=((Vec_Expr_Node *)getbuf(sizeof(Vec_Expr_Node)));
/*#define INIT_SHAPE_PTR(shpp)		shpp=((Shape_Info *)getbuf(sizeof(Shape_Info))); */
/*#define RELEASE_SHAPE_PTR(shpp)		givbuf(shpp); */
#define INIT_SHAPE_PTR(shpp)		shpp=alloc_shape();
#define RELEASE_SHAPE_PTR(shpp)		rls_shape(shpp);
#define INIT_MACRO_PTR(mp)		mp=((Macro *)getbuf(sizeof(Macro)));
#define INIT_CPAIR_PTR(cpp)		cpp=((Context_Pair *)getbuf(sizeof(Context_Pair)));

#ifdef FOOBAR
#define DECLARE_INCSET(s)		/* nop for objective C... */
#define DECLARE_DIMSET(s)		/* nop */
#define DECLARE_OBJARGS(s)		/* nop */

// What is the difference - there seems to be no copy???
#define INIT_INCSET_PTR_FROM_OBJ(isp,incset)	\
	INIT_INCSET_PTR(isp)
#endif // FOOBAR

/* Deep or shallow copy??? */
#define DIMSET_COPY(dsp_to,dsp_fr)	*dsp_to = *dsp_fr

//#define _prt_msg_frag	c_prt_msg_frag

/* is p a code or a precision struct ptr??? */

/*#define CURRENT_INPUT_FILENAME	"(CURRENT_INPUT_FILENAME not implemented)" */
#define CURRENT_INPUT_FILENAME	QRY_FILENAME(CURR_QRY(THIS_QSP))

/*#define ASSIGN_VAR(vname,val)	[[Variable insure : STRINGOBJ(vname) ] setValue : STRINGOBJ(val) ] */

#define ASKIF(p)		askif(QSP_ARG  p )

//#define _prt_msg			c_prt_msg

#define LONGLIST(dp)		longlist(QSP_ARG  dp)

#define PICK_DATA_AREA(p)	pick_data_area(QSP_ARG  p)

#define TRY_OPEN(s,m)		try_open(QSP_ARG  s,m)
#define TRY_HARD(s,m)		try_hard(QSP_ARG  s,m)
#define TRYNICE(s,m)		trynice(QSP_ARG  s,m)

#define CONFIRM(p)		confirm(QSP_ARG  p)

#define DOBJ_COPY(dpto,dpfr)    *dpto = *dpfr


/* added for standard C: */
/* BUT conflict with X11 !? */
typedef int QUIP_BOOL;


/* to make it easy to switch to NSStrings later... */
/* BUT conflicts with X11 !? */
#define Quip_String	const char

#define VAR_OF(s)	var_of(QSP_ARG  s)
#define PICK_VAR(s)	pick_var_(QSP_ARG  s)

/*#define ELEMENT_SIZE(dp)	(siztbl[ MACHINE_PREC(dp) ]) */
#define ELEMENT_SIZE(dp)	OBJ_PREC_MACH_SIZE(dp)

#define PICK_OBJ(pmpt)		pick_obj(QSP_ARG   pmpt)

/* Filetype */
#define FT_NAME(ftp)		(ftp)->ft_item.item_name
#define FT_CODE(ftp)		(ftp)->ft_code
#define FT_FLAGS(ftp)		(ftp)->ft_flags
#define FT_OPEN_FUNC(ftp)	(ftp)->op_func
#define FT_READ_FUNC(ftp)	(ftp)->rd_func
#define FT_WRITE_FUNC(ftp)	(ftp)->wt_func
#define FT_CLOSE_FUNC(ftp)	(ftp)->close_func
#define FT_CONV_FUNC(ftp)	(ftp)->conv_func
#define FT_UNCONV_FUNC(ftp)	(ftp)->unconv_func
#define FT_SEEK_FUNC(ftp)	(ftp)->seek_func
#define FT_INFO_FUNC(ftp)	(ftp)->info_func

#define SET_FT_CODE(ftp,c)		(ftp)->ft_code = c
#define SET_FT_FLAGS(ftp,v)		(ftp)->ft_flags = v
#define SET_FT_OPEN_FUNC(ftp,f)		(ftp)->op_func = f
#define SET_FT_READ_FUNC(ftp,f)		(ftp)->rd_func = f
#define SET_FT_WRITE_FUNC(ftp,f)	(ftp)->wt_func = f
#define SET_FT_CLOSE_FUNC(ftp,f)	(ftp)->close_func = f
#define SET_FT_CONV_FUNC(ftp,f)		(ftp)->conv_func = f
#define SET_FT_UNCONV_FUNC(ftp,f)	(ftp)->unconv_func = f
#define SET_FT_SEEK_FUNC(ftp,f)		(ftp)->seek_func = f
#define SET_FT_INFO_FUNC(ftp,f)		(ftp)->info_func = f

/* Image_File */
#define IF_NAME(ifp)		(ifp)->if_name
#define IF_TYPE(ifp)		(ifp)->if_ftp
#define SET_IF_TYPE(ifp,ftp)	(ifp)->if_ftp = ftp
#define IF_TYPE_CODE(ifp)	FT_CODE( IF_TYPE(ifp) )

/* Debug_Module */
#define DEBUG_NAME(dbmp)		(dbmp)->db_name
#define DEBUG_MASK(dbmp)		(dbmp)->db_mask
#define SET_DEBUG_MASK(dbmp,m)		(dbmp)->db_mask = m
#define DEBUG_FLAGS(dbmp)		(dbmp)->db_flags
#define SET_DEBUG_FLAGS(dbmp,f)		(dbmp)->db_flags = f
#define CLEAR_DEBUG_FLAG_BITS(dbmp,b)	(dbmp)->db_flags &= ~(b)


#endif /* ! _STDC_DEFS_H_ */




