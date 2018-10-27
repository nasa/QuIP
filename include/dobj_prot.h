#ifndef _DOBJ_PROT_H_
#define _DOBJ_PROT_H_

#include "quip_config.h"
#include <stdio.h>
#include "quip_menu.h"
#include "ascii_fmts.h"
#include "data_obj.h"
#include "scalar_value.h"
//#define CONST const	// why needed?

// BUG - this file contains both prototypes for functions which are private
// to the module, and also the external api...



/* contig.c */

extern int is_evenly_spaced(Data_Obj *);
extern void check_contiguity(Data_Obj *);
extern int _is_contiguous(QSP_ARG_DECL  Data_Obj *);
#define is_contiguous(dp) _is_contiguous(QSP_ARG  dp)


/* dobj_util.c */

extern void add_reference(Data_Obj *dp);
extern void remove_reference(Data_Obj *dp);

/* data_obj.c */

extern void		_init_precisions(SINGLE_QSP_ARG_DECL);
extern Precision * 	_get_prec(QSP_ARG_DECL  const char *);

#define init_precisions()	_init_precisions(SINGLE_QSP_ARG)
#define get_prec(s)		_get_prec(QSP_ARG  s)

extern Data_Obj *	_dobj_of(QSP_ARG_DECL  const char *);
extern void		_list_dobjs(QSP_ARG_DECL  FILE *fp);
extern Data_Obj *	_new_dobj(QSP_ARG_DECL  const char *);
extern List *		_dobj_list(SINGLE_QSP_ARG_DECL);

#define dobj_list()	_dobj_list(SINGLE_QSP_ARG)


extern void		_disown_child(QSP_ARG_DECL  Data_Obj * dp);
extern void		_delvec(QSP_ARG_DECL  Data_Obj * dp);
extern void		info_area(QSP_ARG_DECL  Data_Area *ap);
extern void		info_all_dps(SINGLE_QSP_ARG_DECL);
extern void		sizinit(void);
extern void		make_complex(Shape_Info *shpp);
extern int		set_shape_flags(Shape_Info *shpp,uint32_t type_flags);
extern int		auto_shape_flags(Shape_Info *shpp);
extern void		_show_space_used(QSP_ARG_DECL  Data_Obj * dp);
#define show_space_used(dp) _show_space_used(QSP_ARG  dp)
extern void		dobj_iterate(Data_Obj * dp,void (*func)(Data_Obj * ,uint32_t));
extern void		_dpair_iterate(QSP_ARG_DECL  Data_Obj * dp,Data_Obj * dp2,
				void (*func)(QSP_ARG_DECL  Data_Obj * ,uint32_t,Data_Obj * ,uint32_t));
#define dpair_iterate(dp,dp2, func) _dpair_iterate(QSP_ARG  dp,dp2, func)

extern void		_gen_xpose(QSP_ARG_DECL  Data_Obj * dp,int dim1,int dim2);
#define gen_xpose(dp,dim1,dim2) _gen_xpose(QSP_ARG  dp,dim1,dim2)

extern double		get_dobj_size(QSP_ARG_DECL  Data_Obj * dp,int index);
extern const char *	get_dobj_prec_name(QSP_ARG_DECL  Data_Obj * dp);
extern double		get_dobj_il_flg(QSP_ARG_DECL  Data_Obj * dp);
extern void		_dataobj_init(SINGLE_QSP_ARG_DECL);
#define dataobj_init() _dataobj_init(SINGLE_QSP_ARG)

extern void		init_dfuncs(SINGLE_QSP_ARG_DECL);
extern int		same_shape(Shape_Info *,Shape_Info *);

#define disown_child(dp) _disown_child(QSP_ARG  dp)
#define delvec(dp)	_delvec(QSP_ARG  dp)

/* dplist.c */

/* apparently , this file was split off from data_obj.c, bacause
 * some of the prototypes are still listed above!?
 */

//extern const char *	name_for_prec(int prec);
extern Precision *	prec_for_code(prec_t prec);
extern void		_dump_shape(QSP_ARG_DECL  Shape_Info *shpp);
extern void		_list_dobj(QSP_ARG_DECL  Data_Obj * dp);
extern void		_longlist(QSP_ARG_DECL  Data_Obj * dp);

#define dump_shape(shpp) _dump_shape(QSP_ARG  shpp)
#define list_dobj(dp)	_list_dobj(QSP_ARG  dp)
#define longlist(dp)	_longlist(QSP_ARG  dp)

//extern void		describe_shape(Shape_Info *shpp);
extern void _show_obj_dimensions(QSP_ARG_DECL  Data_Obj *dp, Dimension_Set *dsp, Increment_Set *isp);
#define show_obj_dimensions(dp, dsp, isp) _show_obj_dimensions(QSP_ARG  dp, dsp, isp)

/* pars_obj.c */
extern Data_Obj * pars_obj(const char *);

/* arrays.c */
/* formerly in arrays.h */

extern void release_tmp_obj(Data_Obj *);
extern int is_in_string(int c,const char *s);
extern void unlock_children(Data_Obj *dp);

extern void _unlock_all_tmp_objs(SINGLE_QSP_ARG_DECL);
extern void _set_array_base_index(QSP_ARG_DECL  int);
extern void _init_tmp_dps(SINGLE_QSP_ARG_DECL);
extern Data_Obj * _find_free_temp_dp(QSP_ARG_DECL  Data_Obj *dp);
extern Data_Obj * _temp_child(QSP_ARG_DECL  const char *name,Data_Obj * dp);
extern void _make_array_name(QSP_ARG_DECL  char *target_str,int buflen,Data_Obj * dp, index_t index,int which_dim,int subscr_type);
extern Data_Obj * _gen_subscript(QSP_ARG_DECL  Data_Obj * dp, int which_dim,index_t index,int subscr_type);
extern Data_Obj * _reduce_from_end(QSP_ARG_DECL  Data_Obj * dp, index_t index,int subscr_type);
extern Data_Obj * _d_subscript(QSP_ARG_DECL  Data_Obj * dp,index_t index);
extern Data_Obj * _c_subscript(QSP_ARG_DECL  Data_Obj * dp,index_t index);
extern void _reindex(QSP_ARG_DECL  Data_Obj * ,int,index_t);
extern void _list_temp_dps(QSP_ARG_DECL  FILE *fp);

#define unlock_all_tmp_objs()		_unlock_all_tmp_objs(SINGLE_QSP_ARG)
#define set_array_base_index(idx)	_set_array_base_index(QSP_ARG  idx)
#define init_tmp_dps()			_init_tmp_dps(SINGLE_QSP_ARG)
#define find_free_temp_dp(dp)		_find_free_temp_dp(QSP_ARG  dp)
#define temp_child(name,dp)		_temp_child(QSP_ARG  name,dp)
#define make_array_name(target_str,buflen,dp,index,which_dim,subscr_type)		_make_array_name(QSP_ARG  target_str,buflen,dp,index,which_dim,subscr_type)
#define gen_subscript(dp,which_dim,index,subscr_type)		_gen_subscript(QSP_ARG  dp,which_dim,index,subscr_type)
#define reduce_from_end(dp,index,subscr_type)		_reduce_from_end(QSP_ARG  dp,index,subscr_type)
#define d_subscript(dp,index)		_d_subscript(QSP_ARG  dp,index)
#define c_subscript(dp,index)		_c_subscript(QSP_ARG  dp,index)
#define reindex(dp,which_dim,idx)	_reindex(QSP_ARG  dp,which_dim,idx)
#define list_temp_dps(fp)		_list_temp_dps(QSP_ARG  fp)

/* get_obj.c */
/* formerly in get_obj.h */

extern Data_Obj * _hunt_obj(QSP_ARG_DECL  const char *s);
#define hunt_obj(s) _hunt_obj(QSP_ARG  s)
extern Data_Obj * _get_obj(QSP_ARG_DECL  const char *s);
#define get_obj(s)	_get_obj(QSP_ARG  s)
extern Data_Obj * get_vec(QSP_ARG_DECL  const char *s);
extern Data_Obj * img_of(QSP_ARG_DECL  const char *s);
extern Data_Obj * get_seq(QSP_ARG_DECL  const char *s);
extern Data_Obj * get_img(QSP_ARG_DECL  const char *s );


/* data_fns.c */
extern const char *	localname(void);

extern void *		multiply_indexed_data(Data_Obj *dp, dimension_t *offset );
extern void *		indexed_data(Data_Obj *dp, dimension_t offset );
extern void		make_contiguous(Data_Obj *);
extern int		_set_shape_dimensions(QSP_ARG_DECL  Shape_Info *shpp,Dimension_Set *dimensions,Precision *);
#define set_shape_dimensions(shpp,dimensions,prec_p) _set_shape_dimensions(QSP_ARG  shpp,dimensions,prec_p)

extern int		_obj_rename(QSP_ARG_DECL  Data_Obj *dp,const char *newname);
#define obj_rename(dp,newname) _obj_rename(QSP_ARG  dp,newname)
extern Data_Obj *	_mk_scalar(QSP_ARG_DECL  const char *name,Precision *prec_p);
extern void		_assign_scalar_obj(QSP_ARG_DECL  Data_Obj *,Scalar_Value *);
#define assign_scalar_obj(dp,svp) _assign_scalar_obj(QSP_ARG  dp,svp)

extern void		_extract_scalar_value(QSP_ARG_DECL  Scalar_Value *, Data_Obj *);
extern double		_cast_from_scalar_value(QSP_ARG_DECL  Scalar_Value *, Precision *prec_p);
extern void		_cast_dbl_to_scalar_value(QSP_ARG_DECL  Scalar_Value *, Precision *prec_p, double val);
/*extern const char *	string_for_scalar_value(QSP_ARG_DECL  Scalar_Value *, Precision *prec_p); */
extern void		_cast_dbl_to_cpx_scalar(QSP_ARG_DECL  int index, Scalar_Value *, Precision *prec_p, double val);
extern void		_cast_dbl_to_quat_scalar(QSP_ARG_DECL  int index, Scalar_Value *, Precision *prec_p, double val);
extern void		_cast_dbl_to_color_scalar(QSP_ARG_DECL  int index, Scalar_Value *, Precision *prec_p, double val);

#define extract_scalar_value(svp, dp) _extract_scalar_value(QSP_ARG  svp, dp)
#define cast_from_scalar_value(svp, prec_p) _cast_from_scalar_value(QSP_ARG  svp, prec_p)
#define cast_dbl_to_scalar_value(svp, prec_p, val) _cast_dbl_to_scalar_value(QSP_ARG  svp, prec_p, val)
#define cast_dbl_to_cpx_scalar(index, svp, prec_p, val) _cast_dbl_to_cpx_scalar(QSP_ARG  index, svp, prec_p, val)
#define cast_dbl_to_quat_scalar(index, svp, prec_p, val) _cast_dbl_to_quat_scalar(QSP_ARG  index, svp, prec_p, val)
#define cast_dbl_to_color_scalar(index, svp, prec_p, val) _cast_dbl_to_color_scalar(QSP_ARG  index, svp, prec_p, val)

extern Data_Obj *	mk_cscalar(QSP_ARG_DECL  const char *name,double rval, double ival);
extern Data_Obj *	_mk_vec(QSP_ARG_DECL  const char *,dimension_t, dimension_t,Precision *prec_p);
#define mk_vec(name,n_elts,depth,prec_p) _mk_vec(QSP_ARG  name,n_elts,depth,prec_p)

extern Data_Obj *	_dup_obj(QSP_ARG_DECL  Data_Obj *dp,const char *name);
extern int		_is_valid_dname(QSP_ARG_DECL  const char *name);
extern Data_Obj *	_dup_half(QSP_ARG_DECL  Data_Obj *dp,const char *name);
extern Data_Obj *	_dup_dbl(QSP_ARG_DECL  Data_Obj *dp,const char *name);
extern Data_Obj *	_dupdp(QSP_ARG_DECL  Data_Obj *dp);
#define dup_obj(dp,name) _dup_obj(QSP_ARG  dp,name)
#define is_valid_dname(name) _is_valid_dname(QSP_ARG  name)
#define dup_half(dp,name) _dup_half(QSP_ARG  dp,name)
#define dup_dbl(dp,name) _dup_dbl(QSP_ARG  dp,name)
#define dupdp(dp) _dupdp(QSP_ARG  dp)

extern Data_Obj *	_mk_img(QSP_ARG_DECL  const char *,dimension_t,dimension_t,dimension_t ,Precision *prec_p);
extern int		_set_obj_dimensions(QSP_ARG_DECL  Data_Obj *dp,Dimension_Set *dimensions,Precision *);
extern Data_Obj *	_make_obj(QSP_ARG_DECL  const char *name,dimension_t frames, dimension_t rows,dimension_t cols,dimension_t type_dim,Precision * prec_p);
extern Data_Obj *	_comp_replicate(QSP_ARG_DECL  Data_Obj *dp,int n,int allocate_data);
extern Data_Obj *	_make_obj_list(QSP_ARG_DECL  const char *name,List *lp);
#define mk_scalar(name,prec_p) _mk_scalar(QSP_ARG  name,prec_p)
#define mk_img(s,h,w,d,p)	_mk_img(QSP_ARG  s,h,w,d,p)
#define set_obj_dimensions(dp,dimensions,prec_p)	_set_obj_dimensions(QSP_ARG  dp,dimensions,prec_p)
#define make_obj(name,frames,rows,cols,type_dim,prec_p)	_make_obj(QSP_ARG  name,frames,rows,cols,type_dim,prec_p)
#define comp_replicate(dp,n,f)	_comp_replicate(QSP_ARG  dp,n,f)
#define make_obj_list(name,lp) _make_obj_list(QSP_ARG  name,lp)

/* makedobj.c */
extern void	  set_dp_alignment(int);
extern void set_dimension(Dimension_Set *dsp, int idx, dimension_t value);

extern int _cpu_obj_alloc(QSP_ARG_DECL  Data_Obj *dp, dimension_t size, int align );
extern void _cpu_obj_free(QSP_ARG_DECL  Data_Obj *dp );
extern void * _cpu_mem_alloc(QSP_ARG_DECL  Platform_Device *pdp, dimension_t size, int align );
extern void _cpu_mem_free(QSP_ARG_DECL  void *ptr );
extern Data_Obj * _make_dobj_with_shape(QSP_ARG_DECL  const char *name,Dimension_Set *dsp,Precision *prec_p,uint32_t typ_flg);

// what is the difference between make_dobj and _make_dp???
extern Data_Obj * _make_dobj(QSP_ARG_DECL  const char *name,Dimension_Set *,Precision *);
extern Data_Obj * _setup_dp(QSP_ARG_DECL  Data_Obj *dp,Precision *);
extern Data_Obj * _make_dp(QSP_ARG_DECL  const char *name,Dimension_Set *,Precision * );
extern Data_Obj * _init_dp(QSP_ARG_DECL  Data_Obj *dp,Dimension_Set *,Precision *);

#define make_dobj(name,dsp,prec_p)	_make_dobj(QSP_ARG  name,dsp,prec_p)
#define setup_dp(dp,prec_p)		_setup_dp(QSP_ARG  dp,prec_p)
#define make_dp(name,dsp,prec_p)	_make_dp(QSP_ARG  name,dsp,prec_p)
#define init_dp(dp,dsp,prec_p)		_init_dp(QSP_ARG  dp,dsp,prec_p)
#define cpu_obj_alloc(dp,size,align)	_cpu_obj_alloc(QSP_ARG  dp,size,align)
#define cpu_obj_free(dp)		_cpu_obj_free(QSP_ARG  dp)
#define cpu_mem_alloc(pdp,size,align)	_cpu_mem_alloc(QSP_ARG  pdp,size,align)
#define cpu_mem_free(ptr)		_cpu_mem_free(QSP_ARG  ptr)
#define make_dobj_with_shape(name,dsp,prec_p,n)	_make_dobj_with_shape(QSP_ARG  name,dsp,prec_p,n)

// These can probably be local to module...

#define ALLOC_SHAPE		alloc_shape()
extern void alloc_shape_elts(Shape_Info *shpp);
#define INIT_SHAPE_PTR(shpp)		shpp=alloc_shape();

extern void rls_shape_elts(Shape_Info *shpp);

extern Shape_Info *alloc_shape(void);
extern void copy_shape(Shape_Info *dst, Shape_Info *src);
extern void rls_shape(Shape_Info *shpp);
#define RELEASE_SHAPE_PTR(shpp)		rls_shape(shpp);
/*#define RELEASE_SHAPE_PTR(shpp)		givbuf(shpp); */


/* formerly in areas.h */
/* areas.c */

//ITEM_INTERFACE_PROTOTYPES(Data_Area,data_area)

void			push_data_area(Data_Area *);
void			pop_data_area(void);
extern int		dp_addr_cmp(const void *dpp1,const void *dpp2);
extern void		a_init(void);
extern void		set_data_area(Data_Area *);
extern Data_Obj *	search_areas(const char *name);

extern List *		_da_list(SINGLE_QSP_ARG_DECL);
extern Data_Area *	_default_data_area(SINGLE_QSP_ARG_DECL);
extern Data_Area *	_pf_area_init(QSP_ARG_DECL  const char *name,u_char *buffer,uint32_t siz,int nobjs,uint32_t flags, struct platform_device *pdp);
extern Data_Area *	_new_area(QSP_ARG_DECL  const char *s,uint32_t siz,uint32_t n);
extern void		_list_area(QSP_ARG_DECL  Data_Area *ap);
extern void		_data_area_info(QSP_ARG_DECL  Data_Area *ap);
extern void		_show_area_space(QSP_ARG_DECL  Data_Area *ap);
extern Data_Obj *	_area_scalar(QSP_ARG_DECL  Data_Area *ap);

#define da_list()					_da_list(SINGLE_QSP_ARG)
#define default_data_area()				_default_data_area(SINGLE_QSP_ARG)
#define pf_area_init(name,buffer,siz,nobjs,flags,pdp)	_pf_area_init(QSP_ARG  name,buffer,siz,nobjs,flags,pdp)
#define new_area(s,siz,n)				_new_area(QSP_ARG  s,siz,n)
#define list_area(ap)					_list_area(QSP_ARG  ap)
#define data_area_info(ap)				_data_area_info(QSP_ARG  ap)
#define show_area_space(ap)				_show_area_space(QSP_ARG  ap)
#define area_scalar(ap)					_area_scalar(QSP_ARG  ap)

/* formerly in index.h */
extern Data_Obj * _index_data( QSP_ARG_DECL  Data_Obj *dp, const char *index_str );
#define index_data(dp,index_str) _index_data( QSP_ARG  dp,index_str )

/* memops.c */
extern int dp_same_size_query(Data_Obj *dp1,Data_Obj *dp2);
extern int dp_same_size_query_rc(Data_Obj *real_dp,Data_Obj *cpx_dp);
extern int dp_equal_dim(Data_Obj *dp1,Data_Obj *dp2,int index);
extern void mxpose(Data_Obj *dp_to, Data_Obj *dp_fr);

extern int _not_prec(QSP_ARG_DECL  Data_Obj *,prec_t);
extern void _check_vectorization(QSP_ARG_DECL  Data_Obj *dp);
extern void _dp1_vectorize(QSP_ARG_DECL  int,Data_Obj *,void (*func)(QSP_ARG_DECL  Data_Obj *) );
extern void _dp2_vectorize(QSP_ARG_DECL  int,Data_Obj *,Data_Obj *, void (*func)(QSP_ARG_DECL  Data_Obj *,Data_Obj *) );
extern void _getmean(QSP_ARG_DECL  Data_Obj *dp);
extern void _dp_equate(QSP_ARG_DECL  Data_Obj *dp, double v);
extern void _dp_copy(QSP_ARG_DECL  Data_Obj *dp_to, Data_Obj *dp_fr);
extern void _i_rnd(QSP_ARG_DECL  Data_Obj *dp, int imin, int imax);
extern void _dp_uni(QSP_ARG_DECL  Data_Obj *dp);
extern int _dp_same_prec(QSP_ARG_DECL  Data_Obj *dp1,Data_Obj *dp2,const char *whence);
extern int _dp_same_mach_prec(QSP_ARG_DECL  Data_Obj *dp1,Data_Obj *dp2,const char *whence);
extern int _dp_same_pixel_type(QSP_ARG_DECL  Data_Obj *dp1,Data_Obj *dp2,const char *whence);
extern int _dp_same_size(QSP_ARG_DECL  Data_Obj *dp1,Data_Obj *dp2,const char *whence);
extern int _dp_same_dim(QSP_ARG_DECL  Data_Obj *dp1,Data_Obj *dp2,int index,const char *whence);
extern int _dp_same_dims(QSP_ARG_DECL  Data_Obj *dp1,Data_Obj *dp2,int index1,int index2,const char *whence);
extern int _dp_same(QSP_ARG_DECL  Data_Obj *dp1,Data_Obj *dp2,const char *whence);
extern int _dp_equal_dims(QSP_ARG_DECL  Data_Obj *dp1,Data_Obj *dp2,int index1,int index2);

#define not_prec(dp,p)				_not_prec(QSP_ARG  dp,p)
#define check_vectorization(dp)			_check_vectorization(QSP_ARG  dp)
#define dp1_vectorize(i,dp,func)		_dp1_vectorize(QSP_ARG  i,dp,func)
#define dp2_vectorize(i,dp1,dp2,func)		_dp2_vectorize(QSP_ARG  i,dp1,dp2,func)
#define getmean(dp)				_getmean(QSP_ARG  dp)
#define dp_equate(dp,v)				_dp_equate(QSP_ARG  dp,v)
#define dp_copy(dp_to,dp_fr)			_dp_copy(QSP_ARG  dp_to,dp_fr)
#define i_rnd(dp,imin,imax)			_i_rnd(QSP_ARG  dp,imin,imax)
#define dp_uni(dp)				_dp_uni(QSP_ARG  dp)
#define dp_same_prec(dp1,dp2,whence)		_dp_same_prec(QSP_ARG  dp1,dp2,whence)
#define dp_same_mach_prec(dp1,dp2,whence)	_dp_same_mach_prec(QSP_ARG  dp1,dp2,whence)
#define dp_same_pixel_type(dp1,dp2,whence)	_dp_same_pixel_type(QSP_ARG  dp1,dp2,whence)
#define dp_same_size(dp1,dp2,whence)		_dp_same_size(QSP_ARG  dp1,dp2,whence)
#define dp_same_dim(dp1,dp2,index,whence)	_dp_same_dim(QSP_ARG  dp1,dp2,index,whence)
#define dp_same_dims(dp1,dp2,index1,index2,whence)	_dp_same_dims(QSP_ARG  dp1,dp2,index1,index2,whence)
#define dp_same(dp1,dp2,whence)			_dp_same(QSP_ARG  dp1,dp2,whence)
#define dp_equal_dims(dp1,dp2,index1,index2)	_dp_equal_dims(QSP_ARG  dp1,dp2,index1,index2)


#define CHECK_SAME_PREC( dp1, dp2, whence )				\
									\
	if( !_dp_same_prec(DEFAULT_QSP_ARG  dp1,dp2,whence) )		\
		return;
	
#define CHECK_SAME_SIZE( dp1, dp2, whence )				\
									\
	if( !_dp_same_size(DEFAULT_QSP_ARG  dp1,dp2,whence) )		\
		return;

// doesn't seem to be here any more
//extern int dp_same_len(QSP_ARG_DECL  Data_Obj *dp1,Data_Obj *dp2);

/* dfuncs.c  support for pexpr */
extern double obj_exists(QSP_ARG_DECL  const char *);
extern double val_func(QSP_ARG_DECL  Data_Obj *);

extern double row_func(Data_Obj *);
extern double depth_func(Data_Obj *);
extern double col_func(Data_Obj *);
extern double frm_func(Data_Obj *);
extern double seq_func(Data_Obj *);

/* sub_obj.c */

extern void point_obj_to_ext_data(Data_Obj *dp,void *p);
extern void parent_relationship(Data_Obj *parent,Data_Obj *child);
extern void propagate_flag_to_children(Data_Obj *dp, uint32_t flags );

extern Data_Obj *_mk_subseq(QSP_ARG_DECL  const char *name, Data_Obj *parent, index_t *offsets, Dimension_Set *sizes);
extern Data_Obj *_mk_ilace(QSP_ARG_DECL  Data_Obj *parent, const char *name, int parity);
extern int _relocate_with_offsets(QSP_ARG_DECL  Data_Obj *dp,index_t *offsets);
extern int _relocate(QSP_ARG_DECL  Data_Obj *dp,index_t xos,index_t yos,index_t tos);
extern Data_Obj *_mk_subimg(QSP_ARG_DECL  Data_Obj *parent, index_t xos, index_t yos, const char *name, dimension_t rows, dimension_t cols);
extern Data_Obj *_mk_substring(QSP_ARG_DECL  Data_Obj *parent, index_t sos, const char *name, dimension_t len );
extern Data_Obj *_nmk_subimg(QSP_ARG_DECL  Data_Obj *parent, index_t xos, index_t yos, const char *name, dimension_t rows, dimension_t cols, dimension_t tdim);
extern Data_Obj *_make_equivalence(QSP_ARG_DECL  const char *name, Data_Obj *dp, Dimension_Set *dsp, Precision * prec_p);
extern Data_Obj *_make_subsamp(QSP_ARG_DECL  const char *name, Data_Obj *dp, Dimension_Set *sizes, index_t *offsets, incr_t *incrs );

#define mk_subseq(name,parent,offsets,sizes)		_mk_subseq(QSP_ARG  name,parent,offsets,sizes)
#define mk_ilace(parent,name,parity)			_mk_ilace(QSP_ARG  parent,name,parity)
#define relocate_with_offsets(dp,offsets)		_relocate_with_offsets(QSP_ARG  dp,offsets)
#define relocate(dp,xos,yos,tos)			_relocate(QSP_ARG  dp,xos,yos,tos)
#define mk_substring(parent,sos,name,len)		_mk_substring(QSP_ARG  parent,sos,name,len)
#define mk_subimg(parent,xos,yos,name,rows,cols)	_mk_subimg(QSP_ARG  parent,xos,yos,name,rows,cols)
#define nmk_subimg(parent,xos,yos,name,rows,cols,tdim)	_nmk_subimg(QSP_ARG  parent,xos,yos,name,rows,cols,tdim)
#define make_equivalence(name,dp,dsp,prec_p)		_make_equivalence(QSP_ARG  name,dp,dsp,prec_p)
#define make_subsamp(name,dp,sizes,offsets,incrs)	_make_subsamp(QSP_ARG  name,dp,sizes,offsets,incrs)


/* verdata.c */
extern void verdata(SINGLE_QSP_ARG_DECL);


/* ascmenu.c */
#ifdef HAVE_ANY_GPU
extern Data_Obj *_insure_ram_obj_for_reading(QSP_ARG_DECL  Data_Obj *dp);
extern void _release_ram_obj_for_reading(QSP_ARG_DECL  Data_Obj *ram_dp, Data_Obj *dp);
extern Data_Obj *_insure_ram_obj_for_writing(QSP_ARG_DECL  Data_Obj *dp);

#define insure_ram_obj_for_reading(dp) _insure_ram_obj_for_reading(QSP_ARG  dp)
#define release_ram_obj_for_reading(ram_dp, dp) _release_ram_obj_for_reading(QSP_ARG  ram_dp, dp)
#define insure_ram_obj_for_writing(dp) _insure_ram_obj_for_writing(QSP_ARG  dp)
#endif // HAVE_ANY_GPU

/* ascii.c */
extern void _init_dobj_ascii_info(QSP_ARG_DECL  Dobj_Ascii_Info *dai_p);
extern void _format_scalar_obj(QSP_ARG_DECL  char *buf,int buflen,Data_Obj *dp,void *data);
#define format_scalar_obj(buf,buflen,dp,data) _format_scalar_obj(QSP_ARG  buf,buflen,dp,data)

#define init_dobj_ascii_info(dai_p) _init_dobj_ascii_info(QSP_ARG  dai_p)

#define PAD_FOR_EVEN_COLUMNS	1
#define NO_PADDING 		0
extern void _format_scalar_value(QSP_ARG_DECL  char *buf,int buflen,void *data,Precision *prec_p,int pad_flag);
#define format_scalar_value(buf,buflen,data,prec_p,pad_flag) _format_scalar_value(QSP_ARG  buf,buflen,data,prec_p,pad_flag)
extern char * string_for_scalar(QSP_ARG_DECL  void *data,Precision *prec_p);
extern void _pntvec(QSP_ARG_DECL  Data_Obj *dp, FILE *fp);
#define pntvec(dp, fp) _pntvec(QSP_ARG  dp, fp)

extern void _dptrace(QSP_ARG_DECL  Data_Obj *);
extern void _set_max_per_line(QSP_ARG_DECL  int n);
extern void _set_input_format_string(QSP_ARG_DECL  const char *s);
extern void _set_display_precision(QSP_ARG_DECL  int);
extern int _object_is_in_ram(QSP_ARG_DECL  Data_Obj *dp, const char *op_str);

#define dptrace(dp) _dptrace(QSP_ARG  dp)
#define set_max_per_line(n) _set_max_per_line(QSP_ARG  n)
#define set_input_format_string(s) _set_input_format_string(QSP_ARG  s)
#define set_display_precision(n) _set_display_precision(QSP_ARG  n)
#define object_is_in_ram(dp, op_str) _object_is_in_ram(QSP_ARG  dp, op_str)

extern void _read_obj(QSP_ARG_DECL Data_Obj *dp);
extern void _read_ascii_data_from_pipe(QSP_ARG_DECL Data_Obj *dp, Pipe *pp, const char *s, int expect_exact_count);
extern void _read_ascii_data_from_file(QSP_ARG_DECL Data_Obj *dp, FILE *fp, const char *s, int expect_exact_count);

#define read_obj(dp) _read_obj(QSP_ARG dp)
#define read_ascii_data_from_pipe(dp,pp,s,c) _read_ascii_data_from_pipe(QSP_ARG dp,pp,s,c)
#define read_ascii_data_from_file(dp,fp,s,c) _read_ascii_data_from_file(QSP_ARG dp,fp,s,c)

/* datamenu.c */
extern Precision * get_precision(SINGLE_QSP_ARG_DECL);


/* formerly in datamenu/dataprot.h */


/* prototypes for datamenu high level funcs */


/* ascmenu.c */
extern COMMAND_FUNC( asciimenu );


/* datamenu.c */
//extern COMMAND_FUNC( do_area );

extern Data_Obj *req_obj(char *s);
extern void dataport_init(void);
extern void dm_init(SINGLE_QSP_ARG_DECL);


/* ops_menu.c */
extern COMMAND_FUNC( buf_ops );

/* verdatam.c */
extern void verdatam(SINGLE_QSP_ARG_DECL);


// shape_info.c
extern List *		prec_list(SINGLE_QSP_ARG_DECL);
extern Precision *	const_precision(Precision *);
extern Precision *	complex_precision(Precision *);

// fio_menu.c
extern COMMAND_FUNC( do_fio_menu );


// from evaltree.c, but should be moved!?!?
extern void easy_ramp2d(QSP_ARG_DECL  Data_Obj *dst_dp,double start,double dx,double dy);

#endif // ! _DOBJ_PROT_H_
