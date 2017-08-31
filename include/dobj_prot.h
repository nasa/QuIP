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
extern int is_contiguous(QSP_ARG_DECL  Data_Obj *);
extern void check_contiguity(Data_Obj *);

/* data_obj.c */

extern void		init_precisions(SINGLE_QSP_ARG_DECL);
extern Precision * 	get_prec(QSP_ARG_DECL  const char *);

extern Data_Obj *	dobj_of(QSP_ARG_DECL  const char *);
#define DOBJ_OF(s)	dobj_of(QSP_ARG s)
extern void		list_dobjs(QSP_ARG_DECL  FILE *fp);
extern Data_Obj *	new_dobj(QSP_ARG_DECL  const char *);
extern List *		dobj_list(SINGLE_QSP_ARG_DECL);

extern void		disown_child(QSP_ARG_DECL  Data_Obj * dp);
extern void		delvec(QSP_ARG_DECL  Data_Obj * dp);
extern void		info_area(QSP_ARG_DECL  Data_Area *ap);
extern void		info_all_dps(SINGLE_QSP_ARG_DECL);
extern void		sizinit(void);
extern void		make_complex(Shape_Info *shpp);
extern int		set_shape_flags(Shape_Info *shpp,uint32_t type_flags);
extern int		auto_shape_flags(Shape_Info *shpp);
extern void		show_space_used(QSP_ARG_DECL  Data_Obj * dp);
extern void		dobj_iterate(Data_Obj * dp,void (*func)(Data_Obj * ,uint32_t));
extern void		dpair_iterate(QSP_ARG_DECL  Data_Obj * dp,Data_Obj * dp2,
				void (*func)(QSP_ARG_DECL  Data_Obj * ,uint32_t,Data_Obj * ,uint32_t));
extern void		gen_xpose(Data_Obj * dp,int dim1,int dim2);
extern double		get_dobj_size(QSP_ARG_DECL  Data_Obj * dp,int index);
extern const char *	get_dobj_prec_name(QSP_ARG_DECL  Data_Obj * dp);
extern double		get_dobj_il_flg(QSP_ARG_DECL  Data_Obj * dp);
extern void		dataobj_init(SINGLE_QSP_ARG_DECL);
extern void		init_dfuncs(SINGLE_QSP_ARG_DECL);
extern int		same_shape(Shape_Info *,Shape_Info *);

/* dplist.c */

/* apparently , this file was split off from data_obj.c, bacause
 * some of the prototypes are still listed above!?
 */

//extern const char *	name_for_prec(int prec);
extern Precision *	prec_for_code(prec_t prec);
extern void		dump_shape(QSP_ARG_DECL  Shape_Info *shpp);
extern void		list_dobj(QSP_ARG_DECL  Data_Obj * dp);
extern void		longlist(QSP_ARG_DECL  Data_Obj * dp);

//extern void		describe_shape(Shape_Info *shpp);
extern void show_obj_dimensions(QSP_ARG_DECL  Data_Obj *dp, Dimension_Set *dsp, Increment_Set *isp);

#define LONGLIST(dp)	longlist(QSP_ARG  dp)

/* pars_obj.c */
extern Data_Obj * pars_obj(const char *);

/* arrays.c */
/* formerly in arrays.h */

extern void release_tmp_obj(Data_Obj *);
extern void unlock_all_tmp_objs(SINGLE_QSP_ARG_DECL);
extern void set_array_base_index(QSP_ARG_DECL  int);
extern void init_tmp_dps(SINGLE_QSP_ARG_DECL);
extern Data_Obj * find_free_temp_dp(QSP_ARG_DECL  Data_Obj *dp);
extern Data_Obj * temp_child(QSP_ARG_DECL  const char *name,Data_Obj * dp);
extern void make_array_name(QSP_ARG_DECL  char *target_str,int buflen,Data_Obj * dp,
   index_t index,int which_dim,int subscr_type);
extern Data_Obj * gen_subscript(QSP_ARG_DECL  Data_Obj * dp,
   int which_dim,index_t index,int subscr_type);
extern Data_Obj * reduce_from_end(QSP_ARG_DECL  Data_Obj * dp,
   index_t index,int subscr_type);
extern Data_Obj * d_subscript(QSP_ARG_DECL  Data_Obj * dp,index_t index);
extern Data_Obj * c_subscript(QSP_ARG_DECL  Data_Obj * dp,index_t index);
extern int is_in_string(int c,const char *s);
extern void reindex(QSP_ARG_DECL  Data_Obj * ,int,index_t);
extern void list_temp_dps(QSP_ARG_DECL  FILE *fp);
extern void unlock_children(Data_Obj *dp);

/* get_obj.c */
/* formerly in get_obj.h */

extern Data_Obj * hunt_obj(QSP_ARG_DECL  const char *s);
extern Data_Obj * get_obj(QSP_ARG_DECL  const char *s);
#define GET_OBJ(s)	get_obj(QSP_ARG  s)
extern Data_Obj * get_vec(QSP_ARG_DECL  const char *s);
extern Data_Obj * img_of(QSP_ARG_DECL  const char *s);
extern Data_Obj * get_seq(QSP_ARG_DECL  const char *s);
extern Data_Obj * get_img(QSP_ARG_DECL  const char *s );


/* data_fns.c */

extern void *		multiply_indexed_data(Data_Obj *dp, dimension_t *offset );
extern void *		indexed_data(Data_Obj *dp, dimension_t offset );
extern void		make_contiguous(Data_Obj *);
extern int		set_shape_dimensions(QSP_ARG_DECL  Shape_Info *shpp,Dimension_Set *dimensions,Precision *);
extern int		set_obj_dimensions(QSP_ARG_DECL  Data_Obj *dp,Dimension_Set *dimensions,Precision *);
extern int		obj_rename(QSP_ARG_DECL  Data_Obj *dp,const char *newname);
extern Data_Obj *	make_obj_list(QSP_ARG_DECL  const char *name,List *lp);
extern Data_Obj *	make_obj(QSP_ARG_DECL  const char *name,dimension_t frames,
	dimension_t rows,dimension_t cols,dimension_t type_dim,Precision * prec_p);
extern Data_Obj *	mk_scalar(QSP_ARG_DECL  const char *name,Precision *prec_p);
extern void		assign_scalar(QSP_ARG_DECL  Data_Obj *,Scalar_Value *);
extern void		extract_scalar_value(QSP_ARG_DECL  Scalar_Value *, Data_Obj *);
extern double		cast_from_scalar_value(QSP_ARG_DECL  Scalar_Value *, Precision *prec_p);
extern void		cast_to_scalar_value(QSP_ARG_DECL  Scalar_Value *, Precision *prec_p, double val);
/*extern const char *	string_for_scalar_value(QSP_ARG_DECL  Scalar_Value *, Precision *prec_p); */
extern void		cast_to_cpx_scalar(QSP_ARG_DECL  int index, Scalar_Value *, Precision *prec_p, double val);
extern void		cast_to_quat_scalar(QSP_ARG_DECL  int index, Scalar_Value *, Precision *prec_p, double val);
extern void		cast_to_color_scalar(QSP_ARG_DECL  int index, Scalar_Value *, Precision *prec_p, double val);
extern Data_Obj *	mk_cscalar(QSP_ARG_DECL  const char *name,double rval, double ival);
extern Data_Obj *	mk_img(QSP_ARG_DECL  const char *,dimension_t,dimension_t,dimension_t ,Precision *prec_p);
extern Data_Obj *	mk_vec(QSP_ARG_DECL  const char *,dimension_t, dimension_t,Precision *prec_p);
extern Data_Obj *	comp_replicate(QSP_ARG_DECL  Data_Obj *dp,int n,int allocate_data);
extern Data_Obj *	dup_half(QSP_ARG_DECL  Data_Obj *dp,const char *name);
extern Data_Obj *	dup_dbl(QSP_ARG_DECL  Data_Obj *dp,const char *name);
extern Data_Obj *	dup_obj(QSP_ARG_DECL  Data_Obj *dp,const char *name);
extern const char *	localname(void);
extern Data_Obj *	dupdp(QSP_ARG_DECL  Data_Obj *dp);
extern int		is_valid_dname(QSP_ARG_DECL  const char *name);


/* makedobj.c */
extern int cpu_obj_alloc(QSP_ARG_DECL  Data_Obj *dp, dimension_t size, int align );
extern void cpu_obj_free(QSP_ARG_DECL  Data_Obj *dp );
extern void * cpu_mem_alloc(QSP_ARG_DECL  Platform_Device *pdp, dimension_t size, int align );
extern void cpu_mem_free(QSP_ARG_DECL  void *ptr );

extern Data_Obj * make_dobj_with_shape(QSP_ARG_DECL  const char *name,Dimension_Set *,Precision *,uint32_t);
extern void	  set_dp_alignment(int);
// what is the difference between make_dobj and _make_dp???
extern Data_Obj * make_dobj(QSP_ARG_DECL  const char *name,Dimension_Set *,Precision *);
extern Data_Obj * setup_dp(QSP_ARG_DECL  Data_Obj *dp,Precision *);
extern Data_Obj * _make_dp(QSP_ARG_DECL  const char *name,Dimension_Set *,Precision * );
extern Data_Obj * init_dp(QSP_ARG_DECL  Data_Obj *dp,Dimension_Set *,Precision *);
extern void set_dimension(Dimension_Set *dsp, int idx, dimension_t value);

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
extern List *		da_list(SINGLE_QSP_ARG_DECL);
extern void		a_init(void);
extern Data_Area *	default_data_area(SINGLE_QSP_ARG_DECL);
extern void		set_data_area(Data_Area *);
extern Data_Obj *	search_areas(const char *name);
extern Data_Area *	pf_area_init(QSP_ARG_DECL  const char *name,u_char *buffer,uint32_t siz,int nobjs,uint32_t flags, struct platform_device *pdp);
// now only use pf_area_init...
//extern Data_Area *	area_init(QSP_ARG_DECL  const char *name,u_char *buffer,uint32_t siz,int nobjs,uint32_t flags);
extern Data_Area *	new_area(QSP_ARG_DECL  const char *s,uint32_t siz,uint32_t n);
extern void		list_area(QSP_ARG_DECL  Data_Area *ap);
extern void		data_area_info(QSP_ARG_DECL  Data_Area *ap);
extern int		dp_addr_cmp(const void *dpp1,const void *dpp2);
extern void		show_area_space(QSP_ARG_DECL  Data_Area *ap);
extern Data_Obj *	area_scalar(QSP_ARG_DECL  Data_Area *ap);

/* formerly in index.h */
extern Data_Obj * index_data( QSP_ARG_DECL  Data_Obj *dp, const char *index_str );

/* memops.c */
extern int not_prec(QSP_ARG_DECL  Data_Obj *,prec_t);
extern void check_vectorization(QSP_ARG_DECL  Data_Obj *dp);
extern void dp1_vectorize(QSP_ARG_DECL  int,Data_Obj *,void (*func)(Data_Obj *) );
extern void dp2_vectorize(QSP_ARG_DECL  int,Data_Obj *,Data_Obj *,
				void (*func)(Data_Obj *,Data_Obj *) );
extern void getmean(QSP_ARG_DECL  Data_Obj *dp);
extern void dp_equate(QSP_ARG_DECL  Data_Obj *dp, double v);
extern void dp_copy(QSP_ARG_DECL  Data_Obj *dp_to, Data_Obj *dp_fr);
extern void i_rnd(QSP_ARG_DECL  Data_Obj *dp, int imin, int imax);
extern void dp_uni(QSP_ARG_DECL  Data_Obj *dp);
extern void mxpose(Data_Obj *dp_to, Data_Obj *dp_fr);

extern int dp_same_prec(QSP_ARG_DECL  Data_Obj *dp1,Data_Obj *dp2,const char *whence);

#define CHECK_SAME_PREC( dp1, dp2, whence )				\
									\
	if( !dp_same_prec(DEFAULT_QSP_ARG  dp1,dp2,whence) )			\
		return;
	
extern int dp_same_mach_prec(QSP_ARG_DECL  Data_Obj *dp1,Data_Obj *dp2,const char *whence);
extern int dp_same_pixel_type(QSP_ARG_DECL  Data_Obj *dp1,Data_Obj *dp2,const char *whence);
extern int dp_same_size(QSP_ARG_DECL  Data_Obj *dp1,Data_Obj *dp2,const char *whence);

#define CHECK_SAME_SIZE( dp1, dp2, whence )				\
									\
	if( !dp_same_size(DEFAULT_QSP_ARG  dp1,dp2,whence) )			\
		return;
	
extern int dp_same_dim(QSP_ARG_DECL  Data_Obj *dp1,Data_Obj *dp2,int index,const char *whence);
extern int dp_same_dims(QSP_ARG_DECL  Data_Obj *dp1,Data_Obj *dp2,int index1,int index2,const char *whence);
extern int dp_same(QSP_ARG_DECL  Data_Obj *dp1,Data_Obj *dp2,const char *whence);

extern int dp_same_size_query(Data_Obj *dp1,Data_Obj *dp2);
extern int dp_same_size_query_rc(Data_Obj *real_dp,Data_Obj *cpx_dp);
extern int dp_equal_dim(Data_Obj *dp1,Data_Obj *dp2,int index);
extern int dp_equal_dims(QSP_ARG_DECL  Data_Obj *dp1,Data_Obj *dp2,int index1,int index2);
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

extern void parent_relationship(Data_Obj *parent,Data_Obj *child);
extern Data_Obj *mk_subseq(QSP_ARG_DECL  const char *name, Data_Obj *parent,
 index_t *offsets, Dimension_Set *sizes);
extern Data_Obj *mk_ilace(QSP_ARG_DECL  Data_Obj *parent, const char *name, int parity);
extern int __relocate(QSP_ARG_DECL  Data_Obj *dp,index_t *offsets);
extern int _relocate(QSP_ARG_DECL  Data_Obj *dp,index_t xos,index_t yos,index_t tos);
extern Data_Obj *mk_subimg(QSP_ARG_DECL  Data_Obj *parent, index_t xos, index_t yos,
 const char *name, dimension_t rows, dimension_t cols);
extern Data_Obj *nmk_subimg(QSP_ARG_DECL  Data_Obj *parent, index_t xos, index_t yos, 
 const char *name, dimension_t rows, dimension_t cols, dimension_t tdim);
extern Data_Obj *make_equivalence(QSP_ARG_DECL  const char *name, Data_Obj *dp,
 Dimension_Set *dsp, Precision * prec_p);
extern Data_Obj *make_subsamp(QSP_ARG_DECL  const char *name, Data_Obj *dp,
 Dimension_Set *sizes, index_t *offsets, incr_t *incrs );
extern void propagate_flag_to_children(Data_Obj *dp, uint32_t flags );


/* verdata.c */
extern void verdata(SINGLE_QSP_ARG_DECL);


/* ascmenu.c */
#ifdef HAVE_ANY_GPU
extern Data_Obj *insure_ram_obj_for_reading(QSP_ARG_DECL  Data_Obj *dp);
extern void release_ram_obj_for_reading(QSP_ARG_DECL  Data_Obj *ram_dp, Data_Obj *dp);
extern Data_Obj *insure_ram_obj_for_writing(QSP_ARG_DECL  Data_Obj *dp);
#endif // HAVE_ANY_GPU

/* ascii.c */
extern void init_dobj_ascii_info(QSP_ARG_DECL  Dobj_Ascii_Info *dai_p);
extern void format_scalar_obj(QSP_ARG_DECL  char *buf,int buflen,Data_Obj *dp,void *data);
extern void format_scalar_value(QSP_ARG_DECL  char *buf,int buflen,void *data,Precision *prec_p);
extern char * string_for_scalar(QSP_ARG_DECL  void *data,Precision *prec_p);
extern void pntvec(QSP_ARG_DECL  Data_Obj *dp, FILE *fp);
extern void read_ascii_data(QSP_ARG_DECL Data_Obj *dp, FILE *fp, const char *s, int expect_exact_count);
extern void read_obj(QSP_ARG_DECL Data_Obj *dp);
extern void dptrace(QSP_ARG_DECL  Data_Obj *);
extern void set_integer_print_fmt(QSP_ARG_DECL  Number_Fmt fmt_code);
extern void set_max_per_line(QSP_ARG_DECL  int n);
extern void set_input_format_string(QSP_ARG_DECL  const char *s);
extern void set_display_precision(QSP_ARG_DECL  int);
extern int object_is_in_ram(QSP_ARG_DECL  Data_Obj *dp, const char *op_str);

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

