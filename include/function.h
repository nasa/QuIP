#ifndef _FUNCTION_H_
#define _FUNCTION_H_

#ifdef __cplusplus
extern "C" {
#endif

#include "item_type.h"
#include "shape_bits.h"
#include "data_obj.h"
#include "veclib/vecgen.h"

// These function types have to do with the number and type of args
// This is somewhat confusing in that we have also introduced our
// own C-language version of classes, to accommodate sizable objects, etc.
// The precision function string-valued with sizable argument...

typedef enum {
	D0_FUNCTYP,	// 0	no args
	D1_FUNCTYP,	// 1	1 arg, e.g. cos(theta)
	D2_FUNCTYP,	// 2	2 args, e.g. atan2(x,y)
	I1_FUNCTYP,	// 3	1 int arg, e.g.
	STR1_FUNCTYP,	// 4	1 string arg, e.g. strlen(s)
	STR2_FUNCTYP,	// 5	2 string args, e.g. strcmp(s1,s2)
	STR3_FUNCTYP,	// 6	3 string args
	STRV_FUNCTYP,	// 7	string-valued function, one string arg
	SIZE_FUNCTYP,	// 8	sizable object arg
	DOBJ_FUNCTYP,	// 9	double result, 1 scalar data_obj arg
	TS_FUNCTYP,	// 10	time-stamp functions?
	CHAR_FUNCTYP,	// 11	1 char arg
	POSN_FUNCTYP,	// 12	1 positionable arg
	ILACE_FUNCTYP,	// 13	1 interlaceable arg
	STRV2_FUNCTYP,	// 14	string-valued function, two string args
	DOBJV_STR_ARG_FUNCTYP,	// 15	data_obj-valued function, one string arg
	N_FUNC_TYPES	// must be last
} Function_Type;


typedef union {
 	double	      (*d0_func)(void);
 	double	      (*d1_func)(double);
 	double	      (*d2_func)(double, double);
 	int	      (*i1_func)(double);
 	double	      (*sz_func)(QSP_ARG_DECL  Item *);
	const char *  (*strv_func)(QSP_ARG_DECL  const char *);
	const char *  (*strv2_func)(QSP_ARG_DECL  const char *, const char *);
	//void          (*strv_func)(char *, const char *);
	double	      (*ts_func)(QSP_ARG_DECL  Item *,dimension_t frm);
 	double	      (*str1_func)(QSP_ARG_DECL  const char *);
 	double	      (*str2_func)(const char *,const char *);
 	double	      (*str3_func)(const char *,const char *,int);
 	double	      (*dobj_func)(QSP_ARG_DECL  Data_Obj *);
 	Data_Obj *    (*dobjv_str_arg_func)(QSP_ARG_DECL  const char *);
	int           (*char_func)(char);
	double        (*il_func)(QSP_ARG_DECL  Item *);
	double        (*posn_func)(QSP_ARG_DECL  Item *);
} Fn_Func;

struct quip_function {
	const char *	fn_name;
	int		fn_type;
	Fn_Func		fn_u;
	int		fn_vv_code;
	int		fn_vs_code;
	int		fn_vs_code2;
	int		fn_dim_index;	// for size functions
	int		fn_serial;
};

#define FUNC_NAME(funcp)		((funcp)->fn_name)
#define FUNC_DIM_INDEX(funcp)		((funcp)->fn_dim_index)
#define FUNC_SZ_FUNC(funcp)		((funcp)->fn_u.sz_func)
#define FUNC_SERIAL(f)			((f)->fn_serial)
#define FUNC_TYPE(f)			((f)->fn_type)
//#define FUNC_FUNCTION(f)		((f)->fn_name)
#define FUNC_D0_FUNC(f)			((f)->fn_u.d0_func)
#define FUNC_D1_FUNC(f)			((f)->fn_u.d1_func)
#define FUNC_D2_FUNC(f)			((f)->fn_u.d2_func)
#define FUNC_STR1_FUNC(f)		((f)->fn_u.str1_func)
#define FUNC_STR2_FUNC(f)		((f)->fn_u.str2_func)
#define FUNC_STR3_FUNC(f)		((f)->fn_u.str3_func)
#define FUNC_STRV_FUNC(f)		((f)->fn_u.strv_func)
#define FUNC_CHAR_FUNC(f)		((f)->fn_u.char_func)
#define FUNC_VV_CODE(f)			((f)->fn_vv_code)
#define SET_FUNC_VV_CODE(f,c)		(f)->fn_vv_code = c
#define FUNC_VS_CODE(f)			((f)->fn_vs_code)
#define SET_FUNC_VS_CODE(f,c)		(f)->fn_vs_code = c
#define FUNC_VS_CODE2(f)		((f)->fn_vs_code2)
#define SET_FUNC_VS_CODE2(f,c)		(f)->fn_vs_code2 = c

#define FN_NAME(funcp)		(funcp)->fn_name

/* support for size functions */

// sz_func is the array of true size functions...  the others
// seem to be specific for data object subscripting and subsampling!?!?

typedef struct size_functions {
	double (*sz_func)(QSP_ARG_DECL  Item *,int);
	const char *	(*prec_func)(QSP_ARG_DECL  Item *);
} Size_Functions;

#ifndef BUILD_FOR_OBJC
extern const char *default_prec_name(QSP_ARG_DECL  Item *);
#endif // ! BUILD_FOR_OBJC

typedef struct ilace_functions {
	double (*ilace_func)(QSP_ARG_DECL  Item *);
} Interlace_Functions;

typedef struct subscript_functions {
	Item * (*subscript)(QSP_ARG_DECL  Item *,index_t);
	Item * (*csubscript)(QSP_ARG_DECL  Item *,index_t);
} Subscript_Functions;

typedef struct posn_functions {
	double (*posn_func)(QSP_ARG_DECL  Item *, int index);
} Position_Functions;


typedef struct timestamp_functions {
	double	(*timestamp_func[3])(QSP_ARG_DECL  Item *,dimension_t);
} Timestamp_Functions;


/* support for general window functions */

typedef struct genwin_functions {
	void	(*posn_func)(QSP_ARG_DECL  const char *, int, int);
	void	(*show_func)(QSP_ARG_DECL  const char *);
	void	(*unshow_func)(QSP_ARG_DECL  const char *);
	void	(*delete_func)(QSP_ARG_DECL  const char *);
} Genwin_Functions;		



ITEM_INIT_PROT(Quip_Function,function)
ITEM_NEW_PROT(Quip_Function,function)
ITEM_CHECK_PROT(Quip_Function,function)


#define DECLARE_FUNCTION(name,func,code1,code2,code3,type,member,dim_index)	\
										\
										\
{										\
	Quip_Function *func_p;							\
										\
	func_p = new_function(QSP_ARG  #name);					\
	assert(func_p!=NULL);							\
	func_p->fn_type = type;							\
	func_p->fn_u.member = func;						\
	func_p->fn_vv_code = code1;						\
	func_p->fn_vs_code = code2;						\
	func_p->fn_vs_code2 = code3;						\
	func_p->fn_serial = func_serial++;					\
	func_p->fn_dim_index = dim_index;					\
}


#define DECLARE_D0_FUNCTION( name, func, code1, code2, code3 )	\
	DECLARE_FUNCTION(name,func,code1,code2,code3,D0_FUNCTYP,d0_func,-1)

#define DECLARE_D1_FUNCTION( name, func, code1, code2, code3 )	\
	DECLARE_FUNCTION(name,func,code1,code2,code3,D1_FUNCTYP,d1_func,-1)

#define DECLARE_I1_FUNCTION( name, func, code1, code2, code3 )	\
	DECLARE_FUNCTION(name,func,code1,code2,code3,I1_FUNCTYP,i1_func,-1)

#define DECLARE_D2_FUNCTION( name, func, code1, code2, code3 )	\
	DECLARE_FUNCTION(name,func,code1,code2,code3,D2_FUNCTYP,d2_func,-1)

#define DECLARE_SCALAR_FUNCTION( name, func, type, member, dim_index )	\
	DECLARE_FUNCTION( name, func, INVALID_VFC,INVALID_VFC,INVALID_VFC, type, member, dim_index )

#define DECLARE_STR1_FUNCTION( name, func )	\
	DECLARE_SCALAR_FUNCTION(name,func,STR1_FUNCTYP,str1_func,-1)

#define DECLARE_STRV_FUNCTION( name, func, code1, code2, code3 )	\
	DECLARE_FUNCTION(name,func,code1,code2,code3,STRV_FUNCTYP,strv_func,-1)

#define DECLARE_STRV2_FUNCTION( name, func )	\
	DECLARE_FUNCTION(name,func,INVALID_VFC,INVALID_VFC,INVALID_VFC,STRV2_FUNCTYP,strv2_func,-1)

#define DECLARE_CHAR_FUNCTION( name, func, code1, code2, code3 )	\
	DECLARE_FUNCTION(name,func,code1,code2,code3,CHAR_FUNCTYP,char_func,-1)

#define DECLARE_STR2_FUNCTION( name, func )	\
	DECLARE_SCALAR_FUNCTION(name,func,STR2_FUNCTYP,str2_func,-1)

#define DECLARE_STR3_FUNCTION( name, func )	\
	DECLARE_SCALAR_FUNCTION(name,func,STR3_FUNCTYP,str3_func,-1)

#define DECLARE_SIZE_FUNCTION( name, func, dim_index )	\
	DECLARE_SCALAR_FUNCTION(name,func,SIZE_FUNCTYP,sz_func,dim_index)

#define DECLARE_DOBJ_FUNCTION( name, func )	\
	DECLARE_SCALAR_FUNCTION(name,func,DOBJ_FUNCTYP,dobj_func,-1)

#define DECLARE_DOBJV_STR_ARG_FUNCTION( name, func )	\
	DECLARE_SCALAR_FUNCTION(name,func,DOBJV_STR_ARG_FUNCTYP,dobjv_str_arg_func,-1)

#define DECLARE_TS_FUNCTION( name, func )	\
	DECLARE_SCALAR_FUNCTION(name,func,TS_FUNCTYP,ts_func,-1)

#define DECLARE_ILACE_FUNCTION( name, func )	\
	DECLARE_SCALAR_FUNCTION(name,func,ILACE_FUNCTYP,il_func,-1)

#define DECLARE_POSITION_FUNCTION( name, func, dim_index )	\
	DECLARE_SCALAR_FUNCTION(name,func,POSN_FUNCTYP,posn_func,dim_index)

extern int func_serial;

#ifdef BUILD_FOR_OBJC
extern double get_object_size(QSP_ARG_DECL  void *ip,int d_index);
extern const char * get_object_prec_string(QSP_ARG_DECL  void *ip);
//extern const char * precision_string(QSP_ARG_DECL  void *ip);	// IOS_Item or Item
extern Size_Functions *get_size_functions(QSP_ARG_DECL  Item *ip);
#else // ! BUILD_FOR_OBJC
extern double get_object_size(QSP_ARG_DECL  Item *ip,int d_index);
extern const char * get_object_prec_string(QSP_ARG_DECL  Item *ip);
#endif // ! BUILD_FOR_OBJC

extern Size_Functions *get_sizable_functions(QSP_ARG_DECL  Item *ip);
extern Interlace_Functions *get_interlaceable_functions(QSP_ARG_DECL  Item *ip);
extern Timestamp_Functions *get_tsable_functions(QSP_ARG_DECL  Item *ip);
extern Position_Functions *get_positionable_functions(QSP_ARG_DECL  Item *ip);
extern Subscript_Functions *get_subscriptable_functions(QSP_ARG_DECL  Item *ip);


// function.c
//extern Item *find_sizable(QSP_ARG_DECL  const char *name);
#define FIND_FUNC_PROTOTYPE(type_stem)				\
extern Item *find_##type_stem(QSP_ARG_DECL  const char *name);

FIND_FUNC_PROTOTYPE(sizable)
FIND_FUNC_PROTOTYPE(tsable)
FIND_FUNC_PROTOTYPE(positionable)
FIND_FUNC_PROTOTYPE(interlaceable)
FIND_FUNC_PROTOTYPE(subscriptable)

extern Item *check_sizable(QSP_ARG_DECL  const char *name);
extern Item *sub_sizable(QSP_ARG_DECL  Item *ip,index_t index);
extern Item *csub_sizable(QSP_ARG_DECL  Item *ip,index_t index);
extern Item *find_tsable(QSP_ARG_DECL  const char *name);

extern double erfinv(double);
extern float erfinvf(float);

// formerly from psych library...
// now in libinterpreter to support expressions
extern double ptoz( double p );
extern double ztop( double z );

#define ADD_CLASS_PROTOTYPE(type_stem,func_type)		\
extern void add_##type_stem(QSP_ARG_DECL  Item_Type *itp,	\
	func_type *func_str_ptr, Item *(*lookup)(QSP_ARG_DECL  const char *));

#ifdef FOOBAR
//extern void add_sizable(QSP_ARG_DECL  Item_Type *itp,Size_Functions *sfp,
//			Item *(*lookup)(QSP_ARG_DECL  const char *));
//extern void add_tsable(QSP_ARG_DECL  Item_Type *itp,Timestamp_Functions *sfp,
//			Item *(*lookup)(QSP_ARG_DECL  const char *));
#endif // FOOBAR

ADD_CLASS_PROTOTYPE(sizable,Size_Functions)
ADD_CLASS_PROTOTYPE(tsable,Timestamp_Functions)
ADD_CLASS_PROTOTYPE(interlaceable,Interlace_Functions)
ADD_CLASS_PROTOTYPE(positionable,Position_Functions)
ADD_CLASS_PROTOTYPE(subscriptable,Subscript_Functions)


#ifdef __cplusplus
}
#endif

#endif /* ! _FUNCTION_H_ */

