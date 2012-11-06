
#ifndef NO_FUNCTION

#include "data_obj.h"
#include "items.h"
#include "vecgen.h"	/* Vec_Func_Code */

typedef enum {
	VOID_FUNCTYP,
	D1_FUNCTYP,
	D2_FUNCTYP,
	SIZE_FUNCTYP,
	TS_FUNCTYP,
	STR1_FUNCTYP,
	STR2_FUNCTYP,
	STR3_FUNCTYP,
	DOBJ_FUNCTYP,
	N_FUNC_TYPES		// must be last
} Function_Type;

typedef struct function {
	const char *	fn_name;
	Function_Type	fn_type;
	union {
 		double	      (*v_func)(void);
 		double	      (*d_func)(void);
 		double	      (*d1_func)(double);
 		double	      (*d2_func)(double, double);
 		double	      (*sz_func)(QSP_ARG_DECL  Item *);
		double	      (*ts_func)(QSP_ARG_DECL  Item *,dimension_t frm);
 		double	      (*str1_func)(QSP_ARG_DECL  const char *);
 		double	      (*str2_func)(const char *,const char *);
 		double	      (*str3_func)(const char *,const char *,int);
 		double	      (*dobj_func)(Data_Obj *);
	} fn_func;
	int		fn_vv_code;
	int		fn_vs_code;
	int		fn_vs_code2;
} Function;

#define vd_func		fn_func.v_func
#define d0f_func	fn_func.v_func
#define d1f_func	fn_func.d1_func
#define d2f_func	fn_func.d2_func
#define szf_func	fn_func.sz_func
#define tsf_func	fn_func.ts_func
#define str1f_func	fn_func.str1_func
#define str2f_func	fn_func.str2_func
#define str3f_func	fn_func.str3_func
#define dof_func	fn_func.dobj_func

typedef struct function_class {
	int			fc_class;
	Function *		fc_tbl;
} Function_Class;

#define NO_FUNCTION		((Function *)NULL)

/* support for size functions */

typedef struct size_functions {
	double (*sz_func)(Item *,int);
	Item * (*subscript)(Item *,index_t);
	Item * (*csubscript)(Item *,index_t);
	double (*il_func)(Item *);
} Size_Functions;

typedef struct timestamp_functions {
	double	(*ts_func[3])(Item *,dimension_t);
} Timestamp_Functions;

/* support for general window functions */

typedef struct genwin_functions {
	void	(*posn_func)(QSP_ARG_DECL  const char *, int, int);
	void	(*show_func)(QSP_ARG_DECL  const char *);
	void	(*unshow_func)(QSP_ARG_DECL  const char *);
	void	(*delete_func)(QSP_ARG_DECL  const char *);
} Genwin_Functions;		

/* globals */

extern Function math0_functbl[];
extern Function math1_functbl[];
extern Function math2_functbl[];
extern Function data_functbl[];
extern Function size_functbl[];
extern Function timestamp_functbl[];
extern Function str1_functbl[];
extern Function str2_functbl[];
extern Function str3_functbl[];
extern Function misc_functbl[];



extern double	nulld1func(double);
extern double	nulld2func(double,double);
extern double	nullvfunc(void);
extern double	nullszfunc(QSP_ARG_DECL  Item *);
extern double	nulldofunc(Data_Obj *);
extern double	nulls1func(QSP_ARG_DECL  const char *);
extern double	nulls2func(const char *,const char *);

extern int	assgn_func(Function *,const char *,double (*func)(void));
extern void	setdatafunc(const char *,double (*func)(Data_Obj *));
extern void	setstrfunc(const char *,double (*func)(QSP_ARG_DECL  const char *));
extern void	setmiscfunc(const char *,double (*func)(void));

extern void set_obj_funcs(	Data_Obj *(*obj_func)(QSP_ARG_DECL  const char *),
				Data_Obj *(*exist_func)(QSP_ARG_DECL  const char *),
				Data_Obj *(*sub_func)(QSP_ARG_DECL  Data_Obj *,index_t),
				Data_Obj *(*csub_func)(QSP_ARG_DECL  Data_Obj *,index_t));

extern double get_object_size(QSP_ARG_DECL  Item *,int);

extern void add_sizable( QSP_ARG_DECL  Item_Type *, Size_Functions *,
			Item * (*lookup)(QSP_ARG_DECL  const char *));

extern Item *find_sizable(QSP_ARG_DECL  const char *);
extern Item *sub_sizable(QSP_ARG_DECL  Item *,index_t);
extern Item *csub_sizable(QSP_ARG_DECL  Item *,index_t);

extern void add_tsable( QSP_ARG_DECL  Item_Type *, Timestamp_Functions *,
			Item * (*lookup)(QSP_ARG_DECL  const char *));
extern Item *find_tsable(QSP_ARG_DECL  const char *);

extern	int whfunc(Function *table, const char *str);


#endif /* !NO_FUNCTION */

