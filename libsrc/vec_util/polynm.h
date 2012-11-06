

#ifndef NO_POLY

#ifdef INC_VERSION
char VersionId_inc_polynm[] = QUIP_VERSION_STRING;
#endif /* INC_VERSION */

#include "typedefs.h"
#include "data_obj.h"

#define COEFF_TYPE	double

typedef struct coefficient {
	COEFF_TYPE	value;
	long		exponent;
} Coefficient;

typedef struct polynomial {
	Item	poly_item;
#define poly_name	poly_item.item_name

	int order;
	Coefficient *coeff;
} Polynomial;

#define NO_POLY		((Polynomial *)NULL)


ITEM_INTERFACE_PROTOTYPES(Polynomial,polynm)
extern void zap_poly(QSP_ARG_DECL  Polynomial *);

/* other prototypes from legendre.c */

extern Polynomial *new_poly(QSP_ARG_DECL  int);
extern void show_poly(Polynomial *);
extern Polynomial *diff_poly(QSP_ARG_DECL  Polynomial *);
extern Polynomial *scale_poly(QSP_ARG_DECL  Polynomial *,double);
extern Polynomial *exp_poly(QSP_ARG_DECL  Polynomial *,int);
extern Polynomial *legendre_poly(QSP_ARG_DECL  int,int);
extern Polynomial *add_polys(QSP_ARG_DECL  Polynomial *,Polynomial *);
extern Polynomial *mul_polys(QSP_ARG_DECL  Polynomial *,Polynomial *);
extern void factorial(Coefficient *,int);
extern void factorial_series(Coefficient *,int,int);
extern double eval_poly(Polynomial *,double);
extern void tabulate_legendre(QSP_ARG_DECL  Data_Obj *,Data_Obj *,int,int);
extern void tabulate_Pbar(QSP_ARG_DECL  Data_Obj *,Data_Obj *,int,int);
extern void tabulate_polynomial(Data_Obj *,Data_Obj *,Polynomial *);

extern void set_coeff(Coefficient *,double);



#endif /* ! NO_POLY */

