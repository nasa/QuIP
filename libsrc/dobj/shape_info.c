#include "quip_config.h"

#include "quip_prot.h"
#include "dobj_prot.h"

Item_Type *prec_itp=NO_ITEM_TYPE;

#define INIT_VOID_PREC(name,type,code)			\
	INIT_GENERIC_PREC(name,type,code)		\
	SET_PREC_SIZE(prec_p, 0);

#define INIT_PREC(name,type,code)			\
	INIT_GENERIC_PREC(name,type,code)		\
	SET_PREC_SIZE(prec_p, sizeof(type));

#define INIT_GENERIC_PREC(name,type,code)		\
	prec_p = new_prec(QSP_ARG  #name);		\
	SET_PREC_CODE(prec_p, code);			\
	if( (code & PSEUDO_PREC_MASK) == 0 )		\
		SET_PREC_MACH_PREC_PTR(prec_p, prec_p);	\
	else {						\
		SET_PREC_MACH_PREC_PTR(prec_p,		\
		prec_for_code( code & MACH_PREC_MASK ) ); \
	}

static Precision *new_prec(QSP_ARG_DECL  const char *name)
{
	Precision *prec_p;

	prec_p = (Precision *) new_item(QSP_ARG  prec_itp, name, sizeof(Precision) );
	// BUG make sure name is not already in use

//#ifdef CAUTIOUS
//	if( prec_p == NO_PRECISION) ERROR1("CAUTIOUS:  new_prec:  Error creating precision!?");
//#endif /* CAUTIOUS */
	assert( prec_p != NO_PRECISION );

	return(prec_p);
}

Precision *get_prec(QSP_ARG_DECL  const char *name)
{
	return (Precision *)get_item(QSP_ARG  prec_itp, name);
}

void init_precisions(SINGLE_QSP_ARG_DECL)
{
	Precision *prec_p;

	prec_itp = new_item_type(QSP_ARG  "Precision", LIST_CONTAINER);	// used to be hashed, but not many of these?
									// should sort based on access?

	INIT_PREC(byte,char,PREC_BY)
	INIT_PREC(u_byte,u_char,PREC_UBY)
	INIT_PREC(short,short,PREC_IN)
	INIT_PREC(u_short,u_short,PREC_UIN)
	INIT_PREC(int32,int32_t,PREC_DI)
	INIT_PREC(uint32,uint32_t,PREC_UDI)
	INIT_PREC(int64,int64_t,PREC_LI)
	INIT_PREC(uint64,uint64_t,PREC_ULI)
	INIT_PREC(float,float,PREC_SP)
	INIT_PREC(double,double,PREC_DP)

	INIT_PREC(complex,SP_Complex,PREC_CPX)
	INIT_PREC(dblcpx,DP_Complex,PREC_DBLCPX)
	INIT_PREC(quaternion,SP_Quaternion,PREC_QUAT)
	INIT_PREC(dblquat,DP_Quaternion,PREC_DBLQUAT)
	INIT_PREC(char,char,PREC_CHAR)
	INIT_PREC(string,char,PREC_STR)
	INIT_PREC(color,SP_Color,PREC_COLOR)
	INIT_PREC(bit,BITMAP_DATA_TYPE,PREC_BIT)
	INIT_VOID_PREC(void,void,PREC_VOID)

#ifdef USE_LONG_DOUBLE
	INIT_PREC(long_dbl,long double,PREC_LP)
	INIT_PREC(ldblcpx,LP_Complex,PREC_LDBLCPX)
	INIT_PREC(ldblquat,LP_Quaternion,PREC_LDBLQUAT)
#endif // USE_LONG_DOUBLE
}


List *prec_list(SINGLE_QSP_ARG_DECL)
{
	if( prec_itp == NO_ITEM_TYPE ){
		init_precisions(SINGLE_QSP_ARG);
	}

	return item_list(QSP_ARG  prec_itp);
}

Precision *const_precision(Precision *prec_p)
{
	NERROR1("Sorry, const_precision not implemented yet.");
	return NO_PRECISION;
}

Precision *complex_precision(Precision *prec_p)
{
//#ifdef CAUTIOUS
//	if( COMPLEX_PRECISION(PREC_CODE(prec_p)) )
//		NERROR1("CAUTIOUS:  complex_precision:  pass a complex precision!?");
//#endif /* CAUTIOUS */
	assert( ! COMPLEX_PRECISION(PREC_CODE(prec_p)) );

	return PREC_FOR_CODE( PREC_CODE(prec_p) | DT_COMPLEX );
}

