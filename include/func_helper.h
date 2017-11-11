
double evalD0Function( Quip_Function *f );
double evalD1Function(Quip_Function *f,double d);
double evalD2Function(Quip_Function *f,double d1,double d2);
int evalI1Function(Quip_Function *f,double d);
double evalStr1Function(QSP_ARG_DECL  Quip_Function *,const char *s);
double evalStr2Function(Quip_Function *,const char *s1,const char *s2);
double evalStr3Function(Quip_Function *,const char *,const char *,int);
//void evalStrVFunction(Quip_Function *,char *dst, const char *s);
int evalCharFunction(Quip_Function *,char c);

void setD0Code(int);
void setD1Code(int);
void setD2Code(int);
void setStr1Code(int);
void setStr2Code(int);
void setStr3Code(int);
void setDataCode(int);
void setSizeCode(int);

int functionType(Quip_Function *f);

void init_function_classes(void);

