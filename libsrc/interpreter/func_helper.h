
double evalD0Function( Function *f );
double evalD1Function(Function *f,double d);
double evalD2Function(Function *f,double d1,double d2);
int evalI1Function(Function *f,double d);
double evalStr1Function(QSP_ARG_DECL  Function *,const char *s);
double evalStr2Function(Function *,const char *s1,const char *s2);
double evalStr3Function(Function *,const char *,const char *,int);
//void evalStrVFunction(Function *,char *dst, const char *s);
int evalCharFunction(Function *,char c);

void setD0Code(int);
void setD1Code(int);
void setD2Code(int);
void setStr1Code(int);
void setStr2Code(int);
void setStr3Code(int);
void setDataCode(int);
void setSizeCode(int);

int functionType(Function *f);

void init_function_classes(void);

