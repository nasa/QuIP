
#include "query.h"

void digest(QSP_ARG_DECL const char *);
void swallow(QSP_ARG_DECL const char *);
void chew_text(QSP_ARG_DECL const char *);
#define CHEW_TEXT(str)		chew_text(QSP_ARG  str)
#define DIGEST(str)		digest(QSP_ARG str)


