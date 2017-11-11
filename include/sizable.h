// Why is this file only for iOS???

#ifndef _SIZABLE_H_
#define _SIZABLE_H_

#ifdef BUILD_FOR_OBJC

#include "ios_item.h"

typedef struct ios_size_functions {
	double (*sz_func)(QSP_ARG_DECL  IOS_Item *,int);
	const char * (*prec_func)(QSP_ARG_DECL  IOS_Item *);
} IOS_Size_Functions;

typedef struct ios_posn_functions {
	double (*posn_func)(QSP_ARG_DECL  IOS_Item *, int index);
} IOS_Position_Functions;

typedef struct ios_ilace_functions {
	double (*ilace_func)(QSP_ARG_DECL  IOS_Item *);
} IOS_Interlace_Functions;



extern void _add_ios_sizable(QSP_ARG_DECL  IOS_Item_Type *itp,IOS_Size_Functions *sfp,
			IOS_Item *(*lookup)(QSP_ARG_DECL  const char *));
extern void _add_ios_positionable(QSP_ARG_DECL  IOS_Item_Type *itp,IOS_Position_Functions *sfp,
			IOS_Item *(*lookup)(QSP_ARG_DECL  const char *));
extern IOS_Item *find_ios_sizable(QSP_ARG_DECL  const char *name);

extern IOS_Item *check_ios_sizable(QSP_ARG_DECL  const char *name);
extern IOS_Item *check_ios_positionable(QSP_ARG_DECL  const char *name);
extern IOS_Item *check_ios_interlaceable(QSP_ARG_DECL  const char *name);

extern const char *default_prec_name(QSP_ARG_DECL  void *ip);

#define add_ios_sizable(itp,sfp,lookup) _add_ios_sizable(QSP_ARG  itp,sfp,lookup)
#define add_ios_positionable(itp,sfp,lookup) _add_ios_positionable(QSP_ARG  itp,sfp,lookup)

#endif /* BUILD_FOR_OBJC */

#endif /* ! _SIZABLE_H_ */

