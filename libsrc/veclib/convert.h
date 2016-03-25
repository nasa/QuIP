

#define CONVERSION_DECLS( key1, key2 )				\
extern void v##key1##2##key2  ( HOST_CALL_ARG_DECLS );

#define ALL_CONVERSION_DECLS( key )				\
CONVERSION_DECLS( key, by )					\
CONVERSION_DECLS( key, in )					\
CONVERSION_DECLS( key, di )					\
CONVERSION_DECLS( key, sp )					\
CONVERSION_DECLS( key, dp )					\
CONVERSION_DECLS( key, uby )					\
CONVERSION_DECLS( key, uin )					\
CONVERSION_DECLS( key, udi )

/*
void bmconv(Vec_Args *argp);
void bmvmov(Vec_Args *argp);
*/

ALL_CONVERSION_DECLS( by )
ALL_CONVERSION_DECLS( in )
ALL_CONVERSION_DECLS( di )
ALL_CONVERSION_DECLS( sp )
ALL_CONVERSION_DECLS( dp )
ALL_CONVERSION_DECLS( uby )
ALL_CONVERSION_DECLS( uin )
ALL_CONVERSION_DECLS( udi )

