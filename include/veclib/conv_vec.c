// These must be for the old-style conversions?

ALL_UNSIGNED_CONVERSIONS( by, char )
ALL_FLOAT_CONVERSIONS( by, char )
REAL_CONVERSION( by, char, in,  short   )			\
REAL_CONVERSION( by, char, di,  int32_t    )			\
REAL_CONVERSION( by, char, li,  int64_t    )

ALL_UNSIGNED_CONVERSIONS( in, short )
ALL_FLOAT_CONVERSIONS( in, short )
REAL_CONVERSION( in, short, by,  char   )			\
REAL_CONVERSION( in, short, di,  int32_t    )			\
REAL_CONVERSION( in, short, li,  int64_t    )

ALL_UNSIGNED_CONVERSIONS( di, int32_t )
ALL_FLOAT_CONVERSIONS( di, int32_t )
REAL_CONVERSION( di, int32_t, in,  short   )			\
REAL_CONVERSION( di, int32_t, by,  char    )			\
REAL_CONVERSION( di, int32_t, li,  int64_t    )

ALL_UNSIGNED_CONVERSIONS( li, int64_t )
ALL_FLOAT_CONVERSIONS( li, int64_t )
REAL_CONVERSION( li, int64_t, in,  short   )			\
REAL_CONVERSION( li, int64_t, di,  int32_t    )			\
REAL_CONVERSION( li, int64_t, by,  char    )

ALL_SIGNED_CONVERSIONS( uby, u_char )
ALL_FLOAT_CONVERSIONS( uby, u_char )
REAL_CONVERSION( uby, u_char, uin,  u_short   )			\
REAL_CONVERSION( uby, u_char, udi,  uint32_t    )			\
REAL_CONVERSION( uby, u_char, uli,  uint64_t    )

ALL_SIGNED_CONVERSIONS( uin, u_short )
ALL_FLOAT_CONVERSIONS( uin, u_short )
REAL_CONVERSION( uin, u_short, uby,  u_char   )			\
REAL_CONVERSION( uin, u_short, udi,  uint32_t    )			\
REAL_CONVERSION( uin, u_short, uli,  uint64_t    )

ALL_SIGNED_CONVERSIONS( udi, uint32_t )
ALL_FLOAT_CONVERSIONS( udi, uint32_t )
REAL_CONVERSION( udi, uint32_t, uin,  u_short   )			\
REAL_CONVERSION( udi, uint32_t, uby,  u_char    )			\
REAL_CONVERSION( udi, uint32_t, uli,  uint64_t    )

ALL_SIGNED_CONVERSIONS( uli, uint64_t )
ALL_FLOAT_CONVERSIONS( uli, uint64_t )
REAL_CONVERSION( uli, uint64_t, uin,  u_short   )			\
REAL_CONVERSION( uli, uint64_t, udi,  uint32_t    )			\
REAL_CONVERSION( uli, uint64_t, uby,  u_char    )

ALL_UNSIGNED_CONVERSIONS( sp, float )
ALL_SIGNED_CONVERSIONS( sp, float )
REAL_CONVERSION( sp, float, dp, double   )			\

ALL_UNSIGNED_CONVERSIONS( dp, double )
ALL_SIGNED_CONVERSIONS( dp, double )
REAL_CONVERSION( dp, double, sp, float   )			\

