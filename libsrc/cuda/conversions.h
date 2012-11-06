
#define ALL_SIGNED_HOST_CONVERSIONS( src_code , src_type )			\
										\
H_CALL_CONV( h_##src_code##2by, g_##src_code##2by, char, src_type )	\
H_CALL_CONV( h_##src_code##2in, g_##src_code##2in, short, src_type )	\
H_CALL_CONV( h_##src_code##2di, g_##src_code##2di, int32_t, src_type )	\
H_CALL_CONV( h_##src_code##2li, g_##src_code##2li, int64_t, src_type )

#define ALL_UNSIGNED_HOST_CONVERSIONS( src_code , src_type )			\
									\
H_CALL_CONV( h_##src_code##2uby, g_##src_code##2uby, u_char, src_type )	\
H_CALL_CONV( h_##src_code##2uin, g_##src_code##2uin, u_short, src_type )	\
H_CALL_CONV( h_##src_code##2udi, g_##src_code##2udi, uint32_t, src_type )	\
H_CALL_CONV( h_##src_code##2uli, g_##src_code##2uli, uint64_t, src_type )

#define ALL_FLOAT_HOST_CONVERSIONS( src_code , src_type )					\
											\
H_CALL_CONV( h_##src_code##2sp, g_##src_code##2sp, float, src_type )	\
H_CALL_CONV( h_##src_code##2dp, g_##src_code##2dp, double, src_type )

// From sp
H_CALL_CONV( h_sp2dp, g_sp2dp, double, float )
ALL_SIGNED_HOST_CONVERSIONS( sp , float )
ALL_UNSIGNED_HOST_CONVERSIONS( sp , float )

// From dp
H_CALL_CONV( h_dp2sp, g_dp2sp, float , double)
ALL_SIGNED_HOST_CONVERSIONS( dp , double )
ALL_UNSIGNED_HOST_CONVERSIONS( dp , double )

//from by
H_CALL_CONV( h_by2in, g_by2in, short, char )
H_CALL_CONV( h_by2di, g_by2di, int32_t, char )
H_CALL_CONV( h_by2li, g_by2li, int64_t, char )
ALL_UNSIGNED_HOST_CONVERSIONS( by , char )
ALL_FLOAT_HOST_CONVERSIONS( by , char )

//from in
H_CALL_CONV( h_in2by, g_in2by, char, short )
H_CALL_CONV( h_in2di, g_in2di, int32_t, short )
H_CALL_CONV( h_in2li, g_in2li, int64_t, short )
ALL_UNSIGNED_HOST_CONVERSIONS( in , short )
ALL_FLOAT_HOST_CONVERSIONS( in , short )

//from di
H_CALL_CONV( h_di2by, g_di2by, char, int32_t )
H_CALL_CONV( h_di2in, g_di2in, short, int32_t )
H_CALL_CONV( h_di2li, g_di2li, int64_t, int32_t )
ALL_UNSIGNED_HOST_CONVERSIONS( di , int32_t )
ALL_FLOAT_HOST_CONVERSIONS( di , int32_t )

//from li
H_CALL_CONV( h_li2by, g_li2by, char, int64_t )
H_CALL_CONV( h_li2in, g_li2in, short, int64_t )
H_CALL_CONV( h_li2di, g_li2di, int32_t, int64_t )
ALL_UNSIGNED_HOST_CONVERSIONS( li , int64_t )
ALL_FLOAT_HOST_CONVERSIONS( li , int64_t )


//from uby
H_CALL_CONV( h_uby2uin, g_uby2uin, u_short, u_char )
H_CALL_CONV( h_uby2udi, g_uby2udi, uint32_t, u_char )
H_CALL_CONV( h_uby2uli, g_uby2uli, uint64_t, u_char )
ALL_SIGNED_HOST_CONVERSIONS( uby , u_char )
ALL_FLOAT_HOST_CONVERSIONS( uby , u_char )

//from uin
H_CALL_CONV( h_uin2uby, g_uin2uby, u_char, u_short )
H_CALL_CONV( h_uin2udi, g_uin2udi, uint32_t, u_short )
H_CALL_CONV( h_uin2uli, g_uin2uli, uint64_t, u_short )
ALL_SIGNED_HOST_CONVERSIONS( uin , u_short )
ALL_FLOAT_HOST_CONVERSIONS( uin , u_short )

//from udi
H_CALL_CONV( h_udi2uby, g_udi2uby, u_char, uint32_t )
H_CALL_CONV( h_udi2uin, g_udi2uin, u_short, uint32_t )
H_CALL_CONV( h_udi2uli, g_udi2uli, uint64_t, uint32_t )
ALL_SIGNED_HOST_CONVERSIONS( udi , uint32_t )
ALL_FLOAT_HOST_CONVERSIONS( udi , uint32_t )

//from uli
H_CALL_CONV( h_uli2uby, g_uli2uby, u_char, uint64_t )
H_CALL_CONV( h_uli2uin, g_uli2uin, u_short, uint64_t )
H_CALL_CONV( h_uli2udi, g_uli2udi, uint32_t, uint64_t )
ALL_SIGNED_HOST_CONVERSIONS( uli , uint64_t )
ALL_FLOAT_HOST_CONVERSIONS( uli , uint64_t )

