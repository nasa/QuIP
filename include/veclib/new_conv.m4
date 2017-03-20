dnl add ifdef's to inhibit type-to-type "conversions"?

_VEC_FUNC_2V_CONV( vconv2by , char , dst = (char)(src1) )
_VEC_FUNC_2V_CONV( vconv2in , short , dst = (short)(src1) )
_VEC_FUNC_2V_CONV( vconv2di , int32_t , dst = (int32_t)(src1) )
_VEC_FUNC_2V_CONV( vconv2li , int64_t , dst = (int64_t)(src1) )
_VEC_FUNC_2V_CONV( vconv2uby , u_char , dst = (u_char)(src1) )
_VEC_FUNC_2V_CONV( vconv2uin , u_short , dst = (u_short)(src1) )
_VEC_FUNC_2V_CONV( vconv2udi , uint32_t , dst = (uint32_t)(src1) )
_VEC_FUNC_2V_CONV( vconv2uli , uint64_t , dst = (uint64_t)(src1) )
_VEC_FUNC_2V_CONV( vconv2sp , float , dst = (float)(src1) )
_VEC_FUNC_2V_CONV( vconv2dp , double , dst = (double)(src1) )

