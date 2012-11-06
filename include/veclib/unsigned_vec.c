THREE_VEC_METHOD( vibnd , dst=src1<src2?src2:src1 )
TWO_VEC_METHOD( vsign , dst = 1 )
TWO_VEC_METHOD( vabs , dst=src1 )
TWO_VEC_METHOD( rvneg , dst=( (std_type) -((std_signed)src1) ) )
THREE_VEC_METHOD( vbnd , dst=(src1<=src2?src1:src2) )


TWO_VEC_SCALAR_METHOD( vsmnm , dst=(scalar1_val<src1?scalar1_val:src1) )
TWO_VEC_SCALAR_METHOD( vsmxm , dst=(scalar1_val>src1?scalar1_val:src1 ) )
TWO_VEC_SCALAR_METHOD( viclp , dst = ( src1 >= scalar1_val ? src1 : scalar1_val ) )
TWO_VEC_SCALAR_METHOD( vclip , dst = ( src1 <= scalar1_val ? src1 : scalar1_val ) )

EXTREMA_LOCATIONS_METHOD( vmxmg, src1>=extval, src1>extval, extval=src1  )
EXTREMA_LOCATIONS_METHOD( vmnmg, src1<=extval, src1<extval, extval=src1  )

PROJECTION_METHOD_IDX_2( vmnmi ,
	dst = index_base[0] ,
	tmp_ptr = INDEX_VDATA(dst); if( src1<(*tmp_ptr) ) dst=index_base[0] )
PROJECTION_METHOD_IDX_2( vmxmi ,
	dst = index_base[0] ,
	tmp_ptr = INDEX_VDATA(dst); if( src1>(*tmp_ptr) ) dst=index_base[0] )

PROJECTION_METHOD_2( vmxmv , dst = src1 , if( src1 > dst ) dst = src1; )
PROJECTION_METHOD_2( vmnmv , dst = src1 , if( src1 < dst ) dst = src1; )

THREE_VEC_METHOD( vmaxm , dst = ( src1 >= src2 ? src1 : src2 ) )
THREE_VEC_METHOD( vminm , dst = ( src1 <  src2 ? src1 : src2 ) )



