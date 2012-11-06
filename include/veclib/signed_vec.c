THREE_VEC_METHOD( vibnd , dst=(src1<0?(src1<-src2?src1:-src2):(src1<src2?src2:src1)) )
TWO_VEC_METHOD( vsign , if( src1 > 0 ) dst = 1; else if( src1<0 ) dst = (-1); else dst=0; )
TWO_VEC_METHOD( vabs , dst=absfunc(src1) )
THREE_VEC_METHOD( vbnd , dst=(absfunc(src1)<=src2?src1:(src1>src2?src2:-src2)) )


TWO_VEC_SCALAR_METHOD( vsmnm , dst=(absfunc(scalar1_val)<absfunc(src1)?scalar1_val:src1) )
TWO_VEC_SCALAR_METHOD( vsmxm , dst=(absfunc(scalar1_val)>absfunc(src1) ? scalar1_val : src1 ) )
TWO_VEC_SCALAR_METHOD( viclp , dst = ( absfunc(src1) >= scalar1_val ? src1 : ( src1 < -scalar1_val ? -scalar1_val : scalar1_val ) ) )
TWO_VEC_SCALAR_METHOD( vclip , dst = ( absfunc(src1) <= scalar1_val ? src1 : ( src1 < -scalar1_val ? -scalar1_val : scalar1_val ) ) )

EXTREMA_LOCATIONS_METHOD( vmxmg, absfunc(src1)>=extval, absfunc(src1)>extval, extval=absfunc(src1)  )
EXTREMA_LOCATIONS_METHOD( vmnmg, absfunc(src1)<=extval, absfunc(src1)<extval, extval=absfunc(src1)  )

PROJECTION_METHOD_IDX_2( vmnmi ,
	dst = index_base[0] ,
	tmp_ptr = INDEX_VDATA(dst); if( absfunc(src1)<absfunc(*tmp_ptr) ) dst=index_base[0] )
PROJECTION_METHOD_IDX_2( vmxmi ,
	dst = index_base[0] ,
	tmp_ptr = INDEX_VDATA(dst); if( absfunc(src1)>absfunc(*tmp_ptr) ) dst=index_base[0] )

PROJECTION_METHOD_2( vmxmv , dst = absfunc(src1) , if( absfunc(src1) > dst ) dst = absfunc(src1); )
PROJECTION_METHOD_2( vmnmv , dst = absfunc(src1) , if( absfunc(src1) < dst ) dst = absfunc(src1); )

THREE_VEC_METHOD( vmaxm , dst = ( (absfunc(src1)) >= (absfunc(src2)) ? src1 : src2 ) )
THREE_VEC_METHOD( vminm , dst = ( absfunc(src1) < absfunc(src2) ? src1 : src2 ) )

TWO_VEC_METHOD( rvneg , dst = - src1 )


