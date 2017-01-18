include(`../../include/veclib/real_args.m4')

_VEC_FUNC_3V( vibnd, dst=(dest_type)(src1<0?(src1<-src2?src1:-src2):(src1<src2?src2:src1)) )
_VEC_FUNC_2V( vsign, if( src1 > 0 ) dst = 1; else if( src1<0 ) dst = (-1); else dst=0; )
_VEC_FUNC_2V( vabs, dst=(dest_type)absfunc(src1) )
_VEC_FUNC_3V( vbnd, dst=(dest_type)(absfunc(src1)<=src2?src1:(src1>src2?src2:-src2)) )


_VEC_FUNC_2V_SCAL( vsmnm, dst=(dest_type)(absfunc(scalar1_val)<absfunc(src1)?scalar1_val:src1) )
_VEC_FUNC_2V_SCAL( vsmxm, dst=(dest_type)(absfunc(scalar1_val)>absfunc(src1) ? scalar1_val : src1 ) )
_VEC_FUNC_2V_SCAL( viclp, dst = (dest_type)( absfunc(src1) >= scalar1_val ? src1 : ( src1 < -scalar1_val ? -scalar1_val : scalar1_val ) ) )
_VEC_FUNC_2V_SCAL( vclip, dst = (dest_type)( absfunc(src1) <= scalar1_val ? src1 : ( src1 < -scalar1_val ? -scalar1_val : scalar1_val ) ) )

_VEC_FUNC_MM_NOCC( vmxmg, absfunc(src1)>=extval, absfunc(src1)>extval, extval=(std_type)absfunc(src1), absfunc(src_vals[IDX2])>absfunc(src_vals[IDX2+1]), absfunc(src_vals[IDX2])<absfunc(src_vals[IDX2+1]) )

_VEC_FUNC_MM_NOCC( vmnmg, absfunc(src1)<=extval, absfunc(src1)<extval, extval=(std_type)absfunc(src1), absfunc(src_vals[IDX2])<absfunc(src_vals[IDX2+1]), absfunc(src_vals[IDX2])>absfunc(src_vals[IDX2+1]))


_VEC_FUNC_2V_PROJ_IDX( vmnmi, dst = index_base[0], tmp_ptr = INDEX_VDATA(dst); if( absfunc(src1)<absfunc(*tmp_ptr) ) dst=index_base[0], dst = (absfunc(src1) < absfunc(src2) ? IDX2 : IDX3+len1), dst = (absfunc(orig[src1]) < absfunc(orig[src2]) ? src1 : src2 ))
_VEC_FUNC_2V_PROJ_IDX( vmxmi, dst = index_base[0], tmp_ptr = INDEX_VDATA(dst); if( absfunc(src1)>absfunc(*tmp_ptr) ) dst=index_base[0], dst = (absfunc(src1) > absfunc(src2) ? IDX2 : IDX3+len1), dst = (absfunc(orig[src1]) > absfunc(orig[src2]) ? src1 : src2 ))

_VEC_FUNC_2V_PROJ( vmxmv, dst = (dest_type)absfunc(src1), if( absfunc(src1) > dst ) dst = (dest_type)absfunc(src1);, (absfunc(psrc1) > absfunc(psrc2) ? psrc1 : psrc2))

_VEC_FUNC_2V_PROJ( vmnmv, dst = (dest_type)absfunc(src1), if( absfunc(src1) < dst ) dst = (dest_type)absfunc(src1);, (absfunc(psrc1) < absfunc(psrc2) ? psrc1 : psrc2))

_VEC_FUNC_3V( vmaxm, dst = (dest_type) ( (absfunc(src1)) >= (absfunc(src2)) ? src1 : src2 ) )
_VEC_FUNC_3V( vminm, dst = (dest_type) ( absfunc(src1) < absfunc(src2) ? src1 : src2 ) )

_VEC_FUNC_2V( rvneg, dst = (dest_type) ( - src1 ) )

