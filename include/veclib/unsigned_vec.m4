my_include(`veclib/real_args.m4')

_VEC_FUNC_3V( vibnd, dst=src1<src2?src2:src1 )
_VEC_FUNC_2V( vsign, dst=src1>0?1:0 )
_VEC_FUNC_2V( vabs, dst=src1 )
_VEC_FUNC_3V( vbnd, dst=(src1<=src2?src1:src2) )


_VEC_FUNC_2V_SCAL( vsmnm, dst=(scalar1_val<src1?scalar1_val:src1) )
_VEC_FUNC_2V_SCAL( vsmxm, dst=(scalar1_val>src1?scalar1_val:src1 ) )
_VEC_FUNC_2V_SCAL( viclp, dst = ( src1 >= scalar1_val ? src1 : scalar1_val ) )
_VEC_FUNC_2V_SCAL( vclip, dst = ( src1 <= scalar1_val ? src1 : scalar1_val ) )

_VEC_FUNC_MM_NOCC( vmxmg, src1>=extval, src1>extval, extval=src1, src_vals[index2]>src_vals[index2+1], src_vals[index2]<src_vals[index2+1])
_VEC_FUNC_MM_NOCC( vmnmg, src1<=extval, src1<extval, extval=src1, src_vals[index2]<src_vals[index2+1], src_vals[index2]>src_vals[index2+1])

_VEC_FUNC_2V_PROJ_IDX( vmnmi, dst = index_base[0], tmp_ptr = INDEX_VDATA(dst); if( src1<(*tmp_ptr) ) dst=index_base[0], dst = (src1 < src2 ? index2 : index3+len1), dst = (orig[src1] < orig[src2] ? src1 : src2 ))
_VEC_FUNC_2V_PROJ_IDX( vmxmi, dst = index_base[0], tmp_ptr = INDEX_VDATA(dst); if( src1>(*tmp_ptr) ) dst=index_base[0], dst = (src1 > src2 ? index2 : index3+len1), dst = (orig[src1] > orig[src2] ? src1 : src2 ))

_VEC_FUNC_2V_PROJ( vmxmv, dst = src1, if( src1 > dst ) dst = src1;, (psrc1 > psrc2 ? psrc1 : psrc2))
_VEC_FUNC_2V_PROJ( vmnmv, dst = src1, if( src1 < dst ) dst = src1;, (psrc1 < psrc2 ? psrc1 : psrc2))

_VEC_FUNC_3V( vmaxm, dst = ( src1 >= src2 ? src1 : src2 ) )
_VEC_FUNC_3V( vminm, dst = ( src1 <  src2 ? src1 : src2 ) )

_VEC_FUNC_2V( rvneg, dst = (std_type)( - (std_signed) src1 ) )

/***********/

