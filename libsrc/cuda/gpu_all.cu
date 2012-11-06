
/* To find the index of the extremum, we start by doing pairs,
 * then continue using indices in temporary buffers... 
 */

KERN_CALL_MM_NOCC(vmaxg,src_vals[index2.x]>src_vals[index2.x+1],src_vals[index2.x]<src_vals[index2.x+1])
KERN_CALL_MM_NOCC(vming,src_vals[index2.x]<src_vals[index2.x+1],src_vals[index2.x]>src_vals[index2.x+1])

KERN_CALL_MM_IND(vmaxi, dst = (src1 > src2 ? index2.x : index3.x+len1) , dst = (orig[src1] > orig[src2] ? src1 : src2) )
KERN_CALL_MM_IND(vmini, dst = (src1 < src2 ? index2.x : index3.x+len1) , dst = (orig[src1] < orig[src2] ? src1 : src2) )


KERN_CALL_1V_2SCAL(vramp1d, dst = scalar1_val + index1.x * scalar2_val )

// BUG better to go back to a dedicated ramp function...
KERN_CALL_1V_3SCAL(vramp2d, dst = scalar1_val + scalar2_val * index1.x / inc1.x + scalar3_val * index1.y / inc1.y )

KERN_CALL_2V_PROJ( vsum , psrc1 + psrc2 )
KERN_CALL_2V_PROJ( vmaxv , psrc1 > psrc2 ? psrc1 : psrc2 )
KERN_CALL_2V_PROJ( vminv , psrc1 < psrc2 ? psrc1 : psrc2 )

KERN_CALL_3V_PROJ( vdot )

KERN_CALL_VVSLCT( vvv_slct , dst = srcbit ? src1 : src2 )
KERN_CALL_VSSLCT( vvs_slct , dst = srcbit ? src1 : scalar1_val )
KERN_CALL_SSSLCT( vss_slct , dst = srcbit ? scalar1_val : scalar2_val )


