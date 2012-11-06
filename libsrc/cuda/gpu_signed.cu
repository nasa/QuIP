
/* Functions which have to be implemented differently on the gpu */

KERN_CALL_MM(vmxmv, dst = (abs(src1) > abs(src2) ? src1 : src2))
KERN_CALL_MM(vmnmv, dst = (abs(src1) < abs(src2) ? src1 : src2))

KERN_CALL_MM_IND(vmxmi, dst = (abs(src1) > abs(src2) ? index2.x : index3.x+len1) , dst = (abs(orig[src1]) > abs(orig[src2]) ? src1 : src2 ) )
KERN_CALL_MM_IND(vmnmi, dst = (abs(src1) < abs(src2) ? index2.x : index3.x+len1) , dst = (abs(orig[src1]) < abs(orig[src2]) ? src1 : src2 ) )

KERN_CALL_MM_NOCC(vmxmg,abs(src_vals[index2.x])>abs(src_vals[index2.x+1]),abs(src_vals[index2.x])<abs(src_vals[index2.x+1]) )
KERN_CALL_MM_NOCC(vmnmg,abs(src_vals[index2.x])<abs(src_vals[index2.x+1]),abs(src_vals[index2.x])>abs(src_vals[index2.x+1]) )

