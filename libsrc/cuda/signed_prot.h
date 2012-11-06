/* These functions are implemented for all signed precisions */

//Two Vec functions
KERN_PROT_2V( vsign )
KERN_PROT_2V( vabs )
KERN_PROT_2V( rvneg )

KERN_PROT_MM( vmxmv )
KERN_PROT_MM( vmnmv )
KERN_PROT_MM_IND( vmxmi )
KERN_PROT_MM_IND( vmnmi )
KERN_PROT_MM_NOCC( vmxmg )
KERN_PROT_MM_NOCC( vmnmg )

KERN_PROT_3V( vbnd )
KERN_PROT_3V( vmaxm )
KERN_PROT_3V( vminm )
KERN_PROT_2V_SCAL( vsmnm )
KERN_PROT_2V_SCAL( vsmxm )
KERN_PROT_2V_SCAL( viclp )
KERN_PROT_2V_SCAL( vclip )
