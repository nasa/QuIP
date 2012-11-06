/* These functions are implemented for all machine precisions */

KERN_PROT_MM_IND( vmaxi )
KERN_PROT_MM_IND( vmini )

KERN_PROT_MM_NOCC( vmaxg )
KERN_PROT_MM_NOCC( vming )

KERN_PROT_3V( vcmp )
KERN_PROT_3V( vibnd )
KERN_PROT_3V( vmax )
KERN_PROT_3V( vmin )
KERN_PROT_3V( rvadd )
KERN_PROT_3V( rvsub )
KERN_PROT_3V( rvmul )
KERN_PROT_3V( rvdiv )

//Two Vec functions
KERN_PROT_2V( rvmov )
KERN_PROT_2V( rvsqr )
//KERN_PROT_2V( rvrand )

//KERN_PROT_DBM_1S_( vset )
//KERN_PROT_1V_SCAL( vset )
KERN_PROT_1V_SCAL( rvset )
KERN_PROT_1V_2SCAL( vramp1d )
KERN_PROT_1V_3SCAL( vramp2d )

//Two Vec scalar functions
KERN_PROT_2V_SCAL( vscmp )
KERN_PROT_2V_SCAL( vscmp2 )
KERN_PROT_2V_SCAL( vsmax )
KERN_PROT_2V_SCAL( vsmin )
KERN_PROT_2V_SCAL( rvsadd )
KERN_PROT_2V_SCAL( rvssub )
KERN_PROT_2V_SCAL( rvsmul )
KERN_PROT_2V_SCAL( rvsdiv )
KERN_PROT_2V_SCAL( rvsdiv2 )

KERN_PROT_2V_PROJ( vsum )
KERN_PROT_2V_PROJ( vmaxv )
KERN_PROT_2V_PROJ( vminv )
//KERN_PROT_MM( vmaxv )
//KERN_PROT_MM( vminv )

KERN_PROT_3V_PROJ( vdot )

KERN_PROT_VVSLCT( vvv_slct )
KERN_PROT_VSSLCT( vvs_slct )
KERN_PROT_SSSLCT( vss_slct )

KERN_PROT_VVMAP( vvm_lt )
KERN_PROT_VVMAP( vvm_gt )
KERN_PROT_VVMAP( vvm_le )
KERN_PROT_VVMAP( vvm_ge )
KERN_PROT_VVMAP( vvm_eq )
KERN_PROT_VVMAP( vvm_ne )

KERN_PROT_VSMAP( vsm_lt )
KERN_PROT_VSMAP( vsm_gt )
KERN_PROT_VSMAP( vsm_le )
KERN_PROT_VSMAP( vsm_ge )
KERN_PROT_VSMAP( vsm_eq )
KERN_PROT_VSMAP( vsm_ne )

KERN_PROT_5V( vv_vv_lt )
KERN_PROT_5V( vv_vv_gt )
KERN_PROT_5V( vv_vv_le )
KERN_PROT_5V( vv_vv_ge )
KERN_PROT_5V( vv_vv_eq )
KERN_PROT_5V( vv_vv_ne )

KERN_PROT_4V_SCAL( vv_vs_lt )
KERN_PROT_4V_SCAL( vv_vs_gt )
KERN_PROT_4V_SCAL( vv_vs_le )
KERN_PROT_4V_SCAL( vv_vs_ge )
KERN_PROT_4V_SCAL( vv_vs_eq )
KERN_PROT_4V_SCAL( vv_vs_ne )

KERN_PROT_4V_SCAL( vs_vv_lt )
KERN_PROT_4V_SCAL( vs_vv_gt )
KERN_PROT_4V_SCAL( vs_vv_le )
KERN_PROT_4V_SCAL( vs_vv_ge )
KERN_PROT_4V_SCAL( vs_vv_eq )
KERN_PROT_4V_SCAL( vs_vv_ne )

KERN_PROT_3V_2SCAL( vs_vs_lt )
KERN_PROT_3V_2SCAL( vs_vs_gt )
KERN_PROT_3V_2SCAL( vs_vs_le )
KERN_PROT_3V_2SCAL( vs_vs_ge )
KERN_PROT_3V_2SCAL( vs_vs_eq )
KERN_PROT_3V_2SCAL( vs_vs_ne )

KERN_PROT_3V_2SCAL( ss_vv_lt )
KERN_PROT_3V_2SCAL( ss_vv_gt )
KERN_PROT_3V_2SCAL( ss_vv_le )
KERN_PROT_3V_2SCAL( ss_vv_ge )
KERN_PROT_3V_2SCAL( ss_vv_eq )
KERN_PROT_3V_2SCAL( ss_vv_ne )

KERN_PROT_2V_3SCAL( ss_vs_lt )
KERN_PROT_2V_3SCAL( ss_vs_gt )
KERN_PROT_2V_3SCAL( ss_vs_le )
KERN_PROT_2V_3SCAL( ss_vs_ge )
KERN_PROT_2V_3SCAL( ss_vs_eq )
KERN_PROT_2V_3SCAL( ss_vs_ne )

