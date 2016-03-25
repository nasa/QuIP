
//Three vec functions
// Maybe we could do this from include/veclib?

MENU_CALL_3V( vcmp )
MENU_CALL_3V( vibnd )
MENU_CALL_3V_S( vbnd )
MENU_CALL_3V_S( vmaxm )
MENU_CALL_3V_S( vminm )
MENU_CALL_3V( vmax )
MENU_CALL_3V( vmin )
MENU_CALL_3V_RC( vadd )
MENU_CALL_3V( rvsub )
MENU_CALL_3V( rvmul )
MENU_CALL_3V( rvdiv )

MENU_CALL_3V_I( vand )
MENU_CALL_3V_I( vnand )
MENU_CALL_3V_I( vor )
MENU_CALL_3V_I( vxor )
MENU_CALL_3V_I( vmod )
MENU_CALL_3V_I( vshr )
MENU_CALL_3V_I( vshl )

MENU_CALL_2V_I( vtolower )
MENU_CALL_2V_I( vtoupper )
MENU_CALL_2V_I( vislower )
MENU_CALL_2V_I( visupper )
MENU_CALL_2V_I( visalnum )
MENU_CALL_2V_I( visdigit )
MENU_CALL_2V_I( visalpha )
MENU_CALL_2V_I( visspace )
MENU_CALL_2V_I( viscntrl )
MENU_CALL_2V_I( visblank )

// float only
MENU_CALL_3V_F( vatan2 )
MENU_CALL_2V_F( rvexp )
MENU_CALL_3V_F( rvpow )
//MENU_CALL_3V_F( vpow )

//Two vec functions
MENU_CALL_2V_S( vsign )
MENU_CALL_2V_S( vabs )
MENU_CALL_2V_S( rvneg )

MENU_CALL_2V_I( vnot )
MENU_CALL_2V_I( vcomp )

MENU_CALL_1S_1_B( rvset )
MENU_CALL_1S_1V_F( cvset )
MENU_CALL_1S_1V_RC(vset)

MENU_CALL_2V( rvmov )
MENU_CALL_2V( rvsqr )
#ifndef BUILD_FOR_GPU
MENU_CALL_2V( rvrand )
#endif // BUILD_FOR_GPU

// float only
/*
..MENU_CALL_2V_F( vj0 )
..MENU_CALL_2V_F( vj1 )
*/
MENU_CALL_2V_F( vrint)
MENU_CALL_2V_F( vfloor)
MENU_CALL_2V_F( vround)
MENU_CALL_2V_F( vceil)
MENU_CALL_2V_F( vlog)
MENU_CALL_2V_F( vlog10)
MENU_CALL_2V_F( vatan)
MENU_CALL_2V_F( vtan)
MENU_CALL_2V_F( vcos)
MENU_CALL_2V_F( verf)
MENU_CALL_2V_F( vacos)
MENU_CALL_2V_F( vsin)
MENU_CALL_2V_F( vasin)
MENU_CALL_2V_F( vsqrt)

//Two vec scalar functions
MENU_CALL_2V_SCALAR( vscmp)
MENU_CALL_2V_SCALAR( vscmp2)
MENU_CALL_2V_SCALAR_S( vsmnm)
MENU_CALL_2V_SCALAR_S( vsmxm)
MENU_CALL_2V_SCALAR_S( viclp)
MENU_CALL_2V_SCALAR_S( vclip)
MENU_CALL_2V_SCALAR( vsmin)
MENU_CALL_2V_SCALAR( vsmax)
MENU_CALL_2V_SCALAR( rvsadd)
MENU_CALL_2V_SCALAR( rvssub)
MENU_CALL_2V_SCALAR( rvsmul)
MENU_CALL_2V_SCALAR( rvsdiv)
MENU_CALL_2V_SCALAR( rvsdiv2)

MENU_CALL_2V_SCALAR_I( vsand)
//MENU_CALL_2V_SCALAR_I( vsnand)
MENU_CALL_2V_SCALAR_I( vsor)
MENU_CALL_2V_SCALAR_I( vsxor)
MENU_CALL_2V_SCALAR_I( vsmod)
MENU_CALL_2V_SCALAR_I( vsmod2)
MENU_CALL_2V_SCALAR_I( vsshr)
MENU_CALL_2V_SCALAR_I( vsshr2 )
MENU_CALL_2V_SCALAR_I( vsshl)
MENU_CALL_2V_SCALAR_I( vsshl2)

// float only
// now pow, atan2 on gpu?
MENU_CALL_2V_SCALAR_F( vsatan2)
MENU_CALL_2V_SCALAR_F( vsatan22)
MENU_CALL_2V_SCALAR_F( vspow)
MENU_CALL_2V_SCALAR_F( vspow2)

MENU_CALL_2V_CONV( convert)


