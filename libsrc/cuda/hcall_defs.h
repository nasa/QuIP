
#define _KERN_PROT_5V(n,c)		GENERIC_HOST_CALL(n,c,,,,5)
#define _KERN_PROT_4V_SCAL(n,c)		GENERIC_HOST_CALL(n,c,,,1S_,4)
#define _KERN_PROT_3V_2SCAL(n,c)	GENERIC_HOST_CALL(n,c,,,2S_,3)
#define _KERN_PROT_2V_3SCAL(n,c)	GENERIC_HOST_CALL(n,c,,,3S_,2)
#define _KERN_PROT_3V(n,c)		GENERIC_HOST_CALL(n,c,,,,3)
#define _KERN_PROT_CPX_3V(n,c)		GENERIC_HOST_CALL(n,c,,CPX_,,3)
#define _KERN_PROT_2V(n,c)		GENERIC_HOST_CALL(n,c,,,,2)
#define _KERN_PROT_1V_2SCAL(n,c)	GENERIC_HOST_CALL(n,c,,,2S_,1)
#define _KERN_PROT_2V_SCAL(n,c)		GENERIC_HOST_CALL(n,c,,,1S_,2)
// this is vramp2d
#define _KERN_PROT_1V_3SCAL(n,c)	SLOW_HOST_CALL(n,c,,,3S_,1)
#define _KERN_PROT_2V_MIXED(n,c)	GENERIC_HOST_CALL(n,c,,RC_,,2)
#define _KERN_PROT_VVSLCT(n,c)		GENERIC_HOST_CALL(n,c,SBM_,,,3)
#define _KERN_PROT_VSSLCT(n,c)		GENERIC_HOST_CALL(n,c,SBM_,,1S_,2)
#define _KERN_PROT_SSSLCT(n,c)		GENERIC_HOST_CALL(n,c,SBM_,,2S_,1)
#define _KERN_PROT_VVMAP(n,c)		GENERIC_HOST_CALL(n,c,DBM_,,,2SRCS)
// vsm_gt etc
#define _KERN_PROT_VSMAP(n,c)		GENERIC_HOST_CALL(n,c,DBM_,,1S_,1SRC)
// this is vset
#define _KERN_PROT_1V_SCAL(n,c)		GENERIC_HOST_CALL(n,c,,,1S_,1)
#define _KERN_PROT_CPX_1V_SCAL(n,c)	GENERIC_HOST_CALL(n,c,,CPX_,1S_,1)
// this is bit_vset
#define _KERN_PROT_DBM_1S_(n,c)		GENERIC_HOST_CALL(n,c,DBM_,,1S_,)



/* FIXME still need to convert these to generic macros if possible */

#define _KERN_PROT_MM(name,code,type)		H_CALL_MM( h_##code##_##name , g_##code##_##name ,type)
#define _KERN_PROT_MM_IND(name,code,type)	H_CALL_MM_IND( h_##code##_##name , g_##code##_##name ,type)
#define _KERN_PROT_MM_NOCC(name,code,type)	H_CALL_MM_NOCC( h_##code##_##name , g_##code##_##name ,type)
#define _KERN_PROT_2V_PROJ(name,code,type)	H_CALL_PROJ_2V( h_##code##_##name , g_##code##_##name ,type)
#define _KERN_PROT_3V_PROJ(name,code,type)	H_CALL_PROJ_3V( h_##code##_##name , g_##code##_##name ,type)
#ifdef FOOBAR
#endif /* FOOBAR */


#define KERN_PROT_5V(name)		_KP_5V(name, type_code , std_type )
#define KERN_PROT_4V_SCAL(name)	_KP_4V_SCAL(name, type_code , std_type )
#define KERN_PROT_3V_2SCAL(name)	_KP_3V_2SCAL(name, type_code , std_type )
#define KERN_PROT_2V_3SCAL(name)	_KP_2V_3SCAL(name, type_code , std_type )
#define KERN_PROT_3V(name)		_KP_3V(name, type_code , std_type )
#define KERN_PROT_CPX_3V(name)		_KP_CPX_3V(name, type_code , std_cpx )
#define KERN_PROT_2V(name)		_KP_2V(name, type_code , std_type )
#define KERN_PROT_2V_MIXED(name)	_KP_2V_MIXED(name, type_code , std_type , std_cpx )
#define KERN_PROT_1V_SCAL(name)	_KP_1V_SCAL(name, type_code , std_type )
#define KERN_PROT_CPX_1V_SCAL(name)	_KP_CPX_1V_SCAL(name, type_code , std_type )
#define KERN_PROT_DBM_1S_(name)	_KP_DBM_1S_(name, type_code , std_type )
#define KERN_PROT_1V_2SCAL(name)	_KP_1V_2SCAL(name, type_code , std_type )
#define KERN_PROT_1V_3SCAL(name)	_KP_1V_3SCAL(name, type_code , std_type )
#define KERN_PROT_2V_SCAL(name)	_KP_2V_SCAL(name, type_code , std_type )
#define KERN_PROT_MM(name)		_KP_MM(name, type_code , std_type )
#define KERN_PROT_MM_IND(name)	_KP_MM_IND(name, type_code , std_type )
#define KERN_PROT_MM_NOCC(name)	_KP_MM_NOCC(name, type_code , std_type )
#define KERN_PROT_2V_PROJ(name)	_KP_2V_PROJ(name, type_code , std_type )
#define KERN_PROT_3V_PROJ(name)	_KP_3V_PROJ(name, type_code , std_type )
#define KERN_PROT_VVSLCT(name)	_KP_VVSLCT(name, type_code , std_type )
#define KERN_PROT_VSSLCT(name)	_KP_VSSLCT(name, type_code , std_type )
#define KERN_PROT_SSSLCT(name)	_KP_SSSLCT(name, type_code , std_type )
#define KERN_PROT_VVMAP(name)		_KP_VVMAP(name, type_code , std_type )
#define KERN_PROT_VSMAP(name)		_KP_VSMAP(name, type_code , std_type )

/* UGLY - type arg is not needed */
#define _KP_5V( n , c , t )		_KERN_PROT_5V( n , c )
#define _KP_4V_SCAL( n , c , t )	_KERN_PROT_4V_SCAL( n , c )
#define _KP_3V_2SCAL( n , c , t )	_KERN_PROT_3V_2SCAL( n , c )
#define _KP_2V_3SCAL( n , c , t )	_KERN_PROT_2V_3SCAL( n , c )
#define _KP_3V( n , c , t )		_KERN_PROT_3V( n , c )
#define _KP_CPX_3V( n , c , t )		_KERN_PROT_CPX_3V( n , c )
#define _KP_2V( n , c , t )		_KERN_PROT_2V( n , c )
#define _KP_2V_MIXED( n , c , t , ct )	_KERN_PROT_2V_MIXED( n , c )
#define _KP_1V_SCAL( n , c , t )	_KERN_PROT_1V_SCAL( n , c )
#define _KP_CPX_1V_SCAL( n , c , t )	_KERN_PROT_CPX_1V_SCAL( n , c )
#define _KP_DBM_1S_( n , c , t )	_KERN_PROT_DBM_1S_( n , c )
#define _KP_1V_2SCAL( n , c , t )	_KERN_PROT_1V_2SCAL( n , c )
#define _KP_1V_3SCAL( n , c , t )	_KERN_PROT_1V_3SCAL( n , c )
#define _KP_2V_SCAL( n , c , t )	_KERN_PROT_2V_SCAL( n , c )
#define _KP_VVSLCT( n , c , t )		_KERN_PROT_VVSLCT( n , c )
#define _KP_VSSLCT( n , c , t )		_KERN_PROT_VSSLCT( n , c )
#define _KP_SSSLCT( n , c , t )		_KERN_PROT_SSSLCT( n , c )
#define _KP_VVMAP( n , c , t )		_KERN_PROT_VVMAP( n , c )
#define _KP_VSMAP( n , c , t )		_KERN_PROT_VSMAP( n , c )


#define _KP_2V_PROJ( n , c , t )	_KERN_PROT_2V_PROJ( n , c , t )
#define _KP_3V_PROJ( n , c , t )	_KERN_PROT_3V_PROJ( n , c , t )
#define _KP_MM( n , c , t )		_KERN_PROT_MM( n , c , t )
#define _KP_MM_IND( n , c , t )		_KERN_PROT_MM_IND( n , c , t )
#define _KP_MM_NOCC( n , c , t )	_KERN_PROT_MM_NOCC( n , c , t )

