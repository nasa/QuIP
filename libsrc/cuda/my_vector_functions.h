// global vars
extern int max_threads_per_block;

extern void g_fwdfft(QSP_ARG_DECL  Data_Obj *dst_dp, Data_Obj *src1_dp);

// first do the type-less host master calls

#define HOST_PROTOTYPE( name )					\
								\
	extern void g_##name( Vec_Obj_Args *oap );


#define KERN_PROT_5V( name )		HOST_PROTOTYPE( name )
#define KERN_PROT_4V_SCAL( name )	HOST_PROTOTYPE( name )
#define KERN_PROT_3V_2SCAL( name )	HOST_PROTOTYPE( name )
#define KERN_PROT_2V_3SCAL( name )	HOST_PROTOTYPE( name )
#define KERN_PROT_3V( name )		HOST_PROTOTYPE( name )
#define KERN_PROT_CPX_3V( name )	HOST_PROTOTYPE( name )
#define KERN_PROT_2V( name )		HOST_PROTOTYPE( name )
#define KERN_PROT_2V_MIXED( name )	HOST_PROTOTYPE( name )
#define KERN_PROT_1V_SCAL( name )	HOST_PROTOTYPE( name )
#define KERN_PROT_CPX_1V_SCAL( name )	HOST_PROTOTYPE( name )
#define KERN_PROT_DBM_1S_( name )	HOST_PROTOTYPE( name )
#define KERN_PROT_1V_2SCAL( name )	HOST_PROTOTYPE( name )
#define KERN_PROT_1V_3SCAL( name )	HOST_PROTOTYPE( name )
#define KERN_PROT_2V_SCAL( name )	HOST_PROTOTYPE( name )
#define KERN_PROT_MM( name )		HOST_PROTOTYPE( name )
#define KERN_PROT_MM_IND( name )	HOST_PROTOTYPE( name )
#define KERN_PROT_MM_NOCC( name )	HOST_PROTOTYPE( name )
#define KERN_PROT_2V_PROJ( name )	HOST_PROTOTYPE( name )
#define KERN_PROT_3V_PROJ( name )	HOST_PROTOTYPE( name )
#define KERN_PROT_VVSLCT( name )	HOST_PROTOTYPE( name )
#define KERN_PROT_VSSLCT( name )	HOST_PROTOTYPE( name )
#define KERN_PROT_SSSLCT( name )	HOST_PROTOTYPE( name )
#define KERN_PROT_VVMAP( name )		HOST_PROTOTYPE( name )
#define KERN_PROT_VSMAP( name )		HOST_PROTOTYPE( name )

#include "float_prot.h"
#include "all_prot.h"
#include "signed_prot.h"
#include "int_prot.h"
#include "undefs.h"

// Here we have all the funcs w/ real & complex versions
HOST_PROTOTYPE(vadd)
HOST_PROTOTYPE(vset)

#define HOST_TYPED_PROTOTYPE( name , ty )				\
									\
extern void h_##ty##_##name( Vec_Obj_Args *oap );

/* curand.cpp */
extern void g_vuni(Vec_Obj_Args *oap);


/* Here are the conversions */
extern void h_sp2by( Vec_Obj_Args *oap );
extern void h_sp2in( Vec_Obj_Args *oap );
extern void h_sp2di( Vec_Obj_Args *oap );
extern void h_sp2li( Vec_Obj_Args *oap );
extern void h_sp2uby( Vec_Obj_Args *oap );
extern void h_sp2uin( Vec_Obj_Args *oap );
extern void h_sp2udi( Vec_Obj_Args *oap );
extern void h_sp2uli( Vec_Obj_Args *oap );
extern void h_sp2dp( Vec_Obj_Args *oap );

extern void h_dp2by( Vec_Obj_Args *oap );
extern void h_dp2in( Vec_Obj_Args *oap );
extern void h_dp2di( Vec_Obj_Args *oap );
extern void h_dp2li( Vec_Obj_Args *oap );
extern void h_dp2uby( Vec_Obj_Args *oap );
extern void h_dp2uin( Vec_Obj_Args *oap );
extern void h_dp2udi( Vec_Obj_Args *oap );
extern void h_dp2uli( Vec_Obj_Args *oap );
extern void h_dp2sp( Vec_Obj_Args *oap );

extern void h_by2in( Vec_Obj_Args *oap );
extern void h_by2di( Vec_Obj_Args *oap );
extern void h_by2li( Vec_Obj_Args *oap );
extern void h_by2uby( Vec_Obj_Args *oap );
extern void h_by2uin( Vec_Obj_Args *oap );
extern void h_by2udi( Vec_Obj_Args *oap );
extern void h_by2uli( Vec_Obj_Args *oap );
extern void h_by2sp( Vec_Obj_Args *oap );
extern void h_by2dp( Vec_Obj_Args *oap );

extern void h_in2by( Vec_Obj_Args *oap );
extern void h_in2di( Vec_Obj_Args *oap );
extern void h_in2li( Vec_Obj_Args *oap );
extern void h_in2uby( Vec_Obj_Args *oap );
extern void h_in2uin( Vec_Obj_Args *oap );
extern void h_in2udi( Vec_Obj_Args *oap );
extern void h_in2uli( Vec_Obj_Args *oap );
extern void h_in2sp( Vec_Obj_Args *oap );
extern void h_in2dp( Vec_Obj_Args *oap );

extern void h_di2by( Vec_Obj_Args *oap );
extern void h_di2in( Vec_Obj_Args *oap );
extern void h_di2li( Vec_Obj_Args *oap );
extern void h_di2uby( Vec_Obj_Args *oap );
extern void h_di2uin( Vec_Obj_Args *oap );
extern void h_di2udi( Vec_Obj_Args *oap );
extern void h_di2uli( Vec_Obj_Args *oap );
extern void h_di2sp( Vec_Obj_Args *oap );
extern void h_di2dp( Vec_Obj_Args *oap );

extern void h_li2by( Vec_Obj_Args *oap );
extern void h_li2in( Vec_Obj_Args *oap );
extern void h_li2di( Vec_Obj_Args *oap );
extern void h_li2uby( Vec_Obj_Args *oap );
extern void h_li2uin( Vec_Obj_Args *oap );
extern void h_li2udi( Vec_Obj_Args *oap );
extern void h_li2uli( Vec_Obj_Args *oap );
extern void h_li2sp( Vec_Obj_Args *oap );
extern void h_li2dp( Vec_Obj_Args *oap );

extern void h_uby2by( Vec_Obj_Args *oap );
extern void h_uby2in( Vec_Obj_Args *oap );
extern void h_uby2di( Vec_Obj_Args *oap );
extern void h_uby2li( Vec_Obj_Args *oap );
extern void h_uby2uin( Vec_Obj_Args *oap );
extern void h_uby2udi( Vec_Obj_Args *oap );
extern void h_uby2uli( Vec_Obj_Args *oap );
extern void h_uby2sp( Vec_Obj_Args *oap );
extern void h_uby2dp( Vec_Obj_Args *oap );

extern void h_uin2by( Vec_Obj_Args *oap );
extern void h_uin2in( Vec_Obj_Args *oap );
extern void h_uin2di( Vec_Obj_Args *oap );
extern void h_uin2li( Vec_Obj_Args *oap );
extern void h_uin2uby( Vec_Obj_Args *oap );
extern void h_uin2udi( Vec_Obj_Args *oap );
extern void h_uin2uli( Vec_Obj_Args *oap );
extern void h_uin2sp( Vec_Obj_Args *oap );
extern void h_uin2dp( Vec_Obj_Args *oap );

extern void h_udi2by( Vec_Obj_Args *oap );
extern void h_udi2in( Vec_Obj_Args *oap );
extern void h_udi2di( Vec_Obj_Args *oap );
extern void h_udi2li( Vec_Obj_Args *oap );
extern void h_udi2uby( Vec_Obj_Args *oap );
extern void h_udi2uin( Vec_Obj_Args *oap );
extern void h_udi2uli( Vec_Obj_Args *oap );
extern void h_udi2sp( Vec_Obj_Args *oap );
extern void h_udi2dp( Vec_Obj_Args *oap );

extern void h_uli2by( Vec_Obj_Args *oap );
extern void h_uli2in( Vec_Obj_Args *oap );
extern void h_uli2di( Vec_Obj_Args *oap );
extern void h_uli2li( Vec_Obj_Args *oap );
extern void h_uli2uby( Vec_Obj_Args *oap );
extern void h_uli2uin( Vec_Obj_Args *oap );
extern void h_uli2udi( Vec_Obj_Args *oap );
extern void h_uli2sp( Vec_Obj_Args *oap );
extern void h_uli2dp( Vec_Obj_Args *oap );


/* First do the float functions */
#define std_type float
#define type_code sp
#include "prot_defs.h"
#include "all_prot.h"
#include "signed_prot.h"
#include "float_prot.h"
#include "undefs.h"



/* Now do the double functions */
#define std_type double
#define type_code dp
#include "prot_defs.h"
#include "all_prot.h"
#include "signed_prot.h"
#include "float_prot.h"
#include "undefs.h"

/* Now do signed byte */
#define std_type char
#define type_code by
#include "prot_defs.h"
#include "all_prot.h"
#include "signed_prot.h"
#include "int_prot.h"
#include "undefs.h"

/* Now do short int */
#define std_type short
#define type_code in
#include "prot_defs.h"
#include "all_prot.h"
#include "signed_prot.h"
#include "int_prot.h"
#include "undefs.h"


/* Now do int32 */
#define std_type int32_t
#define type_code di
#include "prot_defs.h"
#include "all_prot.h"
#include "signed_prot.h"
#include "int_prot.h"
#include "undefs.h"


/* Now do int64 */
#define std_type int64_t
#define type_code li
#include "prot_defs.h"
#include "all_prot.h"
#include "signed_prot.h"
#include "int_prot.h"
#include "undefs.h"



/* Now do unsigned byte */
#define std_type u_char
#define type_code uby
#include "prot_defs.h"
#include "all_prot.h"
#include "int_prot.h"
#include "undefs.h"


/* Now do u_short int */
#define std_type u_short
#define type_code uin
#include "prot_defs.h"
#include "all_prot.h"
#include "int_prot.h"
#include "undefs.h"

/* Now do uint32 */
#define std_type uint32_t
#define type_code udi
#include "prot_defs.h"
#include "all_prot.h"
#include "int_prot.h"
#include "undefs.h"


/* Now do uint64 */
#define std_type uint64_t
#define type_code uli
#include "prot_defs.h"
#include "all_prot.h"
#include "int_prot.h"
#include "undefs.h"

/* finally bit */
#define std_type bitmap_word
#define type_code bit
#include "prot_defs.h"
#include "bit_prot.h"
#include "undefs.h"

