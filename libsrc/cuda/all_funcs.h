
// SP stuff

#define std_type float
#define std_cpx SP_Complex
#define dest_type float
#define dest_cpx SP_Complex
#define type_code sp
#include "hcall_defs.h"
#define std_scalar	u_f
#define std_cpx_scalar	u_spc
#include "float_prot.h"
#include "all_prot.h"
#include "signed_prot.h"
#include "undefs.h"

// DP stuff

#define std_type double
#define std_cpx DP_Complex
#define dest_type double
#define dest_cpx DP_Complex
#define type_code dp
#include "hcall_defs.h"
#define std_scalar	u_d
#define std_cpx_scalar	u_dpc
#include "float_prot.h"
#include "all_prot.h"
#include "signed_prot.h"
#include "undefs.h"

// BY stuff

#define std_type char
#define dest_type char
#define type_code by
#include "hcall_defs.h"
#define std_scalar	u_b
#include "all_prot.h"
#include "signed_prot.h"
#include "int_prot.h"
#include "undefs.h"


// IN stuff

#define std_type short
#define dest_type short
#define type_code in
#include "hcall_defs.h"
#define std_scalar	u_s
#include "all_prot.h"
#include "signed_prot.h"
#include "int_prot.h"
#include "undefs.h"


// DI stuff

#define std_type int32_t
#define dest_type int32_t
#define type_code di
#include "hcall_defs.h"
#define std_scalar	u_l
#include "all_prot.h"
#include "signed_prot.h"
#include "int_prot.h"
#include "undefs.h"


// LI stuff

#define std_type int64_t
#define dest_type int64_t
#define type_code li
#include "hcall_defs.h"
#define std_scalar	u_ll
#include "all_prot.h"
#include "signed_prot.h"
#include "int_prot.h"
#include "undefs.h"


// UBY stuff

#define std_type u_char
#define dest_type u_char
#define type_code uby
#include "hcall_defs.h"
#define std_scalar	u_ub
#include "all_prot.h"
#include "int_prot.h"
#include "undefs.h"

// UIN stuff

#define std_type u_short
#define dest_type u_short
#define type_code uin
#include "hcall_defs.h"
#define std_scalar	u_us
#include "all_prot.h"
#include "int_prot.h"
#include "undefs.h"


// UDI stuff

#define std_type uint32_t
#define dest_type uint32_t
#define type_code udi
#include "hcall_defs.h"
#define std_scalar	u_ul
#include "all_prot.h"
#include "int_prot.h"
#include "undefs.h"


// ULI stuff

#define std_type uint64_t
#define dest_type uint64_t
#define type_code uli
#include "hcall_defs.h"
#define std_scalar	u_ull
#include "all_prot.h"
#include "int_prot.h"
#include "undefs.h"

// bitmap set

#define std_type bitmap_word
#define dest_type bitmap_word
#define type_code bit
#include "hcall_defs.h"
#define std_scalar bitmap_scalar
#include "bit_prot.h"
#include "undefs.h"

