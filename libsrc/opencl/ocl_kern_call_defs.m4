/* ocl_kern_call_defs.m4 BEGIN */

// These definitions are expanded during C compilation, but are not
// compiled - they are stored to strings and compiled on the fly.

// We have a problem because some gpu's can't handle double (e.g. Iris Pro)
// The typedefs here cause problems, even when the kernel doesn't use them.
// So we take them out and include only when needed...

// BUG the structure definitions have to be kept in sync with the definitions in the header files...
// How could we write this in a way that would insure this?

define(`KERNEL_FUNC_PRELUDE',`					\
								\
typedef unsigned char u_char;					\
typedef unsigned short u_short;					\
typedef unsigned long uint64_t;					\
typedef unsigned int uint32_t;					\
typedef long int64_t;						\
typedef uint32_t dimension_t;					\
typedef int int32_t;						\
typedef int index_type;						\
/*typedef struct { int x; int y; int z; } dim3 ;*/		\
typedef struct { int d5_dim[5]; } dim5 ;			\
typedef unsigned long bitmap_word;				\
typedef struct {						\
	uint32_t	word_offset;				\
	uint32_t	first_indices[5];			\
	uint64_t	first_bit_num;				\
	bitmap_word	valid_bits;				\
} Bitmap_GPU_Word_Info;						\
typedef struct {						\
	uint32_t			n_bitmap_words;		\
	uint32_t			total_size;		\
	int32_t				next_word_idx;		\
	int32_t				this_word_idx;		\
	int32_t				last_word_idx;		\
	Bitmap_GPU_Word_Info 	word_tbl[1];			\
} Bitmap_GPU_Info;						\
EXTRA_PRELUDE(type_code)					\
')

define(`EXTRA_PRELUDE',EXTRA_PRELUDE_$1)

define(`EXTRA_PRELUDE_by',`')
define(`EXTRA_PRELUDE_in',`')
define(`EXTRA_PRELUDE_di',`')
define(`EXTRA_PRELUDE_li',`')
define(`EXTRA_PRELUDE_uby',`')
define(`EXTRA_PRELUDE_uin',`')
define(`EXTRA_PRELUDE_udi',`')
define(`EXTRA_PRELUDE_uli',`')
define(`EXTRA_PRELUDE_ubyin',`')
define(`EXTRA_PRELUDE_uindi',`')
define(`EXTRA_PRELUDE_inby',`')
define(`EXTRA_PRELUDE_bit',`')
define(`EXTRA_PRELUDE_spdp',EXTRA_PRELUDE_sp EXTRA_PRELUDE_dp)

define(`EXTRA_PRELUDE_sp',`								\
											\
typedef struct { float re; float im; } SP_Complex;					\
typedef struct { float re; float _i; float _j; float _k; } SP_Quaternion;		\
')

define(`EXTRA_PRELUDE_dp',`								\
											\
typedef struct { double re; double im; } DP_Complex;					\
typedef struct { double re; double _i; double _j; double _k; } DP_Quaternion;		\
')


/* ocl_kern_call_defs.m4 END */

