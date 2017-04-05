dnl	Here we need to redefine some stuff!

/* all_bit_kernels.m4, before definition:  bit_precision = BIT_PRECISION */
define(`BIT_PRECISION',`')
/* all_bit_kernels.m4, after definition:  bit_precision = BIT_PRECISION */

my_include(`../../include/veclib/fast_defs.m4')

/* all_bit_kernels.m4 including gen_bit_calls.m4, bit_precision = BIT_PRECISION */
my_include(`../../include/veclib/gen_bit_calls.m4')

dnl  BUG?  what about elen and slen???
ifdef(`BUILD_FOR_CUDA',`
my_include(`../../include/veclib/flen_defs.m4')
my_include(`../../include/veclib/gen_bit_calls.m4')
',`')

my_include(`../../include/veclib/eqsp_defs.m4')
my_include(`../../include/veclib/gen_bit_calls.m4')

my_include(`../../include/veclib/slow_defs.m4')
my_include(`../../include/veclib/gen_bit_calls.m4')

/* all_bit_kernels.m4, before undefinition:  bit_precision = BIT_PRECISION */
undefine(`BIT_PRECISION')
/* all_bit_kernels.m4, after undefinition:  bit_precision = BIT_PRECISION */

