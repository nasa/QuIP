#include "quip_config.h"
#include <stdio.h>
#include "quip_prot.h"
#include "function.h"
#include "rn.h"
#include "veclib_api.h"
#include "my_vl2.h"
#include "veclib/vl2_port.h"
#include "veclib/vl2_veclib_prot.h"

#include "vl2_kernels.c"

// That declares the kernels - now the host-side functions
// But for cpu, the kernels ARE the host functions...
// So we only need the untyped functions...

#include "vl2_host_funcs.c"

#include "vl2_typtbl.c"


