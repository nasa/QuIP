
//#include <Random123/philox.h>
//#include <Random123/threefry.h>
#include "threefry.h"

__kernel void test_fast_vuni(__global float *a) {
	union {
	    threefry4x32_ctr_t c;
	    int4 i;
	} u;
	unsigned tid = get_global_id(0);

	threefry4x32_key_t k = {{tid, 0xdecafbad, 0xfacebead, 0x12345678}};
	threefry4x32_ctr_t c = {{0, 0xf00dcafe, 0xdeadbeef, 0xbeeff00d}};
	u.c = threefry4x32(c, k);
	long x1 = u.i.x, y1 = u.i.y;
	long x2 = u.i.z, y2 = u.i.w;
	a[tid] = (float) x1 / 0xffffffff;
}

