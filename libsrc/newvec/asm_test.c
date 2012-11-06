
/* This is a test to see how the compiler passes args...
 * compile w/ -S option to dump assembler source
 *
 * NOTE:  supposedly the SSE instructions can now be obtained from gcc using pragma's...
 */

void vec_func(float *v1, float *v2, float *v3, long n)
{
	int n4,resid;
	n4=n/4;
	resid=n%4;
	while(n4--){
		v1[0] = v2[0] + v3[0];
		v1[1] = v2[1] + v3[1];
		v1[2] = v2[2] + v3[2];
		v1[3] = v2[3] + v3[3];
		v1+=4;
		v2+=4;
		v3+=4;
	}
	while(resid--){
		v1[0] = v2[0] + v3[0];
		v1++;
		v2++;
		v3++;
	}
}

void test_func()
{
	float f[8],g[8],h[8];

	vec_func(f,g,h,8);
}




