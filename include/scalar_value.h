#ifndef _SCALAR_VALUE_H_
#define _SCALAR_VALUE_H_

union scalar_value {
	double		u_d;	// this is first to be the default initializer
	long double	u_ld;
	char		u_b;
	short		u_s;
	int32_t		u_l;
	int64_t		u_ll;
	unsigned char	u_ub;
	unsigned short	u_us;
	uint32_t	u_ul;
	uint64_t	u_ull;
	float		u_f;
	float		u_color_comp[3];
	SP_Color	u_color;
	/* Do we need both of these??? */
	float		u_fc[2];
	SP_Complex	u_spc;
	float		u_fq[4];
	SP_Quaternion	u_spq;

	double		u_dc[2];
	DP_Complex	u_dpc;
	double		u_dq[4];
	DP_Quaternion	u_dpq;

	long double	u_lc[2];
	LP_Complex	u_lpc;
	long double	u_lq[4];
	LP_Quaternion	u_lpq;

	bitmap_word	u_bit;	/* should be boolean type... */
	void *		u_vp;	// for string type
};

#define SVAL_FLOAT(svp)		(svp)->u_f
#define SVAL_STD(svp)		(svp)->std_scalar
#define SVAL_STD_CPX(svp)	(svp)->std_cpx_scalar
#define SVAL_STD_QUAT(svp)	(svp)->std_quat_scalar

#endif /* ! _SCALAR_VALUE_H_ */

