
// BUG - get rid of these globals!
// also not thread-safe!

//extern int nlevels;
#define MAX_QUANT_LEVELS	256
//extern float quant_level[MAXLEVELS];
//extern int _ncols, _nrows;
#define MAXCOLS	512	// columns or colors???

typedef struct dither_params {
	int	dp_nlevels;
	int	dp_n_columns;
	int	dp_n_rows;
	float	dp_quant_level[MAX_QUANT_LEVELS];
	float	*dp_desired[3];
	int	dp_thebest[3];
} Dither_Params;

extern Dither_Params dp1; // BUG avoid static global

#define nlevels		(dp1.dp_nlevels)
#define quant_level	(dp1.dp_quant_level)
#define _ncols		(dp1.dp_n_columns)
#define _nrows		(dp1.dp_n_rows)
#define desired		(dp1.dp_desired)
#define thebest		(dp1.dp_thebest)

