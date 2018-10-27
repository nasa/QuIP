#include "quip_config.h"

#include "vec_util.h"

#define BEGIN_CLEAR(level)								\
											\
	for(var[level]=0;var[level] < OBJ_TYPE_DIM(dp,level);var[level]++){		\
		os[level] = var[level] * OBJ_TYPE_INC(dp,level);

void img_clear3d(Data_Obj *dp)
{
	dimension_t var[N_DIMENSIONS];
	incr_t os[N_DIMENSIONS];
	float *ptr;

	ptr = (float *)OBJ_DATA_PTR(dp);

	BEGIN_CLEAR(3)
		BEGIN_CLEAR(2)
			BEGIN_CLEAR(1)
				BEGIN_CLEAR(0)
					*( ptr + os[0] + os[1] + os[2] + os[3] ) = 0.0;

				}
			}
		}
	}
}

#ifdef NOWRAP

#define CHECK_EDGE(level)									\
												\
		if( os[level] < 0 || os[level] >= (incr_t) OBJ_TYPE_DIM(image_dp,level) ) continue;
#else

#define CHECK_EDGE(level)									\
												\
		if( os[level] < 0 ) os[level]+=OBJ_TYPE_DIM(image_dp,level);			\
		else if( os[level] >= OBJ_TYPE_DIM(image_dp,level) )				\
			os[level]-=OBJ_TYPE_DIM(image_dp,level);				\

#endif /* NOWRAP */

#define BEGIN_ITERATION(level)									\
												\
	var[level] = (incr_t ) OBJ_TYPE_DIM(ir_dp, level );					\
	while( var[level] -- ){									\
		os[level] = position[level] + var[level] - OBJ_TYPE_DIM(ir_dp,level)/2;	\
		CHECK_EDGE(level)								\
		os[level] *= OBJ_TYPE_INC(image_dp,level);					\
		iros[level] = var[level] * OBJ_TYPE_INC(ir_dp,level);				\



void add_impulse3d(double amp,Data_Obj *image_dp,Data_Obj *ir_dp,posn_t *position)
{
	float *image_ptr, *irptr;
	incr_t offset,ir_offset;		/* offsets into image */
	incr_t var[N_DIMENSIONS];
	incr_t os[N_DIMENSIONS], iros[N_DIMENSIONS];	/* offsets into impulse response */


	image_ptr = (float *) OBJ_DATA_PTR(image_dp);
	irptr = (float *) OBJ_DATA_PTR(ir_dp);

	BEGIN_ITERATION(3)						/* frames */
		BEGIN_ITERATION(2)					/* rows */
			BEGIN_ITERATION(1)				/* columns */
				BEGIN_ITERATION(0)			/* components */
					/* BUG?  this code is compact but tricky to maintain, because
					 * the following sums have the number of dimensions hard-coded
					 * to 4!?
					 */
					offset = os[0]+os[1]+os[2]+os[3];
					ir_offset = iros[0]+iros[1]+iros[2]+iros[3];

					*(image_ptr+offset) += *(irptr+ir_offset) * amp;
				}
			}
		}
	}
}

#define BEGIN_C(level)								\
										\
	var[level] = OBJ_TYPE_DIM(dpto,level);					\
	while( var[level] -- ){							\
		os[level] = var[level] * OBJ_TYPE_INC(dpfr,level);


void _convolve3d(QSP_ARG_DECL  Data_Obj *dpto,Data_Obj *dpfr,Data_Obj *dpfilt)
{
	float val, *frptr;
	dimension_t offset;
	incr_t	os[N_DIMENSIONS];
	posn_t var[N_DIMENSIONS];

	img_clear3d(dpto);

	frptr = (float *) OBJ_DATA_PTR(dpfr);

	var[4]=0;

	BEGIN_C(3)							/* frms */
		BEGIN_C(2)						/* rows */
			BEGIN_C(1)					/* cols */
				BEGIN_C(0)				/* components */
					offset = os[3]+os[2]+os[1]+os[0];
					val = *(frptr+offset);
					add_impulse3d(val,dpto,dpfilt,var);
				}
			}
		}
	}
}

