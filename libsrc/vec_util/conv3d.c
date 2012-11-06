#include "quip_config.h"

char VersionId_vec_util_conv3d[] = QUIP_VERSION_STRING;

#include "vec_util.h"

#define BEGIN_CLEAR(level)								\
											\
	for(var[level]=0;var[level] < dp->dt_type_dim[level];var[level]++){		\
		os[level] = var[level] * dp->dt_type_inc[level];

void img_clear3d(Data_Obj *dp)
{
	dimension_t var[N_DIMENSIONS];
	incr_t os[N_DIMENSIONS];
	float *ptr;

	ptr = (float *)dp->dt_data;

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
		if( os[level] < 0 || os[level] >= (incr_t) image_dp->dt_type_dim[level] ) continue;
#else

#define CHECK_EDGE(level)									\
												\
		if( os[level] < 0 ) os[level]+=image_dp->dt_type_dim[level];			\
		else if( os[level] >= image_dp->dt_type_dim[level] )				\
			os[level]-=image_dp->dt_type_dim[level];				\

#endif /* NOWRAP */

#define BEGIN_ITERATION(level)									\
												\
	var[level] = (incr_t ) ir_dp->dt_type_dim[ level ];					\
	while( var[level] -- ){									\
		os[level] = position[level] + var[level] - ir_dp->dt_type_dim[level]/2;	\
		CHECK_EDGE(level)								\
		os[level] *= image_dp->dt_type_inc[level];					\
		iros[level] = var[level] * ir_dp->dt_type_inc[level];				\



void add_impulse3d(double amp,Data_Obj *image_dp,Data_Obj *ir_dp,dimension_t *position)
{
	float *image_ptr, *irptr;
	incr_t offset,ir_offset;		/* offsets into image */
	incr_t var[N_DIMENSIONS];
	incr_t os[N_DIMENSIONS], iros[N_DIMENSIONS];	/* offsets into impulse response */


	image_ptr = (float *) image_dp->dt_data;
	irptr = (float *) ir_dp->dt_data;

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
	var[level] = dpto->dt_type_dim[level];					\
	while( var[level] -- ){							\
		os[level] = var[level] * dpfr->dt_type_inc[level];


void convolve3d(QSP_ARG_DECL  Data_Obj *dpto,Data_Obj *dpfr,Data_Obj *dpfilt)
{
	float val, *frptr;
	dimension_t offset;
	incr_t	os[N_DIMENSIONS];
	dimension_t var[N_DIMENSIONS];

	img_clear3d(dpto);

	frptr = (float *) dpfr->dt_data;

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

