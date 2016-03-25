
#ifdef HAVE_CLFFT
// 1-D real-to-complex fft

static void PF_FFT_CALL_NAME(rvfft)(FFT_Args *fap)
{
	cl_int err;

	/* FFT library realted declarations */
	clfftPlanHandle planHandle;
	clfftDim dim = CLFFT_1D;
	size_t clLengths[1];

	/* Setup clFFT. */
	clfftSetupData fftSetup;
	err = clfftInitSetupData(&fftSetup);
	err = clfftSetup(&fftSetup);

	/* Create a default plan for a complex FFT. */
	clLengths[0]=FFT_LEN(fap);
	err = clfftCreateDefaultPlan(	&planHandle,
					OCLDEV_CTX( FFT_PFDEV(fap) ),	// OCL context
					dim,
					clLengths);

	/* Set plan parameters. */
	// need to define the precision in sp_defs.h etc
	err = clfftSetPlanPrecision(planHandle, /*CLFFT_SINGLE*/ MY_CLFFT_PRECISION );
	/*
	err = clfftSetLayout(planHandle, CLFFT_COMPLEX_INTERLEAVED, CLFFT_COMPLEX_INTERLEAVED);
	err = clfftSetResultLocation(planHandle, CLFFT_INPLACE);
	*/
	/*                              input layout       output layout	*/
	err = clfftSetLayout(planHandle, CLFFT_REAL, /*CLFFT_COMPLEX_INTERLEAVED*/ CLFFT_HERMITIAN_INTERLEAVED);

	err = clfftSetResultLocation(planHandle, CLFFT_OUTOFPLACE);

	/* Bake the plan. */
	err = clfftBakePlan(	planHandle,
				1,					// num queues
				&(OCLDEV_QUEUE(FFT_PFDEV(fap))),	// cl_command_queue *
				NULL,					// callback notify func ptr
				NULL					// user data ptr for callback func
				);

	/* Execute the plan. */
	err = clfftEnqueueTransform(	planHandle,
					CLFFT_FORWARD,
					1,		// numQueuesAndEvents
					&(OCLDEV_QUEUE(FFT_PFDEV(fap))),
					0,		// numWaitEvents
					NULL,		// cl_event * waitEvents
					NULL,		// cl_event * outEvents
					(cl_mem *) &(FFT_SRC(fap)),	// cl_mem* inputBuffers
					(cl_mem *) &(FFT_DST(fap)),	// cl_mem* outputBuffers
					NULL		// cl_mem tmpBuffer
					);

	/* Wait for calculations to be finished. */
	err = clFinish(OCLDEV_QUEUE(FFT_PFDEV(fap)));


	//err = clfftTeardown();
} // rvfft

static void PF_FFT_CALL_NAME(rvift)(FFT_Args *fap)
{
	cl_int err;

	/* FFT library realted declarations */
	clfftPlanHandle planHandle;
	clfftDim dim = CLFFT_1D;
	size_t clLengths[1];

	/* Setup clFFT. */
	clfftSetupData fftSetup;
	err = clfftInitSetupData(&fftSetup);
	err = clfftSetup(&fftSetup);

	/* Create a default plan for a complex FFT. */
	clLengths[0]=FFT_LEN(fap);
	err = clfftCreateDefaultPlan(	&planHandle,
					OCLDEV_CTX( FFT_PFDEV(fap) ),	// OCL context
					dim,
					clLengths);

	/* Set plan parameters. */
	err = clfftSetPlanPrecision(planHandle, MY_CLFFT_PRECISION );
	err = clfftSetLayout(planHandle, CLFFT_HERMITIAN_INTERLEAVED, CLFFT_REAL );
	err = clfftSetResultLocation(planHandle, CLFFT_OUTOFPLACE);

	/* Bake the plan. */
	err = clfftBakePlan(	planHandle,
				1,					// num queues
				&(OCLDEV_QUEUE(FFT_PFDEV(fap))),	// cl_command_queue *
				NULL,					// callback notify func ptr
				NULL					// user data ptr for callback func
				);

	/* Execute the plan. */
	err = clfftEnqueueTransform(	planHandle,
					CLFFT_BACKWARD,
					1,		// numQueuesAndEvents
					&(OCLDEV_QUEUE(FFT_PFDEV(fap))),
					0,		// numWaitEvents
					NULL,		// cl_event * waitEvents
					NULL,		// cl_event * outEvents
					(cl_mem *) &(FFT_SRC(fap)),	// cl_mem* inputBuffers
					(cl_mem *) &(FFT_DST(fap)),	// cl_mem* outputBuffers
					NULL		// cl_mem tmpBuffer
					);

	/* Wait for calculations to be finished. */
	err = clFinish(OCLDEV_QUEUE(FFT_PFDEV(fap)));


	//err = clfftTeardown();
} // rvift

//#ifdef FOOBAR
static void PF_FFT_CALL_NAME(cvgft)(FFT_Args *fap,int fft_dir)
{
	cl_int err;

	/* FFT library realted declarations */
	clfftPlanHandle planHandle;
	clfftDim dim = CLFFT_1D;
	size_t clLengths[1];

	/* Setup clFFT. */
	clfftSetupData fftSetup;
	err = clfftInitSetupData(&fftSetup);
	err = clfftSetup(&fftSetup);

	/* Create a default plan for a complex FFT. */
	clLengths[0]=FFT_LEN(fap);
	err = clfftCreateDefaultPlan(	&planHandle,
					OCLDEV_CTX( FFT_PFDEV(fap) ),	// OCL context
					dim,
					clLengths);

	/* Set plan parameters. */
	// need to define the precision in sp_defs.h etc
	err = clfftSetPlanPrecision(planHandle, /*CLFFT_SINGLE*/ MY_CLFFT_PRECISION );
	/*
	err = clfftSetLayout(planHandle, CLFFT_COMPLEX_INTERLEAVED, CLFFT_COMPLEX_INTERLEAVED);
	err = clfftSetResultLocation(planHandle, CLFFT_INPLACE);
	*/
	/*                              input layout       output layout	*/
	err = clfftSetLayout(planHandle, CLFFT_COMPLEX_INTERLEAVED, CLFFT_COMPLEX_INTERLEAVED );

	err = clfftSetResultLocation(planHandle, CLFFT_INPLACE);

	/* Bake the plan. */
	err = clfftBakePlan(	planHandle,
				1,					// num queues
				&(OCLDEV_QUEUE(FFT_PFDEV(fap))),	// cl_command_queue *
				NULL,					// callback notify func ptr
				NULL					// user data ptr for callback func
				);

//fprintf(stderr,"cvfft:  src = 0x%lx, dst = 0x%lx\n",(u_long)FFT_SRC(fap),(u_long)FFT_DST(fap));
// source is null...
	/* Execute the plan. */
	err = clfftEnqueueTransform(	planHandle,
					fft_dir,
					1,		// numQueuesAndEvents
					&(OCLDEV_QUEUE(FFT_PFDEV(fap))),
					0,		// numWaitEvents
					NULL,		// cl_event * waitEvents
					NULL,		// cl_event * outEvents
					(cl_mem *) &(FFT_DST(fap)),	// cl_mem* inputBuffers
					(cl_mem *) &(FFT_DST(fap)),	// cl_mem* outputBuffers
					NULL		// cl_mem tmpBuffer
					);

	/* Wait for calculations to be finished. */
	err = clFinish(OCLDEV_QUEUE(FFT_PFDEV(fap)));


	//err = clfftTeardown();
} // cvgft

static void PF_FFT_CALL_NAME(cvfft)(FFT_Args *fap)
{
	PF_FFT_CALL_NAME(cvgft)(fap, CLFFT_FORWARD );
}

/*
static void PF_FFT_CALL_NAME(cvift)(FFT_Args *fap)
{
	PF_FFT_CALL_NAME(cvgft)(fap, CLFFT_BACKWARD );
}
*/

//#endif // FOOBAR

static void HOST_TYPED_CALL_NAME_REAL( fftrows, type_code )(HOST_CALL_ARG_DECLS)
{
	FFT_Args fa;
	FFT_Args *fap=(&fa);
	dimension_t i;

	if( ! real_row_fft_ok(DEFAULT_QSP_ARG  OA_SRC1(oap),OA_DEST(oap),"rfftrows") ) return;

	SET_FFT_ISI( fap, FWD_FFT );			/* not used, but play it safe */
	SET_FFT_SRC( fap, OBJ_DATA_PTR( OA_SRC1(oap) ) );
	SET_FFT_SINC( fap, OBJ_PXL_INC( OA_SRC1(oap) ) );
	SET_FFT_DST( fap, OBJ_DATA_PTR( OA_DEST(oap) ) );
	SET_FFT_DINC( fap, OBJ_PXL_INC( OA_DEST(oap) ) );	// was /2
	SET_FFT_LEN( fap, OBJ_COLS( OA_SRC1(oap) ) );
	SET_FFT_PFDEV( fap, OBJ_PFDEV( OA_SRC1(oap) ) );

	ROW_LOOP( OA_SRC1(oap), PF_FFT_CALL_NAME(rvfft),std_type,std_cpx )

}

static void HOST_TYPED_CALL_NAME_REAL( fft2d, type_code )(HOST_CALL_ARG_DECLS)
{
	cl_int err;
	size_t strides[2];

	/* FFT library realted declarations */
	clfftPlanHandle planHandle;
	size_t clLengths[2];

	/* Setup clFFT. */
	clfftSetupData fftSetup;
	err = clfftInitSetupData(&fftSetup);
	err = clfftSetup(&fftSetup);

fprintf(stderr,"%s BEGIN\n",STRINGIFY(HOST_TYPED_CALL_NAME_REAL(fft2d,type_code)));
	// First we need to check the shape of the transform
	if( real_fft_type(DEFAULT_QSP_ARG  OA_SRC1(oap),OA_DEST(oap),STRINGIFY(HOST_TYPED_CALL_NAME_REAL(fft2d,type_code)) )
			!= 1 ){
		sprintf(DEFAULT_ERROR_STRING,
"Sorry, for OpenCL 2-D real FFT of image %s, transform %s should have %d rows and %d columns.",
OBJ_NAME(OA_SRC1(oap)), OBJ_NAME(OA_DEST(oap)), OBJ_ROWS(OA_SRC1(oap)), 1+OBJ_COLS(OA_DEST(oap))/2);
		NWARN(DEFAULT_ERROR_STRING);
		return;
	}

	/* Create a default plan for a complex FFT. */

	clLengths[0]=OBJ_COLS(OA_SRC1(oap));
	clLengths[1]=OBJ_ROWS(OA_SRC1(oap));
fprintf(stderr,"lengths = %zu, %zu\n",clLengths[0],clLengths[1]);
	err = clfftCreateDefaultPlan(	&planHandle,
					OCLDEV_CTX( OA_PFDEV(oap) ),	// OCL context
					CLFFT_2D,
					clLengths);

	/* Set plan parameters. */
	// MY_CLFFT_PRECISION is set in sp_defs.h, dp_defs.h...
	err = clfftSetPlanPrecision(planHandle, MY_CLFFT_PRECISION );

	err = clfftSetLayout(planHandle, CLFFT_REAL, CLFFT_HERMITIAN_INTERLEAVED);
	err = clfftSetResultLocation(planHandle, CLFFT_OUTOFPLACE);
	strides[0]=1;
	strides[1]=OBJ_COLS(OA_DEST(oap));
	err = clfftSetPlanOutStride( planHandle, CLFFT_2D, strides );

{
size_t iDist, oDist;
err = clfftGetPlanDistance( planHandle, &iDist, &oDist );
fprintf(stderr,"iDist = %zu\t\toDist = %zu\n", iDist, oDist);
err = clfftGetPlanOutStride( planHandle, CLFFT_2D, strides );
fprintf(stderr,"out_strides = %zu %zu\n", strides[0],strides[1]);
}
	/* Bake the plan. */
	err = clfftBakePlan(	planHandle,
				1,					// num queues
				&(OCLDEV_QUEUE(OA_PFDEV(oap))),	// cl_command_queue *
				NULL,					// callback notify func ptr
				NULL					// user data ptr for callback func
				);

//fprintf(stderr,"cvfft:  src = 0x%lx, dst = 0x%lx\n",(u_long)FFT_SRC(fap),(u_long)FFT_DST(fap));
// source is null...
	/* Execute the plan. */
	err = clfftEnqueueTransform(	planHandle,
					CLFFT_FORWARD,
					1,		// numQueuesAndEvents
					&(OCLDEV_QUEUE(OA_PFDEV(oap))),
					0,		// numWaitEvents
					NULL,		// cl_event * waitEvents
					NULL,		// cl_event * outEvents
					(cl_mem *) &(OBJ_DATA_PTR(OA_SRC1(oap))),	// cl_mem* inputBuffers
					(cl_mem *) &(OBJ_DATA_PTR(OA_DEST(oap))),	// cl_mem* outputBuffers
					NULL		// cl_mem tmpBuffer
					);

	/* Wait for calculations to be finished. */
	err = clFinish(OCLDEV_QUEUE(OA_PFDEV(oap)));


	//err = clfftTeardown();
}
#ifdef FOOBAR
{
	FFT_Args fa;
	FFT_Args *fap=(&fa);

	if( real_fft_check(DEFAULT_QSP_ARG  OA_SRC1(oap),OA_DEST(oap),"rfft2d") < 0 ) return;

	SET_FFT_ISI( fap, FWD_FFT );			/* not used, but play it safe */
	SET_FFT_SINC( fap, OBJ_PXL_INC( OA_SRC1(oap) ) );
	//SET_FFT_DINC( fap, OBJ_PXL_INC( OA_DEST(oap) )/2 );
	SET_FFT_DINC( fap, OBJ_PXL_INC( OA_DEST(oap) ) );
	SET_FFT_PFDEV( fap, OBJ_PFDEV( OA_SRC1(oap) ) );

	if( OBJ_COLS( OA_SRC1(oap) ) > 1 ){		/* more than 1 column ? */
		/* Transform the rows */
		dimension_t i;

		SET_FFT_SRC( fap, OBJ_DATA_PTR( OA_SRC1(oap) ) );
		SET_FFT_SINC( fap, OBJ_PXL_INC( OA_SRC1(oap) ) );
		SET_FFT_DST( fap, OBJ_DATA_PTR( OA_DEST(oap) ) );
		SET_FFT_DINC( fap, OBJ_PXL_INC( OA_DEST(oap) ) );
		SET_FFT_LEN( fap, OBJ_COLS( OA_SRC1(oap) ) );

		ROW_LOOP( OA_SRC1(oap),PF_FFT_CALL_NAME(rvfft), std_type,std_cpx)
	}

	/* Now transform the columns */
	/* BUG wrong if columns == 1 */
	/* Then we should copy into the complex target... */

	SET_FFT_LEN( fap, OBJ_ROWS( OA_SRC1(oap) ) );
	//SET_FFT_DINC( fap, OBJ_ROW_INC( OA_DEST(oap) )/2 );
	SET_FFT_DINC( fap, OBJ_ROW_INC( OA_DEST(oap) ) );

	if( OBJ_ROWS( OA_SRC1(oap) ) > 1 ){			/* more than 1 row? */
		SET_FFT_LEN( fap, OBJ_ROWS( OA_SRC1(oap) ) );

		COLUMN_LOOP(OA_DEST(oap),PF_FFT_CALL_NAME(cvfft))
	}
}
#endif // FOOBAR

static void HOST_TYPED_CALL_NAME_REAL( ift2d, type_code )( HOST_CALL_ARG_DECLS )
{
	cl_int err;

	/* FFT library realted declarations */
	clfftPlanHandle planHandle;
	clfftDim dim = CLFFT_2D;
	size_t clLengths[2];
	size_t strides[2];

	/* Setup clFFT. */
	clfftSetupData fftSetup;
	err = clfftInitSetupData(&fftSetup);
	CLFFT_RESULT_CHECK(clfftInitSetupData)
	err = clfftSetup(&fftSetup);
	CLFFT_RESULT_CHECK(clfftSetup)

	/* Create a default plan for a complex FFT. */
	clLengths[0]=OBJ_COLS(OA_DEST(oap));
	clLengths[1]=OBJ_ROWS(OA_DEST(oap));
	err = clfftCreateDefaultPlan(	&planHandle,
					OCLDEV_CTX( OA_PFDEV(oap) ),	// OCL context
					dim,
					clLengths);
	CLFFT_RESULT_CHECK(clfftCreateDefaultPlan)

	/* Set plan parameters. */
	// need to define the precision in sp_defs.h etc
	err = clfftSetPlanPrecision(planHandle, /*CLFFT_SINGLE*/ MY_CLFFT_PRECISION );
	CLFFT_RESULT_CHECK(clfftSetPlanPrecision)

	err = clfftSetLayout(	planHandle,
				CLFFT_HERMITIAN_INTERLEAVED,	// input layout
				CLFFT_REAL			// output layout
				);
	CLFFT_RESULT_CHECK(clfftSetLayout)

	err = clfftSetResultLocation(planHandle, CLFFT_OUTOFPLACE);
	CLFFT_RESULT_CHECK(clfftSetResultLocation)

	strides[0]=1;
	strides[1]=OBJ_COLS(OA_SRC1(oap));
	err = clfftSetPlanInStride( planHandle, CLFFT_2D, strides );
	CLFFT_RESULT_CHECK(clfftSetPlanInStride)
{
size_t iDist, oDist;
err = clfftGetPlanDistance( planHandle, &iDist, &oDist );
fprintf(stderr,"iDist = %zu\t\toDist = %zu\n", iDist, oDist);
err = clfftGetPlanInStride( planHandle, CLFFT_2D, strides );
fprintf(stderr,"in_strides = %zu %zu\n", strides[0],strides[1]);
}
	/* Bake the plan. */
	/* Bake the plan. */
	err = clfftBakePlan(	planHandle,
				1,					// num queues
				&(OCLDEV_QUEUE(OA_PFDEV(oap))),	// cl_command_queue *
				NULL,					// callback notify func ptr
				NULL					// user data ptr for callback func
				);
	CLFFT_RESULT_CHECK(clfftBakePlan)

//fprintf(stderr,"cvfft:  src = 0x%lx, dst = 0x%lx\n",(u_long)FFT_SRC(fap),(u_long)FFT_DST(fap));
// source is null...
	/* Execute the plan. */
	err = clfftEnqueueTransform(	planHandle,
					CLFFT_BACKWARD,
					1,		// numQueuesAndEvents
					&(OCLDEV_QUEUE(OA_PFDEV(oap))),
					0,		// numWaitEvents
					NULL,		// cl_event * waitEvents
					NULL,		// cl_event * outEvents
					(cl_mem *) &(OBJ_DATA_PTR(OA_SRC1(oap))),	// cl_mem* inputBuffers
					(cl_mem *) &(OBJ_DATA_PTR(OA_DEST(oap))),	// cl_mem* outputBuffers
					NULL		// cl_mem tmpBuffer
					);
	CLFFT_RESULT_CHECK(clfftEnqueueTransform)

	/* Wait for calculations to be finished. */
	err = clFinish(OCLDEV_QUEUE(OA_PFDEV(oap)));
	CLFFT_RESULT_CHECK(clFinish)


	//err = clfftTeardown();
	//CLFFT_RESULT_CHECK(clfftTeardown)
}

#ifdef FOOBAR
{
	FFT_Args fa;
	FFT_Args *fap=(&fa);

	if( real_fft_check(DEFAULT_QSP_ARG  OA_DEST(oap),OA_SRC1(oap),"rift2d") < 0 ) return;

	SET_FFT_ISI( fap, 1 );
	//SET_FFT_DINC( fap, OBJ_ROW_INC( OA_SRC1(oap) ) / 2 );
	SET_FFT_DINC( fap, OBJ_ROW_INC( OA_SRC1(oap) ) );
	SET_FFT_PFDEV( fap, OBJ_PFDEV( OA_SRC1(oap) ) );

	if( OBJ_ROWS( OA_SRC1(oap) ) > 1 ){			/* more than 1 row? */
		/* Transform the columns */
		SET_FFT_LEN( fap, OBJ_ROWS( OA_SRC1(oap) ) );
		COLUMN_LOOP(OA_SRC1(oap),PF_FFT_CALL_NAME(cvift))
	}

	if( OBJ_COLS( OA_SRC1(oap) ) > 1 ){		/* more than 1 column ? */
		dimension_t i;

		SET_FFT_SRC( fap, OBJ_DATA_PTR( OA_SRC1(oap) ) );
		SET_FFT_SINC( fap, OBJ_PXL_INC( OA_SRC1(oap) ) );
		SET_FFT_DST( fap, OBJ_DATA_PTR( OA_DEST(oap) ) );
		SET_FFT_DINC( fap, OBJ_PXL_INC( OA_DEST(oap) ) );
		SET_FFT_LEN( fap, OBJ_COLS( OA_DEST(oap) ) );	/* use the real len */

		ROW_LOOP(OA_DEST(oap),PF_FFT_CALL_NAME(rvift), std_cpx,std_type)
	}
}
#endif // FOOBAR

static void HOST_TYPED_CALL_NAME_CPX( fftrows, type_code )( HOST_CALL_ARG_DECLS, int is_inv )
{
	dimension_t i;
	FFT_Args fa;
	FFT_Args *fap=(&fa);

	if( ! row_fft_ok(DEFAULT_QSP_ARG  OA_SRC1(oap), STRINGIFY(HOST_TYPED_CALL_NAME_CPX(fftrows,type_code)) ) )
		return;

	SET_FFT_ISI( fap, is_inv );

	/* transform the rows */

	SET_FFT_PFDEV( fap, OBJ_PFDEV( OA_SRC1(oap) ) );
	if( OBJ_COLS( OA_SRC1(oap) ) > 1 ){
		SET_FFT_DST( fap, (std_type *)OBJ_DATA_PTR( OA_SRC1(oap) ) );
		SET_FFT_LEN( fap, OBJ_COLS( OA_SRC1(oap) ) );
		// What is pinc??? type units not machine units?
		//SET_FFT_DINC( fap, OBJ_PXL_INC( OA_SRC1(oap) )/2 );
		SET_FFT_DINC( fap, OBJ_PXL_INC( OA_SRC1(oap) ) );

		for (i = 0; i < OBJ_ROWS( OA_SRC1(oap) ); ++i) {
			PF_FFT_CALL_NAME(cvfft)(&fa);
			/* why not std_cpx??? */
			SET_FFT_DST( fap, ((std_type *)FFT_DST(fap)) + OBJ_ROW_INC( OA_SRC1(oap) ) );
		}
	}
}

static void HOST_TYPED_CALL_NAME_REAL( iftrows, type_code )( HOST_CALL_ARG_DECLS )
{
	FFT_Args fa;
	FFT_Args *fap=(&fa);
	dimension_t i;

	if( ! real_row_fft_ok(DEFAULT_QSP_ARG  OA_DEST(oap),OA_SRC1(oap),"r_rowift") ) return;

	SET_FFT_ISI( fap, 1 );
	SET_FFT_SRC( fap, OBJ_DATA_PTR( OA_SRC1(oap) ) );
	SET_FFT_SINC( fap, OBJ_PXL_INC( OA_SRC1(oap) ) );		// used to be /2
	SET_FFT_DST( fap, OBJ_DATA_PTR( OA_DEST(oap) ) );
	SET_FFT_DINC( fap, OBJ_PXL_INC( OA_DEST(oap) ) );
	SET_FFT_LEN( fap, OBJ_COLS( OA_DEST(oap) ) );
	SET_FFT_PFDEV( fap, OBJ_PFDEV( OA_DEST(oap) ) );

	ROW_LOOP(OA_DEST(oap),PF_FFT_CALL_NAME(rvift),std_cpx,std_type)
}



/*
 * Do an in-place complex FFT
 *
 * No SMP version (yet).
 */

static void HOST_TYPED_CALL_NAME_CPX( fft2d, type_code )( HOST_CALL_ARG_DECLS, int is_inv )
{
	dimension_t i;
	FFT_Args fa;
	FFT_Args *fap=(&fa);

	if( ! cpx_fft_ok(DEFAULT_QSP_ARG  OA_SRC1(oap), STRINGIFY(HOST_TYPED_CALL_NAME_CPX(fft2d,type_code)) ) )
		return;

	/* transform the columns */

	SET_FFT_ISI( fap, is_inv );
	SET_FFT_PFDEV( fap, OBJ_PFDEV( OA_SRC1(oap) ) );

	if( OBJ_ROWS( OA_SRC1(oap) ) > 1 ){	/* more than one row */
		SET_FFT_DST( fap, (std_type *)OBJ_DATA_PTR( OA_SRC1(oap) ) );
		SET_FFT_LEN( fap, OBJ_ROWS( OA_SRC1(oap) ) );
		//SET_FFT_DINC( fap, OBJ_ROW_INC( OA_SRC1(oap) )/2 );
		SET_FFT_DINC( fap, OBJ_ROW_INC( OA_SRC1(oap) ) );

		for (i = 0; i < OBJ_COLS( OA_SRC1(oap) ); ++i) {
			PF_FFT_CALL_NAME(cvfft)(&fa);
			/* ((std_type *)fa.dst_addr) += OBJ_PXL_INC( OA_SRC1(oap) ); */
			SET_FFT_DST( fap, ((std_cpx *)FFT_DST(fap)) + OBJ_PXL_INC( OA_SRC1(oap) ) );
		}
	}

	/* transform the rows */

	if( OBJ_COLS( OA_SRC1(oap) ) > 1 ){
		SET_FFT_DST( fap, (std_type *)OBJ_DATA_PTR( OA_SRC1(oap) ) );
		SET_FFT_LEN( fap, OBJ_COLS( OA_SRC1(oap) ) );
		/* pixel inc used to be in machine units,
		 * now it's in type units!? */
		//SET_FFT_DINC( fap, OBJ_PXL_INC( OA_SRC1(oap) )/2 );
		SET_FFT_DINC( fap, OBJ_PXL_INC( OA_SRC1(oap) ) );

		for (i = 0; i < OBJ_ROWS( OA_SRC1(oap) ); ++i) {
			PF_FFT_CALL_NAME(cvfft)(&fa);
			SET_FFT_DST( fap, ((std_cpx *)FFT_DST(fap)) + OBJ_ROW_INC( OA_SRC1(oap) ) );
		}
	}
}


#else // ! HAVE_CLFFT

#define NO_CLFFT_MSG(funcname_str)					\
	sprintf(DEFAULT_ERROR_STRING,"%s:  Sorry, no clFFT support in this build.",funcname_str);	\
	NWARN(DEFAULT_ERROR_STRING);

static void HOST_TYPED_CALL_NAME_CPX( fft2d, type_code )( HOST_CALL_ARG_DECLS, int is_inv )
{ NO_CLFFT_MSG(STRINGIFY(HOST_TYPED_CALL_NAME_CPX(fft2d,type_code))) }

static void HOST_TYPED_CALL_NAME_REAL( fft2d, type_code )( HOST_CALL_ARG_DECLS )
{ NO_CLFFT_MSG(STRINGIFY(HOST_TYPED_CALL_NAME_REAL(fft2d,type_code))) }

static void HOST_TYPED_CALL_NAME_REAL( ift2d, type_code )( HOST_CALL_ARG_DECLS )
{ NO_CLFFT_MSG(STRINGIFY(HOST_TYPED_CALL_NAME_REAL(ift2d,type_code))) }

static void HOST_TYPED_CALL_NAME_CPX( fftrows, type_code )( HOST_CALL_ARG_DECLS, int is_inv )
{ NO_CLFFT_MSG(STRINGIFY(HOST_TYPED_CALL_NAME_CPX(fftrows,type_code))) }

static void HOST_TYPED_CALL_NAME_REAL( fftrows, type_code )( HOST_CALL_ARG_DECLS )
{ NO_CLFFT_MSG(STRINGIFY(HOST_TYPED_CALL_NAME_REAL(fftrows,type_code))) }

static void HOST_TYPED_CALL_NAME_REAL( iftrows, type_code )( HOST_CALL_ARG_DECLS )
{ NO_CLFFT_MSG(STRINGIFY(HOST_TYPED_CALL_NAME_REAL(iftrows,type_code))) }


#endif // ! HAVE_CLFFT

