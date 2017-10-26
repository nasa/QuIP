include(`../../include/veclib/ocl_port.m4')
my_include(`../../include/veclib/ocl_veclib_prot.m4')
my_include(`ocl_kern_args.m4')

void h_ocl_sp_vuni(HOST_CALL_ARG_DECLS)
{
	size_t global_work_size[3] = {1, 1, 1};
	int ki_idx=0;

	// BUG should make sure destination is contiguous!

	cl_int status;

	if( ! ocl_sp_vuni_inited ){
		if( ocl_sp_vuni_init( OBJ_PFDEV(OA_DEST(oap)) ) < 0 ) return;
	}

	global_work_size[0] = OBJ_N_MACH_ELTS(OA_DEST(oap));

	_SET_KERNEL_ARG( ocl_sp_vuni_kernel, void *, &(OBJ_DATA_PTR( OA_DEST(oap))) )
	_SET_KERNEL_ARG( ocl_sp_vuni_kernel, int, &ocl_sp_vuni_counter )
	ocl_sp_vuni_counter ++;

	status = clEnqueueNDRangeKernel(				
		OCLDEV_QUEUE( OBJ_PFDEV(OA_DEST(oap)) ),
		ocl_sp_vuni_kernel,
		1,	/* work_dim, 1-3 */
		NULL,
		global_work_size,
		/*local_work_size*/ NULL,
		0,	/* num_events_in_wait_list */
		NULL,	/* event_wait_list */
		NULL	/* event */
		);
	if( status != CL_SUCCESS )
		report_ocl_error(DEFAULT_QSP_ARG  status, "clEnqueueNDRangeKernel" );
}

void h_ocl_dp_vuni(HOST_CALL_ARG_DECLS)
{
	NWARN("Sorry, dp_vuni not implemented for OpenCL");
}

