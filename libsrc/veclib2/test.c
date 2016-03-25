
main(int ac, char **av)
{
	cl_uint num = 0;

	clGetDeviceIDs(NULL,CL_DEVICE_TYPE_GPU,0,NULL,&num);

	cl_device_id devices[num];

	clGetDeviceIDs(NULL,CL_DEVICE_TYPE_GPU,num,devices,NULL);

 

	cl_context ctx = clCreateContext(NULL,num,devices,NULL,NULL,NULL);
}

