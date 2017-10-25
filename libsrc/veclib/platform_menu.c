
#include "quip_config.h"
#include "quip_prot.h"
#include "pf_viewer.h"	// has to come first to pick up glew.h first
#include "platform.h"
#include "ocl_platform.h"


// BUG not thread-safe / add to query_stack!
Platform_Device *curr_pdp=NULL;
List *pdp_stack=NULL;

static COMMAND_FUNC( do_list_pfs )
{
	list_platforms(tell_msgfile());
}

static COMMAND_FUNC( do_list_pfdevs )
{
	Compute_Platform *cpp;

	cpp = pick_platform("");
	if( cpp == NULL ) return;

	// should we list devices for a single platform?
	//push_pfdev_context(QSP_ARG  PF_CONTEXT(cpp) );
	list_item_context(PF_CONTEXT(cpp));
	//if( pop_pfdev_context(SINGLE_QSP_ARG) == NULL )
	//	ERROR1("do_list_pfdevs:  Failed to pop platform device context!?");
}

static COMMAND_FUNC( do_list_all_pfdevs )
{
	Compute_Platform *cpp;
	List *lp;
	Node *np;

	lp = platform_list();
	if( lp == NULL ) {
		return;
	}

	np = QLIST_HEAD(lp);
	while( np != NULL ){
		cpp = (Compute_Platform *) NODE_DATA(np);
		sprintf(msg_str,"%s platform:",PLATFORM_NAME(cpp));
		prt_msg(msg_str);
		list_item_context(PF_CONTEXT(cpp));
		prt_msg("");

		np = NODE_NEXT(np);
	}
}

void _select_pfdev( QSP_ARG_DECL  Platform_Device *pdp )
{
	curr_pdp = pdp;	// select_pfdev

	// How do we specify host-mapped objects???  BUG!
	set_data_area( PFDEV_AREA(pdp,PFDEV_GLOBAL_AREA_INDEX) );
}

void _push_pfdev( QSP_ARG_DECL  Platform_Device *pdp )
{

	if( curr_pdp != NULL ){
		Node *np;

		np = mk_node(pdp);
		if( pdp_stack == NULL )
			pdp_stack = new_list();
		addHead(pdp_stack,np);
	}
	select_pfdev(pdp);
}

Platform_Device * _pop_pfdev(SINGLE_QSP_ARG_DECL)
{
	Node *np;
	Platform_Device *pdp;

	if( curr_pdp == NULL ){
		WARN("pop_pfdev:  nothing to pop (no current device)!?");
		return NULL;
	}
	curr_pdp = NULL;
	if( pdp_stack == NULL ) return NULL;
	if( QLIST_HEAD(pdp_stack) == NULL ) return NULL;

	np = remHead(pdp_stack);
	pdp = NODE_DATA(np);
	rls_node(np);
	select_pfdev(pdp);
	return pdp;
}


#define pick_platform_device()	_pick_platform_device(SINGLE_QSP_ARG)

static Platform_Device *_pick_platform_device(SINGLE_QSP_ARG_DECL)
{
	Platform_Device *pdp;
	Compute_Platform *cpp;

	cpp = pick_platform("");

	if( cpp == NULL ){
		const char *s;
		s=NAMEOF("dummy word");
		// Now use it just to suppress a compiler warning...
		sprintf(ERROR_STRING,"Ignoring \"%s\"",s);
		advise(ERROR_STRING);
		return NULL;
	}

	push_pfdev_context( QSP_ARG  PF_CONTEXT(cpp) );
	pdp = pick_pfdev("");
	pop_pfdev_context( SINGLE_QSP_ARG );
	return pdp;
}

static COMMAND_FUNC( do_select_pfdev )
{
	Platform_Device *pdp;

	pdp = pick_platform_device();
	if( pdp == NULL ) return;
	select_pfdev(pdp);
}

static COMMAND_FUNC( do_obj_dnload )
{
	Data_Obj *dpto, *dpfr;

	dpto = pick_obj("destination RAM object");
	dpfr = pick_obj("source GPU object");

	if( dpto == NULL || dpfr == NULL ) return;

	//ocl_obj_dnload(QSP_ARG  dpto,dpfr);
	gen_obj_dnload(QSP_ARG  dpto, dpfr);
}

static COMMAND_FUNC( do_obj_upload )
{
	Data_Obj *dpto, *dpfr;

	dpto = pick_obj("destination GPU object");
	dpfr = pick_obj("source RAM object");

	if( dpto == NULL || dpfr == NULL ) return;

	//ocl_obj_upload(QSP_ARG  dpto,dpfr);
	gen_obj_upload(QSP_ARG  dpto,dpfr);
}

static COMMAND_FUNC(do_show_pfdev)
{
	if( curr_pdp == NULL ){
		advise("No platform device selected.");
	} else {
		sprintf(ERROR_STRING,"Current platform device is:  %s (%s)",
			PFDEV_NAME(curr_pdp),
			PLATFORM_NAME(PFDEV_PLATFORM(curr_pdp))
			);
		advise(ERROR_STRING);
	}
}

static Platform_Device *find_pfdev( QSP_ARG_DECL  platform_type typ )
{
	List *cp_lp, *pfd_lp;
	Node *cp_np, *pfd_np;
	Compute_Platform *cpp;
	Platform_Device *pdp;

	cp_lp = platform_list();
	cp_np = QLIST_HEAD(cp_lp);
	while( cp_np != NULL ){
		cpp = NODE_DATA(cp_np);
		// We need to push a context before we can get a list of devices...
		push_pfdev_context( QSP_ARG  PF_CONTEXT(cpp) );

		pfd_lp = pfdev_list();
		if( pfd_lp == NULL ) return NULL;
		//pfd_np = QLIST_HEAD(pfd_lp);
		pfd_np = QLIST_TAIL(pfd_lp);
		while( pfd_np != NULL ){
			pdp = (Platform_Device *) NODE_DATA(pfd_np);
			if( PF_TYPE( PFDEV_PLATFORM(pdp) ) == typ ){
				// Some computers have multiple devices,
				// how should we decide which to use?
				// For example, the 2015 MacBook Pro has
				// and AMD gpu (which we use), and something
				// called Iris_Pro - which is not usable!?
				// But Iris_Pro comes up first in the list...
				pop_pfdev_context( SINGLE_QSP_ARG );
				return pdp;
			}
			//pfd_np = NODE_NEXT(pfd_np);
			pfd_np = NODE_PREV(pfd_np);
		}

		pop_pfdev_context( SINGLE_QSP_ARG );
		cp_np = NODE_NEXT(cp_np);
	}

	return NULL;
}

static const char *dev_type_names[]={"cuda","openCL"};
#define N_DEVICE_TYPES	(sizeof(dev_type_names)/sizeof(char *))

static COMMAND_FUNC(do_set_dev_type)
{
	int i;
	Platform_Device *pdp=NULL;

	i=WHICH_ONE( "software interface",N_DEVICE_TYPES , dev_type_names);
	if( i < 0 ) return;

	if( i == 0 )
		pdp = find_pfdev(QSP_ARG  PLATFORM_CUDA);
	else if( i == 1 )
		pdp = find_pfdev(QSP_ARG  PLATFORM_OPENCL);

	if( pdp == NULL ){
		sprintf(ERROR_STRING,"No %s device found!?",dev_type_names[i]);
		WARN(ERROR_STRING);
		return;
	}

	sprintf(ERROR_STRING,"Using %s device %s.",dev_type_names[i],PFDEV_NAME(pdp));
	advise(ERROR_STRING);

	select_pfdev(pdp);
}

// We call this if the user has not set DEFAULT_PLATFORM and DEFAULT_GPU in the enviroment...

static void check_platform_defaults(SINGLE_QSP_ARG_DECL)
{
	Platform_Device *pdp;
	Variable *vp1, *vp2;

	vp1 = var_of("DEFAULT_PLATFORM");
	vp2 = var_of("DEFAULT_GPU");

	if( vp1 != NULL && vp2 != NULL ) return;	// already set by user

	pdp = find_pfdev(QSP_ARG  PLATFORM_OPENCL);
	if( pdp == NULL ) pdp = find_pfdev(QSP_ARG  PLATFORM_CUDA);

	if( pdp == NULL ) return;
	assign_var("DEFAULT_PLATFORM",PLATFORM_NAME(PFDEV_PLATFORM(pdp)));
	assign_var("DEFAULT_GPU",PFDEV_NAME(pdp));
}

static COMMAND_FUNC( do_pfdev_info )
{
	Platform_Device *pdp;

	pdp = pick_platform_device();
	if( pdp == NULL ) return;

	(* PF_DEVINFO_FN(PFDEV_PLATFORM(pdp)))(QSP_ARG  pdp);
}

static COMMAND_FUNC( do_pf_info )
{
	Compute_Platform *cdp;

	cdp = pick_platform("");
	if( cdp == NULL ) return;

	sprintf(MSG_STR,"Platform name:  %s",PLATFORM_NAME(cdp));
	prt_msg(MSG_STR);

	(* PF_INFO_FN(cdp))(QSP_ARG  cdp);
}

static COMMAND_FUNC(do_push_pfdev)
{
	Platform_Device *pdp;

	pdp = pick_platform_device();
	if( pdp == NULL ) return;

	push_pfdev( curr_pdp );
}

static COMMAND_FUNC(do_pop_pfdev)
{
	Platform_Device *pdp;

	pdp = pop_pfdev();
	if( pdp == NULL ) WARN("nothing popped!?");
}


#define ADD_CMD(s,f,h)	ADD_COMMAND(platform_menu,s,f,h)

MENU_BEGIN(platform)
//ADD_CMD( list,		do_list_platforms,	list available platforms )
// should we have a platform info command?
ADD_CMD( list,		do_list_pfs,		list platforms )
ADD_CMD( info,		do_pf_info,		print device information )
ADD_CMD( list_devices,	do_list_pfdevs,		list devices for one platform )
ADD_CMD( list_all,	do_list_all_pfdevs,	list all devices from all platforms )
ADD_CMD( device_info,	do_pfdev_info,		print device information )
ADD_CMD( select,	do_select_pfdev,	select platform/device )
ADD_CMD( push_device,	do_push_pfdev,		select platform/device while remembering previous )
ADD_CMD( pop_device,	do_pop_pfdev,		restore previous device )
ADD_CMD( device_type,	do_set_dev_type,	use device of specified type )
ADD_CMD( show,		do_show_pfdev,		show current default platform )
ADD_CMD( upload,	do_obj_upload,		upload a data object to a device )
ADD_CMD( dnload,	do_obj_dnload,		download a data object from a device )
#ifdef FOOBAR
// To remove dependencies, these functions should go
// into the viewmenu library...
#ifndef BUILD_FOR_OBJC
ADD_CMD( viewer,	do_new_pf_vwr,		create a new platform viewer )
// gl_buffer moved to opengl menu
ADD_CMD( gl_buffer,	do_new_gl_buffer,	create a new GL buffer )
ADD_CMD( load,		do_load_pf_vwr,		load platform viewer with an image )
#endif // ! BUILD_FOR_OBJC
#endif // FOOBAR
MENU_END(platform)

// We use ifdef's to decide which platforms to initialize here...
// That makes it difficult to include this function in a program
// That doesn't need OpenCL or Cuda...
// But we don't know how to have these platforms register
// themselves, so we can't take this out...

void init_all_platforms(SINGLE_QSP_ARG_DECL)
{
	static int inited=0;
	if( inited ) return;

	vl2_init_platform(SINGLE_QSP_ARG);

#ifdef HAVE_OPENCL
	ocl_init_platform(SINGLE_QSP_ARG);
#endif // HAVE_OPENCL

#ifdef HAVE_CUDA
	cu2_init_platform(SINGLE_QSP_ARG);
#endif // HAVE_CUDA

#ifdef HAVE_METAL
	mtl_init_platform(SINGLE_QSP_ARG);
#endif // HAVE_METAL

	check_platform_defaults(SINGLE_QSP_ARG);

	inited=1;
} // init_all_platforms

COMMAND_FUNC( do_platform_menu )
{
	static int inited=0;
	if( ! inited ){
		init_all_platforms(SINGLE_QSP_ARG);
		inited=1;
	}
	PUSH_MENU(platform);
}

