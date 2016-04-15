#include "quip_config.h"
#include <stdio.h>
#include "pf_viewer.h"	// has to come first to pick up glew.h first
#include "quip_prot.h"
#include "platform.h"
#include "ocl_platform.h"

ITEM_INTERFACE_DECLARATIONS( Platform_Device, pfdev )
ITEM_INTERFACE_DECLARATIONS( Compute_Platform, platform )

Item_Context *create_pfdev_context(QSP_ARG_DECL  const char *name)
{
	if( pfdev_itp == NO_ITEM_TYPE )
		init_pfdevs(SINGLE_QSP_ARG);

	return create_item_context(QSP_ARG  pfdev_itp, name );
}

void push_pfdev_context(QSP_ARG_DECL  Item_Context *icp )
{
	push_item_context(QSP_ARG  pfdev_itp, icp );
}

Item_Context *pop_pfdev_context(SINGLE_QSP_ARG_DECL)
{
	return pop_item_context(QSP_ARG  pfdev_itp);
}


static void init_platform_defaults(QSP_ARG_DECL  Compute_Platform *cpp, platform_type t )
{
	SET_PF_TYPE(cpp,t);

	// default value is null...
	//SET_PF_DISPATCH_FN(cpp,NULL);
	//SET_PF_DISPATCH_TBL(cpp,NULL);
	SET_PF_MEM_UPLOAD_FN(cpp,NULL);
	SET_PF_MEM_DNLOAD_FN(cpp,NULL);
	SET_PF_ALLOC_FN(cpp,NULL);
	SET_PF_FREE_FN(cpp,NULL);
	SET_PF_OFFSET_DATA_FN(cpp,NULL);
	SET_PF_UPDATE_OFFSET_FN(cpp,NULL);
	SET_PF_MAPBUF_FN(cpp,NULL);
	SET_PF_UNMAPBUF_FN(cpp,NULL);
	SET_PF_REGBUF_FN(cpp,NULL);

	SET_PF_FUNC_TBL(cpp,NULL);

	switch(t){
		case PLATFORM_CPU:
			break;
#ifdef HAVE_OPENCL
		case PLATFORM_OPENCL:
			// allocate the memory structures
			PF_OPD(cpp) = getbuf(sizeof(*PF_OPD(cpp)));
			break;
#endif // HAVE_OPENCL

#ifdef HAVE_CUDA
		case PLATFORM_CUDA:
			break;
#endif // HAVE_CUDA

//#ifdef CAUTIOUS
		default:
//			ERROR1("CAUTIOUS:  init_platform:  Unexpected platform type code!?");
			assert( AERROR("Unexpected platform type code!?") );
			break;
//#endif // CAUTIOUS
	}

}

Compute_Platform *creat_platform(QSP_ARG_DECL  const char *name, platform_type t)
{
	Compute_Platform *cpp;
	Item_Context *icp;

	cpp = new_platform(QSP_ARG  name);
//	if( cpp == NULL ){
//		sprintf(ERROR_STRING,
//"CAUTIOUS:  creat_platform:  error creating platform %s!?",name);
//		ERROR1(ERROR_STRING);
//	}
	assert( cpp != NULL );

	icp = create_pfdev_context(QSP_ARG  name );
//#ifdef CAUTIOUS
//	if( icp == NO_ITEM_CONTEXT )
//		ERROR1("CAUTIOUS:  creat_platform:  Failed to create platform device context!?");
//#endif // CAUTIOUS
	assert( icp != NO_ITEM_CONTEXT );

	SET_PF_CONTEXT(cpp,icp);

	init_platform_defaults(QSP_ARG  cpp, t );

	return cpp;

} // creat_platform

void gen_obj_upload(QSP_ARG_DECL  Data_Obj *dpto, Data_Obj *dpfr)
{
	size_t siz;

	CHECK_NOT_RAM("gen_obj_upload","destination",dpto)
	CHECK_RAM("gen_obj_upload","source",dpfr)
	CHECK_CONTIG_DATA("gen_obj_upload","source",dpfr)
	CHECK_CONTIG_DATA("gen_obj_upload","destination",dpto)
	CHECK_SAME_SIZE(dpto,dpfr,"gen_obj_upload")
	CHECK_SAME_PREC(dpto,dpfr,"gen_obj_upload")

#ifdef FOOBAR
	if( IS_BITMAP(dpto) )
		siz = BITMAP_WORD_COUNT(dpto) * PREC_SIZE( PREC_FOR_CODE(BITMAP_MACH_PREC) );
	else
		siz = OBJ_N_MACH_ELTS(dpto) * PREC_SIZE( OBJ_MACH_PREC_PTR(dpto) );
#endif /* FOOBAR */
	siz = OBJ_N_TYPE_ELTS(dpto) * PREC_SIZE( OBJ_MACH_PREC_PTR(dpto) );

//#ifdef CAUTIOUS
//	if( PF_MEM_UPLOAD_FN(PFDEV_PLATFORM(OBJ_PFDEV(dpto))) == NULL ){
//		sprintf(ERROR_STRING,
//	"CAUTIOUS:  gen_obj_dnload:  Platform %s has a null upload function!?",
//			PLATFORM_NAME(OBJ_PLATFORM(dpto)));
//		ERROR1(ERROR_STRING);
//	}
//#endif // CAUTIOUS
	assert( PF_MEM_UPLOAD_FN(PFDEV_PLATFORM(OBJ_PFDEV(dpto))) != NULL );

	( * PF_MEM_UPLOAD_FN(OBJ_PLATFORM(dpto)) )
		(QSP_ARG  OBJ_DATA_PTR(dpto), OBJ_DATA_PTR(dpfr), siz, OBJ_PFDEV(dpto) );
}

void gen_obj_dnload(QSP_ARG_DECL  Data_Obj *dpto,Data_Obj *dpfr)
{
	size_t siz;

	CHECK_RAM("gen_obj_dnload","destination",dpto)
	// Really need to check that this object has the right platform?
	CHECK_NOT_RAM("gen_obj_dnload","source",dpfr)

	CHECK_CONTIG_DATA("gen_obj_dnload","source",dpfr)
	CHECK_CONTIG_DATA("gen_obj_dnload","destination",dpto)

	CHECK_SAME_SIZE(dpto,dpfr,"gen_obj_dnload")
	CHECK_SAME_PREC(dpto,dpfr,"gen_obj_dnload")

	/* TEST - does this work for bitmaps? */
	// For complex, PREC_SIZE returns the size of 1 complex,
	// so we need to use OBJ_N_TYPE_ELTS...
	//siz = OBJ_N_MACH_ELTS(dpto) * PREC_SIZE( OBJ_PREC_PTR(dpto) );
	// BUT - for bit, N_TYPE_ELTS is the number of bits?
//	siz = OBJ_N_TYPE_ELTS(dpto) * PREC_SIZE( OBJ_PREC_PTR(dpto) );

//fprintf(stderr,"gen_obj_dnload:  siz = %d, n_type_elts = %d, prec = %s\n",
//siz,OBJ_N_TYPE_ELTS(dpto),PREC_NAME(OBJ_PREC_PTR(dpto)) );

	siz = OBJ_N_MACH_ELTS(dpto) * OBJ_MACH_PREC_SIZE( dpto );
//fprintf(stderr,"gen_obj_dnload:  siz = %d, n_mach_elts = %d, mach_prec = %s\n",
//siz,OBJ_N_MACH_ELTS(dpto),PREC_NAME(OBJ_MACH_PREC_PTR(dpto)) );


//#ifdef CAUTIOUS
//	if( PF_MEM_DNLOAD_FN(PFDEV_PLATFORM(OBJ_PFDEV(dpfr))) == NULL ){
//		sprintf(ERROR_STRING,
//	"CAUTIOUS:  gen_obj_dnload:  Platform %s has a null download function!?",
//			PLATFORM_NAME(OBJ_PLATFORM(dpfr)));
//		ERROR1(ERROR_STRING);
//	}
//#endif // CAUTIOUS
	assert( PF_MEM_DNLOAD_FN(PFDEV_PLATFORM(OBJ_PFDEV(dpfr))) != NULL );

	( * PF_MEM_DNLOAD_FN(OBJ_PLATFORM(dpfr)) )
		(QSP_ARG  OBJ_DATA_PTR(dpto), OBJ_DATA_PTR(dpfr), siz, OBJ_PFDEV(dpfr) );
}

