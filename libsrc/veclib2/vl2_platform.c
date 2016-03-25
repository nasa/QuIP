#include "quip_config.h"


// includes, system
#include <stdio.h>

#ifdef HAVE_STDLIB_H
#include <stdlib.h>
#endif

#ifdef HAVE_STRING_H
#include <string.h>
#endif

#ifdef HAVE_MATH_H
#include <math.h>
#endif

#include "quip_prot.h"
#include "my_vl2.h"
#include "veclib/vl2_veclib_prot.h"

// When is xxx_init_platform called?

static void (*vl2_mem_upload)(QSP_ARG_DECL  void *dst, void *src, size_t siz, struct platform_device *pdp )=NULL;
static void (*vl2_mem_dnload)(QSP_ARG_DECL  void *dst, void *src, size_t siz, struct platform_device *pdp ) = NULL;
static int (*vl2_mem_alloc)(QSP_ARG_DECL  Data_Obj *dp, dimension_t size, int align) = cpu_mem_alloc;
static void (*vl2_mem_free)(QSP_ARG_DECL  Data_Obj *dp) = cpu_mem_free;
static void (*vl2_offset_data)(QSP_ARG_DECL  Data_Obj *dp, index_t o ) = default_offset_data_func;

// Update the offsets in a child after the parent is relocated

static void vl2_update_offset(QSP_ARG_DECL  Data_Obj *dp )
{
	// We don't need to SET_OBJ_OFFSET, because the child offset
	// is relative to the parent...

	// OBJ_OFFSET is in bytes, not pixels?

//fprintf(stderr,"vl2_update_offset:  obj = %s, obj_offset = 0x%x, prec_size = %d\n",
//OBJ_NAME(dp),OBJ_OFFSET(dp),PREC_SIZE(OBJ_PREC_PTR(dp)));

	// change the base pointer...
	// we originally scaled the offset by PREC_SIZE, but it appears
	// the offset is kept in bytes.
	SET_OBJ_DATA_PTR(dp,
	((char *)OBJ_DATA_PTR(OBJ_PARENT(dp)))+OBJ_OFFSET(dp) /* *PREC_SIZE(OBJ_PREC_PTR(dp))*/ );
}

static int vl2_map_buf(QSP_ARG_DECL Data_Obj *dp){ return 0; }
static int vl2_unmap_buf(QSP_ARG_DECL Data_Obj *dp){ return 0; }
static int vl2_register_buf(QSP_ARG_DECL Data_Obj *dp){ return 0; }

static void vl2_dev_info(QSP_ARG_DECL  Platform_Device *pdp)
{
	sprintf(MSG_STR,"%s:",PFDEV_NAME(pdp));
	prt_msg(MSG_STR);
	prt_msg("Sorry, vl2 device info not implemented yet!?");
}

static void vl2_info(QSP_ARG_DECL  Compute_Platform *cdp)
{
	sprintf(MSG_STR,"%s:",PLATFORM_NAME(cdp));
	prt_msg(MSG_STR);
	prt_msg("Sorry, vl2 platform info not implemented yet!?");
}

static void init_vl2_pfdevs(QSP_ARG_DECL  Compute_Platform *cpp)
{
	Platform_Device *pdp;

	pdp = new_pfdev(QSP_ARG  "CPU1");
	SET_PFDEV_PLATFORM(pdp,cpp);

	SET_PFDEV_MAX_DIMS(pdp,DEFAULT_PFDEV_MAX_DIMS);

	// set the data area for the device?
	if( ram_area_p == NO_AREA ){
		ram_area_p = pf_area_init(QSP_ARG  "ram",NULL,0L,MAX_RAM_CHUNKS,DA_RAM,pdp);
	}

	SET_PFDEV_AREA(pdp,PFDEV_GLOBAL_AREA_INDEX,ram_area_p);
	SET_AREA_PFDEV( ram_area_p, pdp );
}

void vl2_init_platform(SINGLE_QSP_ARG_DECL)
{
	Compute_Platform *cpp;
	static int inited=0;

	if( inited ){
		/* As long as we have not completely replaced the
		 * previous veclib, we can call this from vl2 or veclib...
		 */
		return;
	}

	cpp = creat_platform(QSP_ARG  "CPU", PLATFORM_CPU );

	SET_PLATFORM_FUNCTIONS(cpp,vl2)
	SET_PF_FUNC_TBL(cpp,vl2_vfa_tbl);

	push_pfdev_context(QSP_ARG  PF_CONTEXT(cpp) );

	init_vl2_pfdevs(QSP_ARG  cpp);

	if( pop_pfdev_context(SINGLE_QSP_ARG) == NO_ITEM_CONTEXT )
		ERROR1("init_ocl_platform:  Failed to pop platform device context!?");

	check_vl2_vfa_tbl(SINGLE_QSP_ARG);

	inited=1;
}

