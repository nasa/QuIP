#include "quip_config.h"

char VersionId_dataf_areas[] = QUIP_VERSION_STRING;

/*
 * Data areas were introduced back when the sky warrior had
 * to have it's objects living in special VME memory.
 * The construct is pretty much obsolete nowadays, but it
 * is not worth the trouble to get rid of it at present...
 */

#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include "query.h"
#include "items.h"
#include "getbuf.h"
#include "savestr.h"
#include "data_obj.h"

Data_Area *curr_ap=NO_AREA, *ram_area=NO_AREA;

#define DATA_AREA_STACK_SIZE	256
static Data_Area *data_area_stack[DATA_AREA_STACK_SIZE];
static int i_stack=0;

static int n_areas=0;

ITEM_INTERFACE_DECLARATIONS(Data_Area,data_area)

void push_data_area(Data_Area *ap)
{
	/* This test is not CAUTIOUS because we could encounter
	 * infinite recursion in a script, or something similar...
	 */
	if( i_stack >= DATA_AREA_STACK_SIZE ){
		sprintf(DEFAULT_ERROR_STRING,
	"Data area stack size limit (%d) reached.",DATA_AREA_STACK_SIZE);
		NWARN(DEFAULT_ERROR_STRING);
		advise("The value can be increased in areas.c");
		NERROR1("push_data_area() failed.");
	}
#ifdef CAUTIOUS
	if( i_stack < 0 ){
		sprintf(DEFAULT_ERROR_STRING,
	"CAUTIOUS:  push_data_area:  negative i_stack = %d",i_stack);
		NERROR1(DEFAULT_ERROR_STRING);
	}
#endif /* CAUTIOUS */
	data_area_stack[i_stack++]=curr_ap;
	curr_ap=ap;
}

void pop_data_area(void)
{
	i_stack--;
#ifdef CAUTIOUS
	if( i_stack < 0 )
		NERROR1("CAUTIOUS:  Data area stack underflow.");
#endif
	curr_ap = data_area_stack[i_stack];
}

List *da_list(SINGLE_QSP_ARG_DECL)
{
	if( data_area_itp == NO_ITEM_TYPE ) data_area_init(SINGLE_QSP_ARG);
	return( item_list(QSP_ARG  data_area_itp) );
}

void a_init(void)
{
	if( n_areas==0 ){
		curr_ap=NO_AREA;
		sizinit();
	}
}

Data_Area *default_data_area(SINGLE_QSP_ARG_DECL)
{
	if( curr_ap == NO_AREA ) dataobj_init(SINGLE_QSP_ARG);
	return(curr_ap);
}

void set_data_area(Data_Area *ap)
{
	curr_ap = ap;
}

Data_Area *			/**/
area_init( QSP_ARG_DECL  const char *name, u_char *buffer, uint32_t siz, int nobjs, uint32_t flags )
{
	Data_Area *ap;
	int is_ram=0;

	if( flags == DA_RAM ) is_ram=1;

	a_init();

	ap = new_data_area(QSP_ARG  name);
	if( ap == NO_AREA ) return(ap);

	if( buffer == NULL ){
		ap->da_ma_p = NO_MEMORY_AREA;
	} else {
		ap->da_ma_p =
			(Memory_Area *) getbuf(sizeof(Memory_Area));
		ap->da_base = buffer;
		ap->da_memsiz=siz;
		ap->da_memfree=siz;

	/* this doesn't have to be nobjs exactly, but it's a good approx */
		freeinit(&ap->da_freelist,(count_t)nobjs,siz);
	}

	ap->da_flags = flags;

	curr_ap=ap;

	return(ap);
}

Data_Area *
new_area( QSP_ARG_DECL  const char *s, uint32_t siz, int n )
{
	u_char *buf;
	Data_Area *ap;

	buf=(u_char *)getbuf(siz);
#ifdef CAUTIOUS
	if( buf==((u_char *)NULL) ) NERROR1("no mem for data area");
#endif /* CAUTIOUS */

	ap=area_init(QSP_ARG  s,buf,siz,n,DA_RAM);
	if( ap==NO_AREA ) givbuf((char *)buf);

	return(ap);
}

void data_area_info( Data_Area *ap )
{
	sprintf(msg_str,"AREA %-30s",ap->da_name);
	prt_msg_frag(msg_str);
	switch(ap->da_flags & DA_TYPE_MASK ){
		case DA_RAM:
			prt_msg("   ram");
			break;
		case DA_CUDA_GLOBAL:
			prt_msg("   CUDA device global memory");
			break;
		case DA_CUDA_HOST:
			prt_msg("   CUDA device host memory");
			break;
		case DA_CUDA_HOST_MAPPED:
			prt_msg("   CUDA device host mapped  memory");
			break;
#ifdef DA_CUDA_CONSTANT
		case DA_CUDA_CONSTANT:
			prt_msg("   CUDA device constant memory");
			break;
#endif /* defined DA_CUDA_CONSTANT */

#ifdef CAUTIOUS
		default:
			sprintf(DEFAULT_ERROR_STRING,"CAUTIOUS:  data_area_info %s:  Bad data area type flag 0x%x (flags = 0x%x)",
				ap->da_name,ap->da_flags & DA_TYPE_MASK, ap->da_flags);
			NWARN(DEFAULT_ERROR_STRING);
			break;
#endif /* CAUTIOUS */
	}
}

void list_area( Data_Area *ap )
{
	sprintf(msg_str,"AREA %-10s",ap->da_name);
	prt_msg_frag(msg_str);
	if( ap->da_memsiz != 0 ){
		sprintf(msg_str,"   0x%-10x total bytes, 0x%-10x remaining",
			ap->da_memsiz,
			ap->da_memfree);
		prt_msg(msg_str);
	} else prt_msg("");
}

/* This const business causes a lot of warnings if not done right...
 * The way it is here doesn't complain on the newest gcc, but complains
 * on a slightly older version, while taking the const out of the typedef
 * and putting down in the cast shuts up the older version but makes
 * the newer one complain...  We assume the newer one is correct, this
 * seems to make sense as best as I can figure out the semantics
 * of the const keyword...  jbm
 */

typedef CONST Data_Obj * DataPtr;

int dp_addr_cmp( CONST void *dpp1, CONST void *dpp2 )
{
	CONST Data_Obj *dp1,*dp2;

	dp1= *(CONST DataPtr *)dpp1;

	dp2= *(CONST DataPtr *)dpp2;

	if( dp1->dt_data > dp2->dt_data ) return(1);
	else if( dp1->dt_data < dp2->dt_data ) return(-1);
	else return(0);
}

void show_area_space( QSP_ARG_DECL  Data_Area *ap )
{
	List *lp;
	Node *np;
	Data_Obj **dp_list;
	Data_Obj *dp;
	int n,i;

	if( data_area_itp == NO_ITEM_TYPE ) data_area_init(SINGLE_QSP_ARG);

	lp=dobj_list(SINGLE_QSP_ARG);
	if( lp==NO_LIST ) return;

	n=eltcount(lp);
	if( n == 0 ){
		sprintf(msg_str,"Area %s has no objects.",ap->da_name);
		prt_msg(msg_str);
	} else {
		sprintf(msg_str,"Area %s:  %d object%s:",ap->da_name,n,n==1?"":"s");
		prt_msg(msg_str);
	}

	dp_list = (Data_Obj **) getbuf( n * sizeof(Data_Obj *) );
	if( dp_list == NULL ) mem_err("show_area_space");

	np=lp->l_head;
	i=0;
	while( np != NO_NODE ){
		dp = (Data_Obj *) np->n_data;
		if( dp->dt_ap == ap ){
			dp_list[i] = dp;
			i++;
		}
		np=np->n_next;
	}
	if( i == 0 ) return;

#ifdef PC
	qsort((void *)dp_list,(size_t)i,sizeof(Data_Obj *),dp_addr_cmp);
#else /* PC */
	qsort((char *)dp_list,(int)i,sizeof(Data_Obj *),dp_addr_cmp);
#endif /* PC */

	for(i=0;i<n;i++)
		show_space_used( dp_list[i] );
	
	givbuf(dp_list);
}

