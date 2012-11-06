#include "quip_config.h"

char VersionId_dataf_dplist[] = QUIP_VERSION_STRING;

/* information display functions */

#include <stdio.h>
#ifdef HAVE_STRING_H
#include <string.h>
#endif

#include "data_obj.h"
#include "debug.h"

/* local prototypes */
static void list_dp_flags(Data_Obj *);
static void list_sizes(Data_Obj *);
static void list_relatives(Data_Obj *);
static void init_mach_prec(mach_prec prec);
static void init_pseudo_prec(pseudo_prec prec);

const char *name_for_prec(int prec)
{
	int pp;

	if( prec < N_MACHINE_PRECS )
		return(prec_name[prec]);
	pp = prec >> N_MACHPREC_BITS;
	return(prec_name[N_MACHINE_PRECS+pp]);
}

void describe_shape(Shape_Info *shpp)
{
	/*
	sprintf(msg_str,"0x%lx:  ",(int_for_addr)shpp);
	prt_msg_frag(msg_str);
	*/

	if( shpp->si_prec == PREC_VOID ){
		prt_msg("void (no shape)");
		return;
	}

	if( HYPER_SEQ_SHAPE(shpp) )
		sprintf(msg_str,"hyperseq, %3u sequences          ",
			shpp->si_seqs);
	else if( SEQUENCE_SHAPE(shpp) )
		sprintf(msg_str,"sequence, %3u %dx%d frames             ",
			shpp->si_frames,shpp->si_rows,shpp->si_cols);
	else if( IMAGE_SHAPE(shpp) ){
		/* used to have a special case for bitmaps here, but
		 * apparently no longer needed...
		 */
		sprintf(msg_str,"image, %4u rows   %4u columns  ",
			shpp->si_rows,shpp->si_cols);
	} else if( ROWVEC_SHAPE(shpp) )
		sprintf(msg_str,"row vector, %4u elements        ",shpp->si_cols);
	else if( COLVEC_SHAPE(shpp) )
		sprintf(msg_str,"column vector, %4u elements     ",shpp->si_rows);
	else if( PIXEL_SHAPE(shpp) )
		sprintf(msg_str,"scalar                           ");
	else if( UNKNOWN_SHAPE(shpp) )
		sprintf(msg_str,"shape unknown at this time       ");
	else if( VOID_SHAPE(shpp) )
		sprintf(msg_str,"void shape                            ");
#ifdef CAUTIOUS
	else {
		sprintf(DEFAULT_ERROR_STRING,"CAUTIOUS:  describe_shape:  unrecognized object type flag 0x%x",
			shpp->si_flags);
		NWARN(DEFAULT_ERROR_STRING);
		sprintf(msg_str,"                                 ");
	}
#endif /* CAUTIOUS */
	prt_msg_frag(msg_str);

	if( BITMAP_PRECISION(shpp->si_prec) ){
#ifdef CAUTIOUS
		if( (shpp->si_prec & MACH_PREC_MASK) != BITMAP_MACH_PREC ){
			sprintf(DEFAULT_ERROR_STRING,
		"CAUTIOUS:  describe_shape:  prec = 0x%x, BIT pseudo precision is set, but machine precision is not %s!?",
				shpp->si_prec,name_for_prec(BITMAP_MACH_PREC));
			NERROR1(DEFAULT_ERROR_STRING);
		}
#endif /* CAUTIOUS */
		prt_msg("     bit");
		return;
	} else if( STRING_PRECISION(shpp->si_prec) || CHAR_PRECISION(shpp->si_prec) ){
#ifdef CAUTIOUS
		if( (shpp->si_prec & MACH_PREC_MASK) != PREC_BY ){
			sprintf(DEFAULT_ERROR_STRING,
		"CAUTIOUS:  describe_shape:  prec = 0x%x, STRING or CHAR  pseudo precision is set, but machine precision is not byte!?",
				shpp->si_prec);
			NERROR1(DEFAULT_ERROR_STRING);
		}
#endif /* CAUTIOUS */
		if( STRING_PRECISION(shpp->si_prec) )
			prt_msg("     string");
		else if( CHAR_PRECISION(shpp->si_prec) )
			prt_msg("       char");
		return;
	}

	sprintf(msg_str,"     %s",prec_name[ shpp->si_prec & MACH_PREC_MASK ] );
	prt_msg_frag(msg_str);
	if( COMPLEX_PRECISION(shpp->si_prec) ){
		if( (shpp->si_prec & MACH_PREC_MASK) == PREC_SP )
			sprintf(msg_str,", complex");
		else if( (shpp->si_prec & MACH_PREC_MASK) == PREC_DP )
			sprintf(msg_str,", dblcpx");
#ifdef CAUTIOUS
		else {
			sprintf(msg_str,", unknown_precision_cpx");
			NWARN("CAUTIOUS:  describe_shape:  unexpected complex machine precision!?");
		}
#endif /* CAUTIOUS */
	} else if( QUAT_PRECISION(shpp->si_prec) ){
		if( (shpp->si_prec & MACH_PREC_MASK) == PREC_SP )
			sprintf(msg_str,", quaternion");
		else if( (shpp->si_prec & MACH_PREC_MASK) == PREC_DP )
			sprintf(msg_str,", dblquat");
#ifdef CAUTIOUS
		else {
			sprintf(msg_str,", unknown_precision_quaternion");
			NWARN("CAUTIOUS:  describe_shape:  unexpected quaternion machine precision!?");
		}
#endif /* CAUTIOUS */
	} else {
		sprintf(msg_str,", real");
	}

	if( shpp->si_comps > 1 ){
		prt_msg_frag(msg_str);
		sprintf(msg_str,", %d components",shpp->si_comps);
	}

	if( INTERLACED_SHAPE(shpp) ){
		prt_msg_frag(msg_str);
		sprintf(msg_str,", interlaced");
	}

	prt_msg(msg_str);
}

void dump_shape(Shape_Info *shpp)
{
	int i;

	sprintf(msg_str,"shpp = 0x%lx",(int_for_addr)shpp);
	prt_msg(msg_str);

	describe_shape(shpp);
	sprintf(msg_str,"prec = 0x%x",shpp->si_prec);
	prt_msg(msg_str);
	for(i=0;i<N_DIMENSIONS;i++){
		sprintf(msg_str,"dim[%d] = %d (%d), ",i,shpp->si_type_dim[i],shpp->si_mach_dim[i]);
		if( i == N_DIMENSIONS-1 )
			prt_msg(msg_str);
		else
			prt_msg_frag(msg_str);
	}

	sprintf(msg_str,"n_elts = 0x%x (0x%x)",shpp->si_n_type_elts,shpp->si_n_mach_elts);
	prt_msg(msg_str);
	sprintf(msg_str,"mindim = 0x%x",shpp->si_mindim);
	prt_msg(msg_str);
	sprintf(msg_str,"maxdim = 0x%x",shpp->si_maxdim);
	prt_msg(msg_str);
	sprintf(msg_str,"flags = 0x%x",shpp->si_flags);
	prt_msg(msg_str);
	sprintf(msg_str,"last_subi = 0x%x",shpp->si_last_subi);
	prt_msg(msg_str);
}

void listone(Data_Obj *dp)
{
	char string[128];

	if( dp->dt_ap == NO_AREA )
		sprintf(string,"(no data area):%s", dp->dt_name );
	else
		sprintf(string,"%s:%s", dp->dt_ap->da_name, dp->dt_name );
	sprintf(msg_str,"%-20s",string);
	prt_msg_frag(msg_str);
	describe_shape(&dp->dt_shape);

	/*
	if( dp->dt_extra != NULL ){
		sprintf(msg_str,"Decl node has addr 0x%lx\n",
			(int_for_addr)dp->dt_extra);
		prt_msg(msg_str);
	}
	*/
}

/*
 * Print out information about a data object
 */

struct _flagtbl {
	const char *flagname;
	int flagmask;
} flagtbl[N_DP_FLAGS]={
	{	"sequence",		DT_SEQUENCE		},
	{	"image",		DT_IMAGE		},
	{	"row vector",		DT_ROWVEC		},
	{	"column vector",	DT_COLVEC		},
	{	"scalar",		DT_SCALAR		},
	{	"hypersequence",	DT_HYPER_SEQ		},
	{	"unknown",		DT_UNKNOWN_SHAPE	},
	{	"string",		DT_STRING		},

	{	"char",			DT_CHAR			},
	{	"quaternion",		DT_QUAT			},
	{	"complex",		DT_COMPLEX		},
	{	"multidimensional",	DT_MULTIDIM		},

	{	"bitmap",		DT_BIT			},
	{	"not data owner",	DT_NO_DATA		},
	{	"zombie",		DT_ZOMBIE		},

	{	"contiguous",		DT_CONTIG		},
	{	"contiguity checked",	DT_CHECKED		},
	{	"evenly spaced",	DT_EVENLY		},

	{	"data aligned",		DT_ALIGNED		},
	{	"locked",		DT_LOCKED		},
	{	"assigned",		DT_ASSIGNED		},

	{	"temporary object",	DT_TEMP			},
	{	"void type",		DT_VOID			},
	{	"exported",		DT_EXPORTED		},
	{	"read-only",		DT_RDONLY		},
	{	"volatile",		DT_VOLATILE		},
	{	"interlaced",		DT_INTERLACED		},
	{	"obj_list",		DT_OBJ_LIST		},
	{	"static",		DT_STATIC		},
	{	"GL buffer",		DT_GL_BUF		},
	{	"GL buffer is mapped",	DT_BUF_MAPPED		},
};

static void list_dp_flags(Data_Obj *dp)
{
	int i;
	uint32_t flags;

	sprintf(msg_str,"\tflags (0x%x):",(unsigned) dp->dt_flags);
	prt_msg(msg_str);

	/* We keep a copy of flags, and clear each bit as we display its
	 * description...  then if there are any bits left at the end, we know
	 * something has been left out of the table.
	 */

	flags = dp->dt_flags;
	for(i=0;i<N_DP_FLAGS;i++){

#ifdef CAUTIOUS
		if( flagtbl[i].flagmask == 0 ){
			sprintf(DEFAULT_ERROR_STRING,"CAUTIOUS:  list_dp_flags:  flagtbl[%d].flagmask = 0!?",i);
			NWARN(DEFAULT_ERROR_STRING);
			sprintf(DEFAULT_ERROR_STRING,"make sure flagtbl has %d initialization entries in dplist.c",N_DP_FLAGS);
			NERROR1(DEFAULT_ERROR_STRING);
		}
#endif /* CAUTIOUS */

		if( flags & flagtbl[i].flagmask ){
			sprintf(msg_str,"\t\t%s (0x%x)",flagtbl[i].flagname,
				flagtbl[i].flagmask);
			prt_msg(msg_str);

			flags &= ~flagtbl[i].flagmask;
		}
	}
	fflush(stdout);

#ifdef CAUTIOUS
	if( flags ){	/* any bits still set */
		sprintf(DEFAULT_ERROR_STRING,"CAUTIOUS:  list_dp_flags:  unhandled flag bit(s) 0x%x!?",flags);
		NWARN(DEFAULT_ERROR_STRING);
	}
#endif /* CAUTIOUS */
}

static void show_dimensions(Data_Obj *dp, dimension_t *dim_array, incr_t *inc_array)
{
	int i;
	char dn[32];

	/* this makes the singularity check somewhat superfluous,
	 * but we'll leave it in for now...
	 */
#ifdef DEBUG
if( debug & debug_data ){

	for(i=N_DIMENSIONS-1;i>dp->dt_maxdim;i--){
		strcpy(dn,dimension_name[i]);
		if( dim_array[i] > 1 ) strcat(dn,"s");
		sprintf(msg_str,"\t%4u %12s   inc=%d", dim_array[i],
			dn, inc_array[i]);
		prt_msg(msg_str);
	}
}
#endif /* DEBUG */

	/* show only the dimensions which are between mindim and maxdim */

	for(i=dp->dt_maxdim;i>=dp->dt_mindim;i--){
		strcpy(dn,dimension_name[i]);
		if( dim_array[i] > 1 ) strcat(dn,"s");
		sprintf(msg_str,"\t%4u %12s   inc=%d", dim_array[i],
			dn, inc_array[i]);
		prt_msg(msg_str);
	}

#ifdef DEBUG
if( debug & debug_data ){
	for(i=dp->dt_mindim-1;i>=0;i--){
		strcpy(dn,dimension_name[i]);
		if( dim_array[i] > 1 ) strcat(dn,"s");
		sprintf(msg_str,"\t%4u %12s   inc=%d", dim_array[i],
			dn, inc_array[i]);
		prt_msg(msg_str);
	}
}
#endif /* DEBUG */

}

static void list_sizes(Data_Obj *dp)
{
	sprintf(msg_str,"\tmindim = %d, maxdim = %d",
		dp->dt_mindim,dp->dt_maxdim);
	prt_msg(msg_str);

	show_dimensions(dp,dp->dt_type_dim,dp->dt_type_inc);
	if( debug & debug_data ){
		prt_msg("machine type dimensions:");
		show_dimensions(dp,dp->dt_mach_dim,dp->dt_mach_inc);
	}
}


static void list_relatives(Data_Obj *dp)
{
	if( dp->dt_parent != NO_OBJ ){
		sprintf(msg_str,"\tparent data object:  %s",
			dp->dt_parent->dt_name);
		prt_msg(msg_str);

		sprintf(msg_str,"\tdata offset:  0x%x", dp->dt_offset);
		prt_msg(msg_str);
	}
	if( dp->dt_children != NO_LIST &&
		dp->dt_children->l_head != NO_NODE ){

		Node *np;
		Data_Obj *dp2;

		sprintf(msg_str,"\tsubobjects:");
		prt_msg(msg_str);
		np = dp->dt_children->l_head;
		while( np != NO_NODE ){
			dp2=(Data_Obj *) np->n_data;
			sprintf(msg_str,"\t\t%s",dp2->dt_name);
			prt_msg(msg_str);
			np=np->n_next;
		}
	}
}

void list_context(QSP_ARG_DECL  Data_Obj *dp)
{
	Item_Context *icp;
	Node *np;
	Item *ip;

	/* objects don't keep a ptr to their context,
	 * so we search all the contexts until we find it.
	 *
	 * But subscripted objects won't show up,
	 * so if the object has a parent, list the context
	 * of the parent instead.
	 */

	if( dp->dt_parent != NO_OBJ ){
		list_context(QSP_ARG  dp->dt_parent);
		return;
	}

	/* BUG this is the list of the current context stack,
	 * not ALL the contexts!?
	 */
	np=CONTEXT_LIST(dobj_itp)->l_head;
#ifdef CAUTIOUS
	if( np == NO_NODE )
		NERROR1("CAUTIOUS:  list_context:  no data object context");
#endif /* CAUTIOUS */
	while(np!=NO_NODE){
		icp=(Item_Context *)np->n_data;
		/* can we search this context only? */
/*
sprintf(error_string,
"Searching context %s for object %s",icp->ic_name,dp->dt_name);
advise(error_string);
*/
		ip=fetch_name(dp->dt_name,icp->ic_nsp);
		if( ((Data_Obj *)ip) == dp ){	/* found it! */
			sprintf(msg_str,"\titem context:  %s",icp->ic_name);
			prt_msg(msg_str);
			return;
		}
		np=np->n_next;
	}
#ifdef CAUTIOUS
	sprintf(error_string,
"CAUTIOUS:  list_context:  No item context found for object %s!?",
		dp->dt_name);
	NWARN(error_string);
#endif /* CAUTIOUS */
}

static void list_data(Data_Obj *dp)
{
	dimension_t n;

	if( IS_BITMAP(dp) )
		n = BITMAP_WORD_COUNT(dp);
	else
		n = dp->dt_n_mach_elts;

	sprintf(msg_str,"\t%d %s element%s",n,name_for_prec(MACHINE_PREC(dp)),n==1?"":"s");
	prt_msg(msg_str);

	sprintf(msg_str,"\tdata at   0x%lx",(int_for_addr)dp->dt_data);
	prt_msg(msg_str);
	if( IS_BITMAP(dp) ){
		sprintf(msg_str,"\t\tbit0 = %d",dp->dt_bit0);
		prt_msg(msg_str);
	}
}

#ifdef DEBUG

static void list_increments(Data_Obj *dp)
{
	int i;

	for(i=0;i<N_DIMENSIONS;i++){
		sprintf(msg_str,"\tincr[%d] = %d (%d)",i,dp->dt_type_inc[i],dp->dt_mach_inc[i]);
		prt_msg(msg_str);
	}
}
#endif /* DEBUG */

void longlist(QSP_ARG_DECL  Data_Obj *dp)
{
	listone(dp);
	list_context(QSP_ARG  dp);
	list_sizes(dp);
	list_data(dp);
	list_relatives(dp);
	list_dp_flags(dp);
#ifdef DEBUG
if( debug & debug_data ){
list_increments(dp);
dump_shape(&dp->dt_shape);
}
#endif /* DEBUG */
}

void info_area(QSP_ARG_DECL  Data_Area *ap)
{
	List *lp;
	Node *np;
	Data_Obj *dp;

	lp=dobj_list(SINGLE_QSP_ARG);
	if( lp==NO_LIST ) return;
	np=lp->l_head;
	while( np != NO_NODE ){
		dp = (Data_Obj *)np->n_data;
		if( dp->dt_ap == ap )
			listone( dp );
		np=np->n_next;
	}
}

void info_all_dps(SINGLE_QSP_ARG_DECL)
{
	List *lp;
	Node *np;

	lp=da_list(SINGLE_QSP_ARG);
	if( lp==NO_LIST ) return;
	np=lp->l_head;
	while( np != NO_NODE ){
		info_area( QSP_ARG  (Data_Area *) np->n_data );
		np=np->n_next;
	}
}

static void init_mach_prec(mach_prec p)
{
	switch(p){
		case PREC_NONE: siztbl[p]=0;			prec_name[p]="no_prec";	break;
		case PREC_BY: siztbl[p]=sizeof(char);		prec_name[p]="byte";	break;
		case PREC_IN: siztbl[p]=sizeof(short);		prec_name[p]="short";	break;
		case PREC_DI: siztbl[p]=sizeof(int32_t);	prec_name[p]="int32";	break;
		case PREC_LI: siztbl[p]=sizeof(int64_t);	prec_name[p]="int64";	break;
		case PREC_SP: siztbl[p]=sizeof(float);		prec_name[p]="float";	break;
		case PREC_DP: siztbl[p]=sizeof(double);		prec_name[p]="double";	break;
		case PREC_UBY: siztbl[p]=sizeof(char);		prec_name[p]="u_byte";	break;
		case PREC_UIN: siztbl[p]=sizeof(short);		prec_name[p]="u_short";	break;
		case PREC_UDI: siztbl[p]=sizeof(uint32_t);	prec_name[p]="uint32";	break;
		case PREC_ULI: siztbl[p]=sizeof(uint64_t);	prec_name[p]="uint64";	break;
		case N_MACHINE_PRECS:							break;
	}
}


/* these are not real precisions, but are used by
 * the vector expression parser...
 * They don't need a siztbl entry...
 */

static void init_pseudo_prec(pseudo_prec pp)
{
	switch(pp){
		case PP_NORM:	prec_name[N_MACHINE_PRECS+pp]="normal";	break;
		case PP_COLOR:	prec_name[N_MACHINE_PRECS+pp]="color";	break;
		case PP_MIXED:	prec_name[N_MACHINE_PRECS+pp]="mixed";	break;
		case PP_VOID:	prec_name[N_MACHINE_PRECS+pp]="void";	break;
		case PP_CPX:	prec_name[N_MACHINE_PRECS+pp]="complex";break;
		case PP_QUAT:	prec_name[N_MACHINE_PRECS+pp]="quaternion";break;
		case PP_CHAR:	prec_name[N_MACHINE_PRECS+pp]="char";	break;
		case PP_DBLCPX:	prec_name[N_MACHINE_PRECS+pp]="dblcpx";	break;
		case PP_DBLQUAT:prec_name[N_MACHINE_PRECS+pp]="dblquat";break;
		case PP_STRING:	prec_name[N_MACHINE_PRECS+pp]="string";	break;
		case PP_BIT:	prec_name[N_MACHINE_PRECS+pp]="bit";	break;
		case PP_LIST:	prec_name[N_MACHINE_PRECS+pp]="list";	break;
		case PP_ANY:	prec_name[N_MACHINE_PRECS+pp]="any";	break;
		case N_PSEUDO_PRECS:			break;
	}
}

void sizinit(void)			/**/
{
	/* mach_prec */ int mp;
	/* pseudo_prec */ int pp;

	for(mp=FIRST_MACH_PREC;mp<N_MACHINE_PRECS;mp++)
		init_mach_prec((mach_prec)mp);

	for(pp=FIRST_PSEUDO_PREC;pp<N_PSEUDO_PRECS;pp++)
		init_pseudo_prec((pseudo_prec)pp);
}

void show_space_used(Data_Obj *dp)
{
	sprintf(msg_str,"%s:\t\t0x%lx",dp->dt_name,(int_for_addr)dp->dt_data);
	prt_msg(msg_str);
}


