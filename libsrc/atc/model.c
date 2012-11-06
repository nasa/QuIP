#include "quip_config.h"

#ifdef HAVE_X11

char VersionId_atc_model[] = QUIP_VERSION_STRING;

#include <math.h>
#include <stdlib.h>	/* drand48() */
#include "vec_util.h"	/* used for saliency map filtering */
#include "items.h"
#include "debug.h"
#include "nvf_api.h"	/* fft2d etc */
#include "vecgen.h"	/* FVSET etc */
#include "rn.h"
#include "xsupp.h"	/* event_loop() */

#include "conflict.h"
#include "draw.h"
#include "query.h"

#ifdef DEBUG
u_long debug_model=0;
#endif /* DEBUG */

double assumed_min_speed=MIN_SPEED;
double assumed_max_speed=MAX_SPEED;

/* BUG these vars are used to pass values from paths_are_in_conflict() to
 * check_pair_for_conflit()...
 */

static atc_type travel_distance, travel_time;
static atc_type min_distance;


#define CONFLICT_FOUND_VAR_NAME	"conflict_found"

#define ANY_ALTITUDE (-1)
static Altitude curr_alt=ANY_ALTITUDE;

List *crossing_lp=NO_LIST;

static int fixation_delay=0;	/* if this is >0, pause model after each fixation */
				/* this is so we can watch! */


atc_type heading_ecc_thresh = DEFAULT_HEADING_ECC_THRESH;

static boolean display_model_state=TRUE;
static boolean show_crossings=TRUE;
static int pretty_plot=15;		/* slows things down, but looks nice */
static int ugliness=0;

static Flight_Path *model_resp1,*model_resp2;
static List *candidate_lp=NO_LIST;	/* list of Point ptrs */
static u_long n_model_fixations;
static u_long n_recorded_fixations;

/* global vars for forgetting */
static Data_Obj *p_forgetting_dp[N_INFO_TYPES]={NO_OBJ,NO_OBJ,NO_OBJ};
static Info_Type forget_info_index;
static age_t max_forgetting_age;
static float *p_forgetting_array;
static const char *info_type_names[N_INFO_TYPES]={NULL,NULL,NULL};
static atc_type fixation_theta;

static Data_Obj *fixation_dp=NO_OBJ;
static u_long max_recorded_fixations;

static Strategy *curr_stratp=NO_STRATEGY;


static Item_Type *mo_itp=NO_ITEM_TYPE;

static Point *last_fixp, *this_fixp;
/*static Point curr_fix; */
static Point curr_fix, saved_fix;

/* local prototypes */
static void set_strat_params(SINGLE_QSP_ARG_DECL);
static Point * get_saliency(SINGLE_QSP_ARG_DECL);
static void get_crossing_candidates(SINGLE_QSP_ARG_DECL);
static void redraw_crossings(SINGLE_QSP_ARG_DECL);

ITEM_INTERFACE_DECLARATIONS(Strategy,strat)
ITEM_INTERFACE_DECLARATIONS(Pair,pair)
ITEM_INTERFACE_DECLARATIONS(Model_Obj,model_object)


#define PICK_MODEL_OBJECT(pmpt)		pick_model_object(QSP_ARG  pmpt)
#define PICK_STRAT(pmpt)		pick_strat(QSP_ARG  pmpt)

#define LAST_OBJ_FIXATED	(last_plane_fixated!=NO_FLIGHT_PATH ? last_plane_fixated : last_tag_fixated)
			
static Indexed_Name scan_names[N_SCAN_TYPES]={
	{	RANDOM,		"random"	},
	{	NEAREST,	"nearest"	},
	{	CLOCKWISE,	"clockwise"	},
	{	MAX_INFO,	"max_info"	}
};

int pick_indexed_name( QSP_ARG_DECL   char *prompt, int n, Indexed_Name *inp )
{
	const char **choices;
	int i;

	choices = (const char **)(getbuf( n * sizeof(char *) ));
	for(i=0;i<n;i++)
		choices[i] = inp[i].in_name;

	i = WHICH_ONE( prompt, n, choices );

	givbuf(choices);

	if( i < 0 ) return(i);

	return( inp[i].in_index );
}

char *tell_indexed_name(Indexed_Name *list,int n,int index)
{
	while(n--){
		if( list->in_index == index ) return( list->in_name);
	}
	return(NULL);
}

static COMMAND_FUNC( get_scan_type)
{
	int i;

	i=pick_indexed_name( QSP_ARG  "scan type",N_SCAN_TYPES,scan_names);

	if( curr_stratp == NO_STRATEGY ) return;

	curr_stratp->strat_scan = (Scan_Type)i;
}

static COMMAND_FUNC( set_use_crossings )
{
	int cflag;

	cflag = ASKIF("check altitudes of known crossings first?");

	if( curr_stratp == NO_STRATEGY ) return;

	if( cflag )
		curr_stratp->strat_flags |= STRAT_USES_CROSSINGS;
	else
		curr_stratp->strat_flags &= ~STRAT_USES_CROSSINGS;
}

static COMMAND_FUNC( set_use_density )
{
	int sflag;

	sflag = ASKIF("use density of traffic to determine saliency?");

	if( curr_stratp == NO_STRATEGY ) return;

	if( sflag )
		curr_stratp->strat_flags |= STRAT_USES_SALIENCY;
	else
		curr_stratp->strat_flags &= ~STRAT_USES_SALIENCY;
}

static COMMAND_FUNC( set_use_altitude )
{
	int aflag;

	aflag = ASKIF("check headings of craft at similar altitudes?");

	if( curr_stratp == NO_STRATEGY ) return;

	if( aflag )
		curr_stratp->strat_flags |= STRAT_USES_ALTITUDE;
	else
		curr_stratp->strat_flags &= ~STRAT_USES_ALTITUDE;
}

static Command strat_param_ctbl[]={
{ "scan_type",		get_scan_type,		"specify scan type"				},
{ "use_crossings",	set_use_crossings,	"enable use of crossing info"			},
{ "use_altitude",	set_use_altitude,	"preferentially search at current alt"		},
{ "use_density",	set_use_density,	"enable use of density-based saliency map"	},
{ "quit",		popcmd,			"exit submenu"					},
{ NULL_COMMAND											}
};

static void set_strat_params(SINGLE_QSP_ARG_DECL)
{
	PUSHCMD(strat_param_ctbl,"strat_params");
}

static COMMAND_FUNC( create_strategy )
{
	const char *s;
	Strategy *save_sp;

	save_sp = curr_stratp;

	s=NAMEOF("name for new strategy");
	curr_stratp = new_strat(QSP_ARG  s);

	set_strat_params(SINGLE_QSP_ARG);

	if( curr_stratp == NO_STRATEGY ){
		advise("reinstating previous strategy");
		curr_stratp = save_sp;
	}
}

List *model_list(SINGLE_QSP_ARG_DECL)
{
	return( item_list(QSP_ARG  mo_itp) );
}

void apply_to_models(QSP_ARG_DECL  void (*func)(QSP_ARG_DECL  Model_Obj *))
{
	Node *np;
	List *lp;

	lp = model_list(SINGLE_QSP_ARG);
	if( lp == NO_LIST ) return;

	np = lp->l_head;
	while(np!=NO_NODE){
		Model_Obj *mop;
		mop = (Model_Obj *)(np->n_data);
		(*func)(QSP_ARG  mop);
		np = np->n_next;
	}
}

static void apply_to_pairs( QSP_ARG_DECL  void (*func)(QSP_ARG_DECL  Pair *) )
{
	List *lp;
	Node *np;
	Pair *prp;

	lp = item_list(QSP_ARG  pair_itp);
	if( lp==NO_LIST ) return;

	np=lp->l_head;
	while(np!=NO_NODE){
		prp=(Pair *)(np->n_data);
		(*func)(QSP_ARG  prp);
		np=np->n_next;
	}
}

static void rls_model_object(QSP_ARG_DECL  Model_Obj *mop)
{
	del_model_object(QSP_ARG  mop->mo_name);
	rls_str(mop->mo_name);
}

static void rls_pair(QSP_ARG_DECL  Pair *prp)
{
	del_pair(QSP_ARG  prp->pr_name);
	rls_str(prp->pr_name);
}

void clear_models(SINGLE_QSP_ARG_DECL)
{
	List *lp; Node *np;

	if( mo_itp == NO_ITEM_TYPE ) return;

	lp=item_list(QSP_ARG  mo_itp);
	if( lp==NO_LIST ) return;

	while( (np=remHead(lp)) != NO_NODE ){
		Model_Obj *mop;

		mop=(Model_Obj *)np->n_data;
		rls_model_object(QSP_ARG  mop);
		lp=item_list(QSP_ARG  mo_itp);
	}

	lp=item_list(QSP_ARG  pair_itp);
	if( lp==NO_LIST ) return;

	while( (np=remHead(lp)) != NO_NODE ){
		Pair *prp;
		prp=(Pair *)(np->n_data);
		rls_pair(QSP_ARG  prp);
		lp=item_list(QSP_ARG  pair_itp);
	}
}

void init_model_object(QSP_ARG_DECL  Flight_Path *fpp)
{
	Model_Obj *mop;

	mop = new_model_object(QSP_ARG  fpp->fp_name);
	if( mop == NO_MODEL_OBJECT ){
		sprintf(ERROR_STRING,"Unable to create model object for flight %s",fpp->fp_name);
		WARN(ERROR_STRING);
		return;
	}
	mop->mo_fpp = fpp;
	mop->mo_flags = ALT_COLOR ? ALTITUDE_BIT : 0;
}

void init_model_objects(SINGLE_QSP_ARG_DECL)
{
	apply_to_planes(QSP_ARG  init_model_object);
}

static void model_object_info(QSP_ARG_DECL  Model_Obj *mop)
{
	sprintf(msg_str,"%s at %3.1f %3.1f",mop->mo_name,
		mop->mo_fpp->fp_plane_loc.p_x,
		mop->mo_fpp->fp_plane_loc.p_y);
	prt_msg_frag(msg_str);

	sprintf(msg_str,"\t(%3.1f, %3.1f)",
		mop->mo_plane_dist,mop->mo_tag_dist);
	prt_msg_frag(msg_str);

	if( HEADING_KNOWN(mop) )
		sprintf(msg_str,"   hd: %3.0f",mop->mo_fpp->fp_theta);
	else
		sprintf(msg_str,"   hd: N/A");
	prt_msg_frag(msg_str);

	if( ALT_KNOWN(mop) )
		sprintf(msg_str,"   alt: %3ld",mop->mo_fpp->fp_altitude);
	else
		sprintf(msg_str,"   alt: N/A");
	prt_msg_frag(msg_str);

	if( SPEED_KNOWN(mop) )
		sprintf(msg_str,"   speed: %3.0f",mop->mo_fpp->fp_speed);
	else
		sprintf(msg_str,"   speed: N/A");

	prt_msg(msg_str);
}

COMMAND_FUNC( model_state_info )
{
	apply_to_models(QSP_ARG  model_object_info);
}

/* probably should go in draw.c */

static void render_model_obj(QSP_ARG_DECL  Model_Obj *mop)
{
	u_long save_altcolor;

	default_plane_color = GRAY;

	save_altcolor = ALT_COLOR;

	if( ALT_KNOWN(mop) )
		atc_flags |= ALT_COLOR_BIT;

	if( HEADING_KNOWN(mop) )
		draw_plane(mop->mo_fpp,color_of(mop->mo_fpp));
	else
		draw_plane_loc(mop->mo_fpp,color_of(mop->mo_fpp));

	draw_model_tag(mop,DATA_TAG_COLOR);

	DRAW_TAG_LINE(mop->mo_fpp,TAG_LINE_COLOR);

	if( save_altcolor == 0 )
		atc_flags &= ~ALT_COLOR_BIT;
}

/* See if the symbol is close to the current fixation;
 * If it is, then set the HEADING flag.
 * The list has been sorted by increasing distance from fixation,
 * so when the min distance is greater than the threshold, we
 * can stop.
 */

void update_headings(SINGLE_QSP_ARG_DECL)
{
	List *lp;
	Node *np;

	lp = model_list(SINGLE_QSP_ARG);
	if( lp == NO_LIST ) return;

	np=lp->l_head;

	while( np != NO_NODE ){
		Model_Obj *mop;

		mop=(Model_Obj *)(np->n_data);
		if( mop->mo_plane_dist < heading_ecc_thresh ){
			if( display_model_state ){
				u_long save;

				if( ! HEADING_KNOWN(mop) )
					/* erase circle */
					draw_plane_loc(mop->mo_fpp,BLACK);

				/* We usually don't need to redraw this,
				 * but we do so in case we now know the
				 * altitude; if we do we will display the
				 * icon w/ color coding.
				 */

				save=ALT_COLOR;
				if( ALT_KNOWN(mop) ) atc_flags |= ALT_COLOR_BIT;
				draw_plane(mop->mo_fpp,color_of(mop->mo_fpp));
				if( save == 0 ) atc_flags &= ~ALT_COLOR_BIT;
			}
			mop->mo_flags |= HEADING_BIT;
			mop->mo_info_age[HEADING]=0;

		} else if( mop->mo_tag_dist > heading_ecc_thresh ){
			return;
		}

		np = np->n_next;
	}
}

		
	
static void compute_eccentricity(QSP_ARG_DECL  Model_Obj *mop)
{
	Flight_Path *fpp;

	fpp = mop->mo_fpp;
	mop->mo_plane_dist = dist(this_fixp,&fpp->fp_plane_loc);
	mop->mo_tag_dist = dist(this_fixp,&fpp->fp_tag_loc);
}

static void compute_eccentricities(SINGLE_QSP_ARG_DECL)
{
	apply_to_models(QSP_ARG   compute_eccentricity );
}

static void sort_models(SINGLE_QSP_ARG_DECL)
{
	Node *np;
	List *lp;

	lp = item_list(QSP_ARG  mo_itp);
	np=lp->l_head;
	while(np!=NO_NODE){
		Model_Obj *mop;
		atc_type d;

		mop=(Model_Obj *)(np->n_data);
		d = MIN( mop->mo_plane_dist , mop->mo_tag_dist );
		np->n_pri = (short int)(- floor(d));
		np=np->n_next;
	}
	p_sort(lp);

	/* BUG make sure that we have sorted correctly! */
}

void report_model_rt(SINGLE_QSP_ARG_DECL)
{
	end_simulation_run(QSP_ARG  "Conflict FOUND");
}

void end_simulation_run(QSP_ARG_DECL  char *reason)
{
	char str[16];

	if( verbose ){
		sprintf(ERROR_STRING,"%s after %ld fixations, %s - %s",reason,
			n_model_fixations,model_resp1->fp_name,model_resp2->fp_name);
		advise(ERROR_STRING);
	}

	sprintf(str,"%ld",n_model_fixations);
	ASSIGN_VAR("n_model_fixations",str);
}

static void age_model_info(QSP_ARG_DECL  Model_Obj *mop)
{
	int i;

	for(i=0;i<N_INFO_TYPES;i++)
		if( mop->mo_info_age[i] >= 0 )
			mop->mo_info_age[i] ++;
}

static void age_information(SINGLE_QSP_ARG_DECL)
{
	apply_to_models(QSP_ARG   age_model_info );
}

static void refresh_drawn_crossing(QSP_ARG_DECL  Pair *prp)
{
	if( ! HAS_CROSSING(prp) ) return;
	if( ! CROSSING_DRAWN(prp) ) return;
	draw_crossing(prp,CROSSING_COLOR);
}

/* This is tricky - erasing this crossing may partially obliterate other
 * crossings involving the aircraft - so we redraw any drawn crossings
 * involving either of the pairs.
 */

static void erase_crossing(QSP_ARG_DECL  Pair *prp)
{
	draw_crossing(prp,BLACK);
	apply_to_pairs( QSP_ARG  refresh_drawn_crossing );
}

static Model_Obj *model_of_interest;
static List *interest_lp;

static void check_object_membership(QSP_ARG_DECL  Pair *prp)
{
	if( prp->pr_mop1 == model_of_interest || prp->pr_mop2 == model_of_interest ){
		Node *np;

		np = mk_node(prp);
		addHead(interest_lp,np);
	}
}

static List *pairs_with_object(QSP_ARG_DECL  Model_Obj *mop)
{
	interest_lp = new_list();
	model_of_interest = mop;
	apply_to_pairs( QSP_ARG  check_object_membership );
	return( interest_lp );
}

static void forget_something(QSP_ARG_DECL  Model_Obj *mop)
{
	age_t age;
	float prob,sample;
	List *lp; Node *np;
#ifdef LINUXPPC
extern double drand48(void);	/* until we get compiler flags right for include file... */
#endif /* LINUXPPC */

	age = mop->mo_info_age[forget_info_index];
	if( age < 0 ) return;
	age = MIN(age,max_forgetting_age);
	prob = p_forgetting_array[ age ];
	sample = drand48();

	if( sample < prob ){

		/* forget it! */
if( verbose ){
sprintf(ERROR_STRING,"forgetting %s of flight %s",info_type_names[forget_info_index],
mop->mo_name);
advise(ERROR_STRING);
}

		mop->mo_info_age[ forget_info_index ] = (-1);

		if( display_model_state ){
			/* erase the old info before we change the flags */
			switch( forget_info_index ){
				case HEADING:
					lp=pairs_with_object(QSP_ARG  mop);
					np=lp->l_head;
					while(np!=NO_NODE){
						Pair *prp;
						prp=(Pair *)np->n_data;
						if( CROSSING_DRAWN(prp) ){
if( verbose ){
sprintf(ERROR_STRING,"forgetting (erasing) crossing for pair %s",prp->pr_name);
advise(ERROR_STRING);
}
							erase_crossing(QSP_ARG  prp);
						}
						np=np->n_next;
					}
					dellist(lp);
					draw_plane(mop->mo_fpp,BLACK);
					break;
				case ALTITUDE:
				case SPEED:
					draw_model_tag(mop,BLACK);
					break;
				case N_INFO_TYPES: break; /* silence compiler */
			}
		}
		/* this is redundant, but we still use it for now... */
		mop->mo_flags &= ~( 1 << forget_info_index );

		if( display_model_state ){
			/* now represent the new state */
			switch( forget_info_index ){
				case HEADING:
					draw_plane_loc(mop->mo_fpp,color_of(mop->mo_fpp));
					break;
				case ALTITUDE:
				case SPEED:
					draw_model_tag(mop,DATA_TAG_COLOR);
					break;
				case N_INFO_TYPES: break; /* silence compiler */
			}
		}
	}
}

/* Forget information...
 *
 * The implementation is fairly general:  foreach information type,
 * a data vector can be specified which gives the probability of forgetting
 * as a function of age (in fixations).  If no vector has been specified,
 * there is no forgetting.
 *
 * For algorithms which look for conflicts with a particular plane, the
 * algorithm is free to reset the age to 0 to inhibit forgetting; we
 * might think of this as rehearsal.
 */

static void forget_things(SINGLE_QSP_ARG_DECL)
{
	int i;

	for(i=0;i<N_INFO_TYPES;i++){
		/* We don't forget any altitude info if color coding is on... */
		if( i!=ALTITUDE || ! ALT_COLOR ){
			if( p_forgetting_dp[i] != NO_OBJ ){	/* forgetting specified? */
				forget_info_index = (Info_Type)i;		/* set global */
				max_forgetting_age = (age_t)(p_forgetting_dp[i]->dt_cols-1);
				p_forgetting_array = (float *)(p_forgetting_dp[i]->dt_data);
				apply_to_models(QSP_ARG   forget_something );
			}
		}
	}
}

static void record_fixation(Point *ptp)
{
	float *data;

	/* For now, we set the time to be the fixation number... */

	data = (float *)(fixation_dp->dt_data);
	data += n_recorded_fixations * fixation_dp->dt_pinc;
	n_recorded_fixations++;

	data[0] = (float) n_model_fixations;	/* the time */
	data[1] = ptp->p_x;
	data[2] = ptp->p_y;
}

/* Update the model information.
 * Add knowledge of all headings for all planes within distance threshold
 * heading_ecc_thresh...
 * If the fixation is within a data block, add speed and heading.
 * In the future, we might like to do speed and heading
 * separately...  As it is, we aren't going to be too concerned
 * with speed, because the speeds don't vary that much.
 */

void make_fixation(QSP_ARG_DECL  Point *ptp )
{
	Model_Obj *mop;
	List *lp;

	if( fixation_dp!=NO_OBJ && n_recorded_fixations < max_recorded_fixations )
		record_fixation(ptp);

	n_model_fixations++;

	if( this_fixp != NO_POINT ){	/* have an old fixation */
		saved_fix = *this_fixp;
		last_fixp = &saved_fix;
	}
	curr_fix = *ptp;	/* make these global so we can access them */
	this_fixp = &curr_fix;

#ifdef DEBUG
if( debug & debug_model ){
sprintf(ERROR_STRING,"fixation coords %g %g",this_fixp->p_x,this_fixp->p_y);
advise(ERROR_STRING);
}
#endif /* DEBUG */

#ifdef CAUTIOUS
	if( mo_itp == NO_ITEM_TYPE ){
WARN("CAUTIOUS:  make_fixation:  model object type not initialized");
		return;
	}
#endif /* CAUTIOUS */

	compute_eccentricities(SINGLE_QSP_ARG);
	sort_models(SINGLE_QSP_ARG);	/* sort the list based on distance */

	/* before we update info based on this tag, age all other info */
	age_information(SINGLE_QSP_ARG);

	/* If we are fixating a data tag, update alt and speed */
	lp = model_list(SINGLE_QSP_ARG);
	if( lp == NO_LIST ) return;

	mop = (Model_Obj *)(lp->l_head->n_data);
	if( inside_tag(mop->mo_fpp,this_fixp) ){
		if( display_model_state )
			draw_model_tag(mop,BLACK);

		mop->mo_flags |= ALTITUDE_BIT;
		mop->mo_info_age[ALTITUDE]=0;

		mop->mo_flags |= SPEED_BIT;
		mop->mo_info_age[SPEED]=0;

		if( display_model_state )
			draw_model_tag(mop,DATA_TAG_COLOR);
	}

	/* Make the altitude of the currently fixated object the
	 * current altitude, in case we are doing altitude-based search.
	 */

	if( mop->mo_flags & ALTITUDE_BIT ){
		curr_alt = mop->mo_fpp->fp_altitude;
if( verbose ){
sprintf(ERROR_STRING,"Setting current altitude to %d",(int)curr_alt);
advise(ERROR_STRING);
}
	}

	/* update all headings within heading_ecc_thresh.
	 * We do this after checking the tag so that the ALT flag
	 * will be set - we will draw the plane w/ color coding if
	 * we know the altitude, regardless of the display condition.
	 */
	update_headings(SINGLE_QSP_ARG);

	forget_things(SINGLE_QSP_ARG);

	if( display_model_state ){
		if( show_crossings ){
			crossing_lp = new_list();
			get_crossing_candidates(SINGLE_QSP_ARG);
			dellist(crossing_lp);
		}

		if( last_fixp != NO_POINT ){
			/* erase last fixation circle */
			erase_fixation_indicator(last_fixp);
			ugliness++;

			/* to redraw stuff under it
			 * We don't try to be smart here about what
			 * to redraw, we just redraw everything...
			 * Only do this if fixation_delay is turned
			 * on, or we are running interactive...
			 */
			if( fixation_delay > 0 ||
				atc_state == CLICK_FIXING ||
				atc_state == STEPPING ||
				(pretty_plot >= 0 && ugliness >= pretty_plot) ){

				draw_fixation_indicator(this_fixp);
				draw_region(NULL_QSP);
				redraw_crossings(SINGLE_QSP_ARG);
				apply_to_models(QSP_ARG   render_model_obj );
				ugliness = 0;
			}
		}
		draw_fixation_indicator(this_fixp);
	}
}

void show_point(Point *ptp)
{
	sprintf(msg_str,"%g %g",ptp->p_x,ptp->p_y);
	prt_msg(msg_str);
}

void show_vector(Vector *vp)
{
	sprintf(msg_str,"%g %g",vp->v_x,vp->v_y);
	prt_msg(msg_str);
}

/* Check a pair for conflict.
 * We assume that we have complete knowledge...
 *
 * We consider the problem in the reference frame of one of the planes,
 * by subtracting it's velocity from each...
 * We then have a point and a line parameterized by time.
 * we wish to find the distance of the point from the line,
 * and the time at which that distance happens.
 */

static void check_pair_for_conflict(QSP_ARG_DECL  Pair *prp)
{
	Flight_Path *fpp1,*fpp2;

	fpp1 = prp->pr_mop1->mo_fpp;
	fpp2 = prp->pr_mop2->mo_fpp;

	if( paths_are_in_conflict(fpp1,fpp2) ){
		model_resp1 = fpp1;
		model_resp2 = fpp2;

		prp->pr_flags |= IN_CONFLICT;
		prp->pr_min_dist = min_distance;
		prp->pr_conflict_time = travel_time;
	}
}

int paths_are_in_conflict(Flight_Path *fpp1, Flight_Path *fpp2)
{
	Point org, p0;
	Vector vel,normal,dvec;
	atc_type speed_sq, speed;

	if( fpp1->fp_altitude != fpp2->fp_altitude ) return(0);

	/* consider the situation from the point of view of mop1 */

	org = fpp1->fp_plane_loc;
	p0  = fpp2->fp_plane_loc;

	vel.v_x = fpp2->fp_vel.v_x - fpp1->fp_vel.v_x;
	vel.v_y = fpp2->fp_vel.v_y - fpp1->fp_vel.v_y;

	speed_sq = DOT_PROD(&vel,&vel);
	/* we don't expect this to be zero... */
	speed = sqrt(speed_sq);

	/* equation of the line is p = p0 + t vel
	 * We can get the distance of the point from the line
	 * by dotting the difference vector (org-p0) with a normal
	 * unit vector...
	 */

	/* rotate vel by 90 deg */
	PERPENDICULAR(&normal,&vel);
	SCALE_VECTOR(&normal,1/speed);

	dvec.v_x = org.p_x - p0.p_x;
	dvec.v_y = org.p_y - p0.p_y;

	min_distance = fabs( DOT_PROD(&dvec,&normal) );

	if( min_distance >= SEPARATION_MINIMUM ) return(0);

	/* figure out what time this will occur */

	/* we dot the difference vector with a parallel vector
	 * (we use "norm" even thought it's not the normal vector
	 * any more...)
	 */
	normal.v_x = vel.v_x / speed;
	normal.v_y = vel.v_y / speed;

	travel_distance = DOT_PROD(&dvec,&normal);
	travel_time = travel_distance / speed;

	/* We don't care if there is a conflict going
	 * back in time, but we need to be sure that
	 * we have the signs right!!
	 */

	if( travel_time < 0 )
		return(0);

	/* BUG we need to do something sensible with the units */

	/* make sure that the conflict is within the region */
	/* get the conflict point in the world frame */ 
	vel.v_x = fpp1->fp_vel.v_x;
	vel.v_y = fpp1->fp_vel.v_y;

	SCALE_VECTOR(&vel,travel_time);
	p0.p_x = fpp1->fp_plane_loc.p_x + vel.v_x;
	p0.p_y = fpp1->fp_plane_loc.p_y + vel.v_y;

#ifdef DEBUG
if( debug & debug_model ){
sprintf(msg_str,"Flights %s and %s separation of %g at time %g",
fpp1->fp_name,fpp2->fp_name,min_distance,travel_time);
advise(msg_str);
}
#endif /* DEBUG */

	if( center_p == NO_POINT ) init_center();

	if( DIST(&p0,center_p) > (max_y/2) ){
		/* advise("conflict occurs outside of region"); */
		return(0);
	}

	return(1);
}

/* Scan the model list.  For all locations where we have complete
 * knowledge, check against all others which follow in the list.
 */

/* Determine if tracks cross.
 * We can call this routine even if we only know heading.
 * We still reject pairs for which the tracks cross, but
 * the crossing point could not be reached simultaneously
 * for any reasonable speeds.
 *
 * We solve for the coefficients (times) to the intersection point.
 * If both coefficients are positive, then both planes pass
 * over the intersection point in the future.
 */

Point *crossing_point(Flight_Path *fpp1, Flight_Path *fpp2)
{
	Vector dv1,dv2,nv1,nv2;
	atc_type dist1,dist2,speed1,speed2,denom;
	static Point crossing_pt;

	/* Compute dv1,dv2, unit vectors in the direction of travel */

	dv1 = fpp1->fp_vel;
	dv2 = fpp2->fp_vel;

	speed1 = sqrt( DOT_PROD(&dv1,&dv1) );
	SCALE_VECTOR(&dv1,1/speed1);

	speed2 = sqrt( DOT_PROD(&dv2,&dv2) );
	SCALE_VECTOR(&dv2,1/speed2);

	/* p1 + dist1 dv1 = p2 + dist2 dv2
	 *
	 * p1.nv1 = p2.nv1 + dist2 dv2.nv1
	 *
	 * p1.nv2 + dist1 dv1.nv2 = p2.nv2
	 */
	PERPENDICULAR(&nv1,&dv1);
	PERPENDICULAR(&nv2,&dv2);

	denom = DOT_PROD(&dv2,&nv1);

	if( denom == 0.0 ){
		Vector delta;
		atc_type loc2, course_separation, same_dir;
		Flight_Path *fpp_arr[2];
		atc_type spd_arr[2];
		int fast_i,slow_i;
		atc_type rel_speed,dt;

		/* courses are parallel - see if they are on the same line */
		/* compute difference vector from fpp1 to fpp2 */
		delta.v_x = fpp2->fp_plane_loc.p_x - fpp1->fp_plane_loc.p_x;
		delta.v_y = fpp2->fp_plane_loc.p_y - fpp1->fp_plane_loc.p_y;

		/* length of component perp. to course */
		course_separation = fabs( DOT_PROD(&delta,&nv1) );

#define MIN_PARALLEL_COURSE_SEPARATION	SEPARATION_MINIMUM

		if( course_separation > MIN_PARALLEL_COURSE_SEPARATION )
			return(NO_POINT);

		/* we will get here often in the routes condition,
		 * where many planes are on the same route.
		 */

		/* Probably what we want to do in the case of parallel courses, is
		 * to project both flights onto one or another, then calculate
		 * the time of coincidence...
		 *
		 * We adopt a 1-D coordinate system with the origin at
		 * the current location of the first plane, and the coord
		 * incresing in the direction of that plane's velocity.
		 */

		loc2 = DOT_PROD(&delta,&dv1);

		/* same_dir should be 1 or -1, as the flights are
		 * heading in the same, or opposite directions. (resp.)
		 */

		same_dir = DOT_PROD(&dv1,&dv2);

		if( same_dir ){
			if( loc2 > 0 ){		/* fpp1 pointed towards fpp2 */
				if( speed1 < speed2 ){
					return(NO_POINT);
				}
				fast_i=0;
				slow_i=1;
			} else {		/* fpp2 pointed towards fpp1 */
				if( speed2 < speed1 ){
					return(NO_POINT);
				}
				fast_i=1;
				slow_i=0;
			}

			/* one of the planes is going to overtake the other */

			fpp_arr[0] = fpp1;
			fpp_arr[1] = fpp2;
			spd_arr[0] = speed1;
			spd_arr[1] = speed2;

			/* compute relative speed */

			rel_speed = spd_arr[fast_i] - spd_arr[slow_i];

			/* compute time to closest approach */

			dt = fabs(loc2) / rel_speed;

			dv1 = fpp1->fp_vel;
			SCALE_VECTOR(&dv1,dt);
			crossing_pt = fpp1->fp_plane_loc;
			DISPLACE_POINT(&crossing_pt,&dv1);

			if( inbounds(&crossing_pt) ){
sprintf(DEFAULT_ERROR_STRING,"flight %s overtakes flight %s in-bounds",
fpp_arr[fast_i]->fp_name,fpp_arr[slow_i]->fp_name);
NWARN(DEFAULT_ERROR_STRING);
				return(&crossing_pt);
			} else {
				return(NO_POINT);
			}
		} else {	/* opposite directions */
			if( loc2 > 0 ){	/* fpp1 pointed towards fpp2, head-on */
sprintf(DEFAULT_ERROR_STRING,"flights %s and %s are flying head-on, NEED LOGIC!!!",
fpp1->fp_name,fpp2->fp_name);
NWARN(DEFAULT_ERROR_STRING);
				return(NO_POINT);
			} else {
sprintf(DEFAULT_ERROR_STRING,"flights %s and %s are flying away from each other",
fpp1->fp_name,fpp2->fp_name);
advise(DEFAULT_ERROR_STRING);
				return(NO_POINT);
			}
		}
	}

	/* If denom is not 0, then the courses cross somewhere.
	 * dist1 and dist2 are the (signed) distances to this crossing
	 * point from each of the planes' current location.
	 * (positive distance means that the plane is approaching
	 * the crossing point.)
	 */

	dist2 = ( DOT_PROD((&fpp1->fp_plane_locv),&nv1) -
		DOT_PROD((&fpp2->fp_plane_locv),&nv1) ) / denom;
	

	/* this could probably be computed more efficiently if we
	 * thought about it...   denom doesn't really need to be
	 * computed (it's just the sine of the angle between the
	 * tracks) but the brute-force method protects us against
	 * a sign error...
	 *
	 * p1.nv2 + dist1 dv1.nv2 = p2.nv2
	 */

	denom = DOT_PROD(&dv1,&nv2);
	dist1 = ( DOT_PROD(&fpp2->fp_plane_locv,&nv2) -
		DOT_PROD(&fpp1->fp_plane_locv,&nv2) ) / denom;
	
	if( dist1 > 0 && dist2 > 0 ){
		atc_type max_near_time, min_far_time;

		/* a and b are the times (in seconds) to get to the crossing point */

		/* Both planes pass crossing point in the future.
		 * a and b are the distances.  Check that they are
		 * not so disparate that the planes could never conflict.
		 */

		/* the longest time the nearer plane might take */
		max_near_time = MIN(dist1,dist2) / assumed_min_speed;

		/* the shortest time the further plane might take */
		min_far_time = MAX(dist1,dist2) / assumed_max_speed;

		if( max_near_time < min_far_time )
			/* the tracks do cross, but could never result
			 * in a conclict
			 */
			return(NO_POINT);
		else {
			/* Now find the crossing point */

			crossing_pt = fpp1->fp_plane_loc;
			SCALE_VECTOR(&dv1,dist1);
			DISPLACE_POINT(&crossing_pt,&dv1);

			if( inbounds(&crossing_pt) )
				return(&crossing_pt);
		}
	}
	return(NO_POINT);
}

static void tracks_cross(QSP_ARG_DECL  Pair *prp)
{
	Flight_Path *fpp1,*fpp2;
	Point *crossing_ptp;

	fpp1=prp->pr_mop1->mo_fpp;
	fpp2=prp->pr_mop2->mo_fpp;

	crossing_ptp = crossing_point(fpp1,fpp2);
	if( crossing_ptp != NO_POINT ){
		prp->pr_crossing_pt = *crossing_ptp;
		prp->pr_flags |= TRACKS_CROSS;
	} else {
		prp->pr_flags &= ~TRACKS_CROSS;	/* probably unnecessary? */
	}
}


static void apply_to_model_pairs( QSP_ARG_DECL  void (*func)(QSP_ARG_DECL  Model_Obj *,Model_Obj *) )
{
	List *lp;
	Node *np1,*np2;
	Model_Obj *mop1,*mop2;

	lp = model_list(SINGLE_QSP_ARG);
	if( lp == NO_LIST ) return;

	np1=lp->l_head;
	while(np1!=NO_NODE){
		mop1 = (Model_Obj *)(np1->n_data);
		np2 = np1->n_next;
		while(np2!=NO_NODE){
			mop2 = (Model_Obj *)(np2->n_data);
			(*func)(QSP_ARG  mop1,mop2);
			np2 = np2->n_next;
		}
		np1=np1->n_next;
	}
}

static void create_pair( QSP_ARG_DECL  Model_Obj *mop1, Model_Obj *mop2 )
{
	Pair *prp;
	char name[32];

	/* We want the name of this pair to be the same,
	 * regardless of the order of the arguments.
	 */

	if( strcmp(mop1->mo_name,mop2->mo_name) )
		sprintf(name,"%s.%s",mop1->mo_name,mop2->mo_name);
	else
		sprintf(name,"%s.%s",mop2->mo_name,mop1->mo_name);

	prp = new_pair(QSP_ARG  name);

#ifdef CAUTIOUS
	if( prp == NO_PAIR )
		ERROR1("CAUTIOUS:  couldn't create new pair");
#endif /* CAUTIOUS */

	prp->pr_mop1 = mop1;
	prp->pr_mop2 = mop2;
	prp->pr_flags = 0;
}

void setup_pairs(SINGLE_QSP_ARG_DECL)
{
	apply_to_model_pairs( QSP_ARG  create_pair );
	apply_to_pairs( QSP_ARG  tracks_cross );
	apply_to_pairs( QSP_ARG  check_pair_for_conflict );
}

/* Redraw all known crossings (used for pretty plot).
 *
 * We assume that if the crossing_drawn flag is set, that this
 * really is a crossing we want to display!
 */

static void redraw_crossing(QSP_ARG_DECL  Pair *prp)
{
	if( CROSSING_DRAWN(prp) )
		draw_crossing(prp,CROSSING_COLOR);
}

static void redraw_crossings(SINGLE_QSP_ARG_DECL)
{
	apply_to_pairs( QSP_ARG  redraw_crossing );
}

static void find_crossing( QSP_ARG_DECL  Pair *prp )
{
	Model_Obj *mop1, *mop2 ;
	Node *np;

	if( ! HAS_CROSSING(prp) ) return;	/* don't bother */

	mop1=prp->pr_mop1;
	mop2=prp->pr_mop2;

	if( HEADING_KNOWN(mop1) && HEADING_KNOWN(mop2) ){

		/* Only look at these if we need tag info.
		 *
		 * We don't worry about an object occurring on
		 * the list more than once...  If will pick
		 * a random element from the list for the next
		 * fixation, multiple occurrences will skew
		 * the probability distribution in a useful way.
		 */

		/* don't bother if we know the altitudes (perhaps from color coding)
		 * and they are different.
		 */

		if( ALT_KNOWN(mop1) && ALT_KNOWN(mop2) ){
			if( mop1->mo_fpp->fp_altitude != mop2->mo_fpp->fp_altitude ){
				if( display_model_state && CROSSING_DRAWN(prp) ){
					erase_crossing(QSP_ARG  prp);
				}
				
				return;
			}
		}

		/* If we get to here, either:
		 * one or both altitudes are not known...
		 * OR the altitudes are the same but we don't know the speeds
		 * OR we do know the speeds, in which case we can assume that they do
		 * not constitute a conflict pair.  But if we know everything, why
		 * would we want to look at them again?
		 *
		 * BUG?  One feature we might want to add is memory for a pair being
		 * rejected, even after the altitudes have been forgotten...
		 */

		if( display_model_state && ! CROSSING_DRAWN(prp) ){
			draw_crossing(prp,CROSSING_COLOR);
		}

		/* BUG?  We might want to change this logic a bit if we model separate
		 * fixations for speed and altitude...
		 */

		if( ! COMPLETE_KNOWLEDGE(mop1) ){
			np = mk_node(&mop1->mo_fpp->fp_tag_loc);
			addHead(crossing_lp,np);
		}
		if( ! COMPLETE_KNOWLEDGE(mop2) ){
			np = mk_node(&mop2->mo_fpp->fp_tag_loc);
			addHead(crossing_lp,np);
		}

#ifdef NOT_READY
		/* if we might visit this crossing, keep a record so we can refresh
		 * its memory
		 */
		if( ( ! COMPLETE_KNOWLEDGE(mop1) ) || ( ! COMPLETE_KNOWLEDGE(mop1) ) ){
			np = mk_node(prp);
			addHead(crossing_lp,np);
		}
#endif /* NOT_READY */
	}
}


/* This routine is central to the performance of the model,
 * and embodies the various observer strategies, which are:
 *
 * Depth first:		select a target,
 *				then examine all others for conflicts
 *
 * Conflict search algorithms:
 *
 *	2D track:	examine all planes whose 2d ray cross
 *	alt based:	examine all planes at the same altitude
 *
 * Target selection algorithms:
 *
 * Random:		visit a new target at random
 *
 * Smart Random:	visit a new target for which we have no info
 *
 * Nearest:		visit nearest target with no info
 *
 * Clockwise:		visit targets in clockwise order
 *
 * Guided:		use partial information
 *				if paths cross, check altitude
 */



/*
 * Make up a list of all crossing pairs that we could know about now.
 */

static void get_crossing_candidates(SINGLE_QSP_ARG_DECL)
{
	/* Look for pairs where we know the headings, and
	 * the tracks cross.
	 */

	apply_to_pairs( QSP_ARG  find_crossing );
}

/* make up a list of all tags where we don't know altitude.
 * if there are none, make up list where we don't know speed.
 */

static void find_alt_ignorance( QSP_ARG_DECL  Model_Obj *mop )
{
	Node *np;

	if( ALT_KNOWN(mop) ) return;

	np = mk_node(&mop->mo_fpp->fp_tag_loc);
	addHead(candidate_lp,np);
}

static void find_speed_ignorance(QSP_ARG_DECL  Model_Obj *mop )
{
	Node *np;

	if( SPEED_KNOWN(mop) ) return;

	np = mk_node(&mop->mo_fpp->fp_tag_loc);
	addHead(candidate_lp,np);
}

static void get_tag_candidates(SINGLE_QSP_ARG_DECL)
{
	if( candidate_lp != NO_LIST ) dellist(candidate_lp);
	candidate_lp = new_list();

	apply_to_models(QSP_ARG   find_alt_ignorance );
	if( eltcount(candidate_lp) <= 0 )
		apply_to_models(QSP_ARG   find_speed_ignorance );

	if( eltcount(candidate_lp) <= 0 )
		ERROR1("get_tag_candidates:  all tag info appears to be known!?");
}

/* Make up a list of all planes for which we don't know heading...
 */

static Altitude global_want_alt=ANY_ALTITUDE;

static void find_heading_ignorance(QSP_ARG_DECL  Model_Obj *mop)
{
	Node *np;

	if( HEADING_KNOWN(mop) ) return;

	if( global_want_alt != ANY_ALTITUDE &&
		mop->mo_fpp->fp_altitude != global_want_alt )

		return;
		
	np = mk_node(&mop->mo_fpp->fp_plane_loc);
	addHead(candidate_lp,np);
}

static void get_heading_candidates(QSP_ARG_DECL  Altitude want_alt)
{
	if( USES_SALIENCY(curr_stratp) ){
		Point *ptp;
retry:
		ptp = get_saliency(SINGLE_QSP_ARG);	/* compute the saliency map */
		if( ptp != NO_POINT ){
			Node *np;
			np = mk_node(ptp);
			addHead(candidate_lp,np);
			return;
		} else if( curr_alt != ANY_ALTITUDE ){
			/* We might have found nothing because there
			 * are no unknown targets at the current altitude.
			 */
			curr_alt = ANY_ALTITUDE;
			goto retry;
		}
		/* We might return NO_POINT if the max
		 * is not unique, then we just fall through.
		 */
	}
	global_want_alt = want_alt;
	apply_to_models(QSP_ARG   find_heading_ignorance );
}

/* Find anything, but be reasonable:
 * only visit locations where there is some information to be gained.
 */

static void find_anything(QSP_ARG_DECL  Model_Obj *mop)
{
	Node *np;

	if( ! HEADING_KNOWN(mop) ){
		np = mk_node(&mop->mo_fpp->fp_plane_loc);
		addHead(candidate_lp,np);
	}
	if( (!ALT_KNOWN(mop)) || (!SPEED_KNOWN(mop)) ){
		np = mk_node(&mop->mo_fpp->fp_tag_loc);
		addHead(candidate_lp,np);
	}
}

static void get_all_candidates(SINGLE_QSP_ARG_DECL)	/* just get everything */
{
	apply_to_models(QSP_ARG   find_anything );
}

static void apply_to_loc_list(List *lp, void (*func)(Node *) )
{
	Node *np;

	np=lp->l_head;
	while(np!=NO_NODE){
		(*func)(np);
		np=np->n_next;
	}
}


#define POLAR_ANGLE_OF( ptp )						\
									\
	RADIANS_TO_DEGREES( atan2( (ptp)->p_y - center_p->p_y,		\
				(ptp)->p_x - center_p->p_x ) )


static void polar_priority( Node *np )
{
	atc_type polar_angle;
	Point *ptp;

	ptp = (Point *)(np->n_data);
	polar_angle = POLAR_ANGLE_OF( ptp );
	polar_angle -= fixation_theta;
	while( polar_angle < 0 ) polar_angle += 360;
	np->n_pri = (short int)polar_angle;
}

static void distance_priority( Node *np )
{
	int d;
	Point *ptp;

	ptp = (Point *)(np->n_data);
	d = (int)(DIST( ptp, this_fixp ));
	if( d < 1 ) d=10000;	/* don't stay in the same place */
	np->n_pri = (-d);
}

static void random_priority( Node *np )
{
	np->n_pri = (short) rn(1000);
}


/* Choose from a list of candidates, using the current strategy.
 * We assign priorities to the list nodes based on scan strategy,
 * then sort the list and return the head.  This is a bit wasteful,
 * because we do not need the entire list reordered, plus p_sort
 * is currently implemented with an inefficient bubble sort.
 * It would be better to set up an array of pointers to the locations,
 * and sort with qsort()...   For now we'll leave it be, and later
 * do some profiling to see if it is worth the trouble to make this
 * more efficient.
 */

static Point * choose_location_from_list(List *lp)
{
	if( curr_stratp == NO_STRATEGY ){
		NWARN("choose_location_from_list:  No strategy selected!?");
		return(NO_POINT);
	}

	switch( curr_stratp->strat_scan ){
		case NEAREST:
			/* assign node priorities based on distance from the current loc */
			apply_to_loc_list( lp, distance_priority );
			break;

		case CLOCKWISE:
			/* assign node priorities to be the angle minus the current angle */
			fixation_theta = POLAR_ANGLE_OF( this_fixp );
			apply_to_loc_list( lp, polar_priority );
			break;

		case RANDOM:
			/* THis is a very wasteful way of doing this,
			 * because we could just pick out the nth list
			 * element, where n is a random number; instead
			 * we prioritize the whole list and sort...
			 * All for the sake of uniformity...
			 */
			apply_to_loc_list( lp, random_priority );
			break;

		case MAX_INFO:
			/* What we really want to do is find the screen centroid
			 * of the ungathered information...
			 */
			 NERROR1("MAX_INFO not implemented!?");
			 break;
		case N_SCAN_TYPES:
		default:
			NERROR1("bad case in scan type switch, shouldn't happen");
			break;
	}

	p_sort(lp);	/* currently this is a bubble sort...
			 * We ought to be able to do much better!
			 */

	return( (Point *) lp->l_head->n_data );
}

#define MAPSIZE	1024L
static Data_Obj *saliency_dp=NO_OBJ, *sfilter_dp=NO_OBJ;
static Data_Obj *saliency_xform=NO_OBJ;
static Data_Obj *zero_dp=NO_OBJ;
static float saliency_hzoom, saliency_vzoom;
#define ZERO_SCALAR_NAME	"zero_scalar"

static void init_saliency(Data_Obj *sdp,Data_Obj *fdp)
{
	/* BUG we use 768x1024 here, should get it from somewhere!? */
	saliency_hzoom = ((double)sdp->dt_cols)/((double)1024);
	saliency_vzoom = ((double)sdp->dt_rows)/((double)1024);

	if( ! IS_CONTIGUOUS(sdp) ){
		sprintf(DEFAULT_ERROR_STRING,"saliency map %s should be contiguous",
			sdp->dt_name);
		NWARN(DEFAULT_ERROR_STRING);
		return;
	}
	/* We ought to check here that fdp is an appropriate filter for sdp... */
	saliency_dp = sdp;
	sfilter_dp = fdp;
}

static int n_noted;

static void note_saliency(QSP_ARG_DECL  Model_Obj *mop)
{
	long ix,iy;
	float *fltp;

	if( HEADING_KNOWN(mop) ){
		return;
	}

	if( curr_alt != ANY_ALTITUDE &&
		mop->mo_fpp->fp_altitude != curr_alt ){
		return;
	}
		
	/* OK, this is a plane we are interested in. */

	ix = (long) rint(saliency_hzoom * mop->mo_fpp->fp_plane_loc.p_x);
	iy = (long) rint(saliency_vzoom * mop->mo_fpp->fp_plane_loc.p_y);
#ifdef CAUTIOUS
	if( ix<0 || ix >= saliency_dp->dt_cols ){
		sprintf(ERROR_STRING,
		"CAUTIOUS:  note_saliency:  ix (%ld) out of range!?",ix);
		WARN(ERROR_STRING);
		return;
	}
	/* We really think that iy whould be < 768, but this test
	 * is sufficient to prevent a seg viol.
	 */
	if( iy<0 || iy >= saliency_dp->dt_rows ){
		sprintf(ERROR_STRING,
		"CAUTIOUS:  note_saliency:  iy (%ld) out of range!?",iy);
		WARN(ERROR_STRING);
		return;
	}
#endif /* CAUTIOUS */

	fltp = saliency_dp->dt_data;

	/* We would use ix*saliency_dp->dt_pinc, but here we assume that
	 * the image is contiguous so pinc=1;
	 */
	fltp += iy*saliency_dp->dt_rinc + ix;

	*fltp += 1.0;		/* we do this instead of setting in case
				 * two icons have the same location
				 */

	n_noted++;
}

/* get_saliency
 *
 * This routine attempts to capture the effect that clumps of many planes
 * are more interesting than regions with few planes.  We construct a
 * saliency map by creating an impulse for each craft with unknown heading,
 * and then applying a Gaussian low-pass filter, and moving gax the the position
 * with the max value.
 */

static Data_Obj *indices_dp=NO_OBJ,*ntimes_dp=NO_OBJ,*maxval_dp=NO_OBJ;
static Point salient_point;

static Point * get_saliency(SINGLE_QSP_ARG_DECL)
{
	char sal_xf_name[LLEN];
	long *lp;
	float *fltp;
	u_long max_i,ix,iy;
	Vec_Obj_Args oargs;

	if( saliency_dp == NO_OBJ ){
		static int warned=0;
		if( !warned ){
			WARN("Need to specify the saliency map image and filter to compute saliency");
			warned=1;
		}
		return(NO_POINT);
	}
	/* set the map to zero */
	setvarg1(&oargs,saliency_dp);
	if( zero_dp == NO_OBJ ){
		/* create the zero scalar */
		Scalar_Value sv;
		zero_dp = mk_scalar(QSP_ARG  ZERO_SCALAR_NAME,PREC_SP);
		sv.u_f = 0.0;
		assign_scalar(QSP_ARG  zero_dp,&sv);
	}
	oargs.oa_s1 = zero_dp;
	perf_vfunc(QSP_ARG  FVSET,&oargs);
	/* this is a hard way to do dp_const() !? */

	/* Now, scan the aircraft and add an impulse for each one with
	 * unknown heading.
	 */
	n_noted=0;
	apply_to_models(QSP_ARG   note_saliency );
	if( n_noted == 0 ) return(NO_POINT);

	/* now we apply the filter */
	if( saliency_xform==NO_OBJ ){
		Dimension_Set dimset;

		dimset = sfilter_dp->dt_type_dimset;	/* or dt_mach_dimset?? */

		sprintf(sal_xf_name,"%s.xform",saliency_dp->dt_name);
		if( (saliency_xform=make_dobj(QSP_ARG  sal_xf_name,&dimset,
			PREC_SP|COMPLEX_PREC_BITS)) == NO_OBJ ){
			WARN("error creating saliency transform");
			return(NO_POINT);
		}
	}
	fft2d(saliency_xform,saliency_dp);
	/* BUG real arg must be 3rd...  it would be nice if this weren't so! */
	setvarg3(&oargs,saliency_xform,saliency_xform,sfilter_dp);
	perf_vfunc(QSP_ARG  FVMUL,&oargs);
	ift2d(saliency_dp,saliency_xform);


	/* now find the max */

	if( indices_dp==NO_OBJ ){
		Dimension_Set dimset;
		int i;

		for(i=0;i<N_DIMENSIONS;i++)
			dimset.ds_dimension[i]=1;
#define NTIMES_NAME	"saliency_ntimes"
#define INDICES_NAME	"saliency_indices"
#define MAXVAL_NAME	"saliency_maxval"
		ntimes_dp = make_dobj(QSP_ARG  NTIMES_NAME,&dimset,PREC_DI);
		maxval_dp = make_dobj(QSP_ARG  MAXVAL_NAME,&dimset,PREC_SP);
		dimset.ds_dimension[1]=saliency_dp->dt_n_mach_elts;
		indices_dp = make_dobj(QSP_ARG  INDICES_NAME,&dimset,PREC_DI);
	}
	setvarg2(&oargs,indices_dp,saliency_dp);
	oargs.oa_s1 = maxval_dp;
	oargs.oa_s2 = ntimes_dp;
	perf_vfunc(QSP_ARG  FVMAXG,&oargs);

	/* now see if we have a unique max */
	lp = ntimes_dp->dt_data;
	if( *lp != 1 ){
		/* this could either happen because  the targets are very
		 * far apart, or because there are NO targets...
		 * in the former case, we should just fixate one of the targets,
		 * otherwise we should note that there are no targets.
		 */
		WARN("saliency map does not have a unique maximum!?");
		return(NO_POINT);
	}
	/* now get the index of the max */
	lp = indices_dp->dt_data;
	fltp = saliency_dp->dt_data;

	max_i = *lp;

sprintf(ERROR_STRING,"fltp[%ld] = %g",max_i,fltp[max_i]);
advise(ERROR_STRING);
	max_i -- ;	/* fortran style indices */
sprintf(ERROR_STRING,"fltp[%ld] = %g",max_i,fltp[max_i]);
advise(ERROR_STRING);
	/* now convert the index to coordinates */

	ix = max_i % saliency_dp->dt_cols;
	iy = max_i / saliency_dp->dt_cols;

	salient_point.p_x = (float)ix / saliency_hzoom;
	salient_point.p_y = (float)iy / saliency_vzoom;

	return(&salient_point);
}

void plan_saccade(QSP_ARG_DECL  Point *ptp)
{
	int n;
	Point *new_ptp;
	Altitude preferred_alt;

	if( curr_stratp == NO_STRATEGY ){
		WARN("plan_saccade:  no strategy selected");
		return;
	}

	/* release old candidates */
	if( candidate_lp != NO_LIST ) dellist(candidate_lp);
	candidate_lp = new_list();

	/* When altitude is color-coded, it is easy to inspect
	 * the headings of all the flights at a given altitude.
	 *
	 * If we have a current color, and we don't know the headings
	 * of other planes with that color, then we look at them.
	 *
	 * If we have any crossings we look at them first...
	 */

	if( SEES_CROSSINGS( curr_stratp ) ){
		/* make a list of targets that will help find crossings */

		/* first, see if there are crossings we know about for which
		 * we don't know the altitude.
		 */

		crossing_lp = candidate_lp;
		get_crossing_candidates(SINGLE_QSP_ARG);
	}

	/* If we have some known crossings, deal with them first.
	 * This is a somewhat arbitrary choice, and might better be
	 * a strategy parameter...
	 */

	if( eltcount( candidate_lp ) > 0 )
		goto make_choice;


	/*
	 * Curr_alt is the altitude of the last plane fixated...
	 */

	if( USES_ALT_KNOWLEDGE(curr_stratp) && curr_alt > 0 ){
		preferred_alt = curr_alt;
	} else {
		preferred_alt = ANY_ALTITUDE;
	}

	if( SEES_CROSSINGS( curr_stratp ) ){
		/* We don't have any known crossings to investigate;
		 * look somewhere we don't know about heading.
		 */
		get_heading_candidates(QSP_ARG  preferred_alt);
	}

	/* If we have something to look at that might generate a crossing
	 * at the preferred altitude, look at it now...
	 *
	 * If we have a preferred altitude, and don't have any headings
	 * to look at, it is time to switch altitudes!
	 */

	if( eltcount( candidate_lp ) > 0 )
		goto make_choice;

	if( preferred_alt != ANY_ALTITUDE ){
		preferred_alt = ANY_ALTITUDE;
		get_heading_candidates(QSP_ARG  preferred_alt);
	}

	if( eltcount( candidate_lp ) > 0 )
		goto make_choice;

	if( SEES_CROSSINGS( curr_stratp ) ){
		/* This must mean we know all the headings.
		 * It seems very unlikely that we could
		 * 1) know all the headings
		 * 2) not know the answer
		 * and
		 * 3) not have a tag we need to look at
		 *
		 * But perhaps in future experiments we
		 * may have conditions with no conflicts,
		 * or planes that change course and create
		 * conflicts later.
		 *
		 * At any rate, we know all the headings,
		 * so we ought to look at a tag or something...
		 */
		get_tag_candidates(SINGLE_QSP_ARG);
		n = eltcount( candidate_lp );
#ifdef CAUTIOUS
		if( n <= 0 )
			ERROR1("CAUTIOUS:  No tags to fixate!?");
#endif /* CAUTIOUS */
	}

	if( eltcount( candidate_lp ) == 0 ){	/* nothing to look at yet... */
		/* just make up a list with everything on it */
		get_all_candidates(SINGLE_QSP_ARG);
	}

make_choice:

	new_ptp = choose_location_from_list( candidate_lp );

	/* If we have found a crossing, then the new fixation relates to that
	 * crossing, and we really should not forget any of the information
	 * that allowed that to happen...  We would like to zero the ages
	 * on the relevant flight paths...  but since the list is a list
	 * of possible fixation locations, we really have a hard time
	 * figuring out why this one is here!?
	 *
	 * We might use another list (crossing_lp?) to remember all crossings represented
	 * in candidate_lp, but once the candidates have been sorted how
	 * can we refer back to the list of crossings?
	 */

	*ptp = *new_ptp;
}

static COMMAND_FUNC( do_init_model )
{
	clear_models(SINGLE_QSP_ARG);		/* get rid of any old models */
	init_model_objects(SINGLE_QSP_ARG);	/* create empty model objects for this display */
	setup_pairs(SINGLE_QSP_ARG);		/* create and initialize pair objects */

	this_fixp=last_fixp=NO_POINT;
	n_model_fixations = 0;
	n_recorded_fixations = 0;

	if( fixation_dp != NO_OBJ )
		max_recorded_fixations = fixation_dp->dt_cols;
	else
		max_recorded_fixations = 0;

	ASSIGN_VAR(CONFLICT_FOUND_VAR_NAME,"0");
}

static int global_conflict_flag;

void known_conflict_query(QSP_ARG_DECL  Pair *prp)
{
	if(	HAS_CONFLICT(prp) &&
		COMPLETE_KNOWLEDGE(prp->pr_mop1) &&
		COMPLETE_KNOWLEDGE(prp->pr_mop2) ){

		model_resp1 = prp->pr_mop1->mo_fpp;
		model_resp2 = prp->pr_mop2->mo_fpp;

#ifdef DEBUG
if( debug & debug_model ){
sprintf(ERROR_STRING,"Flights %s and %s found to be in conflict!!!",
prp->pr_mop1->mo_name,prp->pr_mop2->mo_name);
advise(ERROR_STRING);
}
#endif /* DEBUG */

		if( display_model_state ){
			draw_plane(prp->pr_mop1->mo_fpp,WHITE);
			draw_plane(prp->pr_mop2->mo_fpp,WHITE);
		}

		global_conflict_flag = 1;
	}
}

int conflict_found(SINGLE_QSP_ARG_DECL)
{
	global_conflict_flag = 0;
	apply_to_pairs( QSP_ARG  known_conflict_query );

	if( global_conflict_flag )
		ASSIGN_VAR(CONFLICT_FOUND_VAR_NAME,"1");

	return( global_conflict_flag );
}

COMMAND_FUNC( run_model )
{
	Point new_fixation;

	do_init_model(SINGLE_QSP_ARG);

	/* initial fixation is at the center */
	if( center_p == NO_POINT ) init_center();
	make_fixation( QSP_ARG  center_p );

	while( conflict_found(SINGLE_QSP_ARG)==0 ){
		plan_saccade(QSP_ARG  &new_fixation );
		make_fixation( QSP_ARG  &new_fixation );
		if( fixation_delay > 0 )
			delay(fixation_delay);
#ifdef DEBUG
if( debug & debug_model ) model_state_info(SINGLE_QSP_ARG);
#endif /* DEBUG */

		/* want to check_events() here, to catch ^C */
		/* but this won't work since the atc window isn't a viewer!? */
		event_loop(SINGLE_QSP_ARG);
	}
	report_model_rt(SINGLE_QSP_ARG);
}

static COMMAND_FUNC( disp_mode )
{
	display_model_state = ASKIF("display model state");
}

#define N_FIXATION_TYPES	3
static const char *fixation_types[N_FIXATION_TYPES]={ "location", "plane", "tag" };

static COMMAND_FUNC( do_fixate )
{
	Flight_Path *fpp;
	Point fix;
	int i;

	i = WHICH_ONE("target type",N_FIXATION_TYPES,fixation_types);
	if( i < 0 ) return;

	switch(i){
		case 0:
			fix.p_x = HOW_MUCH("fixation x");
			fix.p_y = HOW_MUCH("fixation y");
			make_fixation( QSP_ARG  &fix );
			break;
		case 1:
			fpp = PICK_FLIGHT_PATH("");
			if( fpp == NO_FLIGHT_PATH ) return;
			make_fixation(QSP_ARG  &fpp->fp_plane_loc);
			break;
		case 2:
			fpp = PICK_FLIGHT_PATH("");
			if( fpp == NO_FLIGHT_PATH ) return;
			make_fixation(QSP_ARG  &fpp->fp_tag_loc);
			break;
	}
	/* We need to call conflict_found here to set the script variable... */
	if( conflict_found(SINGLE_QSP_ARG) && verbose )
		advise("Conflict found!");
}

static COMMAND_FUNC( do_search )
{
	if( conflict_found(SINGLE_QSP_ARG) )
		report_model_rt(SINGLE_QSP_ARG);
}

static COMMAND_FUNC( do_plan )
{
	Point fix;

	plan_saccade(QSP_ARG  &fix);

	if( verbose ){
		sprintf(msg_str,"Next saccade planned for %g %g",fix.p_x,fix.p_y);
		prt_msg(msg_str);
	}

	/* put the values to script vars too */
	sprintf(msg_str,"%g",fix.p_x);
	ASSIGN_VAR("fix_x",msg_str);
	sprintf(msg_str,"%g",fix.p_y);
	ASSIGN_VAR("fix_y",msg_str);
}

static COMMAND_FUNC( render_model )
{
	clear_atc_screen();
	draw_region(NULL_QSP);
	apply_to_models(QSP_ARG   render_model_obj );
}

static COMMAND_FUNC( do_info )
{
	Model_Obj *mop;

	mop = PICK_MODEL_OBJECT("");
	if( mop== NO_MODEL_OBJECT ) return;

	model_object_info(QSP_ARG  mop);
}

void pair_info( QSP_ARG_DECL  Pair *prp )
{
	prt_msg_frag(prp->pr_name);
	if( HAS_CROSSING(prp) )
		sprintf(msg_str,"\tcrossing at %4.0f %4.0f",prp->pr_crossing_pt.p_x,
			prp->pr_crossing_pt.p_y);
	else
		sprintf(msg_str,"\t                     ");

	prt_msg_frag(msg_str);

	if( HAS_CONFLICT(prp) ){
		sprintf(msg_str,"\tCONFLICT:  min_dist = %g  time = %g",
			prp->pr_min_dist,prp->pr_conflict_time);
		prt_msg(msg_str);
	} else
		prt_msg("");
}

COMMAND_FUNC( all_pair_info )
{
	apply_to_pairs( QSP_ARG  pair_info );
}

static COMMAND_FUNC( set_strategy )
{
	Strategy *stratp;

	stratp = PICK_STRAT("");
	if( stratp == NO_STRATEGY ) return;

	curr_stratp = stratp;
}

/* These routines are useful for confirming that the stimuli really
 * only have one conflict.
 */

static void show_conflict(QSP_ARG_DECL  Pair *prp)
{
	if( ! HAS_CONFLICT(prp) ) return;

	sprintf(ERROR_STRING,
"Flights %s and %s have minimum separation of %g at time %g",
		prp->pr_mop1->mo_name,
		prp->pr_mop2->mo_name,
		prp->pr_min_dist,
		prp->pr_conflict_time);
	advise(ERROR_STRING);
	draw_conflict(QSP_ARG  prp);
}

static COMMAND_FUNC( show_conflicts )
{
	sprintf(ERROR_STRING,"Separation min is 10 kts (%g pixels)",
		SEPARATION_MINIMUM);
	advise(ERROR_STRING);
	sprintf(ERROR_STRING,"%g pixels per knot",PIXELS_PER_KNOT);
	advise(ERROR_STRING);

	apply_to_pairs(QSP_ARG  show_conflict);
}

static COMMAND_FUNC( set_fix_delay )
{
	fixation_delay = HOW_MANY("fixation delay in milliseconds");
}

static COMMAND_FUNC( set_ecc_thresh )
{
	int t;

	t = HOW_MANY("heading eccentricity threshold (in pixels)");

	if( t < 0 || t > max_y ){
		WARN("ridiculous value for heading threshold");
		return;
	}

	heading_ecc_thresh = t;
}

static void init_info_type_names(void)
{
	int i;
	for(i=0;i<N_INFO_TYPES;i++){
		switch(i){
			case HEADING:	info_type_names[i]="heading";	break;
			case ALTITUDE:	info_type_names[i]="altitude";	break;
			case SPEED:	info_type_names[i]="speed";	break;
			case N_INFO_TYPES: break;	/* just to suppress compiler warning */
		}
	}
}

static COMMAND_FUNC( set_forget_prob )
{
	int i;
	Data_Obj *dp;

	if( info_type_names[0] == NULL ) init_info_type_names();

	i = WHICH_ONE("information type",N_INFO_TYPES,info_type_names);
	dp = PICK_OBJ("vector of forgetting probabilities");

	if( i < 0 ) return;
	if( dp == NO_OBJ ) return;

	if( MACHINE_PREC(dp) != PREC_SP ){
		sprintf(ERROR_STRING,"Probability vector %s has precision %s, should be %s",
			dp->dt_name,prec_name[MACHINE_PREC(dp)],prec_name[PREC_SP]);
		WARN(ERROR_STRING);
		return;
	}
	if( ! IS_CONTIGUOUS(dp) ){
		sprintf(ERROR_STRING,"Probability vector %s is not contiguous!?",
			dp->dt_name);
		WARN(ERROR_STRING);
		return;
	}

	p_forgetting_dp[i] = dp;
}

static COMMAND_FUNC( set_remem )
{
	int i;

	if( info_type_names[0] == NULL ) init_info_type_names();

	i = WHICH_ONE("information type",N_INFO_TYPES,info_type_names);
	if( i < 0 ) return;

	p_forgetting_dp[i] = NO_OBJ;
}

static void strat_info(Strategy *stratp)
{
	sprintf(msg_str,"Strategy %s:",stratp->strat_name);
	prt_msg(msg_str);

	sprintf(msg_str,"\t%s scanning pattern",
		tell_indexed_name(scan_names,N_SCAN_TYPES,stratp->strat_scan) );
	prt_msg(msg_str);

	if( SEES_CROSSINGS(stratp) )
		prt_msg("\tinvestigates known crossings first");
	else
		prt_msg("\tignores known crossings");

	if( USES_ALT_KNOWLEDGE(stratp) )
		prt_msg("\tinvestigates flights at similar altitudes");
	else
		prt_msg("\tignores altitude knowledge");

	if( USES_SALIENCY(stratp) )
		prt_msg("\tuses aircraft density to compute saliency");
	else
		prt_msg("\tignores aircraft density");
}

static COMMAND_FUNC( do_strat_info )
{
	Strategy *stratp;

	stratp = PICK_STRAT("");
	if( stratp==NO_STRATEGY ) return;

	strat_info(stratp);
}

static COMMAND_FUNC( do_saliency )
{
	Data_Obj *sdp,*fdp;
	Point *ptp;

	sdp = PICK_OBJ("saliency map image");
	fdp = PICK_OBJ("saliency filter xform");

	if( sdp==NO_OBJ || fdp==NO_OBJ ) return;

	init_saliency(sdp,fdp);
	ptp = get_saliency(SINGLE_QSP_ARG);
	if( ptp != NO_POINT ){
		sprintf(msg_str,"Most salient point is at %g %g",ptp->p_x,ptp->p_y);
		prt_msg(msg_str);
	} else {
		prt_msg("No salient points");
	}
}

#define INSURE_CURR_STRAT						\
									\
	if( curr_stratp == NO_STRATEGY ){				\
		WARN("No strategy currently selected");			\
		return;							\
	}

static COMMAND_FUNC( tell_strategy )
{
	INSURE_CURR_STRAT

	sprintf(msg_str,"Current strategy is %s",curr_stratp->strat_name);
	prt_msg(msg_str);
}

static COMMAND_FUNC( edit_strategy )
{
	if( curr_stratp == NO_STRATEGY )
		WARN("edit_strategy:  no strategy currently specified!?");

	set_strat_params(SINGLE_QSP_ARG);
}

static COMMAND_FUNC( do_list_strats ){list_strats(SINGLE_QSP_ARG);}

static Command strat_ctbl[]={
{ "new",	create_strategy,	"define new strategy"			},
{ "list",	do_list_strats,		"list all existing strategies"		},
{ "info",	do_strat_info,		"list information about a strategy"	},
{ "select",	set_strategy,		"select strategy for next simulation"	},
{ "tell",	tell_strategy,		"report current strategy"		},
{ "edit",	edit_strategy,		"edit current strategy"			},
{ "quit",	popcmd,			"exit submenu"				},
{ NULL_COMMAND									}
};

static COMMAND_FUNC( strat_menu )
{
	PUSHCMD(strat_ctbl,"strategy");
}

static Command test_ctbl[]={
{ "state",		model_state_info,"print entire model state to screen"		},
{ "pairs",		all_pair_info,	"display information about all pairs"		},
{ "info",		do_info,	"display information about one model object"	},
{ "saliency",		do_saliency,	"compute saliency map"				},
{ "quit",	popcmd,			"exit submenu"				},
{ NULL_COMMAND									}
};

static COMMAND_FUNC( test_menu ) { PUSHCMD(test_ctbl,"model_test"); }

static COMMAND_FUNC( set_pretty )
{
	pretty_plot=HOW_MANY("number of refixations between model redraws");
}

static Command param_ctbl[]={
{ "delay",		set_fix_delay,	"set fixation delay for visualization"		},
{ "eccentricity",	set_ecc_thresh,	"set eccentricity limit for heading perception"	},
{ "forget",		set_forget_prob,"specify forgetting probabilities"		},
{ "remember",		set_remem,	"disable forgetting for info type"		},
{ "conflicts",		show_conflicts,	"display conflicts"				},
{ "pretty_plot",	set_pretty,	"enable/disable pretty plotting"		},
{ "quit",		popcmd,		"exit submenu"					},
{ NULL_COMMAND										}
};

static COMMAND_FUNC( param_menu ) { PUSHCMD(param_ctbl,"params"); }

static boolean good_vector( QSP_ARG_DECL   Data_Obj *dp, u_long ncomps )
{
	if( MACHINE_PREC(dp) != PREC_SP ){
		sprintf(ERROR_STRING,"Precision of object %s is %s, should be %s for fixation record",
			dp->dt_name,prec_name[MACHINE_PREC(dp)],prec_name[PREC_SP]);
		WARN(ERROR_STRING);
		return(0);
	}
	if( ! IS_CONTIGUOUS(dp) ){
		sprintf(ERROR_STRING,
	"Sorry, data object %s must be contiguous to hold fixation data",dp->dt_name);
		WARN(ERROR_STRING);
		return(0);
	}
	if( dp->dt_comps != ncomps ){
		sprintf(ERROR_STRING,
	"Fixation data object %s should have 3 components",dp->dt_name);
		WARN(ERROR_STRING);
		return(0);
	}
	if( ! IS_VECTOR(dp) ){
		sprintf(ERROR_STRING,
	"Fixation data object %s should be a vector",dp->dt_name);
		WARN(ERROR_STRING);
		return(0);
	}
	return(1);
}


/* is this a vector we can use for fixations? */

boolean good_fixation_vector(QSP_ARG_DECL  Data_Obj *dp)
{
	return( good_vector(QSP_ARG  dp,3L) );
}

boolean good_scanpath_vector(QSP_ARG_DECL  Data_Obj *dp)
{
	return( good_vector(QSP_ARG  dp,2L) );
}



static COMMAND_FUNC( do_set_fix_obj )
{
	Data_Obj *dp;

	dp = PICK_OBJ("data vector for fixation coords");
	if( dp == NO_OBJ ) return;

	if( ! good_fixation_vector(QSP_ARG  dp) ) return;

	fixation_dp = dp;
}

static COMMAND_FUNC( do_play_fix )
{
	Data_Obj *dp;
	u_long n;
	float *fltp;

	dp = PICK_OBJ("vector of fixation locations");
	if( dp== NO_OBJ ) return;

	if( dp->dt_comps != 2 ){
		sprintf(ERROR_STRING,"vector %s has %d components, should have 2",
			dp->dt_name,dp->dt_comps);
		WARN(ERROR_STRING);
		return;
	}
	if( MACHINE_PREC(dp) != PREC_SP ){
		sprintf(ERROR_STRING,"vector %s has %s precision, should be %s",
			dp->dt_name,prec_name[MACHINE_PREC(dp)],prec_name[PREC_SP]);
		WARN(ERROR_STRING);
		return;
	}
	n=dp->dt_cols;
	fltp = (float *)(dp->dt_data);
	while(n--){
		make_fixation( QSP_ARG  (Point *) fltp );
		fltp += dp->dt_pinc;
	}
}

/* multiple fixation runs */
static Command multi_ctbl[]={
{ "run",		run_model,	"run full model on the current set of flights"	},
{ "fixation_obj",	do_set_fix_obj,	"specify data object for fixation data"		},
{ "play_fix",		do_play_fix,	"run model from fixation transcript"		}, 
{ "step",		step_model,	"step model with mouse clicks"			},
{ "click",		click_fixations,"enter fixations using mouse"			},
{ "quit",		popcmd,		"exit submenu"					},
{ NULL_COMMAND										}
};

static COMMAND_FUNC( multi_menu )
{
	PUSHCMD(multi_ctbl,"multi");
}

/* single fixation runs */
static Command sgl_ctbl[]={
{ "one_click",		sgl_click_fix,	"enter a single fixation using mouse"		},
{ "fixate",		do_fixate,	"make a fixation"				},
{ "search",		do_search,	"search knowledge database for conflict"	},
{ "plan",		do_plan,	"plan next saccade (result in $fix_x, $fix_y)"	},
{ "quit",		popcmd,		"exit submenu"					},
{ NULL_COMMAND										}
};

static COMMAND_FUNC( sgl_menu )
{
	PUSHCMD(sgl_ctbl,"single");
}

static Command model_ctbl[]={
{ "init",		do_init_model,	"initialize model state"			},
{ "display_model",	disp_mode,	"enable/disable graphical model display"	},
{ "strategy",		strat_menu,	"set search strategy submenu"			},
{ "single",		sgl_menu,	"single fixation commands"			},
{ "multi",		multi_menu,	"multiple fixation commands"			},
{ "render",		render_model,	"redisplay model state"				},
{ "test",		test_menu,	"model component test submenu"			},
{ "params",		param_menu,	"miscellaneous simulation parameters"		},
{ "quit",		popcmd,		"exit submenu"					},
{ NULL_COMMAND										}
};

COMMAND_FUNC( model_menu )
{
#ifdef DEBUG
	if( debug_model == 0 ) debug_model=add_debug_module(QSP_ARG  "model");
#endif /* DEBUG */

	PUSHCMD(model_ctbl,"model");
}

#endif /* HAVE_X11 */

