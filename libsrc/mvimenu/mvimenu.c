#include "quip_config.h"

/* generic movie library interface */

#ifdef HAVE_TIME_H
#include <time.h>		/* ctime() */
#endif

#ifdef HAVE_SYS_STAT_H
#include <sys/stat.h>		/* ctime() */
#endif

#include "quip_prot.h"
#include "seq_api.h"		/* seq_menu() prototype */
#include "history.h"	/* init_hist_from_item_list() prototype */
#include "gmovie.h"
#include "function.h"
#include "xmvi.h"
#include "seq.h"	// BUG - need to separate public & private

//#ifdef BUILD_FOR_OBJC
//#include "sizable.h"
//#endif // BUILD_FOR_OBJC

static const char *movie_dir=NULL;	// BUG not thread-safe

/* globals */
int n_refresh=1;
#ifdef QUIP_DEBUG
int mvi_debug=(-1);
#endif /* QUIP_DEBUG */

static Seq_Module mvi_sm;
static Item_Class *playable_iclp=NULL;

/* local prototypes */
ITEM_INTERFACE_PROTOTYPES(Movie,mvi)
#define pick_mvi(p)	_pick_mvi(QSP_ARG  p)

#define mvi_of(s)		_mvi_of(QSP_ARG  s)
#define get_mvi(s)		_get_mvi(QSP_ARG  s)
#define del_mvi(s)		_del_mvi(QSP_ARG  s)
#define new_mvi(s)		_new_mvi(QSP_ARG  s)
#define list_mvis(fp)		_list_mvis(QSP_ARG  fp)
#define init_mvis()		_init_mvis(SINGLE_QSP_ARG)

static const char *get_movie_name(SINGLE_QSP_ARG_DECL);

static COMMAND_FUNC( do_set_mvidir );
static void set_mvidir(QSP_ARG_DECL  const char *);
static const char *get_playable_name(SINGLE_QSP_ARG_DECL);
static double get_mvi_size(QSP_ARG_DECL  Item *mvip,int index);
static const char * get_mvi_prec_string(QSP_ARG_DECL  Item *mvip);
static double get_mvi_il_flg(QSP_ARG_DECL  Item *mvip);
static void play_movie(QSP_ARG_DECL  const char *s);
static COMMAND_FUNC( do_start_movie );
static void ref_mvi(Movie *mvip);
static COMMAND_FUNC( do_set_nrefresh );
static COMMAND_FUNC( do_movie_info );
static COMMAND_FUNC( do_play_movie );
static int is_ready_to_play(QSP_ARG_DECL  Movie *mvip);
static COMMAND_FUNC( do_shuttle );
static COMMAND_FUNC( do_getframe );
static COMMAND_FUNC( do_getfields );
static COMMAND_FUNC( do_del_mvi );
static Movie *open_if(QSP_ARG_DECL  const char *s);
#define OPEN_IF(s)	open_if(QSP_ARG  s)
static int obj_prec_ok(QSP_ARG_DECL  Data_Obj *dp);
static Movie *create_writable_movie(QSP_ARG_DECL  const char *moviename,uint32_t n_fields);


static Size_Functions mvi_sf={
	get_mvi_size,
	get_mvi_prec_string
};

static Interlace_Functions mvi_if={
	get_mvi_il_flg
};

ITEM_INTERFACE_DECLARATIONS(Movie,mvi,0)

static Movie_Module *the_mmp=NULL;

List *movie_list(SINGLE_QSP_ARG_DECL)
{
	return(item_list(mvi_itp));
}

int movie_ok(void)
{
	if( the_mmp == NULL ){
		NWARN("No movie module loaded");
		return(0);
	}
	return(1);
}


/*
 * set up a new movie structure
 */

static int clobber_movies=1;

Movie *create_movie(QSP_ARG_DECL  const char *moviename)
{
	Movie *mvip;

	mvip=mvi_of(moviename);
	if( mvip!=NULL ){
		/* overwrite an existing movie only if clobber flag set */
		if( clobber_movies ){
			if( verbose ){
				sprintf(ERROR_STRING,
			"create_movie:  clobbering existing movie %s",moviename);
				advise(ERROR_STRING);
			}
			del_mvi(mvip);
			/* maybe do something module-specific here */
			/* what does rtv do, that's the question! */
		} else {
			sprintf(ERROR_STRING,"NOT clobbering existing movie %s",moviename);
			advise(ERROR_STRING);
			return(NULL);
		}
	}

	mvip=new_mvi(moviename);
	if( mvip == NULL )
		return(mvip);

	SET_MOVIE_FLAGS(mvip, 0);
	SET_MOVIE_OFFSET(mvip, 0);
	SET_MOVIE_FILENAME(mvip, NULL);
	SET_MOVIE_SHAPE(mvip, ALLOC_SHAPE );

	return(mvip);
}


/*
 * set up a movie structure for capturing or single frame animation
 */

static Movie *create_writable_movie(QSP_ARG_DECL  const char *moviename,uint32_t n_fields)
{
	Movie *mvip;

	mvip=create_movie(QSP_ARG  moviename);
	if( mvip == NULL ) return(mvip);

	if( !movie_ok() ) return(NULL);

#ifdef QUIP_DEBUG
if( debug & mvi_debug ) advise("calling movie setup_func");
#endif /* QUIP_DEBUG */

	if( (*the_mmp->setup_func)(QSP_ARG  mvip,n_fields) < 0 ){	/* device specific */
		delete_movie(QSP_ARG  mvip);
		return(NULL);
	}

	return(mvip);
}

static COMMAND_FUNC( do_movie_info )
{
	Movie *mvip;

	mvip=pick_mvi("");
	if( mvip == NULL ) return;

	printf("Movie %s:\n",MOVIE_NAME(mvip));
	printf("\t%d frames\n\t%d rows\n\t%d columns\n",
		MOVIE_FRAMES(mvip),MOVIE_HEIGHT(mvip),MOVIE_WIDTH(mvip));
	printf("\t%d bytes per pixel\n",
		MOVIE_DEPTH(mvip));
	printf("\tLast played:  %s",ctime( & MOVIE_TIME(mvip)));
	if( IS_REFERENCED(mvip) )
		printf("\treferenced by at least one sequence\n");
	if( IS_ASSEMBLING(mvip) )
		printf("\tcurrently being assembled\n");
	if( IS_RECORDING(mvip) )
		printf("\tcurrently being recorded\n");

	if( !movie_ok() ) return;

#ifdef QUIP_DEBUG
if( debug & mvi_debug ) advise("calling movie info_func");
#endif /* QUIP_DEBUG */

	(*the_mmp->info_func)(QSP_ARG  mvip);		/* device specific */
}

/* automatically open a movie if necessary */

static Movie *open_if(QSP_ARG_DECL  const char *s)
{
	Movie *mvip;

	mvip=mvi_of(s);

	if( mvip == NULL ){
		if( verbose ){
			sprintf(ERROR_STRING,
			"Trying to open movie file %s",s);
			advise(ERROR_STRING);
		}

#ifdef QUIP_DEBUG
if( debug & mvi_debug ) advise("calling movie open_func");
#endif /* QUIP_DEBUG */

/* use pathname here... */
		(*the_mmp->open_func)(QSP_ARG  s);		/* device specific */
		mvip = mvi_of(s);
	}
	if( mvip == NULL ){
		sprintf(ERROR_STRING,"No movie \"%s\"",s);
		WARN(ERROR_STRING);
		return(mvip);
	}

	if( IS_ASSEMBLING(mvip) ){
		WARN("can't play a movie that is being assembled");
		return(NULL);
	}

	SET_MOVIE_TIME(mvip, time(NULL) );

	return(mvip);
}

static COMMAND_FUNC( do_open_mvi )
{
	const char *s;
	Movie *mvip;

	s=NAMEOF("movie filename");
	mvip = mvi_of(s);
	if( mvip != NULL ){
		sprintf(ERROR_STRING,"Movie %s was already open!?",s);
		advise(ERROR_STRING);
		SET_MOVIE_TIME(mvip, time(NULL) );
		return;
	}
	/*mvip =*/ OPEN_IF(s);
}

static COMMAND_FUNC( do_openif_mvi )
{
	const char *s;
	//Movie *mvip;

	s=NAMEOF("movie filename");
	/*mvip =*/ OPEN_IF(s);
}

#define MOVIENAME_PMPT	"movie name"

static const char *get_movie_name(SINGLE_QSP_ARG_DECL)
{
#ifdef HISTORY
	if( intractive(SINGLE_QSP_ARG) ){
		List *lp;
		lp = movie_list(SINGLE_QSP_ARG);
		if( lp != NULL )
			init_hist_from_item_list(QSP_ARG  MOVIENAME_PMPT,lp);
	}
#endif // HISTORY

	return( NAMEOF(MOVIENAME_PMPT) );
}

static COMMAND_FUNC( do_play_movie )
{
	const char *s;
	Member_Info *mip;

	s = get_playable_name(SINGLE_QSP_ARG);

	if( !movie_ok() ) return;

	mip = get_member_info(playable_iclp,s);

	/*
	 * On some implementations (cosmo, sirius), the movies are
	 * regular disk files, and therefore are not necessarily in
	 * our database at this point.  When that is the case, mip
	 * will have a value of NULL, as returned by
	 * get_member_info().  Therefore it is legal to pass
	 * a name to play_movie even if there is not movie object
	 * already loaded.  On the other hand, in the RTV implementation,
	 * all of the playable movies are already in the database,
	 * so in that system play_movie() must check for a bogus name!
	 */

	if( mip == NULL || mip->mi_itp == mvi_itp )
		play_movie(QSP_ARG  s);
	else
		show_sequence(QSP_ARG  s);
}

static void play_movie(QSP_ARG_DECL  const char *s)
{
	Movie *mvip;

	mvip = OPEN_IF(s);
	if( mvip == NULL ) return;

	if( ! is_ready_to_play(QSP_ARG  mvip) ) return;


#ifdef QUIP_DEBUG
if( debug & mvi_debug ) advise("calling movie play_func");
#endif /* QUIP_DEBUG */

	(*the_mmp->play_func)(QSP_ARG  mvip);			/* device specific */

#ifdef QUIP_DEBUG
if( debug & mvi_debug ) advise("calling movie wait_func");
#endif /* QUIP_DEBUG */

	(*the_mmp->wait_func)();			/* device specific */
}

static int is_ready_to_play(QSP_ARG_DECL  Movie *mvip)
{

#ifdef CAUTIOUS
	if( IS_RECORDING(mvip) ){
		WARN("CAUTIOUS:  can't play a movie that is being recorded");
		return(0);
	}
	if( IS_ASSEMBLING(mvip) ){
		WARN("CAUTIOUS:  can't play a movie that is being assembled");
		return(0);
	}
#endif /* CAUTIOUS */


#ifdef QUIP_DEBUG
if( debug & mvi_debug ) advise("calling movie setup_play_func");
#endif /* QUIP_DEBUG */

	if( (*the_mmp->setup_play_func)(mvip) < 0 )	/* device specific */
		return(0);
	
	/* remember the time this movie was played */
	SET_MOVIE_TIME(mvip, time((time_t *)NULL) );

	return(1);
}

static COMMAND_FUNC( do_shuttle )
{
	Movie *mvip;
	incr_t frame;

	mvip=pick_mvi("");
	frame= (incr_t) HOW_MANY("frame index");

	if( mvip == NULL ) return;

	if( frame < 0 || frame >= (incr_t) MOVIE_FRAMES(mvip) ){
		sprintf(ERROR_STRING,"Frame index %d out of range for movie %s",
			frame,MOVIE_NAME(mvip));
		WARN(ERROR_STRING);
		return;
	}

	if( !movie_ok() ) return;

	if( ! is_ready_to_play(QSP_ARG  mvip) ) return;

	(*the_mmp->shuttle_func)(QSP_ARG  mvip,(index_t)frame);
}

static int obj_prec_ok(QSP_ARG_DECL  Data_Obj *dp)
{
	if( OBJ_PREC(dp) != PREC_BY && OBJ_PREC(dp) != PREC_UBY ){
		WARN("movie frame/field precision should be byte or u_byte");
		return(0);
	}
	if( OBJ_COMPS(dp) != 4 ){
		WARN("movie frame/field objects should have 4 components");
		return(0);
	}
	return(1);
}

/* There is some confusion about what this should do, i.e., frames
 * or fields?  On RTV, we fetch a whole frame into the frame buffer,
 * and reference the two fields using interlaced subimages.
 * With Cosmo, we get a frame from the movie library, which
 * contains two compressed images corresponding to the fields.
 * Thus for cosmo it is more sensible to as for a field...
 */

static COMMAND_FUNC( do_getframe )
{
	Movie *mvip;
	incr_t frame;
	Data_Obj *dp;
	const char *moviename;

	dp = pick_obj("");
	moviename = get_movie_name(SINGLE_QSP_ARG);
	frame = (incr_t) HOW_MANY("frame index");

	mvip = OPEN_IF(moviename);

	if( mvip == NULL || dp == NULL ) return;

	if( frame < 0 || frame >= (incr_t)MOVIE_FRAMES(mvip) ){
		sprintf(ERROR_STRING,
			"frame index %d out of range for movie %s",
			frame,MOVIE_NAME(mvip));
		WARN(ERROR_STRING);
		return;
	}
	if( OBJ_ROWS(dp) != MOVIE_HEIGHT(mvip) ||
		OBJ_COLS(dp) != MOVIE_WIDTH(mvip) ){
		sprintf(ERROR_STRING,
			"frame size mismatch between image %s and movie %s",
			OBJ_NAME(dp),MOVIE_NAME(mvip));
		WARN(ERROR_STRING);
		sprintf(ERROR_STRING,"image %s is %d x %d, movie %s is %d x %d",
			OBJ_NAME(dp),OBJ_ROWS(dp),OBJ_COLS(dp),
			MOVIE_NAME(mvip),MOVIE_HEIGHT(mvip),MOVIE_WIDTH(mvip));
		advise(ERROR_STRING);
		return;
	}
	if( !obj_prec_ok(QSP_ARG  dp) ) return;

	if( !movie_ok() ) return;

	(*the_mmp->frame_func)(QSP_ARG  mvip,(index_t)frame,dp);	/* device specific */
}

/* get a single color component */

static COMMAND_FUNC( do_getframec )
{
	Movie *mvip;
	incr_t frame;
	int comp;
	Data_Obj *dp;
	const char *moviename;

	dp = pick_obj("");
	moviename = get_movie_name(SINGLE_QSP_ARG);
	frame = (incr_t) HOW_MANY("frame index");
	comp =  (int) HOW_MANY("component index");

	mvip = OPEN_IF(moviename);

	if( mvip == NULL || dp == NULL ) return;

	if( frame < 0 || frame >= (incr_t) MOVIE_FRAMES(mvip) ){
		sprintf(ERROR_STRING,
			"frame index %d out of range for movie %s",
			frame,MOVIE_NAME(mvip));
		WARN(ERROR_STRING);
		return;
	}
	if( OBJ_ROWS(dp) != MOVIE_HEIGHT(mvip) ||
		OBJ_COLS(dp) != MOVIE_WIDTH(mvip) ){
		sprintf(ERROR_STRING,
			"frame size mismatch between image %s and movie %s",
			OBJ_NAME(dp),MOVIE_NAME(mvip));
		WARN(ERROR_STRING);
		sprintf(ERROR_STRING,"image %s is %d x %d, movie %s is %d x %d",
			OBJ_NAME(dp),OBJ_ROWS(dp),OBJ_COLS(dp),
			MOVIE_NAME(mvip),MOVIE_HEIGHT(mvip),MOVIE_WIDTH(mvip));
		advise(ERROR_STRING);
		return;
	}
	/* BUG? do we have a way to check how many components the movie has? */

	if( !obj_prec_ok(QSP_ARG  dp) ) return;

	if( !movie_ok() ) return;

	(*the_mmp->framec_func)(QSP_ARG  mvip,(index_t)frame,dp,comp);	/* device specific */
}

static COMMAND_FUNC( do_getfields )
{
	Movie *mvip;
	incr_t field;
	Data_Obj *dp;
	const char *moviename;

	dp = pick_obj("");
	moviename = get_movie_name(SINGLE_QSP_ARG);
	field = (incr_t) HOW_MANY("starting field index");

	mvip = OPEN_IF(moviename);

	if( mvip == NULL || dp == NULL ) return;

	if( field < 0 || field >= (incr_t)MOVIE_FRAMES(mvip)*2 ){
		sprintf(ERROR_STRING,
			"field index %d out of range for movie %s",
			field,MOVIE_NAME(mvip));
		WARN(ERROR_STRING);
		return;
	}
	if( OBJ_ROWS(dp) != MOVIE_HEIGHT(mvip)/2 ||
		OBJ_COLS(dp) != MOVIE_WIDTH(mvip) ){
		sprintf(ERROR_STRING,
			"field size mismatch between image %s and movie %s",
			OBJ_NAME(dp),MOVIE_NAME(mvip));
		WARN(ERROR_STRING);
		sprintf(ERROR_STRING,
		"image %s is %d x %d, movie %s field size is %d x %d",
			OBJ_NAME(dp),OBJ_ROWS(dp),OBJ_COLS(dp),
			MOVIE_NAME(mvip),MOVIE_HEIGHT(mvip)/2,MOVIE_WIDTH(mvip));
		advise(ERROR_STRING);
		return;
	}
	if( !obj_prec_ok(QSP_ARG  dp) ) return;

	if( !movie_ok() ) return;

	(*the_mmp->field_func)(QSP_ARG  mvip,(index_t)field,dp);	/* device specific */
}

static COMMAND_FUNC( do_getfieldc )
{
	Movie *mvip;
	incr_t field;
	int comp;
	Data_Obj *dp;
	const char *moviename;

	dp = pick_obj("");
	moviename = get_movie_name(SINGLE_QSP_ARG);
	field = (incr_t) HOW_MANY("starting field index");
	comp =  (int) HOW_MANY("component index");

	mvip = OPEN_IF(moviename);

	if( mvip == NULL || dp == NULL ) return;

	if( field < 0 || field >= (incr_t)MOVIE_FRAMES(mvip)*2 ){
		sprintf(ERROR_STRING,
			"field index %d out of range for movie %s",
			field,MOVIE_NAME(mvip));
		WARN(ERROR_STRING);
		return;
	}
	if( OBJ_ROWS(dp) != MOVIE_HEIGHT(mvip)/2 ||
		OBJ_COLS(dp) != MOVIE_WIDTH(mvip) ){
		sprintf(ERROR_STRING,
			"field size mismatch between image %s and movie %s",
			OBJ_NAME(dp),MOVIE_NAME(mvip));
		WARN(ERROR_STRING);
		sprintf(ERROR_STRING,
		"image %s is %d x %d, movie %s field size is %d x %d",
			OBJ_NAME(dp),OBJ_ROWS(dp),OBJ_COLS(dp),
			MOVIE_NAME(mvip),MOVIE_HEIGHT(mvip)/2,MOVIE_WIDTH(mvip));
		advise(ERROR_STRING);
		return;
	}
	if( !obj_prec_ok(QSP_ARG  dp) ) return;

	if( !movie_ok() ) return;

	(*the_mmp->fieldc_func)(QSP_ARG  mvip,(index_t)field,dp,comp);	/* device specific */
}

static COMMAND_FUNC( do_del_mvi )
{
	Movie *mvip;

	mvip = pick_mvi("");
	if( mvip == NULL ) return;

	if( IS_ASSEMBLING(mvip) ){
		WARN("can't delete a movie which is being assembled");
		return;
	}
	close_movie(QSP_ARG  mvip);
}

/*
 * this is normally called to close a movie that has been played.
 *
 * An "open" movie is ready to play:
 *	rtv:	all movies are always "open"
 *	cosmo:	open compressed file...
 *	sirius:	open disk file...
 *
 * "Closing" a movie makes in unavailable for playback...
 * Therefore we must delete all sequences which depend on it;
 * call a module-specific close routine before destroying the struct.
 */

void close_movie(QSP_ARG_DECL  Movie *mvip)
{
	/* First must destroy all sequences that reference this movie.
	 * The fact that this flag is set does not insure that there
	 * are actually any seqs to delete, since they could have been
	 * deleted already by someone else...
	 */

	if( IS_REFERENCED(mvip) ){
		Node *np;
		Seq *seqp;
		List *lp;

		lp=seqs_referring(QSP_ARG  mvip);
		np=QLIST_HEAD(lp);
		while(np!=NULL){
			seqp=(Seq *)np->n_data;
			/* BUG? does this delete sequences which depend on seqp? */
			delseq(QSP_ARG  seqp);
			np=np->n_next;
		}
		dellist(lp);
	}

	if( the_mmp != NULL )
		(*the_mmp->close_func)(QSP_ARG  mvip);		/* device specific */

	delete_movie(QSP_ARG  mvip);
}

/* delete a movie from the database */

void delete_movie(QSP_ARG_DECL  Movie *mvip)
{
	if( mvip == NULL ) return;

	del_mvi(mvip);	/* remove from item database */
}

static COMMAND_FUNC( do_set_nrefresh )
{
	int n;

	n= (int) HOW_MANY("refresh count");
	if( n > 0 )
		n_refresh=n;
	else
		WARN("refresh cound must be positive");
}

static COMMAND_FUNC( do_set_mvidir )
{
	const char *s;

	s=NAMEOF("movie directory");
	set_mvidir(QSP_ARG  s);
}

static void set_mvidir(QSP_ARG_DECL  const char *dirname)
{
	struct stat statb;


	/* Make sure the given directory is a valid directory name.  */

	/* first check that the file exists! */
	if( stat(dirname,&statb) < 0 ){
		tell_sys_error(dirname);
		sprintf(ERROR_STRING,"Couldn't set movie directory to %s",dirname);
		WARN(ERROR_STRING);
		return;
	}

	if( ! S_ISDIR(statb.st_mode) ){
		sprintf(ERROR_STRING,
			"%s is not a directory",dirname);
		WARN(ERROR_STRING);
		return;
	}

	if( movie_dir != NULL ){
		sprintf(ERROR_STRING,
	"Resetting movie directory to %s (was %s)",dirname,movie_dir);
		advise(ERROR_STRING);
		rls_str((char *)movie_dir);
	}
	movie_dir = savestr( dirname );
}

static COMMAND_FUNC( do_list_mvis ){ list_mvis(tell_msgfile()); }

#define ADD_CMD(s,f,h)	ADD_COMMAND(playback_menu,s,f,h)

MENU_BEGIN(playback)
ADD_CMD( directory,	do_set_mvidir,		set movie directory )
ADD_CMD( open,		do_open_mvi,		open a movie file )
ADD_CMD( insure,	do_openif_mvi,		open a (possibly opened) movie file )
ADD_CMD( play,		do_play_movie,		play a movie to external video device )
ADD_CMD( shuttle,	do_shuttle,		play a movie frame to external video )
ADD_CMD( close,		do_del_mvi,		close an open movie file )
ADD_CMD( refresh,	do_set_nrefresh,	set # of times to repeat fields )
ADD_CMD( sequence,	do_seq_menu,		sequence submenu )
ADD_CMD( getframe,	do_getframe,		extract a single movie frame )
ADD_CMD( getfield,	do_getfields,		extract movie field(s) )
ADD_CMD( getfrmcomp,	do_getframec,		extract single component of a frame )
ADD_CMD( getfldcomp,	do_getfieldc,		extract component of a field )
ADD_CMD( list,		do_list_mvis,		list movies )
ADD_CMD( info,		do_movie_info,		print info about a movie )
MENU_END(playback)

static COMMAND_FUNC( do_play_menu )
{
	PUSH_MENU(playback);
}

#define MAX_MOVIE_FIELDS	0x40000000	/* a dummy value */

static COMMAND_FUNC( do_start_movie )
{
	const char *s;
	Movie *mvip;

	s=NAMEOF("name for movie");

	mvip = create_writable_movie(QSP_ARG  s,MAX_MOVIE_FIELDS);
	if( mvip == NULL ) return;
	SET_MOVIE_FLAG_BITS(mvip, MVI_ASSEMBLING);
}

static COMMAND_FUNC( do_add_frame )
{
	Data_Obj *dp;
	Movie *mvip;

	mvip = pick_mvi("");
	dp = pick_obj("");

	if( mvip == NULL || dp == NULL ) return;

	if( !movie_ok() ) return;

	(*the_mmp->add_func)(QSP_ARG  mvip,dp);		/* device specific */
}

static COMMAND_FUNC( do_end_movie )
{
	Movie *mvip;

	mvip = pick_mvi("");
	if( mvip == NULL ) return;

	if( ! IS_ASSEMBLING(mvip) ){
		sprintf(ERROR_STRING,"Movie \"%s\" is not being assembled",MOVIE_NAME(mvip));
		WARN(ERROR_STRING);
		return;
	}

	if( !movie_ok() ) return;

	(*the_mmp->end_func)(QSP_ARG  mvip);		/* device specific */
}

static COMMAND_FUNC( do_rec_movie )
{
	const char *s;
	uint32_t n;
	Movie *mvip;

	s=NAMEOF("name for movie");
	n= (uint32_t) HOW_MANY("number of fields");

	if( !movie_ok() ) return;
#ifdef QUIP_DEBUG
if( debug & mvi_debug ) advise("creating writable movie");
#endif /* QUIP_DEBUG */

	mvip = create_writable_movie(QSP_ARG  s,n);
	if( mvip == NULL ) return;
	SET_MOVIE_FLAG_BITS(mvip, MVI_RECORDING);


#ifdef QUIP_DEBUG
if( debug & mvi_debug ) advise("calling movie record_func");
#endif /* QUIP_DEBUG */

	(*the_mmp->record_func)(QSP_ARG  n,mvip);		/* device specific */

	/* Not sure which of these comments apply to the present code !? */
	/* need something else here? */
	/* NOT correct for cosmo (6.2) ... */

#ifdef QUIP_DEBUG
if( debug & mvi_debug ) advise("calling movie end_func!?");
#endif /* QUIP_DEBUG */

	(*the_mmp->end_func)(QSP_ARG  mvip);		/* device specific */
}

static COMMAND_FUNC( do_mon_input )
{
	if( !movie_ok() ) return;


#ifdef QUIP_DEBUG
if( debug & mvi_debug ) advise("calling movie monitor_func");
#endif /* QUIP_DEBUG */

	(*the_mmp->monitor_func)(SINGLE_QSP_ARG);		/* device specific */
}

#undef ADD_CMD
#define ADD_CMD(s,f,h)	ADD_COMMAND(assemble_menu,s,f,h)

MENU_BEGIN(assemble)
ADD_CMD( directory,	do_set_mvidir,	set movie directory )
ADD_CMD( start,		do_start_movie,	assemble movie from still frames )
ADD_CMD( add,		do_add_frame,	add a still frame to a movie )
ADD_CMD( finish,	do_end_movie,	close movie file )
ADD_CMD( record,	do_rec_movie,	record a movie )
ADD_CMD( monitor,	do_mon_input,	monitor input video )
ADD_CMD( list,		do_list_mvis,	list movies )
ADD_CMD( info,		do_movie_info,	print info about a movie )
MENU_END(assemble)

static COMMAND_FUNC( ass_menu )
{
	PUSH_MENU(assemble);
}

static COMMAND_FUNC( dev_menu )
{
	if( !movie_ok() ) return;


#ifdef QUIP_DEBUG
if( debug & mvi_debug ) advise("calling movie device menu");
#endif /* QUIP_DEBUG */

	(*the_mmp->menu_func)(SINGLE_QSP_ARG);
}

static COMMAND_FUNC( report_mm )
{
	if( the_mmp == NULL )
		WARN("No movie module loaded");
	else {
		sprintf(msg_str,"Current movie module:  %s",the_mmp->mm_name);
		prt_msg(msg_str);
	}
}

#undef ADD_CMD
#define ADD_CMD(s,f,h)	ADD_COMMAND(movie_menu,s,f,h)

MENU_BEGIN(movie)
ADD_CMD( playback,	do_play_menu,	movie playback submenu )
ADD_CMD( control,	dev_menu,	device control submenu )
ADD_CMD( assemble,	ass_menu,	movie construction submenu )
ADD_CMD( list,		do_list_mvis,	list movies )
ADD_CMD( tell,		report_mm,	report current movie module )
ADD_CMD( delete,	do_del_mvi,	delete movie )
ADD_CMD( info,		do_movie_info,	print info about a movie )
MENU_END(movie)

/* support size functions */

static double get_mvi_size(QSP_ARG_DECL  Item *ip,int index)
{
	double d;
	Movie *mvip;

	mvip = (Movie *)ip;

	switch(index){
		case 0: d=MOVIE_DEPTH(mvip); break;
		case 1: d=MOVIE_WIDTH(mvip); break;
		case 2: d=MOVIE_HEIGHT(mvip); break;
		case 3: d=MOVIE_FRAMES(mvip); break;
		default:
			sprintf(DEFAULT_ERROR_STRING,"Unsupported movie size function, index %d",index);
			NWARN(DEFAULT_ERROR_STRING);
			d=1.0;
			break;
	}
	return(d);
}

static const char * get_mvi_prec_string(QSP_ARG_DECL  Item *ip)
{
	return "u_byte";
}

static double get_mvi_il_flg(QSP_ARG_DECL  Item *ip)
{
	Movie *mvip;
	mvip = (Movie *) ip;
	if( SHP_FLAGS(MOVIE_SHAPE(mvip)) & DT_INTERLACED ) return(1.0);
	else return(0.0);
}


void add_playable(Item_Type * itp,void *vp)
{
	add_items_to_class(playable_iclp,itp,vp,NULL);
}

#define PLAYABLE_PMPT	"movie or sequence"

static const char *get_playable_name(SINGLE_QSP_ARG_DECL)
{
	const char *s;

#ifdef HISTORY
	if( intractive(SINGLE_QSP_ARG) )
		init_hist_from_class(QSP_ARG  PLAYABLE_PMPT,playable_iclp);
#endif // HISTORY

	s=NAMEOF(PLAYABLE_PMPT);
	return(s);
}

static void ref_mvi(Movie *mvip)
{
	SET_MOVIE_FLAG_BITS(mvip, MVI_REFERENCED);
}

static Movie *lookup_movie(QSP_ARG_DECL  const char *name)
{
	Movie *mvip;

	mvip=get_mvi(name);
	if( mvip == NULL ) return(mvip);

	SET_MOVIE_TIME(mvip, time(NULL));
	return(mvip);
}

COMMAND_FUNC( do_movie_menu )
{

	if( playable_iclp == NULL ){
		if( mvi_itp == NULL ) init_mvis();
		add_sizable(mvi_itp,&mvi_sf,NULL);
		add_interlaceable(mvi_itp,&mvi_if,NULL);

		playable_iclp = new_item_class("playable");
		add_playable(mvi_itp,NULL);

		// add sequences to the playable class
		init_movie_sequences(SINGLE_QSP_ARG);
	}


#ifdef HAVE_X11
	if( the_mmp == NULL ){
		if( verbose )
			advise("no movie module loaded, using X w/ fileio...");
		xmvi_init(SINGLE_QSP_ARG);
	}
#endif /* HAVE_X11 */

	/* initialize the sequencer module */

	if( the_mmp != NULL ){
		mvi_sm.init_func = (int (*)(void *)) the_mmp->setup_play_func;
		mvi_sm.get_func  = (void * (*)(const char *)) lookup_movie;
		mvi_sm.show_func = (void (*)(void *)) the_mmp->play_func;
		mvi_sm.rev_func  = (void (*)(void *)) the_mmp->reverse_func;
		mvi_sm.wait_func = the_mmp->wait_func;
		mvi_sm.ref_func  = (void (*)(void *))ref_mvi;

		load_seq_module(&mvi_sm);
	}

#ifdef QUIP_DEBUG
	if( mvi_debug < 0 )
		mvi_debug = add_debug_module("movie");
#endif /* QUIP_DEBUG */

	PUSH_MENU(movie);
}

void load_movie_module(QSP_ARG_DECL  Movie_Module *mmp)
{
	the_mmp = mmp;

#ifdef QUIP_DEBUG
if( debug & mvi_debug ) advise("calling movie init_func");
#endif /* QUIP_DEBUG */

	(*mmp->init_func)(SINGLE_QSP_ARG);
}

const char *movie_pathname(const char *filename)
{
	/* BUG should be MAXPATHLEN */
	static char pathname[LLEN];

	if( movie_dir != NULL && *movie_dir && *filename != '/' ){
		sprintf(pathname,"%s/%s",movie_dir,filename);
		return(pathname);
	} else {
		return(filename);
	}
}

