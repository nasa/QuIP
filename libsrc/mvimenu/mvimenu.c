#include "quip_config.h"

char VersionId_mvimenu_mvimenu[] = QUIP_VERSION_STRING;

/* generic movie library interface */

#ifdef HAVE_TIME_H
#include <time.h>		/* ctime() */
#endif

#ifdef HAVE_SYS_STAT_H
#include <sys/stat.h>		/* ctime() */
#endif

#include "items.h"
#include "savestr.h"
#include "function.h"		/* add_sizable() prototype */
#include "seq_api.h"		/* seq_menu() prototype */
#include "query.h"		/* intractive() prototype */
#include "../interpreter/history.h"	/* init_hist_from_item_list() prototype */
#include "debug.h"
#include "submenus.h"
#include "version.h"		/* auto_version() */

#include "gmovie.h"
#include "xmvi.h"

static const char *movie_dir=NULL;

/* globals */
int n_refresh=1;
#ifdef DEBUG
int mvi_debug=(-1);
#endif /* DEBUG */

static Seq_Module mvi_sm;
static Item_Class *playable_icp=NO_ITEM_CLASS;

/* local prototypes */
ITEM_INTERFACE_PROTOTYPES(Movie,mvi)
#define MVI_OF(s)		mvi_of(QSP_ARG  s)
#define GET_MVI(s)		get_mvi(QSP_ARG  s)

static const char *get_movie_name(SINGLE_QSP_ARG_DECL);

static COMMAND_FUNC( do_set_mvidir );
static void set_mvidir(QSP_ARG_DECL  const char *);
static const char *get_playable_name(SINGLE_QSP_ARG_DECL);
static double get_mvi_size(Item *mvip,int index);
static double get_mvi_il_flg(Item *mvip);
static void play_movie(QSP_ARG_DECL  const char *s);
static COMMAND_FUNC( do_start_movie );
static void ref_mvi(Movie *mvip);
static COMMAND_FUNC( do_set_nrefresh );
static COMMAND_FUNC( movie_seq_menu );
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
static Movie *create_writable_movie(QSP_ARG_DECL  const char *moviename,u_long n_fields);


static Size_Functions mvi_sf={
	/*(double (*)(Item *,int))*/		get_mvi_size,
	/*(Item * (*)(Item *,index_t))*/	NULL,
	/*(Item * (*)(Item *,index_t))*/	NULL,
	/*(double (*)(Item *))*/		get_mvi_il_flg,
};


ITEM_INTERFACE_DECLARATIONS(Movie,mvi)

static Movie_Module *the_mmp=NO_MOVIE_MODULE;

List *movie_list(SINGLE_QSP_ARG_DECL)
{
	return(item_list(QSP_ARG  mvi_itp));
}

int movie_ok(void)
{
	if( the_mmp == NO_MOVIE_MODULE ){
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

	mvip=MVI_OF(moviename);
	if( mvip!=NO_MOVIE ){
		/* overwrite an existing movie only if clobber flag set */
		if( clobber_movies ){
			if( verbose ){
				sprintf(ERROR_STRING,
			"create_movie:  clobbering existing movie %s",moviename);
				advise(ERROR_STRING);
			}
			del_mvi(QSP_ARG  moviename);
			/* maybe do something module-specific here */
			/* what does rtv do, that's the question! */
		} else {
			sprintf(ERROR_STRING,"NOT clobbering existing movie %s",moviename);
			advise(ERROR_STRING);
			return(NO_MOVIE);
		}
	}

	mvip=new_mvi(QSP_ARG  moviename);
	if( mvip == NO_MOVIE )
		return(mvip);

	mvip->mvi_flags = 0;
	mvip->mvi_offset = 0;
	mvip->mvi_filename = NULL;

	return(mvip);
}


/*
 * set up a movie structure for capturing or single frame animation
 */

static Movie *create_writable_movie(QSP_ARG_DECL  const char *moviename,u_long n_fields)
{
	Movie *mvip;

	mvip=create_movie(QSP_ARG  moviename);
	if( mvip == NO_MOVIE ) return(mvip);

	if( !movie_ok() ) return(NO_MOVIE);

#ifdef DEBUG
if( debug & mvi_debug ) advise("calling movie setup_func");
#endif /* DEBUG */

	if( (*the_mmp->setup_func)(QSP_ARG  mvip,n_fields) < 0 ){	/* device specific */
		delete_movie(QSP_ARG  mvip);
		return(NO_MOVIE);
	}

	return(mvip);
}

static COMMAND_FUNC( do_movie_info )
{
	Movie *mvip;

	mvip=PICK_MVI("");
	if( mvip == NO_MOVIE ) return;

	printf("Movie %s:\n",mvip->mvi_name);
	printf("\t%d frames\n\t%d rows\n\t%d columns\n",
		mvip->mvi_nframes,mvip->mvi_height,mvip->mvi_width);
	printf("\t%d bytes per pixel\n",
		mvip->mvi_depth);
	printf("\tLast played:  %s",ctime(&mvip->mvi_time));
	if( IS_REFERENCED(mvip) )
		printf("\treferenced by at least one sequence\n");
	if( IS_ASSEMBLING(mvip) )
		printf("\tcurrently being assembled\n");
	if( IS_RECORDING(mvip) )
		printf("\tcurrently being recorded\n");

	if( !movie_ok() ) return;

#ifdef DEBUG
if( debug & mvi_debug ) advise("calling movie info_func");
#endif /* DEBUG */

	(*the_mmp->info_func)(QSP_ARG  mvip);		/* device specific */
}

/* automatically open a movie if necessary */

static Movie *open_if(QSP_ARG_DECL  const char *s)
{
	Movie *mvip;

	mvip=MVI_OF(s);

	if( mvip == NO_MOVIE ){
		if( verbose ){
			sprintf(ERROR_STRING,
			"Trying to open movie file %s",s);
			advise(ERROR_STRING);
		}

#ifdef DEBUG
if( debug & mvi_debug ) advise("calling movie open_func");
#endif /* DEBUG */

/* use pathname here... */
		(*the_mmp->open_func)(QSP_ARG  s);		/* device specific */
		mvip = MVI_OF(s);
	}
	if( mvip == NO_MOVIE ){
		sprintf(ERROR_STRING,"No movie \"%s\"",s);
		WARN(ERROR_STRING);
		return(mvip);
	}

	if( IS_ASSEMBLING(mvip) ){
		WARN("can't play a movie that is being assembled");
		return(NO_MOVIE);
	}

	mvip->mvi_time = time(NULL);

	return(mvip);
}

static COMMAND_FUNC( do_open_mvi )
{
	const char *s;
	Movie *mvip;

	s=NAMEOF("movie filename");
	mvip = MVI_OF(s);
	if( mvip != NO_MOVIE ){
		sprintf(ERROR_STRING,"Movie %s was already open!?",s);
		advise(ERROR_STRING);
		mvip->mvi_time = time(NULL);
		return;
	}
	mvip = OPEN_IF(s);
}

static COMMAND_FUNC( do_openif_mvi )
{
	const char *s;
	Movie *mvip;

	s=NAMEOF("movie filename");
	mvip = OPEN_IF(s);
}

#define MOVIENAME_PMPT	"movie name"

static const char *get_movie_name(SINGLE_QSP_ARG_DECL)
{
	if( intractive(SINGLE_QSP_ARG) ){
		List *lp;
		lp = movie_list(SINGLE_QSP_ARG);
		if( lp != NO_LIST )
			init_hist_from_item_list(QSP_ARG  MOVIENAME_PMPT,lp);
	}

	return( NAMEOF(MOVIENAME_PMPT) );
}

static COMMAND_FUNC( do_play_movie )
{
	const char *s;
	Member_Info *mip;

	s = get_playable_name(SINGLE_QSP_ARG);

	if( !movie_ok() ) return;

	mip = get_member_info(QSP_ARG  playable_icp,s);

	/*
	 * On some implementations (cosmo, sirius), the movies are
	 * regular disk files, and therefore are not necessarily in
	 * our database at this point.  When that is the case, mip
	 * will have a value of NO_MEMBER_INFO, as returned by
	 * get_member_info().  Therefore it is legal to pass
	 * a name to play_movie even if there is not movie object
	 * already loaded.  On the other hand, in the RTV implementation,
	 * all of the playable movies are already in the database,
	 * so in that system play_movie() must check for a bogus name!
	 */

	if( mip == NO_MEMBER_INFO || mip->mi_itp == mvi_itp )
		play_movie(QSP_ARG  s);
	else
		show_sequence(QSP_ARG  s);
}

static void play_movie(QSP_ARG_DECL  const char *s)
{
	Movie *mvip;

	mvip = OPEN_IF(s);
	if( mvip == NO_MOVIE ) return;

	if( ! is_ready_to_play(QSP_ARG  mvip) ) return;


#ifdef DEBUG
if( debug & mvi_debug ) advise("calling movie play_func");
#endif /* DEBUG */

	(*the_mmp->play_func)(QSP_ARG  mvip);			/* device specific */

#ifdef DEBUG
if( debug & mvi_debug ) advise("calling movie wait_func");
#endif /* DEBUG */

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


#ifdef DEBUG
if( debug & mvi_debug ) advise("calling movie setup_play_func");
#endif /* DEBUG */

	if( (*the_mmp->setup_play_func)(mvip) < 0 )	/* device specific */
		return(0);
	
	/* remember the time this movie was played */
	mvip->mvi_time = time((time_t *)NULL);

	return(1);
}

static COMMAND_FUNC( do_shuttle )
{
	Movie *mvip;
	incr_t frame;

	mvip=PICK_MVI("");
	frame=HOW_MANY("frame index");

	if( mvip == NO_MOVIE ) return;

	if( frame < 0 || frame >= (incr_t) mvip->mvi_nframes ){
		sprintf(ERROR_STRING,"Frame index %d out of range for movie %s",
			frame,mvip->mvi_name);
		WARN(ERROR_STRING);
		return;
	}

	if( !movie_ok() ) return;

	if( ! is_ready_to_play(QSP_ARG  mvip) ) return;

	(*the_mmp->shuttle_func)(QSP_ARG  mvip,(index_t)frame);
}

static int obj_prec_ok(QSP_ARG_DECL  Data_Obj *dp)
{
	if( dp->dt_prec != PREC_BY && dp->dt_prec != PREC_UBY ){
		WARN("movie frame/field precision should be byte or u_byte");
		return(0);
	}
	if( dp->dt_comps != 4 ){
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

	dp = PICK_OBJ("");
	moviename = get_movie_name(SINGLE_QSP_ARG);
	frame = HOW_MANY("frame index");

	mvip = OPEN_IF(moviename);

	if( mvip == NO_MOVIE || dp == NO_OBJ ) return;

	if( frame < 0 || frame >= (incr_t)mvip->mvi_nframes ){
		sprintf(ERROR_STRING,
			"frame index %d out of range for movie %s",
			frame,mvip->mvi_name);
		WARN(ERROR_STRING);
		return;
	}
	if( dp->dt_rows != mvip->mvi_height ||
		dp->dt_cols != mvip->mvi_width ){
		sprintf(ERROR_STRING,
			"frame size mismatch between image %s and movie %s",
			dp->dt_name,mvip->mvi_name);
		WARN(ERROR_STRING);
		sprintf(ERROR_STRING,"image %s is %d x %d, movie %s is %d x %d",
			dp->dt_name,dp->dt_rows,dp->dt_cols,
			mvip->mvi_name,mvip->mvi_height,mvip->mvi_width);
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

	dp = PICK_OBJ("");
	moviename = get_movie_name(SINGLE_QSP_ARG);
	frame = HOW_MANY("frame index");
	comp = HOW_MANY("component index");

	mvip = OPEN_IF(moviename);

	if( mvip == NO_MOVIE || dp == NO_OBJ ) return;

	if( frame < 0 || frame >= (incr_t) mvip->mvi_nframes ){
		sprintf(ERROR_STRING,
			"frame index %d out of range for movie %s",
			frame,mvip->mvi_name);
		WARN(ERROR_STRING);
		return;
	}
	if( dp->dt_rows != mvip->mvi_height ||
		dp->dt_cols != mvip->mvi_width ){
		sprintf(ERROR_STRING,
			"frame size mismatch between image %s and movie %s",
			dp->dt_name,mvip->mvi_name);
		WARN(ERROR_STRING);
		sprintf(ERROR_STRING,"image %s is %d x %d, movie %s is %d x %d",
			dp->dt_name,dp->dt_rows,dp->dt_cols,
			mvip->mvi_name,mvip->mvi_height,mvip->mvi_width);
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

	dp = PICK_OBJ("");
	moviename = get_movie_name(SINGLE_QSP_ARG);
	field = HOW_MANY("starting field index");

	mvip = OPEN_IF(moviename);

	if( mvip == NO_MOVIE || dp == NO_OBJ ) return;

	if( field < 0 || field >= (incr_t)mvip->mvi_nframes*2 ){
		sprintf(ERROR_STRING,
			"field index %d out of range for movie %s",
			field,mvip->mvi_name);
		WARN(ERROR_STRING);
		return;
	}
	if( dp->dt_rows != mvip->mvi_height/2 ||
		dp->dt_cols != mvip->mvi_width ){
		sprintf(ERROR_STRING,
			"field size mismatch between image %s and movie %s",
			dp->dt_name,mvip->mvi_name);
		WARN(ERROR_STRING);
		sprintf(ERROR_STRING,
		"image %s is %d x %d, movie %s field size is %d x %d",
			dp->dt_name,dp->dt_rows,dp->dt_cols,
			mvip->mvi_name,mvip->mvi_height/2,mvip->mvi_width);
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

	dp = PICK_OBJ("");
	moviename = get_movie_name(SINGLE_QSP_ARG);
	field = HOW_MANY("starting field index");
	comp = HOW_MANY("component index");

	mvip = OPEN_IF(moviename);

	if( mvip == NO_MOVIE || dp == NO_OBJ ) return;

	if( field < 0 || field >= (incr_t)mvip->mvi_nframes*2 ){
		sprintf(ERROR_STRING,
			"field index %d out of range for movie %s",
			field,mvip->mvi_name);
		WARN(ERROR_STRING);
		return;
	}
	if( dp->dt_rows != mvip->mvi_height/2 ||
		dp->dt_cols != mvip->mvi_width ){
		sprintf(ERROR_STRING,
			"field size mismatch between image %s and movie %s",
			dp->dt_name,mvip->mvi_name);
		WARN(ERROR_STRING);
		sprintf(ERROR_STRING,
		"image %s is %d x %d, movie %s field size is %d x %d",
			dp->dt_name,dp->dt_rows,dp->dt_cols,
			mvip->mvi_name,mvip->mvi_height/2,mvip->mvi_width);
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

	mvip = PICK_MVI("");
	if( mvip == NO_MOVIE ) return;

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
		np=lp->l_head;
		while(np!=NO_NODE){
			seqp=(Seq *)np->n_data;
			/* BUG? does this delete sequences which depend on seqp? */
			delseq(QSP_ARG  seqp);
			np=np->n_next;
		}
		dellist(lp);
	}

	if( the_mmp != NO_MOVIE_MODULE )
		(*the_mmp->close_func)(QSP_ARG  mvip);		/* device specific */

	delete_movie(QSP_ARG  mvip);
}

/* delete a movie from the database */

void delete_movie(QSP_ARG_DECL  Movie *mvip)
{
	if( mvip == NO_MOVIE ) return;

	del_mvi(QSP_ARG  mvip->mvi_name);	/* remove from item database */
	rls_str((char *)mvip->mvi_name);
}

static COMMAND_FUNC( movie_seq_menu )
{
	seq_menu(SINGLE_QSP_ARG);
}

static COMMAND_FUNC( do_set_nrefresh )
{
	int n;

	n=HOW_MANY("refresh count");
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

static COMMAND_FUNC( do_list_mvis ){ list_mvis(SINGLE_QSP_ARG); }

static Command play_ctbl[]={
{ "directory",	do_set_mvidir,	"set movie directory"			},
{ "open",	do_open_mvi,	"open a movie file"			},
{ "insure",	do_openif_mvi,	"open a (possibly opened) movie file"	},
{ "play",	do_play_movie,	"play a movie to external video device"	},
{ "shuttle",	do_shuttle,	"play a movie frame to external video"	},
{ "close",	do_del_mvi,	"close an open movie file"		},
{ "refresh",	do_set_nrefresh,"set # of times to repeat fields"	},
{ "sequence",	movie_seq_menu,	"sequence submenu"			},
{ "getframe",	do_getframe,	"extract a single movie frame"		},
{ "getfield",	do_getfields,	"extract movie field(s)"		},
{ "getfrmcomp",	do_getframec,	"extract single component of a frame"	},
{ "getfldcomp",	do_getfieldc,	"extract component of a field"		},
{ "list",	do_list_mvis,	"list movies"				},
{ "info",	do_movie_info,	"print info about a movie"		},
{ "quit",	popcmd,		"exit submenu"				},
{ NULL,		NULL,		NULL					}
};

static COMMAND_FUNC( playmenu )
{
	PUSHCMD(play_ctbl,"playback");
}

#define MAX_MOVIE_FIELDS	0x40000000	/* a dummy value */

static COMMAND_FUNC( do_start_movie )
{
	const char *s;
	Movie *mvip;

	s=NAMEOF("name for movie");

	mvip = create_writable_movie(QSP_ARG  s,MAX_MOVIE_FIELDS);
	if( mvip == NO_MOVIE ) return;
	mvip->mvi_flags |= MVI_ASSEMBLING;
}

static COMMAND_FUNC( do_add_frame )
{
	Data_Obj *dp;
	Movie *mvip;

	mvip = PICK_MVI("");
	dp = PICK_OBJ("");

	if( mvip == NO_MOVIE || dp == NO_OBJ ) return;

	if( !movie_ok() ) return;

	(*the_mmp->add_func)(QSP_ARG  mvip,dp);		/* device specific */
}

static COMMAND_FUNC( do_end_movie )
{
	Movie *mvip;

	mvip = PICK_MVI("");
	if( mvip == NO_MOVIE ) return;

	if( ! IS_ASSEMBLING(mvip) ){
		sprintf(ERROR_STRING,"Movie \"%s\" is not being assembled",mvip->mvi_name);
		WARN(ERROR_STRING);
		return;
	}

	if( !movie_ok() ) return;

	(*the_mmp->end_func)(QSP_ARG  mvip);		/* device specific */
}

static COMMAND_FUNC( do_rec_movie )
{
	const char *s;
	u_long n;
	Movie *mvip;

	s=NAMEOF("name for movie");
	n=HOW_MANY("number of fields");

	if( !movie_ok() ) return;
#ifdef DEBUG
if( debug & mvi_debug ) advise("creating writable movie");
#endif /* DEBUG */

	mvip = create_writable_movie(QSP_ARG  s,n);
	if( mvip == NO_MOVIE ) return;
	mvip->mvi_flags |= MVI_RECORDING;


#ifdef DEBUG
if( debug & mvi_debug ) advise("calling movie record_func");
#endif /* DEBUG */

	(*the_mmp->record_func)(QSP_ARG  n,mvip);		/* device specific */

	/* Not sure which of these comments apply to the present code !? */
	/* need something else here? */
	/* NOT correct for cosmo (6.2) ... */

#ifdef DEBUG
if( debug & mvi_debug ) advise("calling movie end_func!?");
#endif /* DEBUG */

	(*the_mmp->end_func)(QSP_ARG  mvip);		/* device specific */
}

static COMMAND_FUNC( do_mon_input )
{
	if( !movie_ok() ) return;


#ifdef DEBUG
if( debug & mvi_debug ) advise("calling movie monitor_func");
#endif /* DEBUG */

	(*the_mmp->monitor_func)(SINGLE_QSP_ARG);		/* device specific */
}

static Command ass_ctbl[]={
{ "directory",	do_set_mvidir,	"set movie directory"			},
{ "start",	do_start_movie,	"assemble movie from still frames"	},
{ "add",	do_add_frame,	"add a still frame to a movie"		},
{ "finish",	do_end_movie,	"close movie file"			},
{ "record",	do_rec_movie,	"record a movie"			},
{ "monitor",	do_mon_input,	"monitor input video"			},
{ "list",	do_list_mvis,	"list movies"				},
{ "info",	do_movie_info,	"print info about a movie"		},
{ "quit",	popcmd,		"exit submenu"				},
{ NULL,		NULL,		NULL					}
};

static COMMAND_FUNC( ass_menu )
{
	PUSHCMD(ass_ctbl,"assemble");
}

static COMMAND_FUNC( dev_menu )
{
	if( !movie_ok() ) return;


#ifdef DEBUG
if( debug & mvi_debug ) advise("calling movie device menu");
#endif /* DEBUG */

	(*the_mmp->menu_func)(SINGLE_QSP_ARG);
}

static COMMAND_FUNC( report_mm )
{
	if( the_mmp == NO_MOVIE_MODULE )
		WARN("No movie module loaded");
	else {
		sprintf(msg_str,"Current movie module:  %s",the_mmp->mm_name);
		prt_msg(msg_str);
	}
}

Command mvi_ctbl[]={
{ "playback",	playmenu,	"movie playback submenu"		},
{ "control",	dev_menu,	"device control submenu",		},
{ "assemble",	ass_menu,	"movie construction submenu"		},
{ "list",	do_list_mvis,	"list movies"				},
{ "tell",	report_mm,	"report current movie module"		},
{ "delete",	do_del_mvi,	"delete movie"				},
{ "info",	do_movie_info,	"print info about a movie"		},
{ "quit",	popcmd,		"exit submenu"				},
{ NULL,		NULL,		NULL					}
};

/* support size functions */

static double get_mvi_size(Item *ip,int index)
{
	double d;
	Movie *mvip;

	mvip = (Movie *)ip;

	switch(index){
		case 0: d=mvip->mvi_depth; break;
		case 1: d=mvip->mvi_width; break;
		case 2: d=mvip->mvi_height; break;
		case 3: d=mvip->mvi_nframes; break;
		default:
			sprintf(DEFAULT_ERROR_STRING,"Unsupported movie size function, index %d",index);
			NWARN(DEFAULT_ERROR_STRING);
			d=1.0;
			break;
	}
	return(d);
}

static double get_mvi_il_flg(Item *ip)
{
	Movie *mvip;

	mvip = (Movie *)ip;
	if( mvip->mvi_shape.si_flags & DT_INTERLACED ) return(1.0);
	else return(0.0);
}


void add_playable(Item_Type * itp,void *vp)
{
	add_items_to_class(playable_icp,itp,vp,NULL);
}

#define PLAYABLE_PMPT	"movie or sequence"

static const char *get_playable_name(SINGLE_QSP_ARG_DECL)
{
	const char *s;

	if( intractive(SINGLE_QSP_ARG) )
		init_hist_from_class(QSP_ARG  PLAYABLE_PMPT,playable_icp);

	s=NAMEOF(PLAYABLE_PMPT);
	return(s);
}

static void ref_mvi(Movie *mvip)
{
	mvip->mvi_flags |= MVI_REFERENCED;
}

Movie *lookup_movie(QSP_ARG_DECL  const char *name)
{
	Movie *mvip;

	mvip=GET_MVI(name);
	if( mvip == NO_MOVIE ) return(mvip);

	mvip->mvi_time = time(NULL);
	return(mvip);
}

COMMAND_FUNC( moviemenu )
{

	if( playable_icp == NO_ITEM_CLASS ){
		if( mvi_itp == NO_ITEM_TYPE ) mvi_init(SINGLE_QSP_ARG);
		add_sizable(QSP_ARG  mvi_itp,&mvi_sf,NULL);

		playable_icp = new_item_class(QSP_ARG  "playable");
		add_playable(mvi_itp,NULL);

		if( mviseq_itp == NO_ITEM_TYPE )
			mviseq_init(SINGLE_QSP_ARG);
		add_playable(mviseq_itp,NULL);

		auto_version(QSP_ARG  "MVIMENU","VersionId_mvimenu");
	}


#ifdef HAVE_X11
	if( the_mmp == NO_MOVIE_MODULE ){
		if( verbose )
			advise("no movie module loaded, using X w/ fileio...");
		xmvi_init(SINGLE_QSP_ARG);
	}
#endif /* HAVE_X11 */

	/* initialize the sequencer module */

	if( the_mmp != NO_MOVIE_MODULE ){
		mvi_sm.init_func = (int (*)(void *)) the_mmp->setup_play_func;
		mvi_sm.get_func  = (void * (*)(const char *)) lookup_movie;
		mvi_sm.show_func = (void (*)(void *)) the_mmp->play_func;
		mvi_sm.rev_func  = (void (*)(void *)) the_mmp->reverse_func;
		mvi_sm.wait_func = the_mmp->wait_func;
		mvi_sm.ref_func  = (void (*)(void *))ref_mvi;

		load_seq_module(&mvi_sm);
	}

#ifdef DEBUG
	if( mvi_debug < 0 )
		mvi_debug = add_debug_module(QSP_ARG  "movie");
#endif /* DEBUG */

	PUSHCMD(mvi_ctbl,"movie");
}

void load_movie_module(QSP_ARG_DECL  Movie_Module *mmp)
{
	the_mmp = mmp;

#ifdef DEBUG
if( debug & mvi_debug ) advise("calling movie init_func");
#endif /* DEBUG */

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

