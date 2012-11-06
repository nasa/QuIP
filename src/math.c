/* arithmetic drill program */

#include <sys/time.h>
#include <sys/timeb.h>
#include <time.h>
#include <signal.h>
#include <stdlib.h>
#include "query.h"
#include "rn.h"
#include "chewtext.h"
#include "debug.h"
#include "submenus.h"

#include "img_file.h"
#include <math.h>

#define min(a,b)	((a)<(b)?(a):(b))

#define MAX_DIFFICULTY	100	/* error percentage */

#define MAX_REACTION_TIME	5000	/* milliseconds */
#define RT_HIST_BIN_WIDTH	20	/* milliseconds */
#define N_RT_HIST_BINS		((MAX_REACTION_TIME/RT_HIST_BIN_WIDTH)+1)

static const char *student_name=NULL;
static FILE *drib_fp=NULL;

typedef struct rt_hist {
	int	rth_n;
	int	rth_bin[N_RT_HIST_BINS];
	double	rth_overflow_total;
	/* derived measure */
	float	rth_avg;	/* centroid */
	float	rth_sd;		/* std deviation */
	time_t	rth_min;
	time_t	rth_max;
} RT_Hist;

typedef struct history {
	int		n_presentations;
	int		n_correct;
	/* derived measure */
	float		hit_rate;
	RT_Hist		hgram, crct_hg, incrct_hg;
} History;

typedef enum {
	OP_PLUS,
	OP_MINUS,
	OP_TIMES,
	OP_DIVIDE
} Operator;

#define N_CHOICES	4

typedef struct problem {
	char *		pb_name;
	Operator	pb_op;
	int		pb_operand[2];
	int		pb_answer;
	History		pb_hist;
	int		pb_difficulty;
	int		pb_ans_index;
	int		pb_choice[N_CHOICES];
} Problem;

#define NO_PROBLEM	((Problem *)NULL)

static Problem *curr_pp=NO_PROBLEM;
static int threshold_hit_rate=50;

static struct timeb start_time, end_time;

ITEM_INTERFACE_DECLARATIONS(Problem,problem)

#define N_OPERATORS	4
static const char *operator_list[N_OPERATORS]={"+","-","x","/"};
static const char *operator_name_list[N_OPERATORS]={"plus","minus","times","divide"};

static Problem *last_presented=NO_PROBLEM;

void init_histogram(RT_Hist *hgp)
{
	int i;

	for(i=0;i<N_RT_HIST_BINS;i++)
		hgp->rth_bin[i] = 0;
	hgp->rth_n = 0;
	hgp->rth_overflow_total = 0.0;
}

void init_history(History *hp)
{
	hp->n_presentations = 0;
	hp->n_correct = 0;
	init_histogram(&hp->hgram);
	init_histogram(&hp->crct_hg);
	init_histogram(&hp->incrct_hg);
}

static void create_problem(int n1,int op, int n2)
{
	Problem *pp;
	char str[128];

	sprintf(str,"%d%s%d",n1,operator_list[op],n2);
	pp = new_problem(DEFAULT_QSP_ARG  str);
	if( pp == NO_PROBLEM ) return;

//sprintf(DEFAULT_ERROR_STRING,"Creating problem %s, problem_itp = 0x%lx",str,(u_long)problem_itp);
//advise(DEFAULT_ERROR_STRING);

	pp->pb_op = op;

	switch( pp->pb_op ){
		case OP_PLUS:
			pp->pb_answer = n1 + n2 ;
			break;
		case OP_MINUS:
			pp->pb_answer = n1 - n2 ;
			break;
		case OP_TIMES:
			pp->pb_answer = n1 * n2 ;
			break;
		case OP_DIVIDE:
			pp->pb_answer = n1 / n2 ;
			break;
	}

	pp->pb_operand[0] = n1;
	pp->pb_operand[1] = n2;
	pp->pb_difficulty = 40 + rn(30)+min(n1,n2);	/* represents the miss rate */

	init_history(&pp->pb_hist);

	/* We don't set the node priorities here, because
	 * the nodes don't stick around...
	 */
}

COMMAND_FUNC( do_create_problem )
{
	int i,n1,n2;

	n1 = HOW_MANY("first operand");
	i=WHICH_ONE("operator",N_OPERATORS,operator_list);
	n2 = HOW_MANY("second operand");

	if( i < 0 ) return;

	create_problem(n1,i,n2);
}

void problem_info(SINGLE_QSP_ARG_DECL)
{
	Problem *pp;

	pp = PICK_PROBLEM("");
	if( pp == NO_PROBLEM ) return;

	sprintf(msg_str,"%s:",pp->pb_name);
	prt_msg(msg_str);

	sprintf(msg_str,"\t%d presentations, %d correct",
		pp->pb_hist.n_presentations,pp->pb_hist.n_correct);
	prt_msg(msg_str);

	sprintf(msg_str,"\tproportion correct:  %g",
		pp->pb_hist.hit_rate);
	prt_msg(msg_str);

	sprintf(msg_str,"\tall responses:");
	prt_msg(msg_str);

	sprintf(msg_str,"\t\tavg. RT %g  (min %ld, max %ld)",
		pp->pb_hist.hgram.rth_avg,
		pp->pb_hist.hgram.rth_min,
		pp->pb_hist.hgram.rth_max);
	prt_msg(msg_str);

	sprintf(msg_str,"\tcorrect responses:");
	prt_msg(msg_str);

	sprintf(msg_str,"\t\tavg. RT %g  (min %ld, max %ld)",
		pp->pb_hist.crct_hg.rth_avg,
		pp->pb_hist.crct_hg.rth_min,
		pp->pb_hist.crct_hg.rth_max);
	prt_msg(msg_str);

	sprintf(msg_str,"\tincorrect responses:");
	prt_msg(msg_str);

	sprintf(msg_str,"\t\tavg. RT %g  (min %ld, max %ld)",
		pp->pb_hist.incrct_hg.rth_avg,
		pp->pb_hist.incrct_hg.rth_min,
		pp->pb_hist.incrct_hg.rth_max);
	prt_msg(msg_str);

}

void log_rt(RT_Hist *rthp, time_t rt)
{
	int i,bin_index;
	double total, center, sos, dev;
	int n;

	if( rthp->rth_n == 0 ) {
		rthp->rth_min =
		rthp->rth_max = rt;
	} else if( rt < rthp->rth_min ){
		rthp->rth_min = rt;
	} else if( rt > rthp->rth_max ){
		rthp->rth_max = rt;
	}
		
	rthp->rth_n ++;
	bin_index = rt / RT_HIST_BIN_WIDTH;

	/* the last bin contains the responses greater than MAX_RESPONSE_TIME,
	 * so we keep their average separately instead of using the bin
	 * center when we compute the average.
	 */

	if( bin_index >= (N_RT_HIST_BINS-1) ){
		bin_index = N_RT_HIST_BINS-1;
		rthp->rth_overflow_total += rt;
	}
	rthp->rth_bin[bin_index] ++;

	/* now compute the average using the bin centers */
	total = 0;
	for(i=0;i<(N_RT_HIST_BINS-1);i++){
		center = RT_HIST_BIN_WIDTH * (i+0.5);
		total += center * rthp->rth_bin[i];
	}
	total += rthp->rth_overflow_total;

	rthp->rth_avg = total / rthp->rth_n;

	sos=0;
	n=0;
	for(i=0;i<(N_RT_HIST_BINS-1);i++){
		center = RT_HIST_BIN_WIDTH * (i+0.5);
		dev = center - rthp->rth_avg;
		sos += dev * dev * rthp->rth_bin[i];
		n += rthp->rth_bin[i];
	}
	rthp->rth_sd = sqrt( sos / n );
} /* end log_rt */

void log_incorrect_response(History *hp, time_t rt)
{
	hp->n_presentations++;
	hp->hit_rate = ((float) hp->n_correct )/((float) hp->n_presentations );
	log_rt(&hp->hgram,rt);
	log_rt(&hp->incrct_hg,rt);
}

void log_correct_response(History *hp, time_t rt)
{
	hp->n_presentations++;
	hp->n_correct++;
	hp->hit_rate = ((float) hp->n_correct )/((float) hp->n_presentations );
	log_rt(&hp->hgram,rt);
	log_rt(&hp->crct_hg,rt);
}

static void dribble_data(Problem *pp, int ans_index, u_long rt )
{
	time_t t;

	if( drib_fp == NULL )
		drib_fp = try_hard(DEFAULT_QSP_ARG  "test.log","a+");

	time(&t);
	fprintf(drib_fp,"%s\t%d\t( %d %d %d %d )\t%ld\t%s",pp->pb_name,
		pp->pb_choice[ans_index],
		pp->pb_choice[0],
		pp->pb_choice[1],
		pp->pb_choice[2],
		pp->pb_choice[3],
		rt,ctime(&t));
	fflush(drib_fp);
}

void present_problem(QSP_ARG_DECL  Problem *pp)
{
	char pmpt[64];
	time_t rt;
	int n;

	sprintf(pmpt,"%d %s %d",
		pp->pb_operand[0],
		operator_list[pp->pb_op],
		pp->pb_operand[1]);

	push_input_file(DEFAULT_QSP_ARG  "-");
	redir( DEFAULT_QSP_ARG  tfile(SGL_DEFAULT_QSP_ARG) );

	ftime(&start_time);

	n = HOW_MANY(pmpt);

	ftime(&end_time);

	popfile(SINGLE_QSP_ARG);

	rt = 1000 * (end_time.time - start_time.time);
	rt += end_time.millitm - start_time.millitm;

	dribble_data(pp,n,rt);

	if( n == pp->pb_answer ) log_correct_response(&pp->pb_hist,rt);
	else log_incorrect_response(&pp->pb_hist,rt);
}

static int get_distractor(Problem *pp)
{
	int distractor=0;

	if( pp->pb_answer < 10 ){
		distractor += rn(N_CHOICES+1) - N_CHOICES/2;
		while( distractor < 0 ) distractor+=10;
		distractor = distractor % 10;
		return(distractor);
	}
	if( pp->pb_answer < 100 ){
		if( rn(2) == 0 ){	/* fiddle the tens digit */
			distractor = pp->pb_answer + (rn(1)-2)*10;
			while( distractor < 0 ) distractor+=10;
			return(distractor);
		} else {
			distractor = pp->pb_answer + rn(N_CHOICES+1) - N_CHOICES/2;
			while( distractor < 0 ) distractor+=10;
			return(distractor);
		}
	}
	/* BUG need to handle results > 100 */
	else {
NWARN("not creating distractor for result > 100");
		return(pp->pb_answer);
	}
}

void my_alarm(int n)
{
	advise("alarm!");
	chew_text(DEFAULT_QSP_ARG  "Alarm");
}

void gpresent_problem(QSP_ARG_DECL  Problem *pp)
{
	char str[LLEN];
	int n;
	int ch[N_CHOICES];
	u_long pbuf[N_CHOICES];
	int i,j;
	struct itimerval value;
	RT_Hist *rthp;
	float target_ms;

	/* First get the choices
	 * We use pb_difficulty to do this.  The initial value
	 * of the difficulty is the min of the two operands.
	 */
	ch[0] = pp->pb_answer;

	/* Now make 3 distractors.
	 * The higher the level, the closer the distractors are
	 * to the answer.
	 */
	for(i=1;i<N_CHOICES;i++){
		/* We won't always want to do this mod 10... */
		n = get_distractor(pp);
		/* now make sure that this n hasn't already been used */
		for(j=0;j<i;j++)
			if( n == ch[j] ){
				i--;
				n = (-1);
			}
		if( n >= 0 ) ch[i] = n;
	}

	/* now permute the order */
	permute(QSP_ARG  pbuf,N_CHOICES);
	/* find the answer index */
	for(i=0;i<N_CHOICES;i++){
		pp->pb_choice[i] = ch[ pbuf[i] ];
		if( pbuf[i]==0 )
			pp->pb_ans_index = i;
	}

	sprintf(str,"Present_Problem %d %s %d %d %d %d %d %d",
		pp->pb_operand[0],
		operator_name_list[pp->pb_op],
		pp->pb_operand[1],
		pp->pb_ans_index,
		ch[ pbuf[0] ],
		ch[ pbuf[1] ],
		ch[ pbuf[2] ],
		ch[ pbuf[3] ] );

	curr_pp = pp;
	ftime(&start_time);
	/* Set an alarm here... */
	rthp = &pp->pb_hist.crct_hg;
	if( rthp->rth_n <= 0 )
		target_ms = 2000;
	else
		target_ms = rthp->rth_avg + rthp->rth_sd;

	value.it_value.tv_sec = floor(target_ms / 1000);
	value.it_value.tv_usec = floor(1000 * (target_ms - (value.it_value.tv_sec*1000)));
	/* stop after countdown if interval is 0 */
	value.it_interval.tv_sec = 0;
	value.it_interval.tv_usec = 0;
	/*
	signal(SIGALRM,my_alarm);
	if( setitimer(ITIMER_REAL,&value,&ovalue) < 0 ){
		perror("setitimer");
	}
	*/

	chew_text(QSP_ARG  str);
}

static void log_response(Problem *pp, int response, u_long rt )
{
	int num;

	if( response == pp->pb_answer ){
		log_correct_response(&pp->pb_hist,rt);
		num=0;
		last_presented = pp;
	}
	else {
		log_incorrect_response(&pp->pb_hist,rt);
		num=25;
	}
	/* get the difficulty from the hit rate
	 * Use a fading-memory filter.
	 * 1/4 of the instantaeous miss rate + 3/4 of the previous value
	 */
/*
sprintf(msg_str,"problem %s, old diff %d, ",pp->pb_name,
pp->pb_difficulty);
prt_msg_frag(msg_str);
*/

	pp->pb_difficulty = num + 0.75*pp->pb_difficulty;
	
/*
sprintf(msg_str,"new diff %d, ", pp->pb_difficulty);
prt_msg(msg_str);
*/

}

static COMMAND_FUNC( do_finish )
{
	time_t rt;
	int i;

	ftime(&end_time);

	i = HOW_MANY("answer index");

	if( i < 0 || i >= N_CHOICES ){
		WARN("bad answer index");
		return;
	}

	rt = 1000 * (end_time.time - start_time.time);
	rt += end_time.millitm - start_time.millitm;

	dribble_data(curr_pp,i,rt);

	log_response(curr_pp,curr_pp->pb_choice[i],rt);
}

COMMAND_FUNC( do_gpresent_problem )
{
	Problem *pp;

	pp=PICK_PROBLEM("");
	if( pp == NO_PROBLEM ) return;

	gpresent_problem(QSP_ARG  pp);
}


COMMAND_FUNC( do_present_problem )
{
	Problem *pp;

	pp=PICK_PROBLEM("");
	if( pp == NO_PROBLEM ) return;

	present_problem(QSP_ARG  pp);
}

static COMMAND_FUNC( do_select_prob )
{
	/* Problem selection is one of the more interesting aspects
	 * of this program.  We don't want to bother presenting problems
	 * that have been mastered, but we don't want to give things
	 * which are too hard.
	 * We begin with a rough ordering of difficulty.  We use
	 * the value of the smaller operand.  This is fine for single
	 * digit problems, but does not generalize well to multidigit
	 * problems, where the presence or absence of a carry influences
	 * difficulty.
	 *
	 * Our preconceived notion of difficulty will be adaptively modified
	 * based on the student's performance.  Incorrect responses will
	 * cause the difficulty to increase, while correct responses will
	 * cause it to decrease.  We would like to have a fading memory
	 * so that if something is gotten correctly 4 times in a row
	 * then its difficulty goes to zero, or something low that
	 * tells us to move on.
	 */
	
	List *lp;
	Node *np;
	Problem *pp;
	char *prob_name;
	int i;

#ifdef CAUTIOUS
	if( problem_itp == NO_ITEM_TYPE )
		ERROR1("CAUTIOUS:  do_select_prob: null problem item type ptr");
#endif /* CAUTIOUS */

	lp = item_list(QSP_ARG  problem_itp);
	if( lp == NO_LIST ){
		prob_name = "none";
		goto done;
	}

	np=lp->l_head;
	while( np != NO_NODE ){
		pp = np->n_data;
		np->n_pri = 100 - pp->pb_difficulty;
		np = np->n_next;
	}

	/* p_sort places the highest priorities at the head of the list */

	p_sort(lp);	/* node priorities represent problem difficulty */

	np=lp->l_head;
	if( np == NO_NODE ){
		prob_name = "none";
		goto done;
	}

	while( np != NO_NODE && np->n_pri > threshold_hit_rate ) {
		if( verbose ){
			pp=np->n_data;
			sprintf(DEFAULT_ERROR_STRING, "select_problem:  %s  pri=%d",
				pp->pb_name,np->n_pri);
			advise(DEFAULT_ERROR_STRING);
		}
		np=np->n_next;
	}

	/* now randomize things a bit */
	i=rn(5);
	while( i-- && np!= NO_NODE )
		np=np->n_next;

	if( np==NO_NODE ) np=lp->l_tail;

//pp=np->n_data;
//sprintf(DEFAULT_ERROR_STRING,"select_problem:  %s  pri=%d",pp->pb_name,np->n_pri);
//advise(DEFAULT_ERROR_STRING);

	pp=np->n_data;

	/* don't present the same problem twice in a row */

	if( pp == last_presented ){
		if( np->n_next != NO_NODE )
			pp=np->n_next->n_data;
		else if( np->n_last != NO_NODE )
			pp=np->n_last->n_data;
	}

	prob_name = pp->pb_name;
done:
	assign_var(QSP_ARG  "next_problem",prob_name);
}

static COMMAND_FUNC( do_set_thresh )
{
	threshold_hit_rate = HOW_MANY("threshold hit rate");
}

static void init_from_file(FILE *fp)
{
	char probname[32];
	int response;
	int ch1,ch2,ch3,ch4;
	u_long rt;
	char day_of_week[8];
	char month[8];
	int day;
	char hhmmss[16];
	int yr;
	Problem *pp;

	if( fseek(fp,0,SEEK_SET) < 0 )
		NWARN("error rewinding dribble file");

	while( fscanf(fp,"%s %d ( %d %d %d %d ) %ld %s %s %d %s %d",
		probname,&response,&ch1,&ch2,&ch3,&ch4,&rt,
		day_of_week,month,&day,hhmmss,&yr) == 12 ){

		pp = problem_of(DEFAULT_QSP_ARG  probname);
		if( pp == NO_PROBLEM ){
			int n1,op,n2;
			char str[32];
			char *s,*t;

			s=probname;
			t=str;
			while( isdigit(*s) ) *t++ = *s++;
			*t=0;
			n1=atoi(str);
			if( *s == '+' ) op=OP_PLUS;
			else if( *s == '-' ) op=OP_MINUS;
			else if( *s == 'x' ) op=OP_TIMES;
			else if( *s == '/' ) op=OP_DIVIDE;
			else {
				sprintf(DEFAULT_ERROR_STRING,"bad op '%c'",*s);
				NWARN(DEFAULT_ERROR_STRING);
			}
			s++;
			t=str;
			while( isdigit(*s) ) *t++ = *s++;
			*t=0;
			n2=atoi(str);

			create_problem(n1,op,n2);
			pp = problem_of(DEFAULT_QSP_ARG  probname);
			if( pp == NO_PROBLEM ){
				sprintf(DEFAULT_ERROR_STRING,"unable to create problem %s",
					probname);
				NERROR1(DEFAULT_ERROR_STRING);
			}
		}
		/* now that we have the problem, log the response */
		log_response(pp,response,rt);
	}
}

static COMMAND_FUNC( set_student )
{
	const char *s;
	char logname[LLEN];

	s=NAMEOF("student name");
	if( student_name != NULL ) rls_str(student_name);

	student_name = savestr(s);

	if( drib_fp != NULL ){
		fclose(drib_fp);
		drib_fp=NULL;
	}
	/* now open the file and read in the old data */
	sprintf(logname,"%s.log",student_name);
	drib_fp = TRY_HARD(logname,"a+");
	init_from_file(drib_fp);
}

static COMMAND_FUNC( do_list_probs ) { list_problems(SINGLE_QSP_ARG); }
static COMMAND_FUNC( do_problem_info ) { problem_info(SINGLE_QSP_ARG); }


Command drill_ctbl[]={
{ "problem",	do_create_problem,	"create a new problem"		},
{ "present",	do_present_problem,	"present a problem"		},
{ "gpresent",	do_gpresent_problem,	"graphical presentation of a problem"		},
{ "finish",	do_finish,		"finish graphical presentation"	},
{ "select",	do_select_prob,		"select next problem"		},
{ "threshold",	do_set_thresh,		"set threshold hit rate"	},
{ "list",	do_list_probs,		"list all problems"		},
{ "info",	do_problem_info,	"report problem statistics"	},
{ "student",	set_student,		"set current student"		},
{ "quit",	popcmd,			"quit program"			},
{ NULL_COMMAND								}
};

static COMMAND_FUNC( drill_menu )
{
	PUSHCMD(drill_ctbl,"drill");
}

Command main_ctbl[]={
{ "drill",	drill_menu,		"math drill submenu"		},
{ "view",	viewmenu,		"image viewer submenu"		},
{ "data",	datamenu,		"data object submenu"		},
{ "fileio",	fiomenu,		"i/o file submenu"		},
{ "compute",	warmenu,		"vector processing submenu"	},
{ "genwin",	genwin_menu,	"viewer/panel submenu"			},
{ "quit",	popcmd,			"quit program"			},
{ NULL_COMMAND								}
};

int main(int ac, char **av)
{
	QSP_DECL

	INIT_QSP

	rcfile(QSP_ARG  av[0]);
	set_args(QSP_ARG  ac,av);

	PUSHCMD(main_ctbl,"drill_main");
	while(1) do_cmd(SINGLE_QSP_ARG);
}

