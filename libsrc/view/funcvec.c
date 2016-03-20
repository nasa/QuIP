#include "quip_config.h"

#include <stdio.h>
#include "quip_prot.h"
#include "cmaps.h"

static Lut_Module *curr_lmp=NO_LUT_MODULE;

static int check_lm(char *);

static int check_lm(str)
char *str;
{
	if( curr_lmp == NO_LUT_MODULE ){
		sprintf(ERROR_STRING,"%s:  no LUT module loaded",str);
		warn(ERROR_STRING);
		return(-1);
	}
	return(0);
}

void dispose_lb(lbp)
Lutbuf *lbp;
{
	if( check_lm("dispose_lb") < 0 ) return;
	(*curr_lmp->dispose_func)(lbp);
}

void init_lb_data(lbp)
Lutbuf *lbp;
{
	if( check_lm("init_lb_data") < 0 ) return;
	(*curr_lmp->init_func)(lbp);
}

void assign_lutbuf(Lutbuf *lbp,Data_Obj *cm_dp)
{
	if( check_lm("assign_lutbuf") < 0 ) return;
	(*curr_lmp->ass_func)(lbp,cm_dp);
}

void read_lutbuf(Data_Obj *cm_dp,Lutbuf *lbp)
{
	if( check_lm("read_lutbuf") < 0 ) return;
	(*curr_lmp->read_func)(cm_dp,lbp);
}

void show_lb_value(lbp,index)
Lutbuf *lbp; int index;
{
	if( check_lm("show_lb_value") < 0 ) return;
	(*curr_lmp->show_func)(lbp,index);
}

void dump_lut(Data_Obj *cm_dp)
{
	if( check_lm("dump_lut") < 0 ) return;
	(*curr_lmp->dump_func)(cm_dp);
}

void lb_extra_info(lbp)
Lutbuf *lbp;
{
	if( check_lm("lb_extra_info") < 0 ) return;
	(*curr_lmp->extra_func)(lbp);
}

void set_n_protect(n)
int n;
{
	if( check_lm("set_n_protect") < 0 ) return;
	(*curr_lmp->protect_func)(n);
}

void no_disp_func(lbp) Lutbuf *lbp;
{ warn("no dispose function loaded in current LUT module"); }

void no_init_func(lbp) Lutbuf *lbp;
{ warn("no init function loaded in current LUT module"); }

void no_ass_func(Lutbuf *lbp,Data_Obj *cm_dp)
{ warn("no assign function loaded in current LUT module"); }

void no_read_func(Data_Obj *cm_dp,Lutbuf *lbp)
{ warn("no read function loaded in current LUT module"); }

void no_show_func(lbp,index) Lutbuf *lbp; int index;
{ warn("no show function loaded in current LUT module"); }

void no_dump_func(Data_Obj *cm_dp)
{ warn("no dump function loaded in current LUT module"); }

void no_extra_func(lbp) Lutbuf *lbp;
{ warn("no extra_info function loaded in current LUT module"); }

void no_prot_func(n) int n;
{ warn("no protect function loaded in current LUT module"); }

Lut_Module *init_lut_module()
{
	Lut_Module *lmp;

	lmp=getbuf(sizeof(*lmp));
	lmp->dispose_func = no_disp_func;
	lmp->init_func = no_init_func;
	lmp->ass_func = no_ass_func;
	lmp->read_func = no_read_func;
	lmp->show_func = no_show_func;
	lmp->dump_func = no_dump_func;
	lmp->extra_func = no_extra_func;
	lmp->protect_func = no_prot_func;

	curr_lmp = lmp;
	return(lmp);
}

void load_lut_module(lmp)
Lut_Module *lmp;
{
	curr_lmp = lmp;
}

