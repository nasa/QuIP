#include "quip_config.h"

char VersionId_gps_gps_menu[] = QUIP_VERSION_STRING;

#include <stdio.h>

#include "query.h"
#include "version.h"		/* auto_version */

#include "gps_data.h"
#include "submenus.h"		/* prototype for gps_menu */

static void format_data(FILE *fp)
{
	int n_f, n_s, n_m, n_h;
	int b1,b2,b3,b4;
	int i=0,c;
	int ub_index;

	n_f = n_s = n_m = n_h = 0;	// quiet compiler

	while(1){
		if( (c=fgetc(fp)) == EOF ) return;
tc_marker_seen:
		while( c != 0xf7 ){
			sprintf(DEFAULT_ERROR_STRING,
				"missing time code marker, expected 0xf7, saw 0x%2x!?",c);
			NWARN(DEFAULT_ERROR_STRING);
			if( c == 0xff ) goto ub_marker_seen;
			if( (c=fgetc(fp)) == EOF ) return;
		}
		if( (c=fgetc(fp)) == EOF ) return;
		n_f = (10 * ((c&0xf0)>>4) ) + (c&0xf);	/* BCD */
		if( (c=fgetc(fp)) == EOF ) return;
		n_s = (10 * ((c&0xf0)>>4) ) + (c&0xf);	/* BCD */
		if( (c=fgetc(fp)) == EOF ) return;
		n_m = (10 * ((c&0xf0)>>4) ) + (c&0xf);	/* BCD */
		if( (c=fgetc(fp)) == EOF ) return;
		n_h = (10 * ((c&0xf0)>>4) ) + (c&0xf);	/* BCD */

		if( c == 0xff ){
			NWARN("UB marker seen 1 char early!?");
			goto ub_marker_seen;
		}

		if( (c=fgetc(fp)) == EOF ) return;
		while( c != 0xff ){
			sprintf(DEFAULT_ERROR_STRING,"missing UB marker, expected 0xff, saw 0x%2x!?",c);
			NWARN(DEFAULT_ERROR_STRING);
			if( c == 0xf7 ) goto tc_marker_seen;
			if( (c=fgetc(fp)) == EOF ) return;
		}
ub_marker_seen:
		if( (c=fgetc(fp)) == EOF ) return;
		b1 = c;
		if( (c=fgetc(fp)) == EOF ) return;
		b2 = c;
		if( (c=fgetc(fp)) == EOF ) return;
		b3 = c;
		if( (c=fgetc(fp)) == EOF ) return;
		b4 = c;

		sprintf(DEFAULT_ERROR_STRING,"%d\t%2d:%2d:%2d:%2d\t\t%2x %2x %2x %2x",
			i,n_h,n_m,n_s,n_f,b1,b2,b3,b4);
		prt_msg_frag(DEFAULT_ERROR_STRING);

		ub_index = ub_index_for(n_f);
		if( ub_index >= 0 ){
			u_long v;

			v = ( b1      & 0x000000ff)
			  | ((b2<<8)  & 0x0000ff00)
			  | ((b3<<16) & 0x00ff0000)
			  | ((b4<<24) & 0xff000000)
			  ;
			sprintf(DEFAULT_ERROR_STRING,"\t%s\t%ld",
				gps_ub_tbl[ub_index].gd_name,v);
			prt_msg(DEFAULT_ERROR_STRING);
		} else {
			prt_msg("");
		}

		i++;
	}
}

static COMMAND_FUNC( do_gps_fmt )
{
	FILE *fp;
	int c;

	fp = TRY_OPEN( NAMEOF("gps dump file"), "r" );
	if( fp == NULL ) return;

	/* seek to the beginning of a record */

	while( (c=fgetc(fp)) != 0xf7 )
		;
	ungetc(c,fp);

	format_data(fp);
	fclose(fp);
}

static void check_special(unsigned char c[], int nf )
{
	int i;
	GPS_UB_Datum *gdp;

	i=ub_index_for(nf);
	if( i < 0 ) return;	/* nothing special for this field */
	/* BUG should do this once and then use table lookup */
	gdp = &gps_ub_tbl[i];

	sprintf(msg_str,"\t%s",gdp->gd_name);
	prt_msg_frag(msg_str);
}

static COMMAND_FUNC( do_ascii_fmt )
{
	const char *fn;
	FILE *fp;
	int level;

	fn = NAMEOF("filename");
	fp=TRY_OPEN(fn,"r");
	if( !fp ) return;

	push_input_file(QSP_ARG  fn);
	redir(QSP_ARG  fp);	/* BUG need input_file() */
	level = tell_qlevel(SINGLE_QSP_ARG);

	while( tell_qlevel(SINGLE_QSP_ARG) >= level ){
		int n_f, n_s, n_m, n_h;

		/* assume we have 5 chars */
		unsigned char c[5];

		c[0] = HOW_MANY("data");
		c[1] = HOW_MANY("data");
		c[2] = HOW_MANY("data");
		c[3] = HOW_MANY("data");
		c[4] = HOW_MANY("data");

		n_f = 0;		// quiet compiler

		if( c[0] == 0xff ){
			sprintf(DEFAULT_ERROR_STRING,"\tUB\t%02x:%02x:%02x:%02x",c[1],c[2],c[3],c[4]);
			prt_msg_frag(DEFAULT_ERROR_STRING);
			check_special(c,n_f);
		} else if( c[0] == 0xf7 ){

			n_f = (10 * ((c[1]&0xf0)>>4) ) + (c[1]&0xf);	/* BCD */
			n_s = (10 * ((c[2]&0xf0)>>4) ) + (c[2]&0xf);	/* BCD */
			n_m = (10 * ((c[3]&0xf0)>>4) ) + (c[3]&0xf);	/* BCD */
			n_h = (10 * ((c[4]&0xf0)>>4) ) + (c[4]&0xf);	/* BCD */
show_time:
			sprintf(DEFAULT_ERROR_STRING,"\n%02d:%02d:%02d:%02d",n_h,n_m,n_s,n_f);
			prt_msg_frag(DEFAULT_ERROR_STRING);
		} else if( c[0] == 0xf5 || /* sometimes we get f5 when we expect f7 */
			   c[0] == 0xf6 ||
			   c[0] == 0xf4
				){
			n_f = n_s = n_m = n_h = 0;
			goto show_time;
		} else {
			sprintf(DEFAULT_ERROR_STRING,"bad marker char 0x%x",c[0]);
			NWARN(DEFAULT_ERROR_STRING);
		}
		/* lookahead here to make sure level is popped at EOF */
		lookahead(SINGLE_QSP_ARG);
	}
	/* is the file closed when popped? */
	prt_msg("");
}

static COMMAND_FUNC( do_redump )
{
	FILE *fp;
	const char *fn;
	int c;

	fn=NAMEOF("dump file");
	fp=TRY_OPEN(fn,"r");
	if( !fp ) return;

	while( (c=fgetc(fp)) != EOF ){
		if( c==0xff || c==0xf7 ){	/* marker byte? */
			prt_msg("");
		}
		sprintf(msg_str,"\t0x%x",c);
		prt_msg_frag(msg_str);
	}
	prt_msg("");
	fclose(fp);
}

static int get_bcd(int v)
{
	int d1,d2;

	d1 = v >> 4;
	d2 = v & 0xf;
	return( d1*10 + d2 );
}

static COMMAND_FUNC( do_traj )
{
	const char *fn;
	FILE *fp;
	int level;
	int expect_tc=1, expect_ub=1;	/* expect either the first time */

	fn = NAMEOF("filename");
	fp=TRY_OPEN(fn,"r");
	if( !fp ) return;

	push_input_file(QSP_ARG  fn);
	redir(QSP_ARG  fp);	/* BUG need input_file() */
	level = tell_qlevel(SINGLE_QSP_ARG);

	while( tell_qlevel(SINGLE_QSP_ARG) >= level ){
		int n_f, n_s, n_m, n_h;

		/* assume we have 5 chars */
		unsigned char c[5];

		c[0] = HOW_MANY("data");
		c[1] = HOW_MANY("data");
		c[2] = HOW_MANY("data");
		c[3] = HOW_MANY("data");
		c[4] = HOW_MANY("data");

		n_f = 0;				// quiet compiler
		n_s = 0;				// quiet compiler
		n_m = 0;				// quiet compiler
		n_h = 0;				// quiet compiler

		if( c[0] == 0xff ){
			int lat_deg, long_deg, alt1,alt2;
			float long_min,long_min_frac1,long_min_frac2;
			float lat_min,lat_min_frac1,lat_min_frac2;

			lat_min_frac1=0.0;		// quiet compiler
			lat_min_frac2=0.0;		// quiet compiler
			lat_min=0.0;			// quiet compiler
			long_min_frac1=0.0;		// quiet compiler
			long_min_frac2=0.0;		// quiet compiler
			long_min=0.0;			// quiet compiler
			alt1=0;				// quiet compiler
			alt2=0;				// quiet compiler
			lat_deg=0;			// quiet compiler
			long_deg=0;			// quiet compiler

			if( ! expect_ub ){
				NWARN("two successive user-byte packets!?");
			}
			if( n_f == 21 ){
				lat_deg = get_bcd(c[2]);
				lat_min = get_bcd(c[1]);
			} else if( n_f == 22 ){
				lat_min_frac1 = get_bcd(c[2]);
				lat_min_frac2 = get_bcd(c[1]);
			} else if( n_f == 23 ){
				long_deg = get_bcd(c[2]);
				long_min = get_bcd(c[1]);
			} else if( n_f == 24 ){
				long_min_frac1 = get_bcd(c[2]);
				long_min_frac2 = get_bcd(c[1]);
			} else if( n_f == 25 ){
				alt1 = get_bcd(c[2]);
				alt2 = get_bcd(c[1]);
			} else if( n_f == 26 ){	/* now it's time to output! */
				sprintf(msg_str,"%02d:%02d:%02d",n_h,n_m,n_s);
				prt_msg_frag(msg_str);

				lat_min += lat_min_frac1/100;
				lat_min += lat_min_frac2/10000;

				long_min += long_min_frac1/100;
				long_min += long_min_frac2/10000;

				alt1 = alt1*100 + alt2;

				sprintf(msg_str,"\tlat %2d deg %2.4f min   long %2d deg %2.4f min   alt %5d",
						lat_deg,lat_min,long_deg,long_min,alt1);
				prt_msg(msg_str);

			}
#ifdef CAUTIOUS
			else {
				lat_min=0;	// quiet compiler
				sprintf(DEFAULT_ERROR_STRING,"CAUTIOUS: do_traj:  unexpected n_f value (%d)",n_f);
				NWARN(DEFAULT_ERROR_STRING);
			}
#endif /* CAUTIOUS */
			expect_tc = 1;
			expect_ub = 0;
		} else if( c[0] == 0xf7 ){
			if( ! expect_tc ){
				NWARN("two successive time code packets!?");
			}
			n_f = (10 * ((c[1]&0xf0)>>4) ) + (c[1]&0xf);	/* BCD */
			n_s = (10 * ((c[2]&0xf0)>>4) ) + (c[2]&0xf);	/* BCD */
			n_m = (10 * ((c[3]&0xf0)>>4) ) + (c[3]&0xf);	/* BCD */
			n_h = (10 * ((c[4]&0xf0)>>4) ) + (c[4]&0xf);	/* BCD */
			expect_tc = 0;
			expect_ub = 1;
		} else {
			sprintf(DEFAULT_ERROR_STRING,"bad marker char 0x%x",c[0]);
			NWARN(DEFAULT_ERROR_STRING);
			if( expect_tc ){
				n_f = -1;
				expect_tc = 0;
				expect_ub = 1;
			} else if( expect_ub ){
				expect_tc = 1;
				expect_ub = 0;
			}
		}
		/* lookahead here to make sure level is popped at EOF */
		lookahead(SINGLE_QSP_ARG);
	}
	/* is the file closed when popped? */
	prt_msg("");
}

Command gps_ctbl[]={
{ "format",	do_gps_fmt,	"format raw gps dump file"	},
{ "fmt2",	do_ascii_fmt,	"format hex dump of gps stream"	},
{ "trajectory",	do_traj,	"format gps trajectory from hex dump of gps stream"	},
{ "redump",	do_redump,	"print a dump file out as ascii"	},
{ "quit",	popcmd,		"exit submenu"			},
{ NULL_COMMAND							}
};

COMMAND_FUNC( gps_menu )
{
	auto_version(QSP_ARG  "GPS","VersionId_gps");
	PUSHCMD(gps_ctbl,"gps");
}

