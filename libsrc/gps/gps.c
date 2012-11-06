#include "quip_config.h"
char VersionId_gps_gps[] = QUIP_VERSION_STRING;

#include "gps_data.h"

GPS_UB_Datum gps_ub_tbl[]={
	{	"tsip_date",	0,	0	},
	{	"tsip_health",	1,	0	},
	{	"nmea_data",	2,	0	},
	{	"sat_nums",	3,	0	},
	{	"fix_mode",	4,	0	},
	{	"fix_time",	5,	0	},
	{	"hdop",		6,	0	},
	{	"tip_angle",	7,	0	},
	{	"roll_angle",	8,	0	},
	{	"latitude",	21,	0	},
	{	"new_data_flag",22,	0	},
	{	"longitude",	23,	0	},
	{	"alt_sign",	24,	0	},
	{	"altitude",	25,	0	},
	{	"up_vel",	26,	0	},
	{	"east_vel",	27,	0	},
	{	"north_vel",	28,	0	}
};

#define N_GPS_DATA	(sizeof(gps_ub_tbl)/sizeof(GPS_UB_Datum))

int ub_index_for(int frameno)
{
	unsigned int i;

	for(i=0;i<N_GPS_DATA;i++)
		if( gps_ub_tbl[i].gd_frm == frameno ) return(i);
	return(-1);
}



