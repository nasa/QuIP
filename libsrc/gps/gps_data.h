
typedef struct gps_ub_datum {
	const char *	gd_name;
	int		gd_frm;
	int		gd_val;
} GPS_UB_Datum;

extern GPS_UB_Datum gps_ub_tbl[];

/* prototypes */

/* gps.c */
extern int ub_index_for(int frame_number);

