
#ifndef _MY_FB_H_
#define _MY_FB_H_

#ifdef HAVE_FB_DEV

#include "data_obj.h"
#include "parport.h"


#include <linux/fb.h>

typedef struct fb_info {
	char *				fbi_name;
	int				fbi_fd;
	Data_Obj *			fbi_dp;
	struct fb_var_screeninfo	fbi_var_info;	/* contains screen info that can be changed */
	struct fb_fix_screeninfo	fbi_fix_info; 	/* contains screen info that is fixed */ 
} FB_Info;


#define fbi_height	fbi_dp->dt_rows
#define fbi_width	fbi_dp->dt_cols
#define fbi_depth	fbi_dp->dt_comps
#define fbi_mem		fbi_dp->dt_data

#define NO_FBI	((FB_Info *)NULL)

ITEM_INTERFACE_PROTOTYPES(FB_Info,fbi)

#define PICK_FBI(pmpt)		pick_fbi(QSP_ARG  pmpt)

#define FBIO_WAITFORVSYNC	_IOW('F', 0x20, u_int32_t)	/* from /usr/src/linux/include/linux/matroxfb.h */

#define MAX_HEADS	2

typedef struct genlock_info {
	ParPort *	gli_ppp;
	FB_Info *	gli_fbip[MAX_HEADS];
	int		gli_n_heads;
	/* struct timeval stuff used to be ifdef LINUX ? */
	struct timeval	gli_tv_pp[2];		/* times of the most recent transition of each type */
	struct timeval	gli_tv_fb[MAX_HEADS][2];
	long		gli_fb_latency[MAX_HEADS];	/* time of fb sync relative to pport pulse */
	long		gli_pp_latency[MAX_HEADS];	/* time of pport sync relative to fb's */
	long		gli_drift[MAX_HEADS];		/* change in fb_latency */
	int		gli_refractory[MAX_HEADS];
} Genlock_Info;

#ifdef HAVE_PARPORT

/* genlock.c */

extern COMMAND_FUNC( do_fbpair_monitor );
extern COMMAND_FUNC( do_genlock );
extern COMMAND_FUNC( halt_genlock );
extern COMMAND_FUNC( report_genlock_status );
extern void get_genlock_mutex(void);
extern void rls_genlock_mutex(void);
extern int init_genlock(SINGLE_QSP_ARG_DECL);

extern int genlock_active;
extern Genlock_Info gli1;
#endif /* HAVE_PARPORT */


extern void show_var_info(FB_Info *);

extern void genlock_vblank(Genlock_Info *);
extern void nc_show_var_info(FB_Info *);

#endif /* HAVE_FB_DEV */

#endif /* _MY_FB_H_ */

