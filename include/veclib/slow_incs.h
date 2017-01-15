
#define SHOW_INCR(inc_p)	fprintf(stderr,"%s:  %d %d %d %d %d\n",#inc_p,\
(inc_p)->d5_dim[0],\
(inc_p)->d5_dim[1],\
(inc_p)->d5_dim[2],\
(inc_p)->d5_dim[3],\
(inc_p)->d5_dim[4]);

#define DECL_SLOW_INCRS_1	dim5 dst_vwxyz_incr;		\
				/*int dim_indices[3];*/
#define DECL_SLOW_INCRS_SRC1	dim5 s1_vwxyz_incr;
#define DECL_SLOW_INCRS_SRC2	dim5 s2_vwxyz_incr;
#define DECL_SLOW_INCRS_SRC3	dim5 s3_vwxyz_incr;
#define DECL_SLOW_INCRS_SRC4	dim5 s4_vwxyz_incr;
#define DECL_SLOW_INCRS_SBM	dim5 sbm_vwxyz_incr;
#define DECL_SLOW_INCRS_DBM	/*dim5 dbm_vwxyz_incr;*/		\
				/*int dim_indices[3];*/

#define DECL_SLOW_INCRS_DBM_SBM	DECL_SLOW_INCRS_DBM DECL_SLOW_INCRS_SBM
#define DECL_SLOW_INCRS_2	DECL_SLOW_INCRS_1 DECL_SLOW_INCRS_SRC1
#define DECL_SLOW_INCRS_CONV	DECL_SLOW_INCRS_1 DECL_SLOW_INCRS_SRC1
#define DECL_SLOW_INCRS_3	DECL_SLOW_INCRS_2 DECL_SLOW_INCRS_SRC2
#define DECL_SLOW_INCRS_4	DECL_SLOW_INCRS_3 DECL_SLOW_INCRS_SRC3
#define DECL_SLOW_INCRS_5	DECL_SLOW_INCRS_4 DECL_SLOW_INCRS_SRC4
#define DECL_SLOW_INCRS_2SRCS	DECL_SLOW_INCRS_SRC1 DECL_SLOW_INCRS_SRC2
#define DECL_SLOW_INCRS_1SRC	DECL_SLOW_INCRS_SRC1
#define DECL_SLOW_INCRS_SBM_1	DECL_SLOW_INCRS_1 DECL_SLOW_INCRS_SBM
#define DECL_SLOW_INCRS_SBM_2	DECL_SLOW_INCRS_2 DECL_SLOW_INCRS_SBM
#define DECL_SLOW_INCRS_SBM_3	DECL_SLOW_INCRS_3 DECL_SLOW_INCRS_SBM
#define DECL_SLOW_INCRS_DBM_		DECL_SLOW_INCRS_DBM
#define DECL_SLOW_INCRS_DBM_1SRC	DECL_SLOW_INCRS_1SRC DECL_SLOW_INCRS_DBM
#define DECL_SLOW_INCRS_DBM_2SRCS	DECL_SLOW_INCRS_2SRCS DECL_SLOW_INCRS_DBM

/////////////////// end of declarations /////////////////


#define REPORT_INCS(incs)					\
	fprintf(stderr,"%s:  %d %d %d %d %d\n",#incs,incs.d5_dim[0],incs.d5_dim[1],incs.d5_dim[2],incs.d5_dim[3],incs.d5_dim[4]);

#define SETUP_SLOW_INCS_1						\
									\
dst_vwxyz_incr.d5_dim[0] = INCREMENT(VA_DEST_INCSET(vap),0);		\
dst_vwxyz_incr.d5_dim[1] = INCREMENT(VA_DEST_INCSET(vap),1);		\
dst_vwxyz_incr.d5_dim[2] = INCREMENT(VA_DEST_INCSET(vap),2);		\
dst_vwxyz_incr.d5_dim[3] = INCREMENT(VA_DEST_INCSET(vap),3);		\
dst_vwxyz_incr.d5_dim[4] = INCREMENT(VA_DEST_INCSET(vap),4);		\
/*REPORT_INCS(dst_vwxyz_incr)*/

// This copy should not be necessary, but it is a historical artifact when we shrank from 5 to 3 dimensions...

#define SETUP_SLOW_INCS_DBM						\
									\
/*dbm_vwxyz_incr.d5_dim[0] = INCREMENT(VA_DEST_INCSET(vap),0);		\
dbm_vwxyz_incr.d5_dim[1] = INCREMENT(VA_DEST_INCSET(vap),1);		\
dbm_vwxyz_incr.d5_dim[2] = INCREMENT(VA_DEST_INCSET(vap),2);		\
dbm_vwxyz_incr.d5_dim[3] = INCREMENT(VA_DEST_INCSET(vap),3);		\
dbm_vwxyz_incr.d5_dim[4] = INCREMENT(VA_DEST_INCSET(vap),4);*/		\
/*REPORT_INCS(dbm_xyz_incr)*/

#define SETUP_SLOW_INCS_SRC1					\
								\
s1_vwxyz_incr.d5_dim[0] = INCREMENT(VA_SRC1_INCSET(vap),0);	\
s1_vwxyz_incr.d5_dim[1] = INCREMENT(VA_SRC1_INCSET(vap),1);	\
s1_vwxyz_incr.d5_dim[2] = INCREMENT(VA_SRC1_INCSET(vap),2);	\
s1_vwxyz_incr.d5_dim[3] = INCREMENT(VA_SRC1_INCSET(vap),3);	\
s1_vwxyz_incr.d5_dim[4] = INCREMENT(VA_SRC1_INCSET(vap),4);

#define SETUP_SLOW_INCS_SRC2					\
								\
s2_vwxyz_incr.d5_dim[0] = INCREMENT(VA_SRC2_INCSET(vap),0);	\
s2_vwxyz_incr.d5_dim[1] = INCREMENT(VA_SRC2_INCSET(vap),1);	\
s2_vwxyz_incr.d5_dim[2] = INCREMENT(VA_SRC2_INCSET(vap),2);	\
s2_vwxyz_incr.d5_dim[3] = INCREMENT(VA_SRC2_INCSET(vap),3);	\
s2_vwxyz_incr.d5_dim[4] = INCREMENT(VA_SRC2_INCSET(vap),4);

#define SETUP_SLOW_INCS_SRC3					\
								\
s3_vwxyz_incr.d5_dim[0] = INCREMENT(VA_SRC3_INCSET(vap),0);	\
s3_vwxyz_incr.d5_dim[1] = INCREMENT(VA_SRC3_INCSET(vap),1);	\
s3_vwxyz_incr.d5_dim[2] = INCREMENT(VA_SRC3_INCSET(vap),2);	\
s3_vwxyz_incr.d5_dim[3] = INCREMENT(VA_SRC3_INCSET(vap),3);	\
s3_vwxyz_incr.d5_dim[4] = INCREMENT(VA_SRC3_INCSET(vap),4);

#define SETUP_SLOW_INCS_SRC4					\
								\
s4_vwxyz_incr.d5_dim[0] = INCREMENT(VA_SRC4_INCSET(vap),0);	\
s4_vwxyz_incr.d5_dim[1] = INCREMENT(VA_SRC4_INCSET(vap),1);	\
s4_vwxyz_incr.d5_dim[2] = INCREMENT(VA_SRC4_INCSET(vap),2);	\
s4_vwxyz_incr.d5_dim[3] = INCREMENT(VA_SRC4_INCSET(vap),3);	\
s4_vwxyz_incr.d5_dim[4] = INCREMENT(VA_SRC4_INCSET(vap),4);

#define SETUP_SLOW_INCS_SRC5					\
								\
sbm_vwxyz_incr.d5_dim[0] = INCREMENT(VA_SRC5_INCSET(vap),0);	\
sbm_vwxyz_incr.d5_dim[1] = INCREMENT(VA_SRC5_INCSET(vap),1);	\
sbm_vwxyz_incr.d5_dim[2] = INCREMENT(VA_SRC5_INCSET(vap),2);	\
sbm_vwxyz_incr.d5_dim[3] = INCREMENT(VA_SRC5_INCSET(vap),3);	\
sbm_vwxyz_incr.d5_dim[4] = INCREMENT(VA_SRC5_INCSET(vap),4);


#define SETUP_SLOW_INCS_SBM	SETUP_SLOW_INCS_SRC5

#define SETUP_SLOW_INCS_2	SETUP_SLOW_INCS_1	\
				SETUP_SLOW_INCS_SRC1

#define SETUP_SLOW_INCS_3	SETUP_SLOW_INCS_2	\
				SETUP_SLOW_INCS_SRC2

#define SETUP_SLOW_INCS_4	SETUP_SLOW_INCS_3	\
				SETUP_SLOW_INCS_SRC3

#define SETUP_SLOW_INCS_5	SETUP_SLOW_INCS_4	\
				SETUP_SLOW_INCS_SRC4


#define SETUP_SLOW_INCS_DBM_2SRCS	SETUP_SLOW_INCS_DBM SETUP_SLOW_INCS_2SRCS
#define SETUP_SLOW_INCS_DBM_1SRC	SETUP_SLOW_INCS_DBM SETUP_SLOW_INCS_SRC1
#define SETUP_SLOW_INCS_DBM_SBM		SETUP_SLOW_INCS_DBM	\
					SETUP_SLOW_INCS_SBM

#define SETUP_SLOW_INCS_2SRCS	SETUP_SLOW_INCS_SRC1 SETUP_SLOW_INCS_SRC2

#define SETUP_SLOW_INCS_DBM_		SETUP_SLOW_INCS_DBM

// Not sure how to handle source bitmaps?
#define SETUP_SLOW_INCS_SBM_1		SETUP_SLOW_INCS_1	\
					SETUP_SLOW_INCS_SBM

#define SETUP_SLOW_INCS_SBM_2		SETUP_SLOW_INCS_2	\
					SETUP_SLOW_INCS_SBM

#define SETUP_SLOW_INCS_SBM_3		SETUP_SLOW_INCS_3	\
					SETUP_SLOW_INCS_SBM

