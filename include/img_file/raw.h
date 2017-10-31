
#ifndef _RAW_H_
#define _RAW_H_

#include "img_file.h"
//#include "fio_prot.h"

FIO_INTERFACE_PROTOTYPES( raw , void )

#define raw_to_dp(a,b)	_raw_to_dp(QSP_ARG  a,b)

extern void	rd_raw_gaps(QSP_ARG_DECL  Data_Obj *dp,Image_File *ifp);
extern void	read_object(QSP_ARG_DECL  Data_Obj *dp,Image_File *ifp);
/* extern void	read_frame(Data_Obj *dp,Image_File *ifp); */
extern int	frag_read(Data_Obj *dp,Image_File *ifp,index_t x_offset,index_t y_offset,index_t t_offset);
extern void	wt_raw_gaps(QSP_ARG_DECL  Data_Obj *dp,Image_File *ifp);
extern void	wt_raw_contig(QSP_ARG_DECL  Data_Obj *dp,Image_File *ifp);
extern void	wt_raw_data(QSP_ARG_DECL  Data_Obj *dp,Image_File *ifp);
extern void	set_raw_sizes(dimension_t arr[N_DIMENSIONS]);


#endif /* _RAW_H_ */

