
#ifdef HAVE_MPEG
int mpeg_unconv( void *hdr_pp, Data_Obj *dp );
int mpeg_conv( Data_Obj *dp, void *hd_pp );
Image_File * mpeg_open( const char *name, int rw );
void mpeg_close( Image_File *ifp );
void mpeg_rd( Data_Obj *dp, Image_File *ifp, index_t x_offset, index_t y_offset,index_t t_offset );
int mpeg_wt( Data_Obj *dp, Image_File *ifp );
void report_mpeg_info( Image_File *ifp );
#endif /* HAVE_MPEG */

