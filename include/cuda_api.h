

#ifdef __cplusplus
extern "C" {
#endif

extern COMMAND_FUNC( do_cuda_menu );
extern void gpu_obj_dnload(QSP_ARG_DECL  Data_Obj *host_dp, Data_Obj *gpu_dp);
extern void gpu_obj_upload(QSP_ARG_DECL  Data_Obj *gpu_dp, Data_Obj *host_dp);

extern void gpu_mem_upload(QSP_ARG_DECL  void *dst, void *src, size_t siz);
extern void gpu_mem_dnload(QSP_ARG_DECL  void *dst, void *src, size_t siz);


#ifdef __cplusplus
}
#endif

