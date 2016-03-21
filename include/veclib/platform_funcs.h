

#ifdef BUILD_FOR_GPU
extern PF_COMMAND_FUNC(obj_dnload);
extern PF_COMMAND_FUNC(obj_upload);
extern PF_COMMAND_FUNC(list_devs);
extern PF_COMMAND_FUNC(set_device);

extern void PF_FUNC_NAME(set_device)(QSP_ARG_DECL  Platform_Device *cdp);
extern Platform_Stream * PF_FUNC_NAME(new_stream)(QSP_ARG_DECL  const char *name);

// streams could be done on a cpu also?
// Or just let OpenCL handle it?
extern void PF_FUNC_NAME(list_streams)(SINGLE_QSP_ARG_DECL);
extern PF_COMMAND_FUNC(new_stream);
extern PF_COMMAND_FUNC(list_streams);
extern PF_COMMAND_FUNC(stream_info);
extern PF_COMMAND_FUNC(sync_stream);
extern PF_COMMAND_FUNC(init_ckpts);
extern PF_COMMAND_FUNC(set_ckpt);
extern PF_COMMAND_FUNC(clear_ckpts);
extern PF_COMMAND_FUNC(show_ckpts);
extern void PF_FUNC_NAME(shutdown)(void);
extern void PF_FUNC_NAME(sync)(SINGLE_QSP_ARG_DECL);
extern void PF_FUNC_NAME(init_dev_memory)(QSP_ARG_DECL  Platform_Device *pdp);
extern void PF_FUNC_NAME(insure_device)(QSP_ARG_DECL  Data_Obj *dp);

#endif // BUILD_FOR_GPU

// These are used by vl2 also

// these are probably redundant!?
extern void PF_FUNC_NAME(init_platform)(SINGLE_QSP_ARG_DECL);
extern void PF_FUNC_NAME(init)(SINGLE_QSP_ARG_DECL);
extern void PF_FUNC_NAME(alloc_data)(QSP_ARG_DECL  Data_Obj *dp, dimension_t size);


