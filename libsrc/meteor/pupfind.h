
extern void compute_diff_image(int newest, int previous, int component_offset);
extern COMMAND_FUNC( setup_diff_computation );
extern void compute_curvature(int newest);
extern COMMAND_FUNC( setup_curv_computation );
extern COMMAND_FUNC( blur_curvature );
extern void _setup_blur(QSP_ARG_DECL  Data_Obj *dp);
#define setup_blur(dp) _setup_blur(QSP_ARG  dp)

