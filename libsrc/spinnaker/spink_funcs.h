
// redefine these macros to declare the prototypes
#define SPINK_WRAPPER_THREE_ARG(my_name,spin_name,decl1,name1,decl2,name2,decl3,name3)	\
extern int _##my_name(QSP_ARG_DECL  decl1 name1, decl2 name2, decl3 name3);

#define SPINK_WRAPPER_TWO_ARG(my_name,spin_name,decl1,name1,decl2,name2)	\
extern int _##my_name(QSP_ARG_DECL  decl1 name1, decl2 name2);

#define SPINK_WRAPPER_ONE_ARG(my_name,spin_name,decl1,name1)	\
extern int _##my_name(QSP_ARG_DECL  decl1 name1);

#include "spink_wrappers.c"

// three arg macros
#define get_enumeration_entry_by_name(a,b,c) _get_enumeration_entry_by_name(QSP_ARG  a,b,c)
#define convert_spink_image(a,b,c) _convert_spink_image(QSP_ARG  a,b,c)
#define fetch_spink_node(a,b,c) _fetch_spink_node(QSP_ARG  a,b,c)
#define spink_get_string(a,b,c) _spink_get_string(QSP_ARG  a,b,c)
#define get_cam_from_list(a,b,c) _get_cam_from_list(QSP_ARG  a,b,c)
#define get_iface_from_list(a,b,c) _get_iface_from_list(QSP_ARG  a,b,c)
#define node_to_string(a,b,c) _node_to_string(QSP_ARG  a,b,c)
#define get_string_value(a,b,c) _get_string_value(QSP_ARG  a,b,c)
#define get_tip_value(a,b,c) _get_tip_value(QSP_ARG  a,b,c)
#define get_entry_symbolic(a,b,c) _get_entry_symbolic(QSP_ARG  a,b,c)
#define get_node_display_name(a,b,c) _get_node_display_name(QSP_ARG  a,b,c)
#define get_node_short_name(a,b,c) _get_node_short_name(QSP_ARG  a,b,c)
#define get_feature_by_index(a,b,c) _get_feature_by_index(QSP_ARG  a,b,c)

#define get_enumeration_int_val(a,b) _get_enumeration_int_val(QSP_ARG  a,b)
#define set_enumeration_int_val(a,b) _set_enumeration_int_val(QSP_ARG  a,b)
#define get_next_image(a,b) _get_next_image(QSP_ARG  a,b)
#define image_is_incomplete(a,b) _image_is_incomplete(QSP_ARG  a,b)
#define get_image_status(a,b) _get_image_status(QSP_ARG  a,b)
#define get_image_width(a,b) _get_image_width(QSP_ARG  a,b)
#define get_image_height(a,b) _get_image_height(QSP_ARG  a,b)
#define get_iface_map(a,b) _get_iface_map(QSP_ARG  a,b)
#define node_is_implemented(a,b) _node_is_implemented(QSP_ARG  a,b)
#define node_is_available(a,b) _node_is_available(QSP_ARG  a,b)
#define node_is_readable(a,b) _node_is_readable(QSP_ARG  a,b)
#define node_is_writeable(a,b) _node_is_writeable(QSP_ARG  a,b)
#define get_iface_cameras(a,b) _get_iface_cameras(QSP_ARG  a,b)
#define get_n_cameras(a,b) _get_n_cameras(QSP_ARG  a,b)
#define get_n_interfaces(a,b) _get_n_interfaces(QSP_ARG  a,b)
#define get_transport_level_map(a,b) _get_transport_level_map(QSP_ARG  a,b)
#define get_iface_list(a,b) _get_iface_list(QSP_ARG  a,b)
#define get_cameras_from_system(a,b) _get_cameras_from_system(QSP_ARG  a,b)
#define get_node_type(a,b) _get_node_type(QSP_ARG  a,b)
#define get_int_value(a,b) _get_int_value(QSP_ARG  a,b)
#define get_float_value(a,b) _get_float_value(QSP_ARG  a,b)
#define get_bool_value(a,b) _get_bool_value(QSP_ARG  a,b)
#define get_current_entry(a,b) _get_current_entry(QSP_ARG  a,b)
#define get_n_features(a,b) _get_n_features(QSP_ARG  a,b)
#define get_device_node_map(a,b) _get_device_node_map(QSP_ARG  a,b)
#define get_stream_node_map(a,b) _get_stream_node_map(QSP_ARG  a,b)
#define get_camera_node_map(a,b) _get_camera_node_map(QSP_ARG  a,b)

// one arg funcs
#define release_spink_image(a) _release_spink_image(QSP_ARG  a)
#define create_empty_image(a) _create_empty_image(QSP_ARG  a)
#define destroy_spink_image(a) _destroy_spink_image(QSP_ARG  a)
#define begin_acquisition(a) _begin_acquisition(QSP_ARG  a)
#define end_acquisition(a) _end_acquisition(QSP_ARG  a)
#define create_empty_cam_list(a) _create_empty_cam_list(QSP_ARG  a)
#define clear_cam_list(a) _clear_cam_list(QSP_ARG  a)
#define destroy_cam_list(a) _destroy_cam_list(QSP_ARG  a)
#define clear_iface_list(a) _clear_iface_list(QSP_ARG  a)
#define destroy_iface_list(a) _destroy_iface_list(QSP_ARG  a)
#define release_spink_interface(a) _release_spink_interface(QSP_ARG  a)
#define release_spink_cam(a) _release_spink_cam(QSP_ARG  a)
#define release_interface(a) _release_interface(QSP_ARG  a)
#define release_spink_system(a) _release_spink_system(QSP_ARG  a)
#define get_spink_system(a) _get_spink_system(QSP_ARG  a)
#define create_empty_iface_list(a) _create_empty_iface_list(QSP_ARG  a)
#define camera_deinit(a) _camera_deinit(QSP_ARG  a)
#define connect_spink_cam(a) _connect_spink_cam(QSP_ARG  a)

#undef SPINK_WRAPPER_THREE_ARG
#undef SPINK_WRAPPER_TWO_ARG
#undef SPINK_WRAPPER_ONE_ARG

