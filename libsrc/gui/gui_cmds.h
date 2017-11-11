#ifndef _GUI_CMDS_H_
#define _GUI_CMDS_H_

extern COMMAND_FUNC( set_panel_label );
//extern COMMAND_FUNC( do_hide_back );
extern COMMAND_FUNC( do_so_info );
extern COMMAND_FUNC( do_get_posn_object );
extern COMMAND_FUNC( do_set_posn_object );
extern COMMAND_FUNC( do_enable_widget );
extern COMMAND_FUNC( do_hide_widget );
extern COMMAND_FUNC( mk_panel );
extern COMMAND_FUNC( do_list_panel_objs );
extern COMMAND_FUNC( mk_menu_button );
extern COMMAND_FUNC( pop_parent );
extern COMMAND_FUNC( end_menu );
extern COMMAND_FUNC( do_normal );
extern COMMAND_FUNC( do_pullright );
extern COMMAND_FUNC( mk_position );
extern COMMAND_FUNC( mk_button );
extern COMMAND_FUNC( mk_toggle );
extern COMMAND_FUNC( mk_label );
extern COMMAND_FUNC( mk_text );
extern COMMAND_FUNC( mk_password );
extern COMMAND_FUNC( mk_edit_box );
extern COMMAND_FUNC( assign_text );
extern COMMAND_FUNC( do_set_prompt );
extern COMMAND_FUNC( do_set_edit_text );
extern COMMAND_FUNC( do_set_text_field );
extern COMMAND_FUNC( mk_gauge );
extern COMMAND_FUNC( set_new_range );
extern COMMAND_FUNC( set_new_pos );
extern COMMAND_FUNC( mk_slider );
extern COMMAND_FUNC( mk_adjuster );
extern COMMAND_FUNC( mk_adjuster_w );
extern COMMAND_FUNC( mk_slider_w );
extern COMMAND_FUNC( mk_message );
extern COMMAND_FUNC( mk_text_box );
extern COMMAND_FUNC( mk_act_ind );
extern COMMAND_FUNC( do_set_active );
extern COMMAND_FUNC( do_show );
extern COMMAND_FUNC( do_unshow );
extern COMMAND_FUNC( do_set_gauge_value );
extern COMMAND_FUNC( do_set_gauge_label );
extern COMMAND_FUNC( do_set_message );
extern COMMAND_FUNC( do_set_label );
extern COMMAND_FUNC( do_append_text );
extern COMMAND_FUNC( do_set_toggle );
extern COMMAND_FUNC( do_add_choice );
extern COMMAND_FUNC( do_del_choice );
extern COMMAND_FUNC( do_set_choice );
extern COMMAND_FUNC( do_get_choice );
extern COMMAND_FUNC( do_set_picks );
extern COMMAND_FUNC( do_clear_choices );
extern COMMAND_FUNC( clear_screen );
extern COMMAND_FUNC( do_pposn );
extern COMMAND_FUNC( do_delete );
extern COMMAND_FUNC( do_notice );
extern COMMAND_FUNC( mk_scroller );
extern COMMAND_FUNC( do_set_scroller );
extern COMMAND_FUNC( do_item_scroller );
extern COMMAND_FUNC( do_file_scroller );
extern COMMAND_FUNC( do_chooser );
extern COMMAND_FUNC( do_mlt_chooser );
extern COMMAND_FUNC( do_picker );
extern COMMAND_FUNC( do_resize_panel );

#ifdef BUILD_FOR_MACOS
extern COMMAND_FUNC( do_menu_bar );
#endif // BUILD_FOR_MACOS

#endif /* ! _GUI_CMDS_H_ */

