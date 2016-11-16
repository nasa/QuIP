SUBDIRS = veclib img_file hips

# remind me what the phony rule does - forces update of quip_version.h?
#.PHONY:	quip_version.h
# this is taken out because update_quip_version.csh can't be run
# by someone with the distribution (no access to git repo)

# Also removed the script from the distribution...
#			update_quip_version.csh	\
#
# How can we make the update script execute when doing a VPATH build???
# $< is first prerequisite...  $^ is all prerequisites

# PROBLEM:  FORCE causes quip_version.h to be made all the time,
# not just when we run make with no args.  In particular, this
# runs when we do make install, which we don't want...

quip_version.h:	update_quip_version.sh FORCE
	./$< $(srcdir)

FORCE:

# obsolete file
#			stdc_defs.h
#			function_basic.h
#			dobj_basic.h
#			qs_basic.h

noinst_HEADERS =			\
					\
			update_quip_version.sh	\
					\
			ascii_fmts.h	\
			camera.h	\
			camera_api.h	\
			cmaps.h		\
			command.h	\
			container.h	\
			container_fwd.h	\
			ctx_pair.h	\
			cuda_api.h	\
			cuda_helper.h	\
			cuda_supp.h	\
			data_obj.h	\
			dct8.h		\
			debug.h		\
			debug_module.h	\
			dict.h		\
			display.h	\
			dispobj.h	\
			dobj_prot.h	\
			fileck.h	\
			fio_api.h	\
			freel.h		\
			function.h	\
			func_helper.h	\
			gen_win.h	\
			genwin_flags.h	\
			getbuf.h	\
			gl_info.h	\
			gl_viewer.h	\
			gmovie.h	\
			hash.h		\
			history.h	\
			identifier.h	\
			img_file.h	\
			ios_config.h	\
			ios_gui.h	\
			item_obj.h	\
			item_prot.h	\
			item_type.h	\
			linear.h	\
			list.h		\
			macro.h		\
			map_ios_item.h	\
			my_curl.h	\
			my_cu2.h	\
			my_encryption.h	\
			my_ocl.h	\
			my_vl2.h	\
			my_parport.h	\
			nav_panel.h	\
			nexpr.h		\
			node.h		\
			nports_api.h	\
			nrm_api.h	\
			ocl_api.h	\
			ocl_platform.h	\
			opengl_utils.h	\
			optimize.h	\
			panel_obj.h	\
			param_api.h	\
			pipe_support.h	\
			pf_viewer.h	\
			platform.h	\
			pointer.h	\
			polh_menu.h	\
			query.h		\
			query_api.h	\
			query_bits.h	\
			query_stack.h	\
			quip_config.h	\
			quip_fwd.h	\
			quip_menu.h	\
			quip_prot.h	\
			quip_version.h	\
			rbtree.h	\
			reference.h	\
			rn.h		\
			rt_sched.h	\
			rv_api.h	\
			screen_obj.h	\
			seq.h		\
			seq_api.h	\
			serbuf.h	\
			serial.h	\
			server.h	\
			shape_bits.h	\
			shape_info.h	\
			sigpush.h	\
			sizable.h	\
			stack.h		\
			strbuf.h	\
			string_ref.h	\
			subrt.h		\
			thread_safe_defs.h	\
			treecode.h	\
			ttyctl.h	\
			typedefs.h	\
			typed_scalar.h	\
			undef_sym.h	\
			variable.h	\
			vec_util.h	\
			veclib_api.h	\
			vectree.h	\
			vec_expr_node.h	\
			view_cmds.h	\
			viewer.h	\
			vt_api.h	\
			warn.h		\
			Xhs.h		\
			xsupp.h		\
			yacc_hack.h

# This returns a non-zero exit status when run from setuid-root program mk_install???
#

# quip version should only be recomputed by the maintainer,
# people building from the distribution don't have access
# to the repo...  Is this a legitimate use for AM_MAINTAINER_MODE?

#quip_version.h:
#	- /bin/csh $(srcdir)/update_quip_version.csh


