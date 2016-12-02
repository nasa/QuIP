#ifndef _QUIP_FWD_H_
#define _QUIP_FWD_H_

// Here we provide forward definitions all of our
// structs and typedefs.

#include "typedefs.h"
typedef uint32_t debug_flag_t;
typedef int QUIP_BOOL;

/* to make it easy to switch to NSStrings later... */
/* BUT conflicts with X11 !? */
#define Quip_String	const char

#define FWD_TYPEDEF(struct_name,typedef_name)			\
struct struct_name; typedef struct struct_name typedef_name;

FWD_TYPEDEF(query,Query)
FWD_TYPEDEF(query_stack,Query_Stack)
FWD_TYPEDEF(typed_scalar,Typed_Scalar)
FWD_TYPEDEF(scalar_expr_node,Scalar_Expr_Node)

FWD_TYPEDEF(mouthful,Mouthful)
FWD_TYPEDEF(my_pipe,Pipe)
FWD_TYPEDEF(list,List)
FWD_TYPEDEF(item_type,Item_Type)
FWD_TYPEDEF(item_context,Item_Context)
FWD_TYPEDEF(menu,Menu)
FWD_TYPEDEF(dictionary,Dictionary)	// obsolete with containers?
FWD_TYPEDEF(item,Item)
FWD_TYPEDEF(node,Node)
FWD_TYPEDEF(string_buf,String_Buf)
FWD_TYPEDEF(macro_arg,Macro_Arg)
FWD_TYPEDEF(macro,Macro)
FWD_TYPEDEF(function,Function)
FWD_TYPEDEF(variable,Variable)
FWD_TYPEDEF(debug_module,Debug_Module)
FWD_TYPEDEF(freelist,FreeList)
FWD_TYPEDEF(container,Container)
FWD_TYPEDEF(data_obj,Data_Obj)
FWD_TYPEDEF(vec_expr_node,Vec_Expr_Node)
FWD_TYPEDEF(input_format_spec,Input_Format_Spec)
FWD_TYPEDEF(shape_info,Shape_Info)
FWD_TYPEDEF(precision,Precision)
FWD_TYPEDEF(dimension_set,Dimension_Set)
FWD_TYPEDEF(increment_set,Increment_Set)
FWD_TYPEDEF(item_class,Item_Class)
FWD_TYPEDEF(member_info,Member_Info)
FWD_TYPEDEF(foreach_loop,Foreach_Loop)
FWD_TYPEDEF(frag_match_info,Frag_Match_Info)
FWD_TYPEDEF(hash_tbl,Hash_Tbl)
FWD_TYPEDEF(command,Command)
FWD_TYPEDEF(vec_obj_args,Vec_Obj_Args)
FWD_TYPEDEF(platform_device,Platform_Device)
FWD_TYPEDEF(curl_info,Curl_Info)

typedef List Stack;

#ifndef BUILD_FOR_OBJC
FWD_TYPEDEF(viewer,Viewer)
#endif // ! BUILD_FOR_OBJC

#include "thread_safe_defs.h"

#define COMMAND_FUNC(name)  void name(SINGLE_QSP_ARG_DECL)

#define ITEM_NEW_PROT(type,stem)	type * new_##stem(QSP_ARG_DECL  const char *name);
#define ITEM_INIT_PROT(type,stem)	void init_##stem##s(SINGLE_QSP_ARG_DECL );
#define ITEM_CHECK_PROT(type,stem)	type * stem##_of(QSP_ARG_DECL  const char *name);
#define ITEM_GET_PROT(type,stem)	type * get_##stem(QSP_ARG_DECL  const char *name);
#define ITEM_LIST_PROT(type,stem)	void  list_##stem##s(SINGLE_QSP_ARG_DECL );
#define ITEM_PICK_PROT(type,stem)	type *pick_##stem(QSP_ARG_DECL  const char *pmpt);
#define ITEM_ENUM_PROT(type,stem)	List *stem##_list(SINGLE_QSP_ARG_DECL);
#define ITEM_DEL_PROT(type,stem)	void del_##stem(QSP_ARG_DECL  type *ip);

#define ITEM_INTERFACE_PROTOTYPES(type,stem)	IIF_PROTS(type,stem,extern)
#define ITEM_INTERFACE_PROTOTYPES_STATIC(type,stem)		\
						IIF_PROTS(type,stem,static)

#define IIF_PROTS(type,stem,storage)				\
								\
storage ITEM_INIT_PROT(type,stem)				\
storage ITEM_NEW_PROT(type,stem)				\
storage ITEM_CHECK_PROT(type,stem)				\
storage ITEM_GET_PROT(type,stem)				\
storage ITEM_LIST_PROT(type,stem)				\
storage ITEM_ENUM_PROT(type,stem)				\
storage ITEM_DEL_PROT(type,stem)				\
storage ITEM_PICK_PROT(type,stem)

#define STRINGIFY(s)	_STRINGIFY(s)
#define _STRINGIFY(s)	#s

#endif // ! _QUIP_FWD_H_

