
/* defns shared by veclib & warlib */

define(`VFCODE_ARG',`vf_code,')
define(`VFCODE_ARG_DECL',`const int vf_code,')
define(`HOST_CALL_ARGS',`VFCODE_ARG  oap')
define(`HOST_CALL_ARG_DECLS',`VFCODE_ARG_DECL  /*const*/ Vec_Obj_Args *oap')

// Why are these called link funcs?  Maybe because they can be chained?
// Kind of a legacy from the old skywarrior library code...
// A vector arg used to just have a length and a stride, but now
// with gpus we have three-dimensional lengths.  But in principle
// there's no reason why we couldn't have full shapes passed...
define(`LINK_FUNC_ARGS',`VFCODE_ARG  vap')
define(`LINK_FUNC_ARG_DECLS',`VFCODE_ARG_DECL  const Vector_Args *vap')


