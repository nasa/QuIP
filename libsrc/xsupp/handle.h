
/* support for relocatable memory objects */

typedef void * pointer;
typedef pointer * Handle;
#define NO_HANDLE ((Handle)NULL)

#define MAX_POINTERS	4096


extern Handle new_hdl(u_long size);
extern void rls_hdl(Handle hdl);

