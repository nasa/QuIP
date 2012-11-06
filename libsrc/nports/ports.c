#include "quip_config.h"

char VersionId_ports_ports[] = QUIP_VERSION_STRING;

#include <stdio.h>

#include "nports.h"
#include "debug.h"
#include "savestr.h"

#include "items.h"
#include "node.h"



debug_flag_t debug_ports=0;

ITEM_INTERFACE_DECLARATIONS(Port,port)

/* del_port deletes given a name */

void delport(QSP_ARG_DECL  Port *mpp)
{
#ifdef CAUTIOUS
	if( mpp==NO_PORT ) ERROR1("delport passed NULL");
#endif /* CAUTIOUS */
	del_item(QSP_ARG  port_itp,mpp);
	rls_str((char *)mpp->mp_name);
}

void portinfo(Port *mpp)
{
	sprintf(msg_str,"Port \"%s\":",mpp->mp_name);
	prt_msg(msg_str);
	sprintf(msg_str,"\tsock = %d, o_sock = %d",mpp->mp_sock,mpp->mp_o_sock);
	prt_msg(msg_str);
	sprintf(msg_str,"\taddrp = 0x%lx",(u_long)mpp->mp_addrp);
	prt_msg(msg_str);

	sprintf(msg_str,"\tflags =  0x%x",mpp->mp_flags);
	prt_msg(msg_str);
	sprintf(msg_str,"\tsleeptime =  %ld",mpp->mp_sleeptime);
	prt_msg(msg_str);

	if( mpp->mp_flags & PORT_SERVER ){
		sprintf(msg_str,"\tSERVER\t\t0x%x",PORT_SERVER);
		prt_msg(msg_str);
	}
	if( mpp->mp_flags & PORT_CLIENT ){
		sprintf(msg_str,"\tCLIENT\t\t0x%x",PORT_CLIENT);
		prt_msg(msg_str);
	}
	if( mpp->mp_flags & PORT_CONNECTED ){
		sprintf(msg_str,"\tCONNECTED\t0x%x",PORT_CONNECTED);
		prt_msg(msg_str);
	}
}

void init_ports(SINGLE_QSP_ARG_DECL)
{
	static int ports_inited=0;

	if( ports_inited ) return;

#ifdef DEBUG
	debug_ports = add_debug_module(QSP_ARG  "ports");
#endif /* DEBUG */

	init_pdt_tbl();
#ifdef PC
	/* Set exit handler called from nice_exit */
	if(do_on_exit(ports_cleanup) == -1)
		ERROR1("init_ports(): do_on_exit() failed");
#endif /* PC */

	verports(SINGLE_QSP_ARG);
#ifdef _WINDLL
	tkdll_init();
#endif /* _WINDLL */
	ports_inited=1;
}

