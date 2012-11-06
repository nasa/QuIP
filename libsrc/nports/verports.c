#include "quip_config.h"

char VersionId_ports_verports[] = QUIP_VERSION_STRING;

#include "version.h"
#include "nports.h"

#ifdef FOOBAR
/* SUNs PC-NFS programmer development toolkit */
char VersionId_ports_ltklib[] = "LTKLIB 4.0 06/08/92";
/* SUNs PC-NFS */
char VersionId_ports_pcnfs[] = "PCNFS 5.0 07/15/93";

extern char VersionId_ports_packets[];
extern char VersionId_ports_portdata[];
extern char VersionId_ports_portmenu[];
extern char VersionId_ports_ports[];
extern char VersionId_ports_sockopts[];
extern char VersionId_ports_verports[];
extern char VersionId_ports_xmitrecv[];

FileVersion ports_files[] = {
	{	VersionId_ports_ltklib,	"ltklib"	},
	{	VersionId_ports_pcnfs,	"pcnfs"		},
	{	VersionId_ports_packets,	"packets.c"	},
	{	VersionId_ports_portdata,	"portdata.c"	},
	{	VersionId_ports_portmenu,	"portmenu.c"	},
	{	VersionId_ports_ports,	"ports.c"	},
	{	VersionId_ports_sockopts,	"sockopts.c"	},
	{	VersionId_ports_xmitrecv,	"xmitrecv.c"	},
	{	VersionId_ports_verports,	"verports.c"	}
};  
 
#define MAX_PORTS_FILES (sizeof(ports_files)/sizeof(FileVersion))
 
void verports()
{
	mkver("PORTS", ports_files, MAX_PORTS_FILES);
} 

#else	/* ! FOOBAR */
 
void verports(SINGLE_QSP_ARG_DECL)
{
	auto_version(QSP_ARG  "PORTS","VersionId_ports");
} 

#endif	/* ! FOOBAR */
