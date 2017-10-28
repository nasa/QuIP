/* New version based on reading /proc/cpuinfo */

#include "quip_config.h"

#include "nvf.h"
#include "quip_prot.h"
#include "my_cpuid.h"

#ifdef HAVE_PROC_CPUINFO

#include <stdio.h>
#include <string.h>

#define cpu_flag_set(string) _cpu_flag_set(QSP_ARG  string)

static int _cpu_flag_set(QSP_ARG_DECL  char *string)
{
	FILE *fp;
	char str[32];

	fp=popen("grep flags /proc/cpuinfo | head -1 | awk -F ':' '{print $2}'","r");
	if( fp == NULL ) {
		sprintf(ERROR_STRING,"cpu_flag_set:  error opening pipe to read /proc/cpuinfo");
		warn(ERROR_STRING);
		return(0);
	}
	while( fscanf(fp,"%s",str) == 1 ){
		if( !strcmp(str,string) ){	/* a match? */
			pclose(fp);
			return(1);
		}
	}
	fclose(fp);
	return(0);
}

int cpu_supports_mmx(void)
{
	return cpu_flag_set("mmx");
}

int cpu_supports_sse(void)
{
	return cpu_flag_set("sse");
}

int cpu_supports_sse2(void)
{
	return cpu_flag_set("sse2");
}

#else /* ! HAVE_PROC_CPUINFO */

/* BUG this stuff might work on Macs if we knew how to test... */

int cpu_supports_mmx(void) { return 1; }
int cpu_supports_sse(void) { return 1; }
int cpu_supports_sse2(void) { return 1; }

#endif /* ! HAVE_PROC_CPUINFO */

COMMAND_FUNC( get_cpu_info )
{
	if( cpu_supports_mmx() )
		prt_msg("CPU supports mmx instructions");
	else
		prt_msg("CPU does not support mmx instructions");

	if( cpu_supports_sse() )
		prt_msg("CPU supports sse instructions");
	else
		prt_msg("CPU does not support sse instructions");

	if( cpu_supports_sse2() )
		prt_msg("CPU supports sse2 instructions");
	else
		prt_msg("CPU does not support sse2 instructions");
}

