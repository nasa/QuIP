#include "quip_config.h"

char VersionId_newvec_cpuid[] = QUIP_VERSION_STRING;

#ifdef LINUX

#ifndef LONG_64_BIT

#include <stdio.h>
#include <string.h>
#include "my_cpuid.h"

void cpu_id1(Cpu_Info *cip, int index)
{
	cip->cpu_data_eax=1234;
	cip->cpu_data_ebx=index;
}

/* This assembler code assumes certain C argument passing conventions,
 * which were determined
 * by examining the assembly code generated for some similar C code.
 * This does not work
 * if the optimizer is invoked...
 */

void cpu_id3(Cpu_Info *cip, int index)
{
	__asm__ (
		"pushl	%ebx\n\t"
		"movl	12(%ebp), %eax\n\t"
		"cpuid\n\t"
		"pushl	%edx\n\t"
		"movl	8(%ebp), %edx\n\t"
		"movl	%eax,(%edx)\n\t"
		"movl	%ebx,4(%edx)\n\t"
		"movl	%ecx,12(%edx)\n\t"
		"popl	%eax\n\t"
		"movl	%eax,8(%edx)\n\t"
		"popl	%ebx\n\t"
	);
}

void show_cpu_info(Cpu_Info *cip)
{
	printf("eax:\t0x%lx\n",cip->cpu_data_eax);
	printf("ebx:\t0x%lx\n",cip->cpu_data_ebx);
	printf("ecx:\t0x%lx\n",cip->cpu_data_ecx);
	printf("edx:\t0x%lx\n",cip->cpu_data_edx);
}

COMMAND_FUNC( get_cpu_info )
{
	Cpu_Info ci1;
	char string[16];
	int i,max_info;

	cpu_id3(&ci1,0);
	strncpy(string,(char *) &ci1.cpu_data_ebx,12);
	string[12]=0;
	printf("vendor:  %s\n",string);
	max_info=ci1.cpu_data_eax;

	for(i=1;i<=max_info;i++){
		printf("\nCPU info #%d:\n\n",i);
		cpu_id3(&ci1,i);
		show_cpu_info(&ci1);
	}
}

int cpu_supports_mmx()		/* really SSE ! */
{
	Cpu_Info ci1;
	char string[16];
	int i,max_info;

	cpu_id3(&ci1,0);
	strncpy(string,(char *) &ci1.cpu_data_ebx,12);
	string[12]=0;
	max_info=ci1.cpu_data_eax;

	for(i=1;i<=max_info;i++){
		cpu_id3(&ci1,i);
	}

	return(1);	/* BUG should do something else here! */
}

#endif /* undef LONG_64_BIT */

#else

int cpu_supports_mmx() { return(0); }

#endif /* ! LINUX */

