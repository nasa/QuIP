
typedef struct cpu_info {
	unsigned long cpu_data_eax;
	unsigned long cpu_data_ebx;
	unsigned long cpu_data_edx;	/* we switch the order of these to be able to cpy the string... */
	unsigned long cpu_data_ecx;
} Cpu_Info;

extern void cpu_id3(Cpu_Info *,int);
extern COMMAND_FUNC( get_cpu_info );

// cpu_flags.c
extern int cpu_supports_mmx(void);
extern int cpu_supports_sse(void);
extern int cpu_supports_sse2(void);


