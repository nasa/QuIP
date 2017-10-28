
typedef struct cpu_info {
	unsigned long cpu_data_eax;
	unsigned long cpu_data_ebx;
	unsigned long cpu_data_edx;	/* we switch the order of these to be able to cpy the string... */
	unsigned long cpu_data_ecx;
} Cpu_Info;

extern void cpu_id3(Cpu_Info *,int);
extern COMMAND_FUNC( get_cpu_info );

// cpu_flags.c
extern int _cpu_supports_mmx(SINGLE_QSP_ARG_DECL);
extern int _cpu_supports_sse(SINGLE_QSP_ARG_DECL);
extern int _cpu_supports_sse2(SINGLE_QSP_ARG_DECL);

#define cpu_supports_mmx() _cpu_supports_mmx(SINGLE_QSP_ARG)
#define cpu_supports_sse() _cpu_supports_sse(SINGLE_QSP_ARG)
#define cpu_supports_sse2() _cpu_supports_sse2(SINGLE_QSP_ARG)

