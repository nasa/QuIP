
/* This file is part of the himemfb device driver */

#define MAX_MC_PIDS	10

typedef struct mem_chunk {
	void *	mc_addr;
	uint32_t	mc_size;
	uint32_t	mc_flags;
#ifdef TRACK_PIDS
	pid_t	mc_pid[MAX_MC_PIDS];
	int	mc_npids;
#endif /* TRACK_PIDS */
} Mem_Chunk;

/* flag bits */
#define AUTO_RELEASE	1	/* release this chunk when the process exits */
#define CHUNK_IN_USE	2

#define MAX_CHUNKS	4

void available_chunk(Mem_Chunk *);
void alloc_chunk(Mem_Chunk *);
#ifdef TRACK_PIDS
int add_mc_pid(Mem_Chunk *);
int sub_mc_pid(Mem_Chunk *);
#endif /* TRACK_PIDS */
int release_chunk(Mem_Chunk *);
void auto_release_chunks(void);
void init_chunk_table(uint32_t size,void *addr);


