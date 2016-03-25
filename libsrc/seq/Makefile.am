all:	libseq.a
	cp libseq.a ../../libs

noinst_LIBRARIES = libseq.a

libseq_a_CPPFLAGS = -I$(top_srcdir)/include -Wall -Wmissing-prototypes

# How to specify seqparse.c built from seqparse.y???
libseq_a_SOURCES =				\
						\
		seqparse.y			\
		seqparse.c			\
		seqprint.c			\
		seqmenu.c


noinst_HEADERS = 				\
		seq.h

