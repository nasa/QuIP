
MAINTAINERCLEANFILES	=	Makefile.in
info_TEXINFOS		=	quip.texi
SUFFIXES		=	.html
HTML_DOCS		=	quip.html

.texinfo.html:
	$(MAKEINFO) --html $<

.PHONY:	html
html:	version.texi $(HTML_DOCS)

#all:	html quip.pdf quip.info
all:	html quip.info


# what is the correct way to list an included file
# that should be part of the distribution?

noinst_HEADERS		= COPYING

