
MAINTAINERCLEANFILES	=	Makefile.in
info_TEXINFOS		=	quip.texi
SUFFIXES		=	.html
HTML_DOCS		=	quip.html

TARGETS			=

#if HAVE_MAKEINFO

#TARGETS			+= pdf html quip.info
TARGETS			+= html quip.info

#endif

.texinfo.html:
	$(MAKEINFO) --html $<

.texi.html:
	$(MAKEINFO) --html $<


# makeinfo --html is not making the html when called from automake
all:	$(TARGETS) pdf
	cp quip.pdf ../docs/quip_manual.pdf
	cp -R quip.html ../docs/quip_web_manual

.PHONY:	html
html:	version.texi $(HTML_DOCS)


# what is the correct way to list an included file
# that should be part of the distribution?

noinst_HEADERS		= COPYING

