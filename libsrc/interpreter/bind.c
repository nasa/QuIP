/*
 * This file performs a few utilities on a.out files,
 * and supports dynamic loading.  This has not really been
 * ported to SGI...
 *
 * Do we really want to include this if the machine is UNIX?
 * For now, no!  SUN only...
 */

#include "quip_config.h"

char VersionId_interpreter_bind[] = QUIP_VERSION_STRING;

#ifdef SUN
#define DYNAMIC_LOAD
#endif /* SUN */

#ifdef DYNAMIC_LOAD

#include <stdio.h>
#include <stdlib.h>

#ifdef SUN
#include <a.out.h>
#endif /* SUN */
#ifdef SGI
#include <libelf.h>
#include <fcntl.h>
#endif /* SGI */

#include "getbuf.h"
#include "nameof.h"
#include "savestr.h"
#include "substr.h"
#include "debug.h"
#include "version.h"

/*
 * this size mask is a hack for sun3
 * to make buffers fall on 0 mod 8 boundary
 */
#define SIZE_MASK	7

/* BUG need to add debug module somewhere... */
#ifdef DEBUG
static int binddebug=0;
#endif /* DEBUG */

static int machtype=(-1);

static void loadfile(char *,char *);

long find_symbol(name)		/* should be addr_t? */
char *name;
{
	char cmd[128], type[64], n2[64];
	char line[128];
	FILE *fp;
	int addr;
	int n;
	char *s;

	s=get_symname();
	sprintf(cmd,"nm %s | egrep %s",s,name);
#ifdef DEBUG
if( debug & binddebug ){
sprintf(ERROR_STRING,"\nexecuting \"%s\"",cmd);
advise(ERROR_STRING);
}
#endif /* DEBUG */
	fp=popen(cmd,"r");
	if( fp==NULL ) {
		WARN("find_symbol: popen failed");
		return(-1);
	}

#ifdef FOO
try:
#endif /* FOO */

	while( fgets(line,128,fp) != NULL ){
		if( (n=sscanf(line,"%x %s %s",&addr,type,n2)) != 3 ){
sprintf(ERROR_STRING,"sscanf returned %d",n);
advise(ERROR_STRING);
sprintf(ERROR_STRING,"input was \"%s\"",line);
advise(ERROR_STRING);
			WARN("error parsing nm output");
			pclose(fp);
			return(-1);
		}
		if( !strcmp(name,n2) ){
			if( verbose ){
				sprintf(ERROR_STRING,
				"matching line, addr= 0x%x:\n%s",addr,line);
				advise(ERROR_STRING);
			}
			goto match_found;
		}
	}
	/* no match found */
	sprintf(ERROR_STRING,"symbol \"%s\" not found",name);
	advise(ERROR_STRING);
	return(-1);

/* why is this commented out? */
#ifdef FOO
	if( (addr=fscanf(fp,"%x %s %s",&addr,type,n2)) != 3 ){
sprintf(ERROR_STRING,"fscanf returned %d",addr);
WARN(ERROR_STRING);
		WARN("error parsing nm output");
		pclose(fp);
		return(-1);
	}
	if( strcmp(name,n2) ) goto try;	/* may be a substring or other match */
#endif /* FOO */

match_found:
	pclose(fp);
	/*
	if( strcmp(type,"T") ){
		sprintf(ERROR_STRING,"symbol \"%s\" is type %s",name,type);
		WARN(ERROR_STRING);
		return(-1);
	}
	*/
	return(addr);
}

void do_load()
{
	char buf[128];
	char *s;

	s=nameof("filename");
	strcpy(buf,s);
	s=nameof("libraries (or \"none\")");
	if( !strcmp(s,"none") ) *s=0;

	loadfile(buf,s);
}

void undef_func()
{
	WARN("function is undefined or not loaded");
}

void do_bind()
{
	char *s, *sel, *help;
	void (*func_addr)();
	long addr;

	s=nameof("routine name");
	addr=find_symbol(s);
#ifdef DEBUG
if( debug & binddebug ){
sprintf(ERROR_STRING,"function \"%s\" found at location 0x%x",s,addr);
advise(ERROR_STRING);
}
#endif /* DEBUG */
	if( addr==(-1) ){
		WARN("symbol not found; binding to dummy function");
		func_addr=undef_func;
	} else func_addr=(void (*)())addr;

	s=nameof("name for menu item");
	/* should make sure that this one is not already in use! */

	sel = savestr(s);
	s=nameof("description string");
	help = savestr(s);

	add_wcmd(sel,func_addr,help);
}

void getmachine()
{
#ifdef SUN
	FILE *fp;
	struct exec e1;

	fp=try_open(get_progfile(),"r");
	if( !fp ) return;

	if( fread(&e1,sizeof(e1),1,fp) != 1 )
		WARN("getmachine:  couldn't read exec");
	else
		machtype = e1.a_machtype;
	fclose(fp);

#elif SGI

	int fd;
	Elf *elf;
	Elf32_Ehdr *ehdr;

	fd=open(get_progfile(),O_RDONLY);
	if( fd < 0 ){
		WARN("error opening program file");
		return;
	}
	if( (elf=elf_begin(fd,ELF_C_READ,(Elf *) 0 ) ) == 0 ){
		WARN("error in elf_begin");
		return;
	}
	if( (ehdr=elf32_getehdr(elf)) == 0 ){
		WARN("error getting elf header");
		return;
	}
	machtype = ehdr->e_machine;

	elf_end(elf);

	close(fd);

#endif /* SGI */
}

static void loadfile(name,libs)
char *name, *libs;
{
	extern char *tempnam();
	int needsize, asize;
	char *loc, *start_loc;
	char *oldloc=NULL;	/* used to determine if we
				 * have to link a second time */
#ifdef SUN
	struct exec e1;
#endif /* SUN */
	FILE *fp;
	char str[256];
	char *tmpname;
	char *sn;		/* name of symbol file */

	/* read size of module */


	sprintf(ERROR_STRING,"loading \"%s\"...  ",name);
	advise(ERROR_STRING);

	fp=try_open(name,"r");
	if( !fp ) return;

#ifdef SUN
	if( fread(&e1,sizeof(e1),1,fp) != 1 ){
		WARN("couldn't read exec");
		goto finis;
	}

	if( machtype==(-1) ) getmachine();
	if( e1.a_machtype != machtype ){
		WARN("wrong machine type");
		goto finis;
	}

	if( N_BADMAG( e1 ) ){
		WARN("bad magic number");
		goto finis;
	}
	if( e1.a_magic != OMAGIC ){
		switch( e1.a_magic ){
			case NMAGIC:
				WARN("magic number is NMAGIC");
				break;
			case ZMAGIC:
				WARN("magic number is ZMAGIC");
				break;
			default:
				WARN("unknown valid magic number!?");
				break;
		}
		WARN("magic number must be OMAGIC");
		goto finis;
	}

	tmpname=tempnam("/tmp","d_ld");
doit:
	needsize = e1.a_text + e1.a_data + e1.a_bss;
	loc = getbuf ( needsize+SIZE_MASK );
	if( loc == NULL ){
		WARN("out of memory");
		goto finis;
	}
#endif /* SUN */

	fclose(fp);
	start_loc =(char *) ( ((int)(loc+SIZE_MASK))&(~SIZE_MASK) );

	sn=get_symname();

	if( start_loc != oldloc ){
		int stat;
		sprintf(str,"ld -x -o %s -A %s -T %lx %s %s",
			tmpname,sn,(u_long)start_loc,name,libs);
#ifdef DEBUG
if( debug & binddebug ){
sprintf(ERROR_STRING,"\nexecuting \"%s\"",str);
advise(ERROR_STRING);
}
#endif /* DEBUG */
		stat=system(str);
		if( stat != 0 ){
			sprintf(ERROR_STRING,"ld exit status:  %d",stat);
			WARN(ERROR_STRING);
			WARN("cancelling load");
			unlink(tmpname);
			givbuf(loc);
			goto finis;
		}
	}

	fp=try_open(tmpname,"r");
	if( !fp ) goto finis;


#ifdef SUN
	if( fread(&e1,sizeof(e1),1,fp) != 1 ){
		WARN("couldn't read exec");
		unlink(tmpname);
		goto finis;
	}
	if( e1.a_machtype != machtype ){
		WARN("wrong machine type");
		unlink(tmpname);
		goto finis;
	}
	if( N_BADMAG( e1 ) ){
		WARN("bad magic number");
		unlink(tmpname);
		goto finis;
	}
	if( e1.a_magic != OMAGIC ){
		WARN("magic number must be OMAGIC");
		unlink(tmpname);
		goto finis;
	}
	if( needsize != e1.a_text + e1.a_data + e1.a_bss ){
#ifdef DEBUG
		if( debug & binddebug ) 
			WARN("size does not agree after linking; reallocating");
#endif /* DEBUG */
		oldloc=start_loc;
		givbuf(loc);
		goto doit;
	}

	asize = e1.a_text + e1.a_data;
	if( fread( start_loc, 1, asize, fp ) != asize ){
		WARN("error reading program");
		goto finis;
	}

	/* zero bss */
	needsize -= asize ;	/* size of bss */
	start_loc += asize;
#ifdef DEBUG
if( debug & binddebug ) {
sprintf(ERROR_STRING,"zeroing %d bss bytes",needsize);
advise(ERROR_STRING);
}
#endif /* DEBUG */
	while( needsize-- ) *start_loc++ = 0;

	/* we don't read the symbol table, since usually we
	 * only want a few symbols anyway.
	 *
	 * Instead we get what we want later with nm & grep
	 */

	advise("Success!");

	/* Is it ok to delete the file here?
	 * this used to be a call to tmp_clean, which
	 * first made sure that tmpname wasn't the prog name!?
	 */
	unlink(tmpname);

#endif /* SUN */

finis:
	fclose(fp);
	return;
}


#endif /* DYNAMIC_LOAD */
