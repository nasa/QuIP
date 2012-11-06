#include "quip_config.h"

char VersionId_viewer_rdplot[] = QUIP_VERSION_STRING;

#ifdef HAVE_X11

#include <stdio.h>

#ifdef HAVE_STRING_H
#include <string.h>
#endif

#include "viewer.h"
#include "data_obj.h"
#include "xsupp.h"

void getone( FILE *fp, int *p )
{
	int i;

	i=getc(fp);
	i |= getc(fp) << 8;
	*p = i;
}

void getpair( FILE *fp, int *px, int *py )
{
	getone(fp,px);
	getone(fp,py);
}

void rdplot(QSP_ARG_DECL  FILE *fp )
{
	int x1,x2,x3,y1,y2,y3,c;
	char modstr[32];
	char *s;

	if( !fp ) return;

	while( (c=getc(fp)) != EOF ){
		switch( c ){
			case 'e': xp_erase(); break;
			case 's':
				getpair(fp,&x1,&y1);
				getpair(fp,&x2,&y2);
				xp_space(x1,y1,x2,y2);
				break;
			case 't':
				{
					char labelstring[256];
					int i=0;

					while( (c=getc(fp)) != '\n' && c!=EOF ){
						labelstring[i++]=c;
					}
					labelstring[i]=0;
					xp_text(labelstring);
				}
				break;
			case 'l':
				getpair(fp,&x1,&y1);
				getpair(fp,&x2,&y2);
				xp_line(x1,y1,x2,y2);
				break;
			case 'c':
				getpair(fp,&x1,&y1);
				getone(fp,&x2);
				xp_move(x1,y1);
				xp_circle(x2);
				break;
			case 'a':
				getpair(fp,&x1,&y1);
				getpair(fp,&x2,&y2);
				getpair(fp,&x3,&y3);
				xp_arc(x1,y1,x2,y2,x3,y3);
				break;
			case 'm':
				getpair(fp,&x1,&y1);
				xp_move(x1,y1);
				break;
			case 'n':
				getpair(fp,&x1,&y1);
				xp_cont(x1,y1);
				break;
			case 'p':
				getpair(fp,&x1,&y1);
				xp_point(x1,y1);
				break;
			case 'f':
				s=modstr;
				while((c=getc(fp)) != '\n' && c != EOF )
					*s++ = c;
				*s=0;
				if( !strcmp(modstr,"solid") )
					xp_select(1);
				else if( !strcmp(modstr,"dotted") )
					xp_select(2);
				else if( !strcmp(modstr,"dotdashed") )
					xp_select(3);
				else if( !strcmp(modstr,"shortdashed") )
					xp_select(4);
				else if( !strcmp(modstr,"longdashed") )
					xp_select(5);
				else WARN("unsupported line color");
				break;
			default:
				sprintf(error_string,
				"unrecognized plot command '%c' (%o)",c,c);
				NWARN(error_string);
				goto plotdun;
		}
	}
plotdun:
	fclose(fp);
}


#endif /* HAVE_X11 */

