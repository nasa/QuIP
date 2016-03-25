/*
 * The short program dumps the flags in the ctype array,
 * so we know how to program the GPU versions of fucntions
 * like iscntrl etc
 */

#include <stdio.h>
#include <ctype.h>

int main(int ac, char **av)
{
	int c;

	printf("CHAR	CNTRL	SPACE	BLANK\n");
	for(c=0;c<128;c++){
		printf("0x%x\t%d\t%d\t%d\n",c,iscntrl(c),isspace(c),isblank(c));
	}
}

