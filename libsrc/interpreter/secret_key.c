/* the function gen_string creates the string used as the secret key.
 * We don't want to distribute this with the open source distribution,
 * rather we should have another version for public distribution that
 * uses a different key.
 */

#define ADD_CHAR(c)				\
						\
	if( i >= buf_len ) NERROR1("gen_string:  buffer too small!?");	\
	buf[i++] = c;

void gen_string(char *buf,int buf_len)
{
	int i;

	/* 'I am the one true root (of all evil)' */

	i=0;
	ADD_CHAR('I');		// 0	I
	ADD_CHAR(' ');		// 1
	ADD_CHAR('a');		// 2	a
	ADD_CHAR('m');		// 3	m
	ADD_CHAR(buf[1]);	// 4
	ADD_CHAR('t');		// 5	t
	ADD_CHAR('h');		// 6	h
	ADD_CHAR('e');		// 7	e
	ADD_CHAR(buf[4]);	// 8
	ADD_CHAR('o');		// 9	o
	ADD_CHAR('n');		// 10	n
	ADD_CHAR(buf[7]);	// 11	e
	ADD_CHAR(buf[8]);	// 12
	ADD_CHAR(buf[5]);	// 13	t
	ADD_CHAR('r');		// 14	r
	ADD_CHAR('u');		// 15	u
	ADD_CHAR(buf[7]);	// 16	e
	ADD_CHAR(buf[12]);	// 17
	ADD_CHAR(buf[14]);	// 18	r
	ADD_CHAR(buf[9]);	// 19	o
	ADD_CHAR(buf[19]);	// 20	o
	ADD_CHAR(buf[13]);	// 21	t
	ADD_CHAR(buf[17]);	// 22
	ADD_CHAR('(');		// 23	(
	ADD_CHAR(buf[19]);	// 24	o
	ADD_CHAR('f');		// 25	f
	ADD_CHAR(buf[17]);	// 26
	ADD_CHAR(buf[2]);	// 27	a
	ADD_CHAR('l');		// 28	l
	ADD_CHAR('l');		// 29	l
	ADD_CHAR(buf[26]);	// 30
	ADD_CHAR(buf[11]);	// 31	e
	ADD_CHAR('v');		// 32	v
	ADD_CHAR('i');		// 33	i
	ADD_CHAR(buf[28]);	// 34	l
	ADD_CHAR(')');		// 35	)
	ADD_CHAR(0);		// 36	terminate string
}

