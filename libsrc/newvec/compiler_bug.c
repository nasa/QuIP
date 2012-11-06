/* This program demonstrates a bug discovered on IA64 machines...
 * gcc treats "long" as a 64 bit number, but won't let you left-shift
 * a bit more than 32 places...
 */

main()
{
	unsigned long long l;
	int s,nb;

	s = sizeof(int);
	printf("sizeof(int) = %d\n",s);

	s = sizeof(l);
	nb = 8 * s;
	printf("unsigned long has %d bytes (%d bits)\n",s,nb);

	/* l=1<<61; */
	l = 0x1000000000000000;
	printf("direct assignment:  l = 0x%lx\n",l);

	l = 1 << (nb-1);
	printf("shift bit %d #1:  l = 0x%lx\n",nb-1,l);

	l = 1 <<  63;
	printf("shift bit %d #2:  l = 0x%lx\n",63,l);
}

