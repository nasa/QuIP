

void inhibit_sigs(void);
#ifdef PC
void sigpush(int sig,void (cdecl *action)(int));
#else
void sigpush(int sig,void (*action)(int));
//void sigpush(int sig,void (*action)());
#endif /* PC */
void sigpop(int sig);

