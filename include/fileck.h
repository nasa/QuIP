#ifndef _FILECK_H_
#define _FILECK_H_

#ifdef __cplusplus
extern "C" {
#endif

extern int path_exists(const char *name);
extern int directory_exists(const char *dirname);
extern int regfile_exists(const char *dirname);
extern int can_write_to(const char *name);
extern int can_read_from(const char *name);
extern int file_exists(const char *pathname);

#ifdef __cplusplus
}
#endif

#endif /* ! _FILECK_H_ */


