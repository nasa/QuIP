
#ifdef HAVE_PNG
extern FIO_UNCONV_FUNC(pngfio);
extern FIO_CONV_FUNC(pngfio);
extern FIO_OPEN_FUNC(pngfio);
extern FIO_CLOSE_FUNC(pngfio);
extern FIO_RD_FUNC(pngfio);
extern FIO_WT_FUNC(pngfio);
extern FIO_INFO_FUNC(pngfio);
extern FIO_SEEK_FUNC(pngfio);

extern void set_bg_color(int bg_color);
extern void _set_color_type(QSP_ARG_DECL  int given_color_type);
	
#else /* !HAVE_PNG */

#ifdef BUILD_FOR_IOS
// Why not just define HAVE_PNG in the ios config file???  BUG?
extern FIO_UNCONV_FUNC(pngfio);
extern FIO_CONV_FUNC(pngfio);
extern FIO_OPEN_FUNC(pngfio);
extern FIO_CLOSE_FUNC(pngfio);
extern FIO_RD_FUNC(pngfio);
extern FIO_WT_FUNC(pngfio);
extern FIO_INFO_FUNC(pngfio);
extern FIO_SEEK_FUNC(pngfio);
#endif // BUILD_FOR_IOS

#endif /* !HAVE_PNG */

