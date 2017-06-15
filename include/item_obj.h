#ifndef _ITEM_OBJ_H_
#define _ITEM_OBJ_H_

//struct item_type;
//typedef struct item_type Item_Type;

struct item {
	const char *	item_name;
	Item_Context *	item_icp;

#ifdef BUILD_FOR_OBJC
	uint64_t	item_magic;
#define QUIP_ITEM_MAGIC	0x8765432187654321
#endif /* BUILD_FOR_OBJC */

};

/* Item */
#define ITEM_NAME(ip)		(ip)->item_name
#define SET_ITEM_NAME(ip,s)	(ip)->item_name = s

#ifdef BUILD_FOR_OBJC
#define ITEM_MAGIC(ip)		(ip)->item_magic
#define SET_ITEM_MAGIC(ip,v)	(ip)->item_magic = v
#endif /* BUILD_FOR_OBJC */

#define ITEM_CTX(ip)		(ip)->item_icp
#define SET_ITEM_CTX(ip,icp)	(ip)->item_icp = icp

#define ITEM_TYPE(ip)		CTX_IT(ITEM_CTX(ip))

#endif /* ! _ITEM_OBJ_H_ */

