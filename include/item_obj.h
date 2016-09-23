#ifndef _ITEM_OBJ_H_
#define _ITEM_OBJ_H_

struct item_type;
typedef struct item_type Item_Type;

typedef struct item {
	const char *	item_name;
#ifdef ITEMS_KNOW_OWN_TYPE
	Item_Type *	item_itp;
#endif /* ITEMS_KNOW_OWN_TYPE */

#ifdef BUILD_FOR_OBJC
	uint64_t	item_magic;
#define QUIP_ITEM_MAGIC	0x8765432187654321
#endif /* BUILD_FOR_OBJC */

} Item;

/* Item */
#define ITEM_NAME(ip)		(ip)->item_name
#define SET_ITEM_NAME(ip,s)	(ip)->item_name = s

#ifdef BUILD_FOR_OBJC
#define ITEM_MAGIC(ip)		(ip)->item_magic
#define SET_ITEM_MAGIC(ip,v)	(ip)->item_magic = v
#endif /* BUILD_FOR_OBJC */

#ifdef ITEMS_KNOW_OWN_TYPE
#define ITEM_TYPE(ip)		(ip)->item_itp
#define SET_ITEM_TYPE(ip,itp)	(ip)->item_itp = itp
#endif /* ITEMS_KNOW_OWN_TYPE */

#define NO_ITEM	((Item *) NULL)


#endif /* ! _ITEM_OBJ_H_ */

