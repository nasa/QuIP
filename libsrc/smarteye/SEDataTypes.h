/**********************************************
* (C) Copyright 2008 Smart Eye AB
**********************************************/

#if !defined(SE_DATA_TYPES__INCLUDED_)
#define SE_DATA_TYPES__INCLUDED_

#ifdef WIN32
typedef unsigned __int64 SEu64;
#else
typedef unsigned long long SEu64;
#endif

typedef unsigned char SEu8;
typedef unsigned short SEu16;
typedef unsigned int SEu32;

typedef float SEf32;
typedef double SEf64;

#ifdef USE32BITFLOAT 
typedef SEf32 SEfloat; // if floats are 32 bits, currently only available for serial connection
#else
typedef SEf64 SEfloat;
#endif

struct SEPoint2D {
	SEfloat x;
	SEfloat y;
};

struct SEVect2D {
	SEfloat x;
	SEfloat y;
};

typedef struct {
	SEfloat x;
	SEfloat y;
	SEfloat z;
} SEPoint3D ;

struct SEVect3D {
	SEfloat x;
	SEfloat y;
	SEfloat z;
};

typedef struct {
	SEu16 size;
	char ptr[1024];
} SEString ;

struct SEWorldIntersectionStruct {
	SEPoint3D worldPoint;     // intersection point in world coordinates
	SEPoint3D objectPoint;    // intersection point in local object coordinates
	SEString objectName;      // name of intersected object
};


typedef enum {
	SEType_u8, 
	SEType_u16,
	SEType_u32,
	SEType_s32,
	SEType_u64,
	SEType_f64,
	SEType_Point2D,
	SEType_Vect2D,
	SEType_Point3D,
	SEType_Vect3D,
	SEType_String,
	SEType_Vector,
	SEType_Struct,
	SEType_WorldIntersection,
	SEType_WorldIntersections,
	SEType_PacketHeader,
	SEType_SubPacketHeader,
	SEType_f32
} SEType;

#ifdef USE32BITFLOAT 
#define	SEType_float SEType_f32
#else
#define SEType_float SEType_f64
#endif

/**
*	A header for the SmartEyePacket
*	- should be 64 bits ( 8 bytes )
* values for packetType:
* 1 - Smart Eye Pro 1.x
* 2 - Smart Eye Pro 2.0 - 2.3
* 3 - Smart Eye Pro 2.4 - 2.5
* 4 - Smart Eye Pro 3.0 - , generic format with subpackets
**/
#define PACKET_HEADER_SIZE 8
typedef struct
{
	SEu32 syncId;		      /**< always 'SEPD' */
	SEu16 packetType;     
	SEu16 length;         /**< number of bytes following this header, that is, not including size of this header  */
} SEPacketHeader;

#define SUB_PACKET_HEADER_SIZE 4
struct SESubPacketHeader
{
	SEu16 id;             /**< Output data identifier, refer to SEOutputDataIds for existing ids */
	SEu16 length;         /**< number of bytes following this header  */
};

#endif // SE_DATA_TYPES__INCLUDED_
