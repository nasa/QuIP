#include "quip_config.h"

#include "query.h"
#include "smarteye_api.h"
#include "smarteye.h"

#include <string.h>		/* memset */
#include <unistd.h>		/* close */

#if defined (_WIN32)

#include <winsock2.h>

#define sockLibInit() initWinSock()
#define sockLibCleanup() WSACleanup()
#define getsockerror() WSAGetLastError()

#else /* ! _WIN32 */

#include <sys/types.h>
#include <sys/socket.h>
#include <netdb.h>
#include <errno.h>

#define sockLibInit()
#define sockLibCleanup()
#define getsockerror() errno
#define closesocket close

#endif /* ! _WIN32 */

//#include <algorithm>
//#include <iostream>

#include <ctype.h>

const int defaultPortUDP = 5001;  /// This is the port number you choose in Smart Eye Pro
const int defaultPortTCP = 5002;  /// This is the port number you choose in Smart Eye Pro

static int socketConnection=(-1);  
static char *packet_buf=NULL;
static int packet_bufsize=0;

#ifdef FOOBAR


/// This is an example function of different ways to handle packet data
/// Modify this function or write your own handler that is called from main function
void userPacketHandler(char* pPacket, const SEu16& packetType)
{

	// This function only cares about packets of type 4
	if (packetType != 4)
		return;

	std::cout << "User packet handler function...\n";

	// Example 1:
	// Look up subpacket of id = SEHeadPosition
	SEPoint3D headPosition;
	if (findDataInPacket(SEHeadPosition, pPacket, headPosition))
		std::cout << "Headposition [" << headPosition.x << ", " << headPosition.y << ", " << headPosition.z << "]\n";
	else
		std::cout << "Did not find SEHeadPosition in data, check packet contents in Smart Eye Pro\n";


	// Example 2:
	// Look up subpacket of id = SEEyelidOpening
	SEf64 eyelidOpening;
	if (findDataInPacket(SEEyelidOpening, pPacket, eyelidOpening))
		std::cout << "Eyelid opening : " << eyelidOpening << "\n";
	else
		std::cout << "Did not find SEEyelidOpening in data, check packet contents in Smart Eye Pro\n";


	// Example 3:
	// Get closest intersection point
	SEWorldIntersectionStruct worldIntersection;
	if (findDataInPacket(SEClosestWorldIntersection, pPacket, worldIntersection)) {
		if (strcmp(worldIntersection.objectName.ptr, "Screen.Surface") != -1) {
			double x = worldIntersection.objectPoint.x;
			double y = worldIntersection.objectPoint.y;
			std::cout << "Screen.Surface intersected at [" << x << ", " << y << "]\n";
		}
	}

	// Example 4:
	// Get all closest intersection point
	SEWorldIntersectionStruct my_worldIntersection[10];
	SEu16 numberOfIntersections;
	if (findDataInPacket(SEAllWorldIntersections, pPacket, my_worldIntersection, numberOfIntersections)) {
		numberOfIntersections = numberOfIntersections<10?numberOfIntersections:10;
		for (int i = 0; i < numberOfIntersections; i++) {
			std::cout << "Intersection " << i << " ";
			std::cout <<  my_worldIntersection[i].objectName.ptr;
			double x = worldIntersection.objectPoint.x;
			double y = worldIntersection.objectPoint.y;
			double z = worldIntersection.objectPoint.z;
			std::cout << " intersected at [" << x << ", " << y << ", " << z << "]\n";
		}
	}

	std::cout << "\n\n";
}
#endif /* FOOBAR */



int connectToSocket(QSP_ARG_DECL  const char *hostname, unsigned short portnum)
{
	struct hostent * hostInfo = gethostbyname(hostname);
	struct sockaddr_in sockAddr;

	if (hostInfo == NULL){
		sprintf(ERROR_STRING,"Unable to look up host '%s'!?",hostname);
		WARN(ERROR_STRING);
		return -1;
	}

	// Prepare socket address
	memset(&sockAddr, 0, sizeof(sockAddr));
	memcpy((char*)&sockAddr.sin_addr, hostInfo->h_addr, hostInfo->h_length);   /* set address */
	sockAddr.sin_family = hostInfo->h_addrtype;
	sockAddr.sin_port = htons((u_short)portnum);

	// Create socket
	unsigned int s = socket(hostInfo->h_addrtype, SOCK_STREAM, 0);
	if( s == -1 ){
		tell_sys_error("socket");
		WARN("connectToSocket:  unable to create socket!?");
		return -1;
	}

	// Try to connect to the specified socket
	int result = connect(s, (struct sockaddr *)&sockAddr, sizeof sockAddr);
	if (result == -1) {
#if defined(_WIN32)
		int err = WSAGetLastError();
#else
		//int err = getsockerror();
#endif
		closesocket(s);
		WARN("connectToSocket:  unable to connect!?");
		return -1;
	}
	return s;
}


/// This function initialized a socket connection with the given
/// host and port

unsigned int bindToSocket(QSP_ARG_DECL  unsigned short portnum)
{
	// Prepare socket address
	struct sockaddr_in sockAddr;
	memset(&sockAddr, 0, sizeof(sockAddr));
	sockAddr.sin_addr.s_addr = htonl(INADDR_ANY);
	sockAddr.sin_family = AF_INET;
	sockAddr.sin_port = htons((u_short)portnum);

	// Create socket
	unsigned int s = socket(AF_INET, SOCK_DGRAM, 0);
	if (s == -1){
		WARN("bindToSocket:  unable to create socket!?");
		return -1;
	}

	// Try to bind to the specified socket
	int result = bind(s, (struct sockaddr *)&sockAddr, sizeof sockAddr);
	if (result == -1) {
		int err;
		err = getsockerror();
		closesocket(s);
		WARN("bindToSocket:  unable to bind socket!?");
		return -1;
	}
	return s;
}

void printUsage()
{
	advise( "\nSocketClient\n(C) Copyright 2009 Smart Eye AB\n\n");
	advise( "Usage: SocketClient UDP|TCP <port> <hostname>\n\n");
	
	advise( "UDP or TCP has to be specified.\n");
	advise( "Default port for UDP is 5001 and default port for TCP is 5002\n");
	advise( "Default host is 127.0.0.1\n");
}


static COMMAND_FUNC( do_se_connect )
{
	const char *hostname;
	int port;

	hostname = NAMEOF("SmartEye host name");
	port = HOW_MANY("port number");

	if( socketConnection != (-1) ){
		WARN("Already connected to SmartEye server!?");
		return;
	}

	socketConnection = connectToSocket(QSP_ARG  hostname, port);

	if (socketConnection < 0) {
		WARN("Could not connect to SmartEye server");
		return;
	}
}

static COMMAND_FUNC( do_se_recv )
{
	// Wait for packet arrival
	// Do this by peeking for packet of size packet header, which is the smallest packet that may arrive
	SEPacketHeader packetHeader;
 
	// MSG_PEEK means that we can read the header again...
	// But why bother?
	int received = recv(socketConnection,
		(char *)&packetHeader, sizeof(SEPacketHeader), MSG_PEEK);

	// Something did arrive, socket is blocking so it would not return until packet arrived
	// Check that packet at least is the size of a packet header

#if defined(_WIN32)
	if(received == -1) {
		int err = getsockerror();
		if(err == 10040) {
			//winsock reports error when receive buffer is smaller than packet, but fills the full buffer
			received = sizeof(SEPacketHeader);
		}
	}
#endif
	if (received != (int)sizeof(SEPacketHeader)) {
		if (received == 0){
			WARN("Socket connection was closet by the SmartEye server");
		} else {
			int err = getsockerror();
			sprintf(ERROR_STRING,"SmartEye communication failure, error = %d",err);
			WARN(ERROR_STRING);
		}

		sockLibCleanup();
		closesocket(socketConnection);
		socketConnection=(-1);	// so code knows no connection now
		return;
	}


	// Interpret packet header
	// readValue function handles the reversed byte order and copies data into packet header
	//int pos = 0;
	 
	int packetSize = sizeof(SEPacketHeader) + packetHeader.length;
	// header.length does not include size of header


	// Allocate memory space for packet to be received
	// BUG we probably don't want to do this every time,
	// we should keep a buffer
	// around and only grow when needed
	if( packetSize > packet_bufsize ){
		if( packet_buf != NULL )
			free(packet_buf);
		packet_buf=malloc(packetSize);
		packet_bufsize=packetSize;
	}
		
	// Receive the packet from stream
	received = recv(socketConnection, packet_buf, packetSize, 0);
	if (received != packetSize) {
		int err;
		err = getsockerror();
		sprintf(ERROR_STRING,"SmartEye communication failure, error = %d",err);
		WARN(ERROR_STRING);
		sockLibCleanup();
		return;
	}

	// Print some general packet information
	if( verbose ){
		sprintf(ERROR_STRING,"Packet type %8d, Total size %8d, Header size %8ld, Data size %8d",
			packetHeader.packetType, packetSize,sizeof(SEPacketHeader),packetHeader.length );

		// Print packet contents
		printPacketContents(QSP_ARG  packet_buf, packetHeader.packetType);
	}

	// Call user handler function with new packet
	//userPacketHandler(packet_buf, packetHeader.packetType);

	// Add other handler functions here if you like
}


#if defined(_WIN32)
int initWinSock(void)
{
	// Initialize the Windows socket library
	WSADATA info;
	if (WSAStartup(0x0002, &info) == SOCKET_ERROR) {
		std::cerr << "Could not initialize socket library.\n";
		exit(1);
	}
	return 0;
}
#endif

#ifdef FOOBAR
/// The following is an example of connecting to a socket and receiving packets
/// Feel free to copy parts of this code into your own client or use this code
/// to build your client from.
int main(int argc, char* argv[])
{
	if (argc == 1) {
		printUsage();
		return 0;
	}

	int arg = 1;  
	bool useUDP = true;
	// Check for UDP/TCP flag
	if (strncmp(argv[arg], "TCP", 3) == 0 || strncmp(argv[arg], "tcp", 3) == 0) 
		useUDP = false;
	else if (strncmp(argv[arg], "UDP", 3) != 0 && strncmp(argv[arg], "udp", 3) != 0) 
	{
		printUsage();
		return 0;
	}
	arg++;

	sockLibInit();

	// Get server name and port number, either default or from command prompt
	unsigned int socketConnection;  
	int portNo;
	if(useUDP)
	{
		portNo = argc > arg ? atoi(argv[arg]) : defaultPortUDP;
		arg++;
		const char * Name = argc > arg ? argv[arg] : defaultHost; // optional command line parameter is tracker host name (default localhost)

		std::cout << "Listening for UDP data on port " << portNo << "\n";
		socketConnection = bindToSocket(portNo);
	}
	else
	{
		portNo = argc > arg ? atoi(argv[arg]) : defaultPortTCP;
		arg++;
		const char * Name = argc > arg ? argv[arg] : defaultHost; // optional command line parameter is tracker host name (default localhost)

		std::cout << "Trying to connect to " << Name << " on port " << portNo << "\n";
		socketConnection = connectToSocket(Name, portNo);
	}

	if (socketConnection == -1) {
		std::cerr << "Could not connect to server\n";
		sockLibCleanup();
		return 1;
	}

	// Infinite loop that handles packet
	// 1. Wait for packet
	// 2. Check for errors
	// 3. Check packet type
	// 4. Interpret contents of packet(print content and call user function)
	// 5. Free packet and return to waiting for packets
std::cerr << "Entering main loop...\n";
	while (true)  {

		// Wait for packet arrival
		// Do this by peeking for packet of size packet header, which is the smallest packet that may arrive
		SEPacketHeader tempPacketHeader;
 
		int received = recv(socketConnection, (char *)&tempPacketHeader, sizeof(SEPacketHeader), MSG_PEEK);

		// Something did arrive, socket is blocking so it would not return until packet arrived
		// Check that packet at least is the size of a packet header

#if defined(_WIN32)
		if(received == -1) {
			int err = getsockerror();
			if(err == 10040) {
				//winsock reports error when receive buffer is smaller than packet, but fills the full buffer
				received = sizeof(SEPacketHeader);
			}
		}
#endif
		if (received != (int)sizeof(SEPacketHeader)) {
			if (received == 0)
	std::cerr << "Socket connection was closet by server\n";
			else {
				int err = getsockerror();
	std::cerr << "Communication failure, error = " << err << "\n";
			}

			sockLibCleanup();
			closesocket(socketConnection);
			return 1;
		}


		// Interpret packet header
		// readValue function handles the reversed byte order and copies data into packet header
		int pos = 0;
	 
		SEPacketHeader packetHeader;
		readValue(packetHeader, (char*)&tempPacketHeader, pos);
		int packetSize = sizeof(SEPacketHeader) + packetHeader.length;    // header.length does not include size of header


		// Allocate memory space for packet to be received
		char* pPacket = (char*) malloc(packetSize);

		
		// Receive the packet from stream
		received = recv(socketConnection, pPacket, packetSize, 0);
		if (received != packetSize) {
			int err;
			err = getsockerror();
			std::cerr << "Communication failure, error = " << err << "\n";
			sockLibCleanup();
			return 1;
		}

		// Print some general packet information
		std::cout << "Packet type = " << packetHeader.packetType << ", Total size = " << packetSize << " (Header size = " << sizeof(SEPacketHeader) << " + Data size = " << packetHeader.length << ")\n";

		// Print packet contents
		printPacketContents(pPacket, packetHeader.packetType);

		// Call user handler function with new packet
		userPacketHandler(pPacket, packetHeader.packetType);

		// Add other handler functions here if you like



		// Free the memory allocated for the packet
		free(pPacket);

	}// One iteration of infinite loop ends here

	return 0;
}
#endif /* FOOBAR */

#define ADD_CMD(s,f,h)	ADD_COMMAND(smarteye_ctbl,s,f,h)

MENU_BEGIN(smarteye)
ADD_CMD(connect,do_se_connect,connect to SmartEye server)
ADD_CMD(recv,do_se_recv,receive a packet from the SmartEye server)
MENU_END(smarteye)

COMMAND_FUNC(smarteye_menu)
{
	PUSH_MENU(smarteye)
}

