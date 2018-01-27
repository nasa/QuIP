//
//  main.m
//  MacQuIP
//
//  Created by Jeff Mulligan on 9/10/13.
//  Copyright (c) 2013 Jeff Mulligan. All rights reserved.
//

#import <Cocoa/Cocoa.h>

#include "quip_prot.h"
#include "quipAppDelegate.h"

#include "quip_start_menu.c"

int main(int argc, char *argv[])
{
	//return NSApplicationMain(argc, (const char **)argv);
    [NSApplication sharedApplication];
    printf("main:  NSApp created, ready to run\n");
    
	init_quip_menu();
	start_quip_with_menu(argc,argv,quip_menu);

    
    quipAppDelegate *qad=[[quipAppDelegate alloc] init];
    printf("qad = 0x%lx\n",(long)qad);
    
    [NSApp setDelegate:qad];
    
    NSMenu *main_menu = [NSApp mainMenu];
    printf("before calling run, mainMenu = 0x%lx\n",(long)main_menu);
    [NSApp run];
    main_menu = [NSApp mainMenu];
    printf("after calling run, mainMenu = 0x%lx\n",(long)main_menu);
    
}
