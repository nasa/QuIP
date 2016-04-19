#!/bin/sh

# Generate a few files that are not part of the git repo

cd ../include
./update_quip_version.sh
cd ../macros/startup
./mac_build_startup.csh demo test
./build_startup_file.csh iquip test


