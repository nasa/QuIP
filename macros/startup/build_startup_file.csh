#!/bin/csh

if( $#argv != 1 && $#argv != 2 ) then
  echo 'usage:  ./build_startup_file.csh demo|ezjet|pvt|csf|iquip|fdm [test]'
  exit
endif

set flavor=$1
if( $flavor != demo && $flavor != ezjet && $flavor != csf && $flavor != pvt && $flavor != iquip && $flavor != fdm ) then
  echo './build_startup_file.csh:  bad flavor requested:  $flavor'
  echo 'usage:  ./build_startup_file.csh <flavor> [test]'
  echo 'known flavors:  demo ezjet csf pvt iquip fdm'
  exit
endif

set out_stem=${flavor}_startup

set test_mode=0
if( $#argv == 2 ) then
  if( $2 == test ) then
    set test_mode=1
  else
    echo 'usage:  ./build_startup_file.csh <flavor> [test]'
    echo 'Second argument must be "test" if present'
    exit
  endif
endif

# Make a single encrypted file containing the startup macros
# and the startup script.
#
# We begin with quip_startup.scr

set outfile=$out_stem.scr
set encfile=$out_stem.enc

cat < /dev/null > $outfile

echo "If var_exists(startup_file_read) 'exit_file'" >> $outfile
#echo 'advise "RESOURCE_DIR = $RESOURCE_DIR"' >> $outfile

# macros from system subdir
set subdir=system
set suffix=mac	# can change from default
set file_list=( builtin msgs vars files ios_helper )
source add_files.csh

set subdir=data
#size.scr
set file_list=( ascii decl hips )
source add_files.csh

# files from gencomp.scr
set subdir=compute
set file_list=( funcs chains )
source add_files.csh

set subdir=view
# dpysize.scr
set file_list=( common luts view ios_plot plotsupp )
source add_files.csh

set subdir=gui
set file_list=( ui )
source add_files.csh

set subdir=numrec
set file_list=( fit_polynomial svd norm_coords )
source add_files.csh

cat >> $outfile << EOF

Start_File '(included text)'

If !var_exists(DISPLAY_WIDTH)
  "Set DISPLAY_WIDTH 200 Set DISPLAY_HEIGHT 200"

#max_warnings -1

EOF
# the max_warnings line was in the fragment above...
#echo "advise 'startup.scr: max_warnings set to -1 in build_startup_file.csh'" >> $outfile

# Add the version identifier - make sure that this matches the rest of the code...
# BUG added --tags because current tag is not "annotated" ???
echo "Set script_version" `git describe --tags` >> $outfile
echo Check_Version_Match >> $outfile
echo Stop_File "'(included text)'" >> $outfile

# BUG?  iquip has $iquip_test_mode ???
# change it to plain test_mode for now...
if( $test_mode != 0 ) then
  echo "Set test_mode 1" >> $outfile
else
  echo "Set test_mode 0" >> $outfile
endif

# BUG these should go in flavor-specific section!
#set subdir=demo
#set file_list=( http anim accel graph paint console gui_demo fio draw view_test port_test encrypt calib arrows csf )
#source add_files.csh

if( $flavor == ezjet || $flavor == pvt ) then
#echo verbose yes >> $outfile	# for debugging
#echo debug query >> $outfile	# for debugging
  set subdir=pvt
  set file_list=( pvt pvt_params pvt_plot pvt_dist pre_pvt )
  source add_files.csh
  #set subdir=sound
  #set file_list=( sound )
  #source add_files.csh
  source add_file.csh sound/sound.mac
endif

# Comment out these two lins to test on unix...
if( $flavor == demo ) then
  # these files were added to make the old demos work:
  set subdir=demo
  #cat ../../macros/demo/login.mac >> $outfile
  #cat ../../macros/demo/gui_demo.mac >> $outfile
  #cat ../../macros/demo/graph.mac >> $outfile
  #cat ../../macros/demo/accel.mac >> $outfile
  #cat ../../macros/demo/draw.mac >> $outfile
  #cat ../../macros/demo/anim.mac >> $outfile
  #cat ../../macros/demo/paint.mac >> $outfile
  #cat ../../macros/demo/console.mac >> $outfile
  #cat ../../macros/demo/view_test.mac >> $outfile
  #cat ../../macros/demo/fio.mac >> $outfile
  set file_list=( login gui_demo graph accel draw anim paint console view_test fio )
  source add_files.csh

  #set subdir=ios
  ##cat ../../macros/ios/utilz.mac >> $outfile
  #set file_list=( utilz )
  #source add_files.csh
  source add_file.csh ios/utilz.mac

  #cat ../../macros/demo/demo.scr >> $outfile
  source add_file.csh demo/demo.scr
else if( $flavor == iquip ) then
  #set macro_dir=ios
  set subdir=ios
  set file_list=( sw_update console server_config cache )
  source add_files.csh
  #foreach file ( sw_update console server_config cache )
  #  echo "# FILE $macro_dir/$file.$suffix BEGIN" >> $outfile
  #  cat ../../macros/$macro_dir/$file.$suffix >> $outfile
  #  echo "# FILE $macro_dir/$file.$suffix END" >> $outfile
  #end

  set subdir=iquip
  #foreach file ( iquip_admin clean setup_iquip iquip )
  #  echo "# FILE $macro_dir/$file.mac BEGIN" >> $outfile
  #  cat ../../macros/$macro_dir/$file.$suffix >> $outfile
  #  echo "# FILE $macro_dir/$file.$suffix END" >> $outfile
  #end
  set file_list=( iquip_admin clean setup_iquip iquip )
  source add_files.csh

  source add_file.csh iquip/files.scr
  source add_file.csh iquip/iquip.scr
else if( $flavor == pvt ) then
  #cat ../../macros/pvt/pvt_main.scr >> $outfile
  source add_file.csh pvt/pvt_main.scr
else if( $flavor == csf ) then
  set subdir=demo
  set file_list=( login gui_demo graph )
  source add_files.csh

  #cat ../../macros/demo/login.mac >> $outfile
  #cat ../../macros/demo/gui_demo.mac >> $outfile
  #cat ../../macros/demo/graph.mac >> $outfile

  set subdir=csf
  set file_list=( csf calib arrows csf_cam sync_files csf_admin psych )
  source add_files.csh

  #cat ../../macros/csf/csf.mac >> $outfile
  #cat ../../macros/csf/calib.mac >> $outfile
  #cat ../../macros/csf/arrows.mac >> $outfile
  #cat ../../macros/csf/csf_cam.mac >> $outfile
  #cat ../../macros/csf/sync_files.mac >> $outfile

  set subdir=ios
  set file_list=( utilz cache console )
  source add_files.csh

  #cat ../../macros/ios/utilz.mac >> $outfile
  #cat ../../macros/ios/cache.mac >> $outfile
  #cat ../../macros/ios/console.mac >> $outfile

  #cat ../../macros/pvt/pvt.mac >> $outfile
  #cat ../../macros/pvt/dashboard.mac >> $outfile
  #cat ../../macros/pvt/pvt_params.mac >> $outfile
  #cat ../../macros/pvt/pvt_dist.mac >> $outfile
  #cat ../../macros/pvt/pvt_plot.mac >> $outfile
  set subdir=pvt
  set file_list=( pvt dashboard pvt_params pvt_dist pvt_plot )
  source add_files.csh

  #cat ../../macros/data/string_edit.mac >> $outfile
  #cat ../../macros/data/set_sizes.mac >> $outfile
  set subdir=data
  set file_list=( string_edit set_sizes )
  source add_files.csh

  source add_file.csh compute/rdp.mac
  source add_file.csh view/opt_ht.mac

#  cat ../../macros/compute/rdp.mac >> $outfile
#  cat ../../macros/view/opt_ht.mac >> $outfile
  #cat ../../macros/csf/csf_admin.mac >> $outfile	# csf
  #cat ../../macros/csf/csf_main.scr >> $outfile
  source add_file.csh csf/csf_main.scr

else if( $flavor == ezjet ) then
  if( $test_mode != 0 ) then
    echo "Set ezjet_test_mode 1" >> $outfile
    cat ../../macros/ezjet/alert_tests.mac >> $outfile
  endif

  set subdir=demo
  set file_list=( demo_util admin )
  source add_files.csh

  set subdir=ios
  set file_list=( utilz console )
  source add_files.csh

  set subdir=ezjet
  set file_list=( questionnaire ezj_qs admin unimp sync_files sw_update participant sleep_diary workload )
  source add_files.csh

  echo "# FILE ezjet/ezjet_init.scr BEGIN" >> $outfile
  cat ../../macros/ezjet/ezjet_init.scr >> $outfile

  #And then the questionnaires:
  # we set the variable to 0 if we want to test a different part
  # of the system, and want to have a shorter start-up file.
  set  load_questionnaires=1
  if( $load_questionnaires ) then
    foreach qfile ( demog cis morn_eve ess Samn-Perelli countermeasures commute_time flying tlx PVT_Distraction workload wl2 sleep_diary_morn sleep_diary_eve sleep_diary_nap sectornum )
      echo "# FILE ezjet/$qfile.scr BEGIN" >> $outfile
      cat ../../macros/ezjet/$qfile.scr >> $outfile
      #cp ../../macros/ezjet/$qfile.scr ~/txt_files/$qfile.txt
    end
    foreach qfile ( skip_pvt )
      echo "# FILE pvt/$qfile.scr BEGIN" >> $outfile
      cat ../../macros/pvt/$qfile.scr >> $outfile
      #cp ../../macros/pvt/$qfile.scr ~/txt_files/$qfile.txt
    end
  else
    echo Warning:  NOT loading questionnaire definitions...
  endif

#  echo "# FILE ios/udids.scr BEGIN" >> $outfile
#  cat ../../macros/ios/udids.scr >> $outfile
  echo "# FILE ezjet/ezjet.scr BEGIN" >> $outfile
  cat ../../macros/ezjet/ezjet.scr >> $outfile
else if( $flavor == fdm ) then
  echo "Set plot_locale ios" >> $outfile
  cat $HOME/exps/uhco13/stimuli/ios_twinkle.mac >> $outfile
  cat $HOME/exps/uhco13/stimuli/tgrat.mac >> $outfile
  cat ../../macros/compute/gaussian.mac >> $outfile
  cat $HOME/exps/uhco13/stimuli/fdm_demos.mac >> $outfile
  cat $HOME/exps/uhco13/stimuli/fdm_ipad.scr >> $outfile
else
  echo "Sorry, unhandled flavor $flavor"
  exit
endif

echo "Set startup_file_read 1" >> $outfile

echo File $outfile complete, encrytping...

quip $outfile $encfile < encrypt_file.scr
#/bin/rm $outfile

# This make sure that we have a current version listing in the executable code
#cd ../../include
#./update_quip_version.sh .



