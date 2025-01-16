#!/bin/bash
# patch_lammps.sh [-e] /path/to/lammps/

do_e_mode=false

while getopts "he" option; do
   case $option in
      e)
         do_e_mode=true;;
      h) # display Help
         echo "patch_lammps.sh [-e] /path/to/lammps/"
         exit;;
   esac
done

# https://stackoverflow.com/a/9472919
shift $(($OPTIND - 1))
lammps_dir=$1

if [ "$lammps_dir" = "" ];
then
    echo "lammps_dir must be provided"
    exit 1
fi

if [ ! -d "$lammps_dir" ]
then
    echo "$lammps_dir doesn't exist"
    exit 1
fi

if [ ! -d "$lammps_dir/cmake" ]
then
    echo "$lammps_dir doesn't look like a LAMMPS source directory"
    exit 1
fi

# Check and produce nice message
if [ ! -f ./ML-CNMP/pair_CNMP.cpp ]; then
    echo "Please run `patch_lammps.sh` from the `ML-CNMP` root directory."
    #exit 1
fi

# Check for double-patch
if grep -q "find_package(Torch REQUIRED)" $lammps_dir/cmake/CMakeLists.txt ; then
    echo "This LAMMPS installation _seems_ to already have been patched; please check it!"
    #exit 1
fi

if [ "$do_e_mode" = true ]; then
    echo "Making source symlinks (-e) for pair_CNMP directory..."
    if [ -d "./ML-CNMP" ]; then
        # Ensure the target directory doesn't already have a symlink or directory with the same name
        if [ ! -e "$lammps_dir/src/ML-CNMP" ]; then
            ln -s "$(realpath -s ./ML-CNMP)" "$lammps_dir/src/ML-CNMP"
            echo "Symlink created."
        else
            echo "Target symlink or directory ML-CNMP already exists in the LAMMPS src directory."
            #exit 1
        fi
    else
        echo "The directory ML-CNMP does not exist in the current path."
        #exit 1
    fi 

    if [ -d "./CNMP" ]; then
        # Ensure the target directory doesn't already have a symlink or directory with the same name
        if [ ! -e "$lammps_dir/potentials/CNMP" ]; then
            ln -s "$(realpath -s ./CNMP)" "$lammps_dir/potentials/CNMP"
            echo "Symlink created."
        else
            echo "Target symlink or directory CNMP already exists in the LAMMPS src directory."
            #exit 1
        fi
    else
        echo "The directory CNMP does not exist in the current path."
        exit 1
    fi    

else
    echo "Copying pair_CNMP directory..."
    # Check if source directory exists and if the target directory does not exist
    if [ -d "./ML-CNMP" ] && [ ! -e "$lammps_dir/src/ML-CNMP" ]; then
        cp -r "./ML-CNMP" "$lammps_dir/src/ML-CNMP"
    else
        echo "Cannot copy: source directory ML-CNMP does not exist or target ML-CNMP already exists."
        exit 1
    fi

    if [ -d "./CNMP" ] && [ ! -e "$lammps_dir/potentials/CNMP" ]; then
        cp -r "./CNMP" "$lammps_dir/potentials/CNMP"
    else
        echo "Cannot copy: source directory CNMP does not exist or target CNMP already exists."
        exit 1
    fi

fi

echo "Updating CMakeLists.txt..."

# Update CMakeLists.txt
# C++11 에서 C++14로 업데이트
#sed -i "s/set(CMAKE_CXX_STANDARD 11)/set(CMAKE_CXX_STANDARD 14)/" $lammps_dir/cmake/CMakeLists.txt

# Add ML-CNMP to the list of packages
# sed -i "/set(STANDARD_PACKAGES/a \ \ ML-CNMP" $lammps_dir/cmake/CMakeLists.txt

# Add libtorch
# cat >> $lammps_dir/cmake/CMakeLists.txt << "EOF2"

#find_package(Torch REQUIRED)
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
#target_link_libraries(lammps PUBLIC "${TORCH_LIBRARIES}")
#EOF2

if [ -f "./CMakeLists.txt" ]; then
    mv "$lammps_dir/cmake/CMakeLists.txt" "$lammps_dir/cmake/CMakeLists.txt.bak"
    cp -rf "./CMakeLists.txt" "$lammps_dir/cmake/CMakeLists.txt"
fi

echo "Done!"