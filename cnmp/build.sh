#!/usr/bin/env bash

usage() {
    echo "Usage: $0 --path <CNMP_lammps_directory>"
    exit 1
}

CONFIG_FILE=""
CHECKPOINT_FILE=""
CNMP_LAMMPS_PATH=""

while [[ $# -gt 0 ]]; do
  case $1 in
    --path)
      CNMP_LAMMPS_PATH="$2"
      shift 2
      ;;
    --config)
      CONFIG_FILE="$2"
      shift 2
      ;;
    --checkpoint)
      CHECKPOINT_FILE="$2"
      shift 2
      ;;
    *)
      echo "Error: Unknown option $1"
      usage
      ;;
  esac
done

if [ -z "$CNMP_LAMMPS_PATH" ]; then
  echo "Error: --path 인자를 지정해야 합니다."
  usage
fi

if [ -n "$CONFIG_FILE" ]; then
  CONFIG_DIR="$(dirname "$0")/CNMP/config"
  mkdir -p "$CONFIG_DIR"
  cp "$CONFIG_FILE" "$CONFIG_DIR"
  echo "Config file copied to $CONFIG_DIR"
fi

if [ -n "$CHECKPOINT_FILE" ]; then
  CHECKPOINT_DIR="$(dirname "$0")/CNMP/checkpoint"
  mkdir -p "$CHECKPOINT_DIR"
  cp "$CHECKPOINT_FILE" "$CHECKPOINT_DIR"
  echo "Checkpoint file copied to $CHECKPOINT_DIR"
fi

sh ./patch_lammps.sh -e "$CNMP_LAMMPS_PATH" 

cd "$CNMP_LAMMPS_PATH" || exit 1
mkdir -p build
cd build || exit 1

export VIRTUAL_ENV=~/base/miniconda3/envs/cignn_env
export CMAKE_PREFIX_PATH="$VIRTUAL_ENV"
export OMPI_CXX="$(which g++)"

cmake ../cmake \
  -DCMAKE_PREFIX_PATH="$(python -c 'import torch;print(torch.utils.cmake_prefix_path)')" \
  -DPKG_ML-CNMP=ON \
  -DPYTHON_EXECUTABLE="$(which python)" \
  -DPYTHON_INCLUDE_DIR="$(python -c "from sysconfig import get_paths; print(get_paths()['include'])")" \
  -DPYTHON_LIBRARY="$VIRTUAL_ENV/lib/libpython3.12.so" \
  -DPython3_ROOT_DIR="$VIRTUAL_ENV" \
  -DPKG_PYTHON=ON \
  -DPKG_OPT=ON

make -j30