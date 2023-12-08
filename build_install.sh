#!/usr/bin/env bash

INSTALL_DIR=/opt
CMAKE=/usr/bin/cmake
CC_DIR=${INSTALL_DIR}/rocm/llvm/bin/clang
CXX_DIR=${INSTALL_DIR}/rocm/llvm/bin/clang++

cd src && rm -rf build && mkdir -p build && cd build
CC=${CC_DIR} CXX=${CXX_DIR} ${CMAKE} -DCMAKE_INSTALL_PREFIX=${INSTALL_DIR}/rocm -DCMAKE_PREFIX_PATH="${INSTALL_DIR}/rocm/include;${INSTALL_DIR}/rocm/lib;${INSTALL_DIR}/rocm/rocdl" -DIMAGE_SUPPORT=OFF -DCMAKE_BUILD_TYPE="RELEASE" .. 
make -j 
make install
