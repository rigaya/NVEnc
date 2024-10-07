#!/bin/sh
# sudo apt install libssl-dev
# curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal
# . $HOME/.cargo/env
# cargo install cargo-c

DOVI_VER=2.1.2
DOVI_SRC=dovi_tool

LIBDOVI_INSTALL_DIR=`pwd`/build_libdovi
if [ ! -e ${DOVI_SRC} ]; then
    wget -O ${DOVI_SRC}.tar.gz https://github.com/quietvoid/dovi_tool/archive/refs/tags/${DOVI_VER}.tar.gz && \
      tar xf ${DOVI_SRC}.tar.gz && \
      rm ${DOVI_SRC}.tar.gz && \
      mv ${DOVI_SRC}-${DOVI_VER} ${DOVI_SRC}
fi
if [ ! -e ${LIBDOVI_INSTALL_DIR} ]; then
    mkdir $LIBDOVI_INSTALL_DIR
fi
cd ${DOVI_SRC}/dolby_vision && \
  cargo cinstall --release --prefix=$LIBDOVI_INSTALL_DIR && \
  find ${LIBDOVI_INSTALL_DIR} -name "libdovi.so*" | xargs rm
