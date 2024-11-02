#!/bin/sh
# sudo apt install libssl-dev
# curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal
# . $HOME/.cargo/env
# cargo install cargo-c

HDR10PLUS_VER=1.6.1
HDR10PLUS_SRC=hdr10plus_tool

LIBHDR10PLUS_INSTALL_DIR=`pwd`/build_libhdr10plus
if [ ! -e ${HDR10PLUS_SRC} ]; then
    wget -O ${HDR10PLUS_SRC}.tar.gz https://github.com/quietvoid/hdr10plus_tool/archive/refs/tags/${HDR10PLUS_VER}.tar.gz && \
      tar xf ${HDR10PLUS_SRC}.tar.gz && \
      rm ${HDR10PLUS_SRC}.tar.gz && \
      mv ${HDR10PLUS_SRC}-${HDR10PLUS_VER} ${HDR10PLUS_SRC}
fi
if [ ! -e ${LIBHDR10PLUS_INSTALL_DIR} ]; then
    mkdir $LIBHDR10PLUS_INSTALL_DIR
fi
cd ${HDR10PLUS_SRC}/hdr10plus && \
  ${CARGO} cinstall --release --prefix=$LIBHDR10PLUS_INSTALL_DIR && \
  find ${LIBHDR10PLUS_INSTALL_DIR} -name "libhdr10plus-rs.so*" | xargs rm
