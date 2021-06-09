#!/bin/sh

PACKAGE_NAME=nvencc
PACKAGE_BIN=nvencc
PACKAGE_OS=
PACKAGE_MAINTAINER=rigaya
PACKAGE_DEPENDS="ffmpeg-libs"
PACKAGE_DESCRIPTION=
PACKAGE_ROOT=.rpmpkg
PACKAGE_VERSION=`git describe --tags | cut -f 1 --delim="-"`
PACKAGE_ARCH=`uname -m`
PACKAGE_LICENSE=MIT

if [ -e /etc/os-release ]; then
    PACKAGE_OS_NAME=`cat /etc/os-release | grep NAME | cut -f 2 --delim="="`
    PACKAGE_OS_VER=`cat /etc/os-release | grep VERSION_ID | cut -f 2 --delim="="`
    PACKAGE_OS="${PACKAGE_OS_NAME}${PACKAGE_OS_VER}"
fi

mkdir -p ${PACKAGE_ROOT}
cp ${PACKAGE_BIN} ${PACKAGE_ROOT}
chmod +x ${PACKAGE_ROOT}/${PACKAGE_BIN}

rm -rf rpmbuild
RPMBUILD_DIR=${HOME}/rpmbuild
PACKAGE_SOURCE_DIR=${RPMBUILD_DIR}/SOURCES
PACKAGE_SPEC_DIR=${RPMBUILD_DIR}/SPEC

mkdir -p ${PACKAGE_SOURCE_DIR}
mkdir -p ${PACKAGE_SPEC_DIR}

WORK_DIR=.tmpwork
rm -rf ${WORK_DIR}
mkdir ${WORK_DIR}

build_pkg/replace.py \
    --rpm \
    -i build_pkg/template.spec \
    -o ${WORK_DIR}/${PACKAGE_NAME}.spec \
    --pkg-name ${PACKAGE_NAME} \
    --pkg-bin ${PACKAGE_BIN} \
    --pkg-version ${PACKAGE_VERSION} \
    --pkg-arch ${PACKAGE_ARCH} \
    --pkg-maintainer ${PACKAGE_MAINTAINER} \
    --pkg-depends ${PACKAGE_DEPENDS} \
    --pkg-desc ${PACKAGE_DESCRIPTION} \
    --pkg-license ${PACKAGE_LICENSE}

cp -rp "${PACKAGE_ROOT}" "${WORK_DIR}/${PACKAGE_NAME}"
cd ${WORK_DIR}
tar czf tmp.tar.gz "${PACKAGE_NAME}/"
mv tmp.tar.gz ${PACKAGE_SOURCE_DIR}/
cd ..
cp -p ${WORK_DIR}/${PACKAGE_NAME}.spec "${PACKAGE_SPEC_DIR}/"
rm -rf ${WORK_DIR}

rpmbuild -ba "${PACKAGE_SPEC_DIR}/${PACKAGE_NAME}.spec"

cp ${RPMBUILD_DIR}/RPMS/${PACKAGE_ARCH}/${PACKAGE_NAME}*.rpm .