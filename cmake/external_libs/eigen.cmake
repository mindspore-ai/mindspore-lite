set(Eigen3_CXXFLAGS "-D_FORTIFY_SOURCE=2 -O2")
set(Eigen3_CFLAGS "-D_FORTIFY_SOURCE=2 -O2")


set(REQ_URL "https://gitlab.com/libeigen/eigen/-/archive/5.0.0/eigen-5.0.0.tar.gz")
set(SHA256 "315c881e19e17542a7d428c5aa37d113c89b9500d350c433797b730cd449c056")

if(MSVC)
    mindspore_add_pkg(Eigen3
            VER 5.0.0
            URL ${REQ_URL}
            SHA256 ${SHA256}
            CMAKE_OPTION -DBUILD_TESTING=OFF)
else()
    mindspore_add_pkg(Eigen3
            VER 5.0.0
            URL ${REQ_URL}
            SHA256 ${SHA256}
            PATCHES ${TOP_DIR}/third_party/patch/eigen/0001-fix-eigen.patch
            CMAKE_OPTION -DBUILD_TESTING=OFF)
endif()
find_package(Eigen3 5.0.0 REQUIRED ${MS_FIND_NO_DEFAULT_PATH})
get_target_property(EIGEN3_INCLUDE_DIR Eigen3::Eigen INTERFACE_INCLUDE_DIRECTORIES)
include_directories(${Eigen3_INC})
include_directories(${EIGEN3_INCLUDE_DIR})
set_property(TARGET Eigen3::Eigen PROPERTY IMPORTED_GLOBAL TRUE)
add_library(mindspore::eigen ALIAS Eigen3::Eigen)
