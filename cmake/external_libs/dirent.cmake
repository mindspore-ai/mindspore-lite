if(ENABLE_GITEE)
    set(REQ_URL "https://gitee.com/mirrors/dirent/repository/archive/1.24.zip")
    set(SHA256 "d762a87c01b63a1bc8d13e573a38dd067a8c290f365398cbc1a9519c8fdc720a")
else()
    set(REQ_URL "https://github.com/tronkko/dirent/archive/refs/tags/1.24.zip")
    set(SHA256 "46fa2833610e60275e30949c9cb4268430f945ca11fdbfa80dfad68de967103a")
endif()


if(MSVC)
    mindspore_add_pkg(dirent
        VER 1.24
        HEAD_ONLY ./include
        RELEASE on
        URL ${REQ_URL}
        SHA256 ${SHA256})
    include_directories(${dirent_INC})
endif()


