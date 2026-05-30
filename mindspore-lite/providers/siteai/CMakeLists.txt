cmake_minimum_required(VERSION 3.12)
project(SiteAi)

##disable external libs
set(MSLITE_DEPS_PYBIND11 on CACHE INTERNAL "setting MSLITE_DEPS_PYBIND11 value")

##enable prune simplest cloud inferenc
set(MSLITE_SIMPLEST_CLOUD_INFERENCE on CACHE INTERNAL "setting MSLITE_SIMPLEST_CLOUD_INFERENCE value")
set(MSLITE_ENABLE_OPENCV on CACHE INTERNAL "setting MSLITE_ENABLE_OPENCV value")