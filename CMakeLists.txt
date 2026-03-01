# CMakeList.txt : CMake project for Lab_3, include source and define
# project specific logic here.
# ======================================
# === DO NOT EDIT ===
# ======================================
cmake_minimum_required(VERSION 3.14)
# This is a preset to allow for easier debugging//dynamic editing. 
if (POLICY CMP0141)
    cmake_policy(SET CMP0141 NEW)
    set(CMAKE_MSVC_DEBUG_INFORMATION_FORMAT "$<IF:$<AND:$<C_COMPILER_ID:MSVC>,$<CXX_COMPILER_ID:MSVC>>,$<$<CONFIG:Debug,RelWithDebInfo>:EditAndContinue>,$<$<CONFIG:Debug,RelWithDebInfo>:ProgramDatabase>>")
endif()
# ======================================
# === DO NOT EDIT ===
# ======================================

# ======================================
# === CAN EDIT: Name of your Project ===
# ======================================
# The name of your project and .sln file. I.e if you named your project as ENGO_363_Lab3, replace that here.

project("Lab_3")

# ======================================
# === CAN EDIT: Name of your Project ===
# ======================================

# ======================================
# === DO NOT EDIT ===
# ======================================
# The below pretty much sets what parts of EIGEN we want (there's some acceleration we do not need that makes it complicated). 
# Fetch Eigen (no need to edit)
include(FetchContent)
set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
set(EIGEN_BUILD_DOC OFF CACHE BOOL "" FORCE)
set(EIGEN_BUILD_TESTING OFF CACHE BOOL "" FORCE)
set(EIGEN_BUILD_PKGCONFIG OFF CACHE BOOL "" FORCE)
set(CMAKE_Fortran_COMPILER NOTFOUND CACHE FILEPATH "" FORCE)
# This then actually downloads it. 
FetchContent_Declare(
        Eigen
        GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
        GIT_TAG 3.4.0
        GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(Eigen)
# ======================================
# === DO NOT EDIT ===
# ======================================

# ======================================
# === CAN EDIT: ADD YOUR SOURCE FILES BELOW IN add_executable() ===
# ======================================
# For every file you have i.e Lab_3.cpp, Lab_3.h, main.cpp, please add here. If the file is in a folder i.e src/main.cpp, input it as that. 
# For the un-"quoted" Lab_3, this sets what your .exe file becomes, so can technically be anything really. 
# It is a good idea for it to match your project name. 

add_executable(Lab_3
        "main.cpp"
        "functions.cpp" "functions.h"
        functions.h
)

# The Lab_3 part should match whatever is in the add_executable.
target_link_libraries(Lab_3 PRIVATE Eigen3::Eigen)
set_property(TARGET Lab_3 PROPERTY CXX_STANDARD 20)

# ======================================
# === CAN EDIT: ADD YOUR SOURCE FILES ABOVE IN add_executable() ===
# ======================================