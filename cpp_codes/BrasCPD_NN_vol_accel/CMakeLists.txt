CMAKE_MINIMUM_REQUIRED(VERSION 3.5)

PROJECT(BRAS_CPD_ACCEL)


########################################################################################################################
###    SET VARIABLES
########################################################################################################################
SET(CMAKE_MINIMUM_REQUIRED_VERSION                           "3.5")
SET(CPP_LIBS                                                 "/SDKS/CPP_LIBS")
SET(EIGEN_ROOT_DIR                                           ${CPP_LIBS}/eigen)
SET(INCLUDE_DIR                                              ${CMAKE_CURRENT_LIST_DIR}/include)
SET(BIN_DIR                                                  ${CMAKE_CURRENT_LIST_DIR}/bin)

########################################################################################################################

########################################################################################################################
###     Get build platform information
########################################################################################################################
cmake_host_system_information(RESULT platform.hostname              QUERY HOSTNAME)
cmake_host_system_information(RESULT platform.os.name               QUERY OS_NAME)
cmake_host_system_information(RESULT platform.processor.description QUERY PROCESSOR_DESCRIPTION)
cmake_host_system_information(RESULT platform.processor.cores       QUERY NUMBER_OF_LOGICAL_CORES)
cmake_host_system_information(RESULT platform.memory.size           QUERY TOTAL_PHYSICAL_MEMORY)
########################################################################################################################

IF (APPLE)
  IF (CMAKE_C_COMPILER_ID MATCHES "Clang")
    MESSAGE( STATUS  "[ BRAS_CPD_ACCEL ] Clang C toolset found." )
    SET(OpenMP_C "${CMAKE_C_COMPILER}")
    SET(OpenMP_C_FLAGS "-fopenmp=libomp -Wno-unused-command-line-argument")
    SET(OpenMP_C_LIB_NAMES "libomp" "libgomp" "libiomp5")
    SET(OpenMP_libomp_LIBRARY ${OpenMP_C_LIB_NAMES})
    SET(OpenMP_libgomp_LIBRARY ${OpenMP_C_LIB_NAMES})
    SET(OpenMP_libiomp5_LIBRARY ${OpenMP_C_LIB_NAMES})
  ENDIF()

  IF (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    MESSAGE( STATUS  "[ BRAS_CPD_ACCEL ] Clang C++ toolset found." )
    SET(OpenMP_CXX "${CMAKE_CXX_COMPILER}")
    SET(OpenMP_CXX_FLAGS "-fopenmp=libomp -Wno-unused-command-line-argument")
    SET(OpenMP_CXX_LIB_NAMES "libomp" "libgomp" "libiomp5")
    SET(OpenMP_libomp_LIBRARY ${OpenMP_CXX_LIB_NAMES})
    SET(OpenMP_libgomp_LIBRARY ${OpenMP_CXX_LIB_NAMES})
    SET(OpenMP_libiomp5_LIBRARY ${OpenMP_CXX_LIB_NAMES})
  ENDIF()
ELSE ()
  ADD_DEFINITIONS(-fopenmp)
ENDIF()


ADD_DEFINITIONS(-Wall -std=c++17)
SET(CMAKE_CXX_STANDARD 17)
SET(CMAKE_CXX_STANDARD_REQUIRED ON)
SET(CMAKE_CXX_EXTENSIONS OFF)

SET(CMAKE_CXX_FLAGS              "${CMAKE_CXX_FLAGS} -fopenmp -Wall -Wextra -std=c++17")
SET(CMAKE_CXX_FLAGS              "${CMAKE_CXX_FLAGS}  -Wno-ignored-attributes -Wno-deprecated-copy")
SET(CMAKE_CXX_FLAGS_RELEASE      "-mtune=native -march=native -O3 -w")
SET(CMAKE_CXX_FLAGS_DEBUG        "-g")

########################################################################################################################
# Eigen
########################################################################################################################
MESSAGE( STATUS  "[ BRAS_CPD_ACCEL ] Configuring Eigen" )

IF (EIGEN_ROOT_DIR)
	 FILE(READ "${EIGEN_ROOT_DIR}/Eigen/src/Core/util/Macros.h" _eigen_version_file)
   STRING(REGEX REPLACE ".*#define EIGEN_WORLD_VERSION ([0-9]+).*" "\\1"
                EIGEN_VERSION_WORLD "${_eigen_version_file}")
	 STRING(REGEX REPLACE ".*#define EIGEN_MAJOR_VERSION ([0-9]+).*" "\\1"
                EIGEN_VERSION_MAJOR "${_eigen_version_file}")
   STRING(REGEX REPLACE ".*#define EIGEN_MINOR_VERSION ([0-9]+).*" "\\1"
                EIGEN_VERSION_MINOR "${_eigen_version_file}")
   SET(EIGEN_VERSION "${EIGEN_VERSION_WORLD}.${EIGEN_VERSION_MAJOR}.${EIGEN_VERSION_MINOR}")
ENDIF()

MESSAGE(STATUS "[ BRAS_CPD_ACCEL ] EIGEN_VERSION                 : ${EIGEN_VERSION}")
MESSAGE(STATUS "[ BRAS_CPD_ACCEL ] Config EIGEN END...")
########################################################################################################################


########################################################################################################################
###     Get build compiler information
########################################################################################################################
SET(compiler.vendor              ${CMAKE_CXX_COMPILER_ID})
SET(compiler.version             ${CMAKE_CXX_COMPILER_VERSION})
SET(compiler.target.architecture ${CMAKE_HOST_SYSTEM_PROCESSOR})

ADD_DEFINITIONS(
   -DCOMPILER_VENDOR="${compiler.vendor}"
   -DCOMPILER_VERSION="${compiler.version}"
   -DCOMPILER_TARGET_ARCHITECTURE="${compiler.target.architecture}"
)

INCLUDE_DIRECTORIES(${EIGEN_ROOT_DIR})
########################################################################################################################



########################################################################################################################
###    SUB DIRECTORIES
########################################################################################################################
###    TUC

ADD_SUBDIRECTORY(test/bras_CPDAccel_test)


MESSAGE(STATUS "[ BRAS_CPD_ACCEL ] C++ Compiler:     {vendor=\"${compiler.vendor}\", version=\"${compiler.version}\", target.architecture=\"${compiler.target.architecture}\"}")
MESSAGE(STATUS "[ BRAS_CPD_ACCEL ] Build Platform:   {hostname=\"${platform.hostname}\", os.name=\"${platform.os.name}\", processor.description=\"${platform.processor.description}\", processor.cores=\"${platform.processor.cores}\", memory.size=\"${platform.memory.size}MB\"}")
