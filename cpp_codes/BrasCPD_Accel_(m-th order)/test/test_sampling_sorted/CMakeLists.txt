CMAKE_MINIMUM_REQUIRED(VERSION ${CMAKE_MINIMUM_REQUIRED_VERSION})

PROJECT(BRASCPD_ACCEL_NN_TEST)

########################################################################################################################
###      SET Variables
########################################################################################################################
STRING(TOUPPER ${PROJECT_NAME} PROJECT_NAME_TOUPPER)  
MESSAGE(STATUS " [ ${PROJECT_NAME_TOUPPER} ] Processing in ${CMAKE_BUILD_TYPE} mode.")
MESSAGE(STATUS " [ ${PROJECT_NAME_TOUPPER} ] SOURCE DIR ==> ${CMAKE_CURRENT_LIST_DIR}")
#MESSAGE(STATUS " [ ${PROJECT_NAME_TOUPPER} ] HEADER DIR ==> ${INCLUDE_DIR_NCOM}")

SET(TEST_SRC_FILE_NAMES    sampling_fibers_sorted.cpp})

#SET(TEST_HEADER_FILE       ${CMAKE_CURRENT_LIST_DIR}/Test${PROJECT_NAME}.hpp)   
########################################################################################################################

########################################################################################################################
### EIGEN LIBRARY
########################################################################################################################
MESSAGE(STATUS " [ ${PROJECT_NAME_TOUPPER} ] EIGEN_ROOT ==> " ${EIGEN_ROOT_DIR})
########################################################################################################################

########################################################################################################################
### OpenMP
########################################################################################################################
MESSAGE(STATUS " [ ${PROJECT_NAME_TOUPPER} ] CONFIGURING OPENMP")
find_package(OpenMP REQUIRED)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()
########################################################################################################################

########################################################################################################################
### CREATE EXECUTABLE FROM SOURCE AND HEADER FILES
########################################################################################################################
SET(LIBRARIES_TO_LINK       OpenMP::OpenMP_CXX)

MESSAGE(STATUS " [ ${PROJECT_NAME_TOUPPER} ] Creating executable brasCPD_NonNegative.cpp")

ADD_EXECUTABLE(sampling_fibers_sorted sampling_fibers_sorted.cpp)

#TARGET_INCLUDE_DIRECTORIES(${TEST} PUBLIC ${INCLUDE_DIR_NCOM})
TARGET_LINK_LIBRARIES(sampling_fibers_sorted ${LIBRARIES_TO_LINK})
SET_TARGET_PROPERTIES(
    sampling_fibers_sorted 
   PROPERTIES
   RUNTIME_OUTPUT_DIRECTORY "${BIN_DIR}"
)
