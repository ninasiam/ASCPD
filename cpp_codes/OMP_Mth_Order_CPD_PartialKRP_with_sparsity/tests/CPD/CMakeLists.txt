CMAKE_MINIMUM_REQUIRED(VERSION ${CMAKE_MINIMUM_REQUIRED_VERSION})

PROJECT(CPD)

########################################################################################################################
###      SET Variables
########################################################################################################################
STRING(TOUPPER ${PROJECT_NAME} PROJECT_NAME_TOUPPER)  
MESSAGE(STATUS " [ ${PROJECT_NAME_TOUPPER} ] Processing in ${CMAKE_BUILD_TYPE} mode.")
MESSAGE(STATUS " [ ${PROJECT_NAME_TOUPPER} ] SOURCE DIR ==> ${CMAKE_CURRENT_LIST_DIR}")
MESSAGE(STATUS " [ ${PROJECT_NAME_TOUPPER} ] HEADER DIR ==> ${INCLUDE_DIR_NCOM}")

#SET(TEST_SRC_FILE_NAMES    "*.cpp")
file(GLOB_RECURSE TEST_SRC_FILE_NAMES_CPP "*.cpp")  # Get FILE_NAME with FILE_EXTENSION

SET (TEST_SRC_FILE_NAMES "")
FOREACH (_cpp_file ${TEST_SRC_FILE_NAMES_CPP})           # Process all elements in list TEST_SRC_FILE_NAMES_CPP
    get_filename_component(_name ${_cpp_file} NAME_WLE)  # Get FILE_NAME 
    list (APPEND TEST_SRC_FILE_NAMES ${_name})           # Add new element in list TEST_SRC_FILE_NAMES
ENDFOREACH()

########################################################################################################################

########################################################################################################################
### EIGEN LIBRARY
########################################################################################################################
MESSAGE(STATUS " [ ${PROJECT_NAME_TOUPPER} ] EIGEN_ROOT ==> " ${EIGEN_ROOT_DIR})
########################################################################################################################

########################################################################################################################
### OpenMP
########################################################################################################################
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

FOREACH(TEST IN LISTS TEST_SRC_FILE_NAMES)

    MESSAGE(STATUS " [ ${PROJECT_NAME_TOUPPER} ] Creating executable for ${TEST}.cpp")

    ADD_EXECUTABLE(${TEST} ${TEST}.cpp)

    #####TARGET_INCLUDE_DIRECTORIES(${TEST} PUBLIC ${INCLUDE_DIR_NCOM})
    TARGET_LINK_LIBRARIES(${LIBRARIES_TO_LINK})
    SET_TARGET_PROPERTIES(
        ${TEST}
        PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${BIN_DIR}"
    )
ENDFOREACH()