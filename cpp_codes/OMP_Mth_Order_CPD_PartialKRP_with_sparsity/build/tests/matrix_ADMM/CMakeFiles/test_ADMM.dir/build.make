# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.16

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/telecom/Desktop/nina/codes/OMP_Mth_Order_CPD_PartialKRP_templated

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/telecom/Desktop/nina/codes/OMP_Mth_Order_CPD_PartialKRP_templated/build

# Include any dependencies generated for this target.
include tests/matrix_ADMM/CMakeFiles/test_ADMM.dir/depend.make

# Include the progress variables for this target.
include tests/matrix_ADMM/CMakeFiles/test_ADMM.dir/progress.make

# Include the compile flags for this target's objects.
include tests/matrix_ADMM/CMakeFiles/test_ADMM.dir/flags.make

tests/matrix_ADMM/CMakeFiles/test_ADMM.dir/test_ADMM.cpp.o: tests/matrix_ADMM/CMakeFiles/test_ADMM.dir/flags.make
tests/matrix_ADMM/CMakeFiles/test_ADMM.dir/test_ADMM.cpp.o: ../tests/matrix_ADMM/test_ADMM.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/telecom/Desktop/nina/codes/OMP_Mth_Order_CPD_PartialKRP_templated/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object tests/matrix_ADMM/CMakeFiles/test_ADMM.dir/test_ADMM.cpp.o"
	cd /home/telecom/Desktop/nina/codes/OMP_Mth_Order_CPD_PartialKRP_templated/build/tests/matrix_ADMM && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/test_ADMM.dir/test_ADMM.cpp.o -c /home/telecom/Desktop/nina/codes/OMP_Mth_Order_CPD_PartialKRP_templated/tests/matrix_ADMM/test_ADMM.cpp

tests/matrix_ADMM/CMakeFiles/test_ADMM.dir/test_ADMM.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/test_ADMM.dir/test_ADMM.cpp.i"
	cd /home/telecom/Desktop/nina/codes/OMP_Mth_Order_CPD_PartialKRP_templated/build/tests/matrix_ADMM && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/telecom/Desktop/nina/codes/OMP_Mth_Order_CPD_PartialKRP_templated/tests/matrix_ADMM/test_ADMM.cpp > CMakeFiles/test_ADMM.dir/test_ADMM.cpp.i

tests/matrix_ADMM/CMakeFiles/test_ADMM.dir/test_ADMM.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/test_ADMM.dir/test_ADMM.cpp.s"
	cd /home/telecom/Desktop/nina/codes/OMP_Mth_Order_CPD_PartialKRP_templated/build/tests/matrix_ADMM && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/telecom/Desktop/nina/codes/OMP_Mth_Order_CPD_PartialKRP_templated/tests/matrix_ADMM/test_ADMM.cpp -o CMakeFiles/test_ADMM.dir/test_ADMM.cpp.s

# Object files for target test_ADMM
test_ADMM_OBJECTS = \
"CMakeFiles/test_ADMM.dir/test_ADMM.cpp.o"

# External object files for target test_ADMM
test_ADMM_EXTERNAL_OBJECTS =

../bin/test_ADMM: tests/matrix_ADMM/CMakeFiles/test_ADMM.dir/test_ADMM.cpp.o
../bin/test_ADMM: tests/matrix_ADMM/CMakeFiles/test_ADMM.dir/build.make
../bin/test_ADMM: tests/matrix_ADMM/CMakeFiles/test_ADMM.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/telecom/Desktop/nina/codes/OMP_Mth_Order_CPD_PartialKRP_templated/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable ../../../bin/test_ADMM"
	cd /home/telecom/Desktop/nina/codes/OMP_Mth_Order_CPD_PartialKRP_templated/build/tests/matrix_ADMM && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/test_ADMM.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
tests/matrix_ADMM/CMakeFiles/test_ADMM.dir/build: ../bin/test_ADMM

.PHONY : tests/matrix_ADMM/CMakeFiles/test_ADMM.dir/build

tests/matrix_ADMM/CMakeFiles/test_ADMM.dir/clean:
	cd /home/telecom/Desktop/nina/codes/OMP_Mth_Order_CPD_PartialKRP_templated/build/tests/matrix_ADMM && $(CMAKE_COMMAND) -P CMakeFiles/test_ADMM.dir/cmake_clean.cmake
.PHONY : tests/matrix_ADMM/CMakeFiles/test_ADMM.dir/clean

tests/matrix_ADMM/CMakeFiles/test_ADMM.dir/depend:
	cd /home/telecom/Desktop/nina/codes/OMP_Mth_Order_CPD_PartialKRP_templated/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/telecom/Desktop/nina/codes/OMP_Mth_Order_CPD_PartialKRP_templated /home/telecom/Desktop/nina/codes/OMP_Mth_Order_CPD_PartialKRP_templated/tests/matrix_ADMM /home/telecom/Desktop/nina/codes/OMP_Mth_Order_CPD_PartialKRP_templated/build /home/telecom/Desktop/nina/codes/OMP_Mth_Order_CPD_PartialKRP_templated/build/tests/matrix_ADMM /home/telecom/Desktop/nina/codes/OMP_Mth_Order_CPD_PartialKRP_templated/build/tests/matrix_ADMM/CMakeFiles/test_ADMM.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : tests/matrix_ADMM/CMakeFiles/test_ADMM.dir/depend

