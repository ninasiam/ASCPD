# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.13

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
CMAKE_SOURCE_DIR = "/home/nina/Documents/uni/nina_s/cpp_codes/BrasCPD_Accel_NN_(transposed_KR)"

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = "/home/nina/Documents/uni/nina_s/cpp_codes/BrasCPD_Accel_NN_(transposed_KR)/build"

# Include any dependencies generated for this target.
include test/test_KR_sampled/CMakeFiles/test_kr_sampled.dir/depend.make

# Include the progress variables for this target.
include test/test_KR_sampled/CMakeFiles/test_kr_sampled.dir/progress.make

# Include the compile flags for this target's objects.
include test/test_KR_sampled/CMakeFiles/test_kr_sampled.dir/flags.make

test/test_KR_sampled/CMakeFiles/test_kr_sampled.dir/test_kr_sampled.cpp.o: test/test_KR_sampled/CMakeFiles/test_kr_sampled.dir/flags.make
test/test_KR_sampled/CMakeFiles/test_kr_sampled.dir/test_kr_sampled.cpp.o: ../test/test_KR_sampled/test_kr_sampled.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir="/home/nina/Documents/uni/nina_s/cpp_codes/BrasCPD_Accel_NN_(transposed_KR)/build/CMakeFiles" --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object test/test_KR_sampled/CMakeFiles/test_kr_sampled.dir/test_kr_sampled.cpp.o"
	cd "/home/nina/Documents/uni/nina_s/cpp_codes/BrasCPD_Accel_NN_(transposed_KR)/build/test/test_KR_sampled" && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/test_kr_sampled.dir/test_kr_sampled.cpp.o -c "/home/nina/Documents/uni/nina_s/cpp_codes/BrasCPD_Accel_NN_(transposed_KR)/test/test_KR_sampled/test_kr_sampled.cpp"

test/test_KR_sampled/CMakeFiles/test_kr_sampled.dir/test_kr_sampled.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/test_kr_sampled.dir/test_kr_sampled.cpp.i"
	cd "/home/nina/Documents/uni/nina_s/cpp_codes/BrasCPD_Accel_NN_(transposed_KR)/build/test/test_KR_sampled" && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E "/home/nina/Documents/uni/nina_s/cpp_codes/BrasCPD_Accel_NN_(transposed_KR)/test/test_KR_sampled/test_kr_sampled.cpp" > CMakeFiles/test_kr_sampled.dir/test_kr_sampled.cpp.i

test/test_KR_sampled/CMakeFiles/test_kr_sampled.dir/test_kr_sampled.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/test_kr_sampled.dir/test_kr_sampled.cpp.s"
	cd "/home/nina/Documents/uni/nina_s/cpp_codes/BrasCPD_Accel_NN_(transposed_KR)/build/test/test_KR_sampled" && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S "/home/nina/Documents/uni/nina_s/cpp_codes/BrasCPD_Accel_NN_(transposed_KR)/test/test_KR_sampled/test_kr_sampled.cpp" -o CMakeFiles/test_kr_sampled.dir/test_kr_sampled.cpp.s

# Object files for target test_kr_sampled
test_kr_sampled_OBJECTS = \
"CMakeFiles/test_kr_sampled.dir/test_kr_sampled.cpp.o"

# External object files for target test_kr_sampled
test_kr_sampled_EXTERNAL_OBJECTS =

../bin/test_kr_sampled: test/test_KR_sampled/CMakeFiles/test_kr_sampled.dir/test_kr_sampled.cpp.o
../bin/test_kr_sampled: test/test_KR_sampled/CMakeFiles/test_kr_sampled.dir/build.make
../bin/test_kr_sampled: /usr/lib/gcc/x86_64-linux-gnu/9/libgomp.so
../bin/test_kr_sampled: /usr/lib/x86_64-linux-gnu/libpthread.so
../bin/test_kr_sampled: test/test_KR_sampled/CMakeFiles/test_kr_sampled.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir="/home/nina/Documents/uni/nina_s/cpp_codes/BrasCPD_Accel_NN_(transposed_KR)/build/CMakeFiles" --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable ../../../bin/test_kr_sampled"
	cd "/home/nina/Documents/uni/nina_s/cpp_codes/BrasCPD_Accel_NN_(transposed_KR)/build/test/test_KR_sampled" && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/test_kr_sampled.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
test/test_KR_sampled/CMakeFiles/test_kr_sampled.dir/build: ../bin/test_kr_sampled

.PHONY : test/test_KR_sampled/CMakeFiles/test_kr_sampled.dir/build

test/test_KR_sampled/CMakeFiles/test_kr_sampled.dir/clean:
	cd "/home/nina/Documents/uni/nina_s/cpp_codes/BrasCPD_Accel_NN_(transposed_KR)/build/test/test_KR_sampled" && $(CMAKE_COMMAND) -P CMakeFiles/test_kr_sampled.dir/cmake_clean.cmake
.PHONY : test/test_KR_sampled/CMakeFiles/test_kr_sampled.dir/clean

test/test_KR_sampled/CMakeFiles/test_kr_sampled.dir/depend:
	cd "/home/nina/Documents/uni/nina_s/cpp_codes/BrasCPD_Accel_NN_(transposed_KR)/build" && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" "/home/nina/Documents/uni/nina_s/cpp_codes/BrasCPD_Accel_NN_(transposed_KR)" "/home/nina/Documents/uni/nina_s/cpp_codes/BrasCPD_Accel_NN_(transposed_KR)/test/test_KR_sampled" "/home/nina/Documents/uni/nina_s/cpp_codes/BrasCPD_Accel_NN_(transposed_KR)/build" "/home/nina/Documents/uni/nina_s/cpp_codes/BrasCPD_Accel_NN_(transposed_KR)/build/test/test_KR_sampled" "/home/nina/Documents/uni/nina_s/cpp_codes/BrasCPD_Accel_NN_(transposed_KR)/build/test/test_KR_sampled/CMakeFiles/test_kr_sampled.dir/DependInfo.cmake" --color=$(COLOR)
.PHONY : test/test_KR_sampled/CMakeFiles/test_kr_sampled.dir/depend
