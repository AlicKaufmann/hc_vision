# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.7

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
CMAKE_SOURCE_DIR = /home/alic/catkin_ws/src/vision

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/alic/catkin_ws/src/vision

# Include any dependencies generated for this target.
include CMakeFiles/vision.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/vision.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/vision.dir/flags.make

CMakeFiles/vision.dir/src/main.cpp.o: CMakeFiles/vision.dir/flags.make
CMakeFiles/vision.dir/src/main.cpp.o: src/main.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/alic/catkin_ws/src/vision/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/vision.dir/src/main.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/vision.dir/src/main.cpp.o -c /home/alic/catkin_ws/src/vision/src/main.cpp

CMakeFiles/vision.dir/src/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/vision.dir/src/main.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/alic/catkin_ws/src/vision/src/main.cpp > CMakeFiles/vision.dir/src/main.cpp.i

CMakeFiles/vision.dir/src/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/vision.dir/src/main.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/alic/catkin_ws/src/vision/src/main.cpp -o CMakeFiles/vision.dir/src/main.cpp.s

CMakeFiles/vision.dir/src/main.cpp.o.requires:

.PHONY : CMakeFiles/vision.dir/src/main.cpp.o.requires

CMakeFiles/vision.dir/src/main.cpp.o.provides: CMakeFiles/vision.dir/src/main.cpp.o.requires
	$(MAKE) -f CMakeFiles/vision.dir/build.make CMakeFiles/vision.dir/src/main.cpp.o.provides.build
.PHONY : CMakeFiles/vision.dir/src/main.cpp.o.provides

CMakeFiles/vision.dir/src/main.cpp.o.provides.build: CMakeFiles/vision.dir/src/main.cpp.o


# Object files for target vision
vision_OBJECTS = \
"CMakeFiles/vision.dir/src/main.cpp.o"

# External object files for target vision
vision_EXTERNAL_OBJECTS =

devel/lib/vision/vision: CMakeFiles/vision.dir/src/main.cpp.o
devel/lib/vision/vision: CMakeFiles/vision.dir/build.make
devel/lib/vision/vision: /opt/ros/lunar/lib/libroscpp.so
devel/lib/vision/vision: /usr/lib/x86_64-linux-gnu/libboost_filesystem.so
devel/lib/vision/vision: /usr/lib/x86_64-linux-gnu/libboost_signals.so
devel/lib/vision/vision: /opt/ros/lunar/lib/librosconsole.so
devel/lib/vision/vision: /opt/ros/lunar/lib/librosconsole_log4cxx.so
devel/lib/vision/vision: /opt/ros/lunar/lib/librosconsole_backend_interface.so
devel/lib/vision/vision: /usr/lib/x86_64-linux-gnu/liblog4cxx.so
devel/lib/vision/vision: /usr/lib/x86_64-linux-gnu/libboost_regex.so
devel/lib/vision/vision: /opt/ros/lunar/lib/libxmlrpcpp.so
devel/lib/vision/vision: /opt/ros/lunar/lib/libroscpp_serialization.so
devel/lib/vision/vision: /opt/ros/lunar/lib/librostime.so
devel/lib/vision/vision: /opt/ros/lunar/lib/libcpp_common.so
devel/lib/vision/vision: /usr/lib/x86_64-linux-gnu/libboost_system.so
devel/lib/vision/vision: /usr/lib/x86_64-linux-gnu/libboost_thread.so
devel/lib/vision/vision: /usr/lib/x86_64-linux-gnu/libboost_chrono.so
devel/lib/vision/vision: /usr/lib/x86_64-linux-gnu/libboost_date_time.so
devel/lib/vision/vision: /usr/lib/x86_64-linux-gnu/libboost_atomic.so
devel/lib/vision/vision: /usr/lib/x86_64-linux-gnu/libpthread.so
devel/lib/vision/vision: /usr/lib/x86_64-linux-gnu/libconsole_bridge.so
devel/lib/vision/vision: CMakeFiles/vision.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/alic/catkin_ws/src/vision/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable devel/lib/vision/vision"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/vision.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/vision.dir/build: devel/lib/vision/vision

.PHONY : CMakeFiles/vision.dir/build

CMakeFiles/vision.dir/requires: CMakeFiles/vision.dir/src/main.cpp.o.requires

.PHONY : CMakeFiles/vision.dir/requires

CMakeFiles/vision.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/vision.dir/cmake_clean.cmake
.PHONY : CMakeFiles/vision.dir/clean

CMakeFiles/vision.dir/depend:
	cd /home/alic/catkin_ws/src/vision && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/alic/catkin_ws/src/vision /home/alic/catkin_ws/src/vision /home/alic/catkin_ws/src/vision /home/alic/catkin_ws/src/vision /home/alic/catkin_ws/src/vision/CMakeFiles/vision.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/vision.dir/depend
