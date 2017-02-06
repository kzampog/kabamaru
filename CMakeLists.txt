cmake_minimum_required(VERSION 2.8)
project(sisyphus)

# compile with C++11 support
add_definitions("-std=c++0x -Dlinux -O3")

find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED)
find_package(PCL 1.8 REQUIRED)
find_package(CUDA 7.0 REQUIRED)

include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

include_directories(include)

set(LIB_NAME "sisyphus")

add_library(${LIB_NAME}_sift_engine SHARED ${PROJECT_BINARY_DIR}/src/sift_engine.cpp)
target_link_libraries(${LIB_NAME}_sift_engine siftgpu ${OpenCV_LIBRARIES} ${CUDA_LIBRARIES})

add_library(${LIB_NAME}_graph_utilities SHARED ${PROJECT_BINARY_DIR}/src/graph_utilities.cpp)

add_library(${LIB_NAME}_io_utilities SHARED ${PROJECT_BINARY_DIR}/src/io_utilities.cpp)
target_link_libraries(${LIB_NAME}_io_utilities ${OpenCV_LIBRARIES})

add_library(${LIB_NAME}_registration_utilities SHARED ${PROJECT_BINARY_DIR}/src/registration_utilities.cpp)
target_link_libraries(${LIB_NAME}_registration_utilities ${LIB_NAME}_sift_engine ${OpenCV_LIBRARIES} ${PCL_LIBRARIES} ${CUDA_LIBRARIES})

add_library(${LIB_NAME}_pointcloud_utilities SHARED ${PROJECT_BINARY_DIR}/src/pointcloud_utilities.cpp)
target_link_libraries(${LIB_NAME}_pointcloud_utilities ${LIB_NAME}_registration_utilities ${OpenCV_LIBRARIES} ${PCL_LIBRARIES} ${CUDA_LIBRARIES})

add_library(${LIB_NAME}_box_fitting SHARED ${PROJECT_BINARY_DIR}/src/box_fitting.cpp)
target_link_libraries(${LIB_NAME}_box_fitting ${LIB_NAME}_pointcloud_utilities ${PCL_LIBRARIES} ${CUDA_LIBRARIES})

add_library(${LIB_NAME}_sift_rgbd_slam SHARED ${PROJECT_BINARY_DIR}/src/sift_rgbd_slam.cpp)
target_link_libraries(${LIB_NAME}_sift_rgbd_slam ${LIB_NAME}_pointcloud_utilities ${LIB_NAME}_io_utilities ${OpenCV_LIBRARIES} ${PCL_LIBRARIES} ${CUDA_LIBRARIES})

add_library(${LIB_NAME}_box_detection SHARED ${PROJECT_BINARY_DIR}/src/box_detection.cpp)
target_link_libraries(${LIB_NAME}_box_detection ${LIB_NAME}_graph_utilities ${LIB_NAME}_pointcloud_utilities ${PCL_LIBRARIES} ${CUDA_LIBRARIES})

set(CMAKE_SKIP_RPATH TRUE)

set_target_properties(${LIB_NAME}_sift_engine ${LIB_NAME}_registration_utilities ${LIB_NAME}_graph_utilities ${LIB_NAME}_io_utilities ${LIB_NAME}_pointcloud_utilities ${LIB_NAME}_sift_rgbd_slam ${LIB_NAME}_box_detection ${LIB_NAME}_box_fitting
	PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib
)

install(TARGETS ${LIB_NAME}_sift_engine ${LIB_NAME}_registration_utilities ${LIB_NAME}_graph_utilities ${LIB_NAME}_io_utilities ${LIB_NAME}_pointcloud_utilities ${LIB_NAME}_sift_rgbd_slam ${LIB_NAME}_box_detection ${LIB_NAME}_box_fitting
	LIBRARY DESTINATION /usr/local/lib
)

install(DIRECTORY include/
	DESTINATION /usr/local/include
)

# uninstall target
configure_file(
	"${CMAKE_CURRENT_SOURCE_DIR}/cmake_uninstall.cmake.in"
	"${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake"
	IMMEDIATE @ONLY)

add_custom_target(uninstall
	COMMAND ${CMAKE_COMMAND} -P ${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake)
