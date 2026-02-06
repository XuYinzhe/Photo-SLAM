cmake_minimum_required(VERSION 3.20 FATAL_ERROR)

# force cuda 12.6 for cuvslam
set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda-12.6")
set(CMAKE_CUDA_COMPILER "${CUDA_TOOLKIT_ROOT_DIR}/bin/nvcc")
set(CMAKE_CUDA_ARCHITECTURES "75;86")

project(photo-slam LANGUAGES CXX CUDA)

if(NOT CMAKE_BUILD_TYPE)
    # set(CMAKE_BUILD_TYPE Debug)
    set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 17)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-deprecated-declarations")

set(CAFFE2_USE_CUDNN on)

# Uncomment the following line if you want to solve CMake Warning "Failed to compute shorthash for libnvrtc.so"
# set(PYTHON_EXECUTABLE "/usr/bin/python3")

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS} -pthread")

# set(ENV{CUDA_LAUNCH_BLOCKING} 1)
# set(ENV{TORCH_USE_CUDA_DSA} 1)

# find_package(CUDA REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(OpenCV 4 REQUIRED)
find_package(jsoncpp REQUIRED)

find_package(OpenGL REQUIRED)
# find_package(glm REQUIRED) # original: /usr/lib/cmake/glm
find_package(glfw3 REQUIRED)

set(ORB_SLAM3_SOURCE_DIR "${PROJECT_SOURCE_DIR}/ORB-SLAM3")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)

# add glm if cannot find automatically
set(GLM_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/glm")
if(NOT TARGET glm::glm)
    add_library(glm INTERFACE)
    target_include_directories(glm INTERFACE ${GLM_INCLUDE_DIR})
    add_library(glm::glm ALIAS glm)
    message(STATUS "Defined manual glm::glm target")
endif()

include_directories(
    ${PROJECT_SOURCE_DIR}
    ${ORB_SLAM3_SOURCE_DIR}
    ${ORB_SLAM3_SOURCE_DIR}/include
    ${ORB_SLAM3_SOURCE_DIR}/include/CameraModels
    ${ORB_SLAM3_SOURCE_DIR}/Thirdparty/Sophus
    ${EIGEN3_INCLUDE_DIR}
    ${OPENGL_INCLUDE_DIR})

##################################################################################
##  Build the gaussian splatting component libraries to ${PROJECT_SOURCE_DIR}/lib
##################################################################################

add_library(simple_knn SHARED
    third_party/simple-knn/simple_knn.cu
    third_party/simple-knn/simple_knn.h
    third_party/simple-knn/spatial.cu
    third_party/simple-knn/spatial.h)
# target_compile_features(simple_knn PUBLIC cxx_std_17)
target_link_libraries(simple_knn "${TORCH_LIBRARIES}")

add_library(cuda_rasterizer SHARED
    include/operate_points.h
    src/operate_points.cu
    include/rasterize_points.h
    src/rasterize_points.cu
    include/stereo_vision.h
    src/stereo_vision.cu
    # cuda_rasterizer/helper_math.h
    # cuda_rasterizer/math.h
    cuda_rasterizer/auxiliary.h
    cuda_rasterizer/backward.cu
    cuda_rasterizer/backward.h
    cuda_rasterizer/config.h
    cuda_rasterizer/forward.cu
    cuda_rasterizer/forward.h
    cuda_rasterizer/operate_points.h
    cuda_rasterizer/rasterizer.h
    cuda_rasterizer/rasterizer_impl.cu
    cuda_rasterizer/rasterizer_impl.h
    cuda_rasterizer/stereo_vision.h)
set_target_properties(cuda_rasterizer PROPERTIES CUDA_ARCHITECTURES "75;86")
# target_compile_features(cuda_rasterizer PUBLIC cxx_std_17)
target_include_directories(cuda_rasterizer PRIVATE
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
target_link_libraries(cuda_rasterizer
    glm::glm
    "${TORCH_LIBRARIES}"
    Eigen3::Eigen)

##################################################################################
##  Build the cc_torch connected components library
##################################################################################

set(CC_TORCH_SRC_DIR ${PROJECT_SOURCE_DIR}/third_party/Connected_components_PyTorch/cpp)

add_library(cc_torch SHARED
    ${CC_TORCH_SRC_DIR}/buf.h
    ${CC_TORCH_SRC_DIR}/buf.cu)

set_target_properties(cc_torch PROPERTIES CUDA_ARCHITECTURES "75;86")

target_include_directories(cc_torch PUBLIC
    ${CC_TORCH_SRC_DIR}
    ${TORCH_INCLUDE_DIRS}
)

target_link_libraries(cc_torch PUBLIC
    ${TORCH_LIBRARIES}
)

##################################################################################
##  Build the cuvslam library to ${PROJECT_SOURCE_DIR}/lib
##################################################################################

set(CUVSLAM_DIR ${PROJECT_SOURCE_DIR}/third_party/cuvlsam)
set(CUVSLAM_INCLUDE_DIR ${CUVSLAM_DIR}/include)
set(CUVSLAM_LIB_DIR ${CUVSLAM_DIR}/lib)
set(CUVSLAM_LIB ${CUVSLAM_LIB_DIR}/libcuvslam.so)

add_library(cuvslam SHARED IMPORTED)
set_target_properties(cuvslam PROPERTIES
    IMPORTED_LOCATION ${CUVSLAM_LIB}
    INTERFACE_INCLUDE_DIRECTORIES ${CUVSLAM_INCLUDE_DIR})

##################################################################################
##  Build the ImGui library to ${PROJECT_SOURCE_DIR}/lib
##################################################################################

add_library(imgui SHARED
    viewer/imgui/imconfig.h
    viewer/imgui/imgui_demo.cpp
    viewer/imgui/imgui_draw.cpp
    viewer/imgui/imgui_impl_glfw.cpp
    viewer/imgui/imgui_impl_glfw.h
    viewer/imgui/imgui_impl_opengl3_loader.h
    viewer/imgui/imgui_impl_opengl3.cpp
    viewer/imgui/imgui_impl_opengl3.h
    viewer/imgui/imgui_internal.h
    viewer/imgui/imgui_tables.cpp
    viewer/imgui/imgui_widgets.cpp
    viewer/imgui/imgui.cpp
    viewer/imgui/imgui.h
    viewer/imgui/imstb_rectpack.h
    viewer/imgui/imstb_textedit.h
    viewer/imgui/imstb_truetype.h)
target_link_libraries(imgui glfw OpenGL::GL)

##################################################################################
##  Build the gaussian mapper library to ${PROJECT_SOURCE_DIR}/lib
##################################################################################

add_library(gaussian_mapper SHARED
    third_party/tinyply/tinyply.h
    third_party/tinyply/tinyply.cpp
    include/gaussian_keyframe.h
    include/gaussian_model.h
    include/gaussian_parameters.h
    include/gaussian_rasterizer.h
    include/gaussian_renderer.h
    include/gaussian_scene.h
    include/gaussian_trainer.h
    include/gaussian_mapper.h
    include/general_utils.h
    include/graphics_utils.h
    include/loss_utils.h
    include/sh_utils.h
    include/tensor_utils.h
    include/metrics_utils.h
    include/camera.h
    include/point_cloud.h
    include/point2d.h
    include/point3d.h
    include/types.h
    src/gaussian_keyframe.cpp
    src/gaussian_model.cpp
    src/gaussian_parameters.cpp
    src/gaussian_rasterizer.cpp
    src/gaussian_renderer.cpp
    src/gaussian_scene.cpp
    src/gaussian_trainer.cpp
    src/gaussian_mapper.cpp
)
target_link_libraries(gaussian_mapper
    ${ORB_SLAM3_SOURCE_DIR}/lib/libORB_SLAM3.so
    ${OpenCV_LIBRARIES}
    jsoncpp_lib
    "${TORCH_LIBRARIES}"
    Eigen3::Eigen
    simple_knn
    cuda_rasterizer
    cc_torch
    cuvslam
)
target_include_directories(gaussian_mapper PUBLIC
    ${CUVSLAM_INCLUDE_DIR}
)

##################################################################################
##  Build the viewer library to ${PROJECT_SOURCE_DIR}/lib
##################################################################################

add_library(gaussian_viewer SHARED
    viewer/drawer_utils.h
    viewer/imgui_viewer.cpp
    viewer/imgui_viewer.h
    viewer/map_drawer.cpp
    viewer/map_drawer.h)
target_link_libraries(gaussian_viewer
    ${ORB_SLAM3_SOURCE_DIR}/lib/libORB_SLAM3.so
    gaussian_mapper
    imgui
    ${OpenCV_LIBRARIES}
    jsoncpp_lib
    "${TORCH_LIBRARIES}"
    glm::glm
    glfw
    OpenGL::GL)

##################################################################################
##  Build the mapping examples to ${PROJECT_SOURCE_DIR}/bin
##################################################################################

add_executable(align_hrlr examples/align_hrlr.cpp)
target_link_libraries(align_hrlr
    gaussian_viewer    
    gaussian_mapper
    ${ORB_SLAM3_SOURCE_DIR}/lib/libORB_SLAM3.so)

add_executable(test_cuvslam examples/test_cuvslam.cpp)
target_link_libraries(
    test_cuvslam
    ${CUVSLAM_LIB}
    ${OpenCV_LIBRARIES}
)