## For specific OpenCV and CUDA versions, you may need to modify the cmake command in this script.
# Since there are multiple versions in my system, I specify the paths to avoid cmake finding the wrong ones. 
# You may not need to do this if you only have one version of OpenCV and CUDA installed.

# OpenCV path variable
OPENCV_DIR=/home/shaun/opencv/opencv-4.10.0/install_12-6_jpg/lib/cmake/opencv4
# CUDA path variable
CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-12.6
# Torch path variable
TORCH_DIR=/home/shaun/libtorch/libtorch-cu118/share/cmake/Torch

# DBoW2
cd ./ORB-SLAM3/Thirdparty/DBoW2
mkdir build
cd build
# cmake .. -DCMAKE_BUILD_TYPE=Release # add OpenCV_DIR definitions if needed, example:
cmake .. -DCMAKE_BUILD_TYPE=Release \
  -DOpenCV_DIR=$OPENCV_DIR \
  -DCUDA_TOOLKIT_ROOT_DIR=$CUDA_TOOLKIT_ROOT_DIR \
  -DCMAKE_CUDA_COMPILER=$CUDA_TOOLKIT_ROOT_DIR/bin/nvcc
make -j8

cd ../../g2o

# g2o
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j8

cd ../../Sophus

# Sophus
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j8

# ORB_SLAM3
cd ../../../Vocabulary
echo "Uncompress vocabulary ..."
tar -xf ORBvoc.txt.tar.gz
cd ..

mkdir build
cd build
# cmake .. -DCMAKE_BUILD_TYPE=Release # add OpenCV_DIR definitions if needed, example:
cmake .. -DCMAKE_BUILD_TYPE=Release \
  -DOpenCV_DIR=$OPENCV_DIR \
  -DCUDA_TOOLKIT_ROOT_DIR=$CUDA_TOOLKIT_ROOT_DIR \
  -DCMAKE_CUDA_COMPILER=$CUDA_TOOLKIT_ROOT_DIR/bin/nvcc
make -j8

# Photo-SLAM
echo "Building Photo-SLAM ..."
cd ../..
mkdir build
cd build
# cmake .. # add Torch_DIR and/or OpenCV_DIR definitions if needed, example:
cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DTorch_DIR=$TORCH_DIR \
  -DOpenCV_DIR=$OPENCV_DIR \
  -DCUDA_TOOLKIT_ROOT_DIR=$CUDA_TOOLKIT_ROOT_DIR \
  -DCMAKE_CUDA_COMPILER=$CUDA_TOOLKIT_ROOT_DIR/bin/nvcc
make -j8

