# DBoW2
cd ./ORB-SLAM3/Thirdparty/DBoW2
mkdir build
cd build
# cmake .. -DCMAKE_BUILD_TYPE=Release # add OpenCV_DIR definitions if needed, example:
cmake .. -DCMAKE_BUILD_TYPE=Release -DOpenCV_DIR=/home/shaun/opencv/opencv-4.10.0/install_12-6/lib/cmake/opencv4/
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
cmake .. -DCMAKE_BUILD_TYPE=Release -DOpenCV_DIR=/home/shaun/opencv/opencv-4.10.0/install_12-6/lib/cmake/opencv4/
make -j8

# Photo-SLAM
echo "Building Photo-SLAM ..."
cd ../..
mkdir build
cd build
# cmake .. # add Torch_DIR and/or OpenCV_DIR definitions if needed, example:
cmake .. -DTorch_DIR=/home/shaun/libtorch/libtorch-cu118/share/cmake/Torch/ -DOpenCV_DIR=/home/shaun/opencv/opencv-4.10.0/install_12-6/lib/cmake/opencv4/
make -j8

