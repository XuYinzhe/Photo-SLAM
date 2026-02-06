echo "Only Building Photo-SLAM ..."
cd build
cmake .. -DTorch_DIR=/home/shaun/libtorch/libtorch-cu118/share/cmake/Torch/ -DOpenCV_DIR=/home/shaun/opencv/opencv-4.10.0/install_12-6/lib/cmake/opencv4/
make