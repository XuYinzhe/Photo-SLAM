echo "Only Building Photo-SLAM ..."
cd build
cmake .. -DTorch_DIR=/home/shaun/libtorch/libtorch-cu118/share/cmake/Torch/ -DOpenCV_DIR=/usr/local/lib/cmake/opencv4/ -DTESTONLY=ON
make