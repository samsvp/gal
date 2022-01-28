if [ ! -d build ]; then
    mkdir build
fi

cd build
make -DCMAKE_BUILD_TYPE=Release ..
make VERBOSE=1