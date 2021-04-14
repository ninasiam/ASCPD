cd build

cmake -DCMAKE_BUILD_TYPE=Release ..

make -j

#export OMP_PLACES=cores

cd ../bin
