# -- Uncomment the following lines if running @ARIS --
# echo "module load cmake"
# module load cmake

# echo "module load gnu/9.2.0"
# module load gnu/9.2.0

if [ ! -d "build" ]; then
  echo "mkdir build"
  echo "--project folder <build> : created"
  mkdir build
fi

cd build

cmake -DCMAKE_BUILD_TYPE=Release ..

make -j

export OMP_PLACES=cores

cd ../bin
