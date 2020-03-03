if [ -d "build" ]; then
  rm -rf build/*
  echo "--project folder <build> : cleaned !"
fi

if [ -d "bin" ]; then
  rm -rf bin/*
  echo "--binary files in project folder <bin> : removed!"  
fi

files=($PWD/*.dat)
###if [[ ${files} =~ \.dat$ ]]; then
if [ -f "$files" ]; then
  rm  *.dat
  echo "--binary project files removed!"  
fi

