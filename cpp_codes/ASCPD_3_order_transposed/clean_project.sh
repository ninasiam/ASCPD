if [ -d "build" ]; then
  rm -rf build/*
  echo "removed files from <build>" 
fi

if [ -d "bin" ]; then
  rm -rf bin/*
  echo "removed files from <bin>"  
fi

