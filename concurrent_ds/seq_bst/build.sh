rm -r build
rm -r bin
mkdir build
mkdir bin

make

mv *.o build
mv main bin
