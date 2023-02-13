rm -r bin
mkdir bin

make

rm *.o
mv main bin
