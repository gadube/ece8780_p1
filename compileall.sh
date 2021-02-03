#!/bin/sh

rm -rf build/*

make clean
sed -i "s/sm\_70/sm\_35/" Makefile
sed -i "s/BLOCK=32/BLOCK=4/" Makefile
make
mv gray ./build/gray_35_4
make clean
sed -i "s/BLOCK=4/BLOCK=8/" Makefile
make
mv gray ./build/gray_35_8
make clean
sed -i "s/BLOCK=8/BLOCK=16/" Makefile
make
mv gray ./build/gray_35_16
make clean
sed -i "s/BLOCK=16/BLOCK=32/" Makefile
make
mv gray ./build/gray_35_32

sed -i "s/sm\_35/sm\_60/" Makefile
make clean
sed -i "s/BLOCK=32/BLOCK=4/" Makefile
make
mv gray ./build/gray_60_4
make clean
sed -i "s/BLOCK=4/BLOCK=8/" Makefile
make
mv gray ./build/gray_60_8
make clean
sed -i "s/BLOCK=8/BLOCK=16/" Makefile
make
mv gray ./build/gray_60_16
make clean
sed -i "s/BLOCK=16/BLOCK=32/" Makefile
make
mv gray ./build/gray_60_32

sed -i "s/sm\_60/sm\_70/" Makefile
make clean
sed -i "s/BLOCK=32/BLOCK=4/" Makefile
make
mv gray ./build/gray_70_4
make clean
sed -i "s/BLOCK=4/BLOCK=8/" Makefile
make
mv gray ./build/gray_70_8
make clean
sed -i "s/BLOCK=8/BLOCK=16/" Makefile
make
mv gray ./build/gray_70_16
make clean
sed -i "s/BLOCK=16/BLOCK=32/" Makefile
make
mv gray ./build/gray_70_32
