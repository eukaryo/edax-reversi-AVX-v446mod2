#!/bin/sh
mkdir bin
cd src
make build ARCH=x64-modern COMP=gcc
cd ../
cp bin/lEdax-x64-modern Edax_mod2
bunzip2 -k eval.dat.bz2
mkdir data
mv eval.dat data
./Edax_mod2 -solve problem/fforum-1-19.obf -n-tasks 1 -level 60 -hash-table-size 23 -verbose 2 -width 200
