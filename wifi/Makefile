CXX = g++
CXXFLAGS = -O2 -g -Wall -std=c++0x

all: bin/main externals/EasyBMP/obj/EasyBMP.o

bin/main: externals/EasyBMP/obj/EasyBMP.o src/new.cpp
	$(CXX) $(CXXFLAGS) -I externals/OBJ-Loader/Source -I externals/embree/include/embree2 -L externals/embree/build \
	-I externals/glm -I externals/gtc  -I externals/gtx \
	-I externals/EasyBMP/include \
	src/new.cpp externals/EasyBMP/obj/EasyBMP.o -o bin/main -lembree

externals/EasyBMP/obj/EasyBMP.o:
	make -C externals/EasyBMP

clean:
	rm -f bin/main
	rm -f externals/EasyBMP/obj/EasyBMP.o externals/EasyBMP/lib/libeasybmp.a