Dimov Ilya Nikolaevich 322

OS: Macos sierra v10.12.6
Processor: 2,9 GHz Intel Core i7
Memory: 16 GB 2133 MHz LPDDR3
Graphics: Radeon Pro 560 4096 MB 
		  Intel HD Graphics 630 1536 MB

How to write settings:
resolution {WIDTH} {HEIGHT}
object_file {filename.obj}
camera_position {x y z}
savename {imgname.bmp}
source_position {x y z}
rotation_around_axes {z y x}

To make the project write "Make" in the current (wifi) directory.
To launch one of the premade (or your own settings file) 
go to bin and write "./main [settingsfile] {--filter}"

Examples: ./main set1 
		  ./main set2 --filter (--filter applies grayscale, 
		  to make the colors less screamy)


Time of execution per setting in seconds:
set1 - 56.656
set2 - 56.32
set3 - 55
setfurniture - 51.88
setmat - 40.80
setsimple - 52.92 //e.g. made with --filter option


Bonus tasks:
1) Complicated polygonal objects - furniture. Corresponds to setfurniture setting "setfurniture". See resfurniture.bmp.
Couches were borrowed from "Couch Set Iridesium".
+3 points;

2) Image postprocessing. Writing --filter as the second argument applies grayscale filter.
+1 point;

3) Windows & Doors. Can bee seen on set1, set2, set3, setmat.
Places the rays can freely traverse.
+1 point;

Total 5 bonus points + 10 base points = 15 points;



Explaination:

The colour palette goes from strong signal to weak - no signal in such a way: white - yellow - orange - magenta - purple - blue.

The walls that have no material or have it listet as "None" are solid walls and are not impregnable by rays, they reflect them instead. Example of a wall with a material can be seen in "setmat" setting. The materials are taken from the corresponding .mtl file.
.obj & .mtl files are looked up in the assets directory.




!!! IF IT IS NOT BUILDING !!!

Embree ray tracing kernel was used in this work and the only problems in generation can only occur on this step as the other libraries are ether header-only or very simple.

To install embree you have to go to externals/embree and follow those commands:
1) rm -rf build
2) mkdir build
3) cd build
4) ccmake .. (here you configure it by pressing "c" and generate the result by pressing "g". Then you press "q" to quit.)
5) make
6) sudo make install

If you do not manage to install it I'm willing to show the programm on my machine. This step is quite complicated and it can not be included in the usual makefile routine as ccmake configuration is made with ncurses and is quite complicated.