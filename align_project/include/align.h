#pragma once

#include "io.h"
#include "matrix.h"

std::tuple<int, int, int, int> pyramid(Image redImage, Image greImage, Image bluImage, int shiftrange,
											int rm0, int rn0, int bm0, int bn0, double scale);

Image align(Image srcImage, bool isPostprocessing, std::string postprocessingType, double fraction, bool isMirror, 
            bool isInterp, bool isSubpixel, double subScale);  

Image mirror(Image srcImage, uint range);

Image sobel_x(Image srcImage);

Image sobel_y(Image srcImage);

Image unsharp(Image srcImage);

Image gray_world(Image srcImage);

Image resize(Image srcImage, double scale);

Image custom(Image srcImage, Matrix<double> kernel);

Image autocontrast(Image srcImage, double fraction);

Image gaussian(Image srcImage, double sigma, int radius);

Image gaussian_separable(Image srcImage, double sigma, int radius);

Image median(Image srcImage, int radius);

Image median_linear(Image srcImage, int radius);

Image median_const(Image srcImage, int radius);

Image canny(Image srcImage, int threshold1, int threshold2);

