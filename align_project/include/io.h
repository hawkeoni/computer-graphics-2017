#pragma once

#include "matrix.h"
#include "EasyBMP.h"

#include <tuple>

typedef Matrix<std::tuple<uint, uint, uint>> Image;


Image load_image(const char*);
void save_image(const Image&, const char*);


class custom_operator
{
public:
    custom_operator(uint rad, Matrix<double> kern) : radius(rad), kernel(kern) {}

    uint radius;
    Matrix<double> kernel;
    std::tuple<uint, uint, uint> operator () (const Image &srcImage) const{
        double Sr = 0, Sg = 0, Sb = 0;
        uint i, j;
        for (i = 0; i < radius * 2 + 1; i++){
            for (j = 0 ; j < radius * 2 + 1; j++){
                Sr += static_cast<double>(std::get<0>(srcImage(i, j))) * kernel(i, j);
                Sg += static_cast<double>(std::get<1>(srcImage(i, j))) * kernel(i, j);
                Sb += static_cast<double>(std::get<2>(srcImage(i, j))) * kernel(i, j);
            }
        }

        Sr = (Sr > 255) ? 255 : Sr;
        Sr = (Sr < 0) ? 0 : Sr; 
        Sg = (Sg > 255) ? 255 : Sg;
        Sg = (Sg < 0) ? 0 : Sg;
        Sb = (Sb > 255) ? 255 : Sb;
        Sb = (Sb < 0) ? 0 : Sb;

        return std::make_tuple(Sr, Sg, Sb);
    }

};