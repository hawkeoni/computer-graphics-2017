#include "align.h"
#include <string>
#include <vector>

#define abs_diff_sqr(x, y) ((x)<(y) ? (((y) - (x)) * ((y) - (x))) : (((x) - (y)) * ((x) - (y))))
#define abs(x) ((x > 0) ? (x) : (-1)*(x))

using std::string;
using std::get;
using std::make_tuple;


std::tuple<int, int, int, int> pyramid(Image redImage, Image greImage, Image bluImage, int shiftrange,
                                            int rm0, int rn0, int bm0, int bn0)
{
    uint height = redImage.n_rows, width = redImage.n_cols, r_min_mse = 4294967295, b_min_mse = 4294967295, r_mse = 0, b_mse = 0;
    int m = 0, n = 0, rm = 0, rn = 0, bm = 0, bn = 0;
    uint i, j;
    if (std::max(width, height) > 400){
        shiftrange = 2;
        auto r = pyramid(resize(redImage, 0.5), resize(greImage, 0.5), resize(bluImage, 0.5), 15, 0, 0, 0, 0);
        rm0 += 2 * get<0>(r); rn0 += 2 * get<1>(r); bm0 +=2 * get<2>(r); bn0 += 2 * get<3>(r);

    }

    Image bigImage(height + 50, width + 50);

    height -= 50; width -= 50;
    for (i = 50; i < height + 50; i++){
        for (j = 50; j < width + 50; j++){
            bigImage(i, j) = greImage(i - 50, j - 50);
        }
    }
    for (m = -shiftrange; m < shiftrange; m++){
        for (n = -shiftrange; n < shiftrange; n++){
            r_mse = 0;
            b_mse = 0;
            for (i = 0; i < height; i++){
                for (j = 0; j < width; j++){                  
                    if ((static_cast<int>(i) - m - rm0 + 50 >= 50) && (static_cast<int>(i) - m - rm0 + 50 <= static_cast<int>(height) + 50) && (static_cast<int>(j) - n - rn0 + 50 >= 50) && (static_cast<int>(j) - n - rn0 + 50 <= static_cast<int>(width) + 50)){          
                           r_mse += abs_diff_sqr(get<0>(redImage(i, j)), get<0>(bigImage(i - m - rm0 + 50, j - n - rn0 + 50)));
                    }                    
                    if ((static_cast<int>(i) - m - bm0 + 50 >= 50) && (static_cast<int>(i) - m - bm0 + 50 <= static_cast<int>(height) + 50) && (static_cast<int>(j) - n - bn0 + 50 >= 50) && (static_cast<int>(j) - n - bn0 + 50 <= static_cast<int>(width) + 50)){
                           b_mse += abs_diff_sqr(get<0>(bluImage(i, j)), get<0>(bigImage(i - m - bm0 + 50, j - n - bn0 + 50)));
                    }
            }
        }
            if (r_min_mse > r_mse*1/((width - abs(m) - abs(rm0))*(height - abs(n) - abs(rn0)))) {
                r_min_mse = r_mse*1/((width - abs(m) - abs(rm0))*(height - abs(n) - abs(rn0)));
                rm = m; rn = n;
            }
            if (b_min_mse > b_mse*1/((width - abs(m) - abs(bm0))*(height - abs(n) - abs(bn0)))){
                b_min_mse = b_mse*1/((width - abs(m) - abs(bm0))*(height - abs(n) - abs(bn0)));
                bm = m; bn = n;
            }
        }
    }
    return make_tuple(rm + rm0, rn + rn0, bm + bm0, bn +bn0);
}

Image align(Image srcImage, bool isPostprocessing, std::string postprocessingType, double fraction, bool isMirror, 
            bool isInterp, bool isSubpixel, double subScale)
{   
    if (isSubpixel) return resize(align(resize(srcImage, subScale), isPostprocessing, postprocessingType, fraction, isMirror, isInterp, false, 0), 
        1.0 / subScale);
    uint height = srcImage.n_rows/3, width = srcImage.n_cols;
    int rm = 0, rn = 0, bm = 0, bn = 0; //, srm = 0, srn = 0, sbm = 0, sbn = 0;
    uint i, j;
    uint delta_h = 0.05 * height, delta_w = width * 0.05;
    Image blu_Image = srcImage.submatrix(delta_h, delta_w, height - delta_h, width - delta_w); //Change this to percent, not to -15..15
    Image gre_Image = srcImage.submatrix(height + delta_h, delta_w, height - delta_h, width - delta_w);
    Image red_Image = srcImage.submatrix(2*height + delta_h, delta_w, height - delta_h, width - delta_w);
    Image &redImage = red_Image, &greImage = gre_Image, &bluImage = blu_Image;    


  
    std::tie(rm, rn, bm, bn) = pyramid(redImage, greImage, bluImage, 15, 0, 0, 0, 0);

    Image bigImage(height + 50, width + 50);
    height -= delta_h;
    width -= delta_w;
    //putting images together
    for (i = 50; i < height + 50; i++){
        for (j = 50; j< width + 50; j++){
            bigImage(i, j) = greImage(i - 50, j - 50);
        }
    }

    for (i = 50; i < height + 50; i++){
        for (j = 50; j< width + 50; j++){
            if ((static_cast<int>(i) - rm >=0) && (static_cast<int>(i) - rm <= static_cast<int>(height) + 50) && (static_cast<int>(j) - rn >= 0) && (static_cast<int>(j) - rn <= static_cast<int>(width) + 50))
            bigImage(i - rm, j - rn) = make_tuple(get<0>(redImage(i - 50, j - 50)), get<1>(bigImage(i - rm, j - rn)), 0);
        }
    }

    for (i = 50; i < height + 50; i++){
        for (j = 50; j< width + 50; j++){
            if ((static_cast<int>(i) - bm >=0) && (static_cast<int>(i) - bm <= static_cast<int>(height) + 50) && (static_cast<int>(j) - bn >= 0) && (static_cast<int>(j) - bn <= static_cast<int>(width) + 50))
            bigImage(i - bm, j - bn) = make_tuple(get<0>(bigImage(i - bm, j - bn)), get<1>(bigImage(i - bm, j - bn)), get<0>(bluImage(i - 50 , j - 50))); 
        }
    }
    
    Image resImage = bigImage.submatrix(50, 50, height, width);    
    if (isMirror) resImage = mirror(resImage, 1);
    if (isPostprocessing){
        if (postprocessingType == "--gray-world"){
            resImage = gray_world(resImage);
            return resImage;
        }
        else if (postprocessingType == "--autocontrast"){
            resImage = autocontrast(resImage, fraction);
            return resImage;
        }
        else if (postprocessingType == "--unsharp"){
            resImage = unsharp(resImage);
            return resImage;
        }
    }
    return resImage;
}

Image mirror(Image srcImage, uint range){
    uint height = srcImage.n_rows, width = srcImage.n_cols;
    Image resImage(height + 2 * range, width + 2 * range);
    uint i, j;
    
    for (i = 0; i < height; i++){
        for (j = 0; j < width; j++){    
            resImage(i + range, j + range) = srcImage(i, j);
        }
    }
    for (i = 0; i < range; i++){
        for (j = 0; j < width; j++){
            resImage(i, j) = resImage(-i + 2 * range, j);
            resImage(height - i - 1, j) = resImage(height + i - 1 - 2 * range, j);
        }
    }
    for (i = 0; i < height; i++){
        for (j = 0; j < range; j++){
             resImage(i, j) = resImage(i, 2 * range - j);
             resImage(i, j) = resImage(i, width + j - 1 - 2 * range);
        }
    }
    
    for (i = 0; i < range; i++){
        for (j = 0; j < range; j++){
            resImage(i, j) = resImage(2 * range - i, 2 * range - j); //(0, 0)
            resImage(i, width - j - 1) = resImage(2 * range - i, width - 1 + j - 2 * range); //(0, width)
            resImage(height - i - 1, width - j - 1) = resImage(height - 1 + i - 2 * range, width - 1 + j - 2 * range); //(height, width)
            resImage(height - i - 1, j) = resImage(height - 1 + i - 2 * range, 2 * range - j); //(height, 0)
        }
    }
    return resImage;

}

Image sobel_x(Image srcImage) {
    Matrix<double> kernel = {{-1, 0, 1},
                             {-2, 0, 2},
                             {-1, 0, 1}};
    return custom(srcImage, kernel);
}

Image sobel_y(Image srcImage) {
    Matrix<double> kernel = {{ 1,  2,  1},
                             { 0,  0,  0},
                             {-1, -2, -1}};
    return custom(srcImage, kernel);
}

Image unsharp(Image srcImage) {
    Matrix<double> kernel = {{-1.0 / 6.0, -2.0 / 3.0, -1.0 / 6.0},
                             {-2.0 / 3.0, 13.0 / 3.0, -2.0 / 3.0},
                             {-1.0 / 6.0, -2.0 / 3.0, -1.0 / 6.0}};
    return custom(srcImage, kernel);
}

Image gray_world(Image srcImage) {
    uint S, Sr = 0, Sg = 0, Sb = 0;
    uint height = srcImage.n_rows, width = srcImage.n_cols;
    uint i, j;
    uint new_red, new_gre, new_blu;
    for (i = 0; i < height; i++){
        for (j = 0; j < width; j++){
            Sr += get<0>(srcImage(i, j));
            Sg += get<1>(srcImage(i, j));
            Sb += get<2>(srcImage(i, j));
        }
    }
    Sr /= height * width;
    Sg /= height * width;
    Sb /= height * width;
    S = (Sr + Sg + Sb) / 3;
    for (i = 0; i < height; i++){
        for (j = 0; j < width; j++){
            new_red = get<0>(srcImage(i, j)) * S / Sr;
            new_red = (new_red < 255) ? new_red : 255;
            new_gre = get<1>(srcImage(i, j)) * S / Sg;
            new_gre = (new_gre < 255) ? new_gre : 255;
            new_blu = get<2>(srcImage(i, j)) * S / Sb;
            new_blu = (new_blu < 255) ? new_blu : 255;
            srcImage(i, j) = make_tuple(new_red,
                                        new_gre,
                                        new_blu
                                        );
        }
    }
    return srcImage;
}

Image resize(Image srcImage, double scale) {
    uint width = srcImage.n_cols, height = srcImage.n_rows;
    uint n_width = static_cast<uint>(scale * width), n_height = static_cast<uint>(scale * height);
    Image resImage(n_height, n_width);
    uint i, j, i_int, j_int;
    double i_diff, j_diff;    
    for (i = 0; i < n_height - static_cast<uint>(1 * scale); i++){
        for (j = 0; j < n_width - static_cast<uint>(1 * scale); j++){            
            i_int = static_cast<uint>(i / scale);
            j_int = static_cast<uint>(j / scale);
            i_diff = i / scale - static_cast<double>(i_int);
            j_diff = j / scale - static_cast<double>(j_int);            
            resImage(i,j) = make_tuple(
                get<0>(srcImage(i_int, j_int)) * (1 - i_diff) * (1 - j_diff) + 
                get<0>(srcImage(i_int + 1, j_int)) * i_diff * (1 - j_diff) + 
                get<0>(srcImage(i_int, j_int + 1)) * (1 - i_diff) * j_diff + 
                get<0>(srcImage(i_int + 1, j_int + 1)) * i_diff * j_diff,

                get<1>(srcImage(i_int, j_int)) * (1 - i_diff) * (1 - j_diff) + 
                get<1>(srcImage(i_int + 1, j_int)) * i_diff * (1 - j_diff) + 
                get<1>(srcImage(i_int, j_int + 1)) * (1 - i_diff) * j_diff + 
                get<1>(srcImage(i_int + 1, j_int + 1)) * i_diff * j_diff,

                get<2>(srcImage(i_int, j_int)) * (1 - i_diff) * (1 - j_diff) + 
                get<2>(srcImage(i_int + 1, j_int)) * i_diff * (1 - j_diff) + 
                get<2>(srcImage(i_int, j_int + 1)) * (1 - i_diff) * j_diff + 
                get<2>(srcImage(i_int + 1, j_int + 1)) * i_diff * j_diff

                );
        }
    }
    return resImage;
}

Image custom(Image srcImage, Matrix<double> kernel) {
    custom_operator helper(1, kernel);
    Image resImage = srcImage.unary_map(helper);
    return resImage;
}

Image autocontrast(Image srcImage, double fraction) {
    uint *histogramm;
    histogramm = static_cast<uint*>(calloc(256, sizeof(uint)));
    uint width = srcImage.n_cols, height = srcImage.n_rows;
    uint i, j;
    double index = 0;

    for  (i = 0; i < height; i++){
        for (j = 0; j < width; j++){
            index = 0.2125 * get<0>(srcImage(i, j)) + 0.7154 * get<1>(srcImage(i, j)) + 0.0721 * get<2>(srcImage(i, j));
            histogramm[int(index)] += 1;
        }
    }
    
    
    uint pixels = static_cast<uint>(fraction * width * height);
    int left = 0, right = 255;
    uint counter = 0;
    
    while (counter < pixels){
        counter += histogramm[left];
        left++;
    }
    counter = 0;
    while (counter < pixels){
        counter += histogramm[right];
        right--;
    }
    double a, b;
    a = (255/static_cast<double>((right - left)));
    b = (-1 * left * 255 / static_cast<double>(right - left));
    for  (i = 0; i < height; i++){
        for (j = 0; j < width; j++){
            int red = static_cast<int>(get<0>(srcImage(i, j)) * a + b);
            int gre = static_cast<int>(get<1>(srcImage(i, j)) * a + b);
            int blu = static_cast<int>(get<2>(srcImage(i, j)) * a + b);
            red = (red > 255) ? 255 : red;
            red = (red < 0) ? 0 : red; 
            gre = (gre > 255) ? 255 : gre;
            gre = (gre < 0) ? 0 : gre;
            blu = (blu > 255) ? 255 : blu;
            blu = (blu < 0) ? 0 : blu;
            srcImage(i, j) = make_tuple(red, gre, blu);
        }
    }
    return srcImage;
}

Image gaussian(Image srcImage, double sigma, int radius)  {
    return srcImage;
}

Image gaussian_separable(Image srcImage, double sigma, int radius) {
    return srcImage;
}

Image median(Image srcImage, int radius) {
    srcImage = mirror(srcImage, radius);
    Image resImage(srcImage);
    uint area = (2 * radius + 1) * (2 * radius + 1);
    using std::vector;
    vector <uint> red, gre, blu;
    red.reserve(area); gre.reserve(area); blu.reserve(area);
    uint i, j, width = srcImage.n_cols, height = srcImage.n_rows;
    int m, n;
    for (i = radius; i < height - radius; i++){
        for (j = radius; j < width - radius; j++){
            red.clear(); gre.clear(); blu.clear();
            for (m = -radius; m <= radius; m++){
                for (n = -radius; n <= radius; n++){                
                    red.push_back(get<0>(srcImage(i + m, j + n)));
                    gre.push_back(get<1>(srcImage(i + m, j + n)));
                    blu.push_back(get<2>(srcImage(i + m, j + n)));
                }
            }
            sort(red.begin(), red.end()); sort(gre.begin(), gre.end()); sort(blu.begin(), blu.end());            
            resImage(i, j) = make_tuple(red[area / 2], gre[area / 2], blu[area / 2]);
        }
    }
    return resImage;
}



Image median_linear(Image srcImage, int radius) {

    
    return srcImage;
}

Image median_const(Image srcImage, int radius) {

    return srcImage;
}

Image canny(Image srcImage, int threshold1, int threshold2) {
    return srcImage;
}