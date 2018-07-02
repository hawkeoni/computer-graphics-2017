#include <string>
#include <vector>
#include <fstream>
#include <cassert>
#include <iostream>
#include <cmath>

#include "classifier.h"
#include "EasyBMP.h"
#include "linear.h"
#include "argvparser.h"
#include "matrix.h"

#define GRAIN 8
#define SEGMENTS 32

using std::string;
using std::vector;
using std::ifstream;
using std::ofstream;
using std::pair;
using std::make_pair;
using std::cout;
using std::cerr;
using std::endl;
using std::acos;
using std::make_pair;

using CommandLineProcessing::ArgvParser;

typedef vector<pair<BMP*, int> > TDataSet;
typedef vector<pair<string, int> > TFileList;
typedef vector<pair<vector<float>, int> > TFeatures;


typedef Matrix<float> matr;

// Load list of files and its labels from 'data_file' and
// stores it in 'file_list'
void LoadFileList(const string& data_file, TFileList* file_list) {
    ifstream stream(data_file.c_str());

    string filename;
    int label;
    
    int char_idx = data_file.size() - 1;
    for (; char_idx >= 0; --char_idx)
        if (data_file[char_idx] == '/' || data_file[char_idx] == '\\')
            break;
    string data_path = data_file.substr(0,char_idx+1);
    
    while(!stream.eof() && !stream.fail()) {
        stream >> filename >> label;
        if (filename.size())
            file_list->push_back(make_pair(data_path + filename, label));
    }

    stream.close();
}

// Load images by list of files 'file_list' and store them in 'data_set'
void LoadImages(const TFileList& file_list, TDataSet* data_set) {
    for (size_t img_idx = 0; img_idx < file_list.size(); ++img_idx) {
            // Create image
        BMP* image = new BMP();
            // Read image from file
        image->ReadFromFile(file_list[img_idx].first.c_str());
            // Add image and it's label to dataset
        data_set->push_back(make_pair(image, file_list[img_idx].second));
    }
}

// Save result of prediction to file
void SavePredictions(const TFileList& file_list,
                     const TLabels& labels, 
                     const string& prediction_file) {
        // Check that list of files and list of labels has equal size 
    assert(file_list.size() == labels.size());
        // Open 'prediction_file' for writing
    ofstream stream(prediction_file.c_str());

        // Write file names and labels to stream
    for (size_t image_idx = 0; image_idx < file_list.size(); ++image_idx)
        stream << file_list[image_idx].first << " " << labels[image_idx] << endl;
    stream.close();
}


matr grayscale(BMP *image){
    int width = image->TellWidth(), height = image->TellHeight();
    matr im(height, width);
    for (int i = 0; i < height; i++){
        for (int j = 0; j < width; j++){
            RGBApixel pixel = image->GetPixel(j, i);
            float y = 0.299 * pixel.Red + 0.587 * pixel.Blue + 0.114 * pixel.Green;
            im(i, j) = y;
        }
    }
    return im;
}

matr sobelX(matr grayscale){
    int height = grayscale.n_rows, width = grayscale.n_cols;
    matr X(height, width);
    for (int i = 1; i < height - 1; i++){
        for (int j = 1; j < width - 1; j++){
            X(i, j) = -1 * grayscale(i, j - 1) + 1 * grayscale(i, j + 1);
        }
    }
    return X.submatrix(1, 1, height - 2, width - 2);
}

matr sobelY(matr grayscale){
    int height = grayscale.n_rows, width = grayscale.n_cols;
    matr Y(height, width);
    for (int i = 1; i < height - 1; i++){
        for (int j = 1; j < width - 1; j++){
            Y(i, j) = 1 * grayscale(i - 1, j) - 1 * grayscale(i + 1, j);
        }
    }
    return Y.submatrix(1, 1, height - 2, width - 2);
}

vector<float> LBP(matr g){
    int height = g.n_rows, width = g.n_cols;
    matr pixel_binary_values(height - 2, width -2);
    for (int i = 1; i < height - 1; i++){
        for (int j = 1; j < width - 1; j++){
            int cur = g(i, j);
            pixel_binary_values(i - 1, j - 1) = pow(2, 0) * (g(i - 1, j - 1) > cur ? 1 : 0);
            pixel_binary_values(i - 1, j - 1) += pow(2, 1) * (g(i - 1, j) > cur ? 1 : 0);
            pixel_binary_values(i - 1, j - 1) += pow(2, 2) * (g(i - 1, j + 1) > cur ? 1 : 0);
            pixel_binary_values(i - 1, j - 1) += pow(2, 3) * (g(i, j + 1) > cur ? 1 : 0);
            pixel_binary_values(i - 1, j - 1) += pow(2, 4) * (g(i + 1, j + 1) > cur ? 1 : 0);
            pixel_binary_values(i - 1, j - 1) += pow(2, 5) * (g(i + 1, j) > cur ? 1 : 0);
            pixel_binary_values(i - 1, j - 1) += pow(2, 6) * (g(i + 1, j - 1) > cur ? 1 : 0);
            pixel_binary_values(i - 1, j - 1) += pow(2, 7) * (g(i, j - 1) > cur ? 1 : 0);
        }
    }
    int size = GRAIN, ni, nj;
    Matrix<vector<float>> cell_histogramm(size, size);
    for (int i = 0; i < size; i++){
        for (int j = 0; j < size; j++){
            for (int k = 0 ; k < 256; k++)
                cell_histogramm(i, j).push_back(0);
            }
        }
    height = pixel_binary_values.n_rows; width = pixel_binary_values.n_cols;
    int cell_h = height / size, cell_w = width / size;
    for (int i = 0; i < height; i++){
        for (int j = 0; j < width; j++){
            ni = i / cell_h; nj = j / cell_w;
            ni = ni >= size ? size - 1 : ni;
            nj = nj >= size ? size - 1 : nj;
            cell_histogramm(ni, nj)[pixel_binary_values(i, j)] += 1;
        }
    }
    float s = 0;
    for (int i = 0; i < size; i++){
        for (int j = 0; j < size; j++){
            s = 0;
            for (int k = 0; k < 256; k++)
                s += cell_histogramm(i, j)[k] * cell_histogramm(i, j)[k];
            s = pow(s, 0.5);
            for (int k = 0; k < 256; k++){
                if (s > 0)
                    cell_histogramm(i, j)[k] /= s;
            }
        }
    }
    vector<float> lbpdesc;
    for (int i = 0; i < size; i++)
        for (int j = 0; j < size; j++)
            lbpdesc.insert(lbpdesc.end(), cell_histogramm(i, j).begin(), cell_histogramm(i, j).end());
    return lbpdesc;
}

vector<float> CF(BMP *image){
    int height = image->TellHeight(), width = image->TellWidth();
    int size = 8;
    int ni, nj;
    Matrix<int> pixel_amount(size, size);
    matr red(size, size), gre(size, size), blu(size, size);
    for (int i = 0; i < size; i++){
        for (int j = 0; j < size; j++){
            red(i, j) = 0; gre(i, j) = 0; blu(i, j) = 0;
            pixel_amount(i, j) = 0;
        }
    }
    int cell_h = height / size, cell_w = width / size;
    RGBApixel p;
    for (int i = 0; i < height; i++){
        for (int j = 0; j < width; j++){
            ni = i / cell_h; nj = j / cell_w;
            ni = ni >= size ? size - 1 : ni;
            nj = nj >= size ? size - 1 : nj;
            p = image->GetPixel(j, i);
            red(ni, nj) += p.Red; gre(ni, nj) += p.Green; blu(ni, nj) += p.Blue;
            pixel_amount(ni, nj) += 1;
        }
    }
    vector<float> colordesc;
    for (int i = 0; i < size; i++){
        for (int j = 0; j < size; j++){
            int amnt = pixel_amount(i, j);
            red(i, j) /= amnt;
            gre(i, j) /= amnt;
            blu(i, j) /= amnt;
            colordesc.push_back(red(i, j) / 255); colordesc.push_back(gre(i, j) / 255); colordesc.push_back(blu(i, j) / 255);
        }
    }
    return colordesc;
}

void ExtractFeatures(const TDataSet& data_set, TFeatures* features) {
    const float pi = 2 * acos(0);
    for (size_t image_idx = 0; image_idx < data_set.size(); ++image_idx) {
       BMP *image = data_set[image_idx].first;
       int width = image->TellWidth(), height = image->TellHeight();
       features = features;
       matr gs = grayscale(image);
       gs.extra_borders(1, 1);
       matr X = sobelX(gs);
       matr Y = sobelY(gs);
       height = X.n_rows; width = X.n_cols;
       matr abs(height, width);
       matr dir(height, width);
       vector<float> lbp = LBP(gs);
       vector<float> clr = CF(image);
       for (int i = 0; i < height; i++){
           for (int j = 0; j < width; j++){
               abs(i, j) = std::pow(X(i, j) * X(i, j) + Y(i, j) * Y(i, j), 0.5);
               dir(i, j) = std::atan2(Y(i, j), X(i, j)) + pi;
           }
       }
       int size = GRAIN, segs = SEGMENTS;
       float delta = 2 * pi / segs; 
       matr norms(size, size);
       int cell_h = height / size, cell_w = width / size;
       Matrix<vector<float>> cell_histogramm(size, size);
       int ni, nj, angle;
       for (int i = 0; i < size; i++){
        for (int j = 0; j < size; j++){
            for (int k = 0 ; k < segs; k++){
                cell_histogramm(i, j).push_back(0);
            }
        }
    }
    for (int i = 0; i < height; i++){
        for (int j = 0; j < width; j++){
            ni = i / cell_h; nj = j / cell_w;
            ni = ni >= size ? size - 1 : ni;
            nj = nj >= size ? size - 1 : nj;
            angle = dir(i, j) / delta;
            if (angle >= segs) angle = segs - 1;
            cell_histogramm(ni, nj)[angle] += abs(i, j);
        }
    }
    float s = 0;
    for (int i = 0; i < size; i++){
        for (int j = 0; j < size; j++){
            s = 0;
            for (int k = 0; k < segs; k++)
                s += cell_histogramm(i, j)[k] * cell_histogramm(i, j)[k];
            s = pow(s, 0.5);
            for (int k = 0; k < segs; k++){
                if (s > 0)
                    cell_histogramm(i, j)[k] /= s;
            }
        }
    }

    vector<float> desc;
    for (int i = 0; i < size; i++){
        for (int j = 0; j < size; j++){            
            desc.insert(desc.end(), cell_histogramm(i,j).begin(), cell_histogramm(i, j).end());
        }
    }
    desc.insert(desc.end(), lbp.begin(), lbp.end());
    desc.insert(desc.end(), clr.begin(), clr.end());
    features->push_back(make_pair(desc, data_set[image_idx].second));
}
return;
}



// Clear dataset structure
void ClearDataset(TDataSet* data_set) {
        // Delete all images from dataset
    for (size_t image_idx = 0; image_idx < data_set->size(); ++image_idx)
        delete (*data_set)[image_idx].first;
        // Clear dataset
    data_set->clear();
}

// Train SVM classifier using data from 'data_file' and save trained model
// to 'model_file'
void TrainClassifier(const string& data_file, const string& model_file) {
        // List of image file names and its labels
    TFileList file_list;
        // Structure of images and its labels
    TDataSet data_set;
        // Structure of features of images and its labels
    TFeatures features;
        // Model which would be trained
    TModel model;
        // Parameters of classifier
    TClassifierParams params;
    
        // Load list of image file names and its labels
    LoadFileList(data_file, &file_list);
        // Load images
    LoadImages(file_list, &data_set);
        // Extract features from images
    ExtractFeatures(data_set, &features);
        // PLACE YOUR CODE HERE
        // You can change parameters of classifier here
    params.C = 0.01;
    TClassifier classifier(params);
        // Train classifier
    classifier.Train(features, &model);
        // Save model to file
    model.Save(model_file);
        // Clear dataset structure
    ClearDataset(&data_set);
}

// Predict data from 'data_file' using model from 'model_file' and
// save predictions to 'prediction_file'
void PredictData(const string& data_file,
                 const string& model_file,
                 const string& prediction_file) {
        // List of image file names and its labels
    TFileList file_list;
        // Structure of images and its labels
    TDataSet data_set;
        // Structure of features of images and its labels
    TFeatures features;
        // List of image labels
    TLabels labels;

        // Load list of image file names and its labels
    LoadFileList(data_file, &file_list);
        // Load images
    LoadImages(file_list, &data_set);
        // Extract features from images
    ExtractFeatures(data_set, &features);
        // Classifier 
    TClassifier classifier = TClassifier(TClassifierParams());
        // Trained model
    TModel model;
        // Load model from file
    model.Load(model_file);
        // Predict images by its features using 'model' and store predictions
        // to 'labels'
    classifier.Predict(features, model, &labels);
        // Save predictions
    SavePredictions(file_list, labels, prediction_file);
        // Clear dataset structure
    ClearDataset(&data_set);
}

int main(int argc, char** argv) {
    // Command line options parser
    ArgvParser cmd;
        // Description of program
    cmd.setIntroductoryDescription("Machine graphics course, task 2. CMC MSU, 2014.");
        // Add help option
    cmd.setHelpOption("h", "help", "Print this help message");
        // Add other options
    cmd.defineOption("data_set", "File with dataset",
        ArgvParser::OptionRequiresValue | ArgvParser::OptionRequired);
    cmd.defineOption("model", "Path to file to save or load model",
        ArgvParser::OptionRequiresValue | ArgvParser::OptionRequired);
    cmd.defineOption("predicted_labels", "Path to file to save prediction results",
        ArgvParser::OptionRequiresValue);
    cmd.defineOption("train", "Train classifier");
    cmd.defineOption("predict", "Predict dataset");
        
        // Add options aliases
    cmd.defineOptionAlternative("data_set", "d");
    cmd.defineOptionAlternative("model", "m");
    cmd.defineOptionAlternative("predicted_labels", "l");
    cmd.defineOptionAlternative("train", "t");
    cmd.defineOptionAlternative("predict", "p");

        // Parse options
    int result = cmd.parse(argc, argv);

        // Check for errors or help option
    if (result) {
        cout << cmd.parseErrorDescription(result) << endl;
        return result;
    }

        // Get values 
    string data_file = cmd.optionValue("data_set");
    string model_file = cmd.optionValue("model");
    bool train = cmd.foundOption("train");
    bool predict = cmd.foundOption("predict");

        // If we need to train classifier

    if (train)
        TrainClassifier(data_file, model_file);
        // If we need to predict data
    if (predict) {
            // You must declare file to save images
        if (!cmd.foundOption("predicted_labels")) {
            cerr << "Error! Option --predicted_labels not found!" << endl;
            return 1;
        }
            // File to save predictions
        string prediction_file = cmd.optionValue("predicted_labels");
            // Predict data
        PredictData(data_file, model_file, prediction_file);
    }
}