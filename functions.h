//
// Created by Peter on 2/4/2026.
//

#ifndef LAB_2_FUNCTIONS_H
#define LAB_2_FUNCTIONS_H
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <iomanip>
#include <vector>
#include <Eigen/Dense>

using namespace std;
using namespace Eigen;

MatrixXd ReadDatatoMatrix(const string& filename);
void WriteMatrixToFile(const MatrixXd& Mat, const string& filename, unsigned int precision);


#endif //LAB_2_FUNCTIONS_H