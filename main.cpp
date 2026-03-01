#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <iomanip>
#include <numeric>
#include <vector>
#include <Eigen/Dense>
#include "functions.h"

using namespace std;
using namespace Eigen;

int main() {
    MatrixXd B = ReadDatatoMatrix("../bMatrix.txt");
    MatrixXd l = ReadDatatoMatrix("../dhs_2026.txt");
    MatrixXd sigma = ReadDatatoMatrix("../stdevs_2026.txt");
    MatrixXd A = B*l.transpose();
}