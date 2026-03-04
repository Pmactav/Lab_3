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
    MatrixXd sigma_mm = ReadDatatoMatrix("../stdevs_2026.txt");
    MatrixXd sigma_m = sigma_mm*(.001);
    MatrixXd P = sigma_m.array().square().inverse().matrix().asDiagonal(); //square st dev, take inverse, store in main diagonal of P
    MatrixXd sigmaMatrix = P.inverse();
    MatrixXd w = B*l;
    MatrixXd M = B * P.inverse() * B.transpose();
    VectorXd k = M.ldlt().solve(w);
    VectorXd v_hat = -P.inverse() * B.transpose() * k;
    //cout << v_hat << endl;
    //VectorXd check = B * v_hat + w;
    //cout << check.transpose() << endl;
    double n = l.rows();
    double c = B.rows();
    double sigma0_sq = (v_hat.transpose()*P*v_hat)(0,0)/(n-c);
    MatrixXd Cl = sigma0_sq*sigmaMatrix;
    MatrixXd Cv = sigma0_sq*(sigmaMatrix-sigmaMatrix*B.transpose()*M.inverse()*B*sigmaMatrix);
    VectorXd std_v = Cv.diagonal().cwiseSqrt();
    VectorXd l_hat = l + v_hat;
    MatrixXd C = Cl - Cv;
    VectorXd check = B * v_hat + w;
    cout << check.transpose() << endl;
    cout << l_hat.transpose() << endl;

    return 0;
}