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
    //Read in data
    MatrixXd B = ReadDatatoMatrix("../bMatrix.txt");
    MatrixXd l = ReadDatatoMatrix("../dhs_2026.txt");
    MatrixXd sigma_mm = ReadDatatoMatrix("../stdevs_2026.txt");
    //Unit conversion, populate vectors to diagonals where required
    MatrixXd sigma_m = sigma_mm*(.001);
    MatrixXd P = sigma_m.array().square().inverse().matrix().asDiagonal(); //square st dev, take inverse, store in main diagonal of P
    MatrixXd stVarMatrix = sigma_m.array().square().matrix().asDiagonal();
    //Begin populating parameters
    MatrixXd w = B*l;
    //MatrixXd test = w.rowwise().sum();
    //cout << test.transpose() << endl;
    MatrixXd M = B * P.inverse() * B.transpose();
    VectorXd k = M.ldlt().solve(w); //create Legrange Multiplier
    VectorXd v_hat = -P.inverse() * B.transpose() * k; //create vector of residuals
    double n = l.rows();
    double c = B.rows();
    double sigma0_sq = (v_hat.transpose()*P*v_hat)(0,0)/(n-c);
    MatrixXd Cl = MatrixXd::Identity(n,n);
    MatrixXd Cv = sigma0_sq*(stVarMatrix-stVarMatrix*B.transpose()*M.inverse()*B*stVarMatrix);
    VectorXd std_v = Cv.diagonal().cwiseSqrt();
    VectorXd l_hat = l + v_hat;
    MatrixXd Cl_hat = Cl - Cv;

    MatrixXd J = ReadDatatoMatrix("../jMatrix.txt");
    //MatrixXd J_l = J + l_hat;
    //VectorXd stationHeight = J_l.colwise().sum()+135.961;
    MatrixXd Cx = J*Cl_hat*J.transpose();
    WriteMatrixToFile(Cx, "../Cx.txt", 10);
    cout << Cx << endl;
    return 0;
}