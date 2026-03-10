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
    MatrixXd sigma_cm = ReadDatatoMatrix("../stdevs_2026.txt");
    //Unit conversion, populate vectors to diagonals where required
    MatrixXd sigma_m = sigma_cm*(.0001);
    MatrixXd P = sigma_m.array().square().inverse().matrix().asDiagonal(); //square st dev, take inverse, store in main diagonal of P
    MatrixXd stVarMatrix = sigma_m.array().square().matrix().asDiagonal();
    //Begin populating parameters and create corrections
    MatrixXd w = B*l;
    MatrixXd M = B * P.inverse() * B.transpose();
    VectorXd k = M.ldlt().solve(w); //create Legrange Multiplier
    VectorXd v_hat = -P.inverse() * B.transpose() * k; //create vector of residuals
    double r = B.rows();
    double sigma0_sq = (v_hat.transpose()*P*v_hat)(0,0)/r;
    // apply corrections and populate variance covariance
    MatrixXd sigma = sigma_m.array().square().matrix().asDiagonal();
    MatrixXd Cl = sigma0_sq * sigma;
    MatrixXd Cv = sigma0_sq*P.inverse()*B.transpose()*(B*P.inverse()*B.transpose()).inverse()*B*P.inverse();
    VectorXd l_hat = l + v_hat;
    MatrixXd Cl_hat = Cl - Cv;
    VectorXd Clstd = Cl_hat.diagonal().array().sqrt();
    //check conditional equation, accounting for floating point precision of type double
    VectorXd zerocheck = (B*v_hat+w).array().abs();
    for (int i = 0; i < zerocheck.size(); i++) {
        if (zerocheck(i) < numeric_limits<double>::epsilon()) {zerocheck(i) = 0;}}
    cout << zerocheck.transpose() << endl;
    //write values to output files
    WriteMatrixToFile(v_hat, "../v_hat_Lab_3.txt", 8);
    WriteMatrixToFile(Clstd, "../Clstd.txt", 6);
    //Task 2 - Compute Station Height and Error
    MatrixXd J = ReadDatatoMatrix("../jMatrix.txt");
    VectorXd J_l = J*l_hat;
    VectorXd stationHeight = J_l.array()+135.961;
    MatrixXd Cx = J*Cl_hat*J.transpose();
    VectorXd Cxstd = Cx.diagonal().array().sqrt();
    //write output files
    WriteMatrixToFile(Cxstd, "../Cxstd.txt", 6);
    WriteMatrixToFile(stationHeight.transpose(), "../stationHeight.txt", 4);
    return 0;
}