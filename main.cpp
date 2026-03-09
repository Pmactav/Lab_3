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
    MatrixXd sigma_m = sigma_mm*(.01);
    MatrixXd P = sigma_m.array().square().inverse().matrix().asDiagonal(); //square st dev, take inverse, store in main diagonal of P
    MatrixXd stVarMatrix = sigma_m.array().square().matrix().asDiagonal();
    //Begin populating parameters and create corrections
    MatrixXd w = B*l;
    MatrixXd M = B * P.inverse() * B.transpose();
    VectorXd k = M.ldlt().solve(w); //create Legrange Multiplier
    VectorXd v_hat = -P.inverse() * B.transpose() * k; //create vector of residuals
    double n = l.rows();
    double r = B.rows();
    double sigma0_sq = (v_hat.transpose()*P*v_hat)(0,0)/r;
    cout << sigma0_sq << endl;
    cout << r << endl;
    double numerator = (v_hat.transpose()*P*v_hat)(0,0);
    cout << "numerator: " << numerator << endl;

    // apply corrections and populate variance covariance
    MatrixXd sigma = sigma_m.array().square().matrix().asDiagonal();
    MatrixXd Cl = sigma0_sq * sigma;
    MatrixXd Cv = sigma0_sq*P.inverse()*B.transpose()*(B*P.inverse()*B.transpose()).inverse()*B*P.inverse();
    VectorXd l_hat = l + v_hat;
    MatrixXd Cl_hat = Cl - Cv;
    VectorXd Cl_hatDiag = Cl_hat.diagonal().array();
    WriteMatrixToFile(Cv, "../Cv.txt", 8);
    WriteMatrixToFile(Cl_hat, "../Cl_hat.txt", 8);
    cout << Cl_hatDiag << endl;
    //Task 2
    MatrixXd J = ReadDatatoMatrix("../jMatrix.txt");
    VectorXd J_l = J*l_hat;
    VectorXd stationHeight = J_l.array()+135.961;
    MatrixXd Cx = J*Cl_hat*J.transpose();
    VectorXd stDevCx = Cx.diagonal().array().sqrt();
    //cout << stDevCx << endl;
    WriteMatrixToFile(Cx, "../Cx.txt", 6);
    //cout << J_l << endl;
    //cout << stationHeight << endl;

    return 0;
}