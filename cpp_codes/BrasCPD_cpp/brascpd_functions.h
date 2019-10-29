#ifndef BRASCPD_FUNCTIONS_H		// if brasCPD_functions.h hasn't been included yet...
#define BRASCPD_FUNCTIONS_H		// #define this so the compiler knows it has been included

#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Core>
#include <iostream>
#include <cstdlib>

using namespace Eigen;

void Set_Info(int* R, int* I, int* J, int* K, const char *file_name);

void Khatri_Rao_Product(const Ref<const MatrixXd> U2, const Ref<const MatrixXd> U1, Ref<MatrixXd> Kr);

void Read_Data(Ref<MatrixXd> A, Ref<MatrixXd> B, Ref<MatrixXd> C, Ref<MatrixXd> X_A_T, Ref<MatrixXd> X_B_T ,Ref<MatrixXd> X_C_T, int I, int J, int K, int R);

void Read_From_File(int nrows, int ncols, Ref<MatrixXd> Mat, const char *file_name, int skip);

double Get_Objective_Value(const Ref<const MatrixXd> C, const Ref<const MatrixXd> X_C_Kr, const Ref<const MatrixXd> A_T_A,
							const Ref<const MatrixXd> B_T_B, const Ref<const MatrixXd> C_T_C, double frob_X);

void Sampling_Operator(int order, VectorXi block_size, VectorXi dims, VectorXi &F_n, VectorXi &kr_idx, int &factor);

#endif
