#include "brascpd_functions.h"
#include <iostream>

using namespace std;
using namespace Eigen;

void Read_Data(Ref<MatrixXd> A, Ref<MatrixXd> B, Ref<MatrixXd> C, Ref<MatrixXd> X_A, Ref<MatrixXd> X_B, Ref<MatrixXd> X_C,
			   int I, int J, int K, int R)
{

	int skip;
	MatrixXd A_true(I, R);
	MatrixXd B_true(J, R);
	MatrixXd C_true(K, R);

	//	<---------------------------		Read Initial Factors from file		--------------------------->	//

	MatrixXd A_T(R, I);
	Read_From_File(I, R, A_T, "Data_cpp/A_init.bin", 0);
	A = A_T.transpose();

	MatrixXd B_T(R, J);
	Read_From_File(J, R, B_T, "Data_cpp/B_init.bin", 0);
	B = B_T.transpose();

	MatrixXd C_T(R, K);
	Read_From_File(K, R, C_T, "Data_cpp/C_init.bin", 0);
	C = C_T.transpose();

	//	<--------------------		Read Matricized Tensor from file		--------------------------->	//
	MatrixXd A_true_T(R, I);
	Read_From_File(I, R, A_true_T, "Data_cpp/A.bin", 0);
	A_true = A_true_T.transpose();

	MatrixXd B_true_T(R, J);
	Read_From_File(J, R, B_true_T, "Data_cpp/B.bin", 0);
	B_true = B_true_T.transpose();

	MatrixXd C_true_T(R, K);
	Read_From_File(K, R, C_true_T, "Data_cpp/C.bin", 0);
	C_true = C_true_T.transpose();

	MatrixXd kr_CB(size_t(K * J), R);
	Khatri_Rao_Product(C_true, B_true, kr_CB);
	X_A.noalias() = A_true * kr_CB.transpose();
    // X_A_T = X_A.transpose();

    MatrixXd kr_CA(size_t(K * I), R);
	Khatri_Rao_Product(C_true, A_true, kr_CA);
	X_B.noalias() = B_true * kr_CA.transpose();
    // X_B_T = X_B.transpose();

	MatrixXd kr_BA(size_t(I * J), R);
	Khatri_Rao_Product(B_true, A_true, kr_BA);
	X_C.noalias() = C_true * kr_BA.transpose();
    // X_C_T = X_C.transpose();
}