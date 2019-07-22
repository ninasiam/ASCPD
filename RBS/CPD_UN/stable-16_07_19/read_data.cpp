/*--------------------------------------------------------------------------------------------------*/
/* 					Function for reading Factors A, B, C and matricized Tensor						*/
/*    						(calls Read_From_File and Read_X_C functions )						   	*/
/*    																							   	*/
/*                                                                           						*/
/* A. P. Liavas																						*/
/* Georgios Lourakis																				*/
/* Georgios Kostoulas																				*/
/* 15/11/2017                                              											*/
/*--------------------------------------------------------------------------------------------------*/
#include "cpd_functions.h"

#include <iostream>
using namespace std;

using namespace Eigen;

void Read_Data(omp_obj omp_var, Ref<MatrixXd> A, Ref<MatrixXd> B, Ref<MatrixXd> C, Ref<MatrixXd> X_A, Ref<MatrixXd> X_B, Ref<MatrixXd> X_C,
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
	MatrixXd X_A_T(size_t(K * J), I);
	Khatri_Rao_Product(omp_var, C_true, B_true, kr_CB);
	X_A.noalias() = A_true * kr_CB.transpose();

	X_B = X_A.transpose();

	MatrixXd kr_BA(size_t(I * J), R);
	MatrixXd X_C_T(size_t(I * J), K);
	Khatri_Rao_Product(omp_var, B_true, A_true, kr_BA);
	X_C.noalias() = C_true * kr_BA.transpose();

	//	<--------------------		Read Matricized Tensor from file		--------------------------->	//
	/*
	MatrixXd X_A_2(n_I, n_J*n_K);
	MatrixXd X_B_2(n_J*n_K, n_I);
	if (comm_sz==1)
		Read_From_File(n_I, n_J*n_K, X_A_2, "Data_cpp/X_C.bin", 0);
	else
		Read_X_C(n_I, n_J, n_K, skip_I, skip_J, skip_K, I, J, X_A_2, "Data_cpp/X_C.bin"); 	// Read X_A ,X_B from one file
	X_B_2 = X_A_2.transpose();


	MatrixXd X_C_2(n_K, n_I*n_J);
	if (comm_sz==1)
		Read_From_File(n_I*n_J, n_K, X_C_T, "Data_cpp/X_C.bin", 0);
	else
		Read_X_C(n_I, n_J, n_K, skip_I, skip_J, skip_K, I, J, X_C_T, "Data_cpp/X_C.bin"); // Read X_C from one file
	X_C_2 = X_C_T.transpose();

	cout << (X_C_2 - X_C).squaredNorm() << " -- " << (X_A_2 - X_A).squaredNorm() << " -- " << (X_B_2 - X_B).squaredNorm() <<  endl << endl;
	*/
}
