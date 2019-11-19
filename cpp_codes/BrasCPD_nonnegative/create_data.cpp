/*--------------------------------------------------------------------------------------------------*/
/* 			Write Factors A_init, B_init, C_init and matricized Tensor X_C	in one file	   			*/
/*                       (Saves Data in Data_cpp Folder)       	                   					*/
/*                         	(Eigen 	: Column Major	)	                       						*/
/* 							(C++ 	: Row Major		)	   		 									*/
/*                 			(Uses Transpose in Write File)											*/
/*				 	                              													*/
/*         																							*/
/* A. P. Liavas																						*/
/* Georgios Lourakis																				*/
/* Georgios Kostoulas																				*/
/* 15/11/2017                                              											*/
/*--------------------------------------------------------------------------------------------------*/
#include <iostream>
#include <fstream>
#include "brascpd_functions.h"

// #define EIGEN_DEFAULT_TO_ROW_MAJOR

using namespace std;
using namespace Eigen;

// void Khatri_Rao_Product(const Ref<const MatrixXd> U2, const Ref<const MatrixXd> U1, Ref<MatrixXd> Kr){
// 	int i,j;
// 	VectorXd temp = VectorXd::Zero(U1.rows());
// 	for (j = 0; j < U2.cols(); j++){
// 		temp = U1.col(j);
// 		for (i = 0; i < U2.rows(); i++)
// 			Kr.block(i*U1.rows(), j, U1.rows(), 1).noalias() =  U2(i,j) * temp;

// 	}
// }

void Write_to_File(int nrows, int ncols, Ref<MatrixXd> Mat, const char *file_name){
	ofstream my_file(file_name, ios::out | ios::binary | ios::trunc);
	if (my_file.is_open()){
		my_file.write((char *) Mat.data(), nrows*ncols*sizeof(double));
		my_file.close();
	}
	else
		cout << "Unable to open file \n";
}

void Write_info_to_File(int* F, int* I, int* J, int* K, const char *file_name){
	ofstream my_file(file_name, ios::out | ios::binary | ios::trunc);
	if (my_file.is_open()){
		my_file.write((char *) F, sizeof(int));
		my_file.write((char *) I, sizeof(int));
		my_file.write((char *) J, sizeof(int));
		my_file.write((char *) K, sizeof(int));
		my_file.close();
	}
	else
		cout << "Unable to open file \n";
}

/* <-----------------------------------------------------------------------------------------------> */
/* < ---------------------------------		MAIN		-------------------------------------------> */
int main(int argc, char **argv){
	int I = 100;//400;//5000;//1008;//3024;//					// | Dimensions
	int J = 100;  //400;//320;//1008;//112;//						// | and rank
	int K = 100;  //50000;//5000;//1008;//3024;//					// | of the
	int F = 20;										// | Tensor


	MatrixXd A(I, F);
	MatrixXd B(J, F);
	MatrixXd C(K, F);
	MatrixXd A_T(F, I);
	MatrixXd B_T(F, J);
	MatrixXd C_T(F, K);
	MatrixXd A_init(I, F);
	MatrixXd B_init(J, F);
	MatrixXd C_init(K, F);
	MatrixXd A_init_T(F, I);
	MatrixXd B_init_T(F, J);
	MatrixXd C_init_T(F, K);

	//	<---------------------------		Print Dimensions and Rank		------------------------------------------->	//
	cout << "F=" << F << ", I=" << I << ", J=" << J << ", K=" << K << endl;

	//	<----------------------		Write Dimensions and Rank in one file each		------------------------------------->	//
	Write_info_to_File(&F, &I, &J, &K, "Data_cpp/info.bin");

	//  <--------------------------------  Create True & Initial Factors ----------------------------------------------->   //

	A = (MatrixXd::Random(I, F) + MatrixXd::Ones(I ,F))/2;	// A~uniform[0,1] (MatrixXd::Random returns uniform random numbers in (-1, 1))
	B = (MatrixXd::Random(J, F) + MatrixXd::Ones(J ,F))/2;	// B~uniform[0,1] (MatrixXd::Random returns uniform random numbers in (-1, 1))
	C = (MatrixXd::Random(K, F) + MatrixXd::Ones(K ,F))/2;	// C~uniform[0,1] (MatrixXd::Random returns uniform random numbers in (-1, 1))

	A_init = (MatrixXd::Random(I, F) + MatrixXd::Ones(I ,F))/2;	// A_init~uniform[0,1] (MatrixXd::Random returns uniform random numbers in (-1, 1))
	B_init = (MatrixXd::Random(J, F) + MatrixXd::Ones(J ,F))/2;	// B_init~uniform[0,1] (MatrixXd::Random returns uniform random numbers in (-1, 1))
	C_init = (MatrixXd::Random(K, F) + MatrixXd::Ones(K ,F))/2;	// C_init~uniform[0,1] (MatrixXd::Random returns uniform random numbers in (-1, 1))

	// A = MatrixXd::Random(I, F);			// A~uniform[-1,1] (MatrixXd::Random returns uniform random numbers in (-1, 1))
	// B = MatrixXd::Random(J, F);			// B~uniform[-1,1] (MatrixXd::Random returns uniform random numbers in (-1, 1))
	// C = MatrixXd::Random(K, F); 		// C~uniform[-1,1] (MatrixXd::Random returns uniform random numbers in (-1, 1))

	// A_init = MatrixXd::Random(I, F); 	// A_init~uniform[-1,1] (MatrixXd::Random returns uniform random numbers in (-1, 1))
	// B_init = MatrixXd::Random(J, F); 	// B_init~uniform[-1,1] (MatrixXd::Random returns uniform random numbers in (-1, 1))
	// C_init = MatrixXd::Random(K, F); 	// C_init~uniform[-1,1] (MatrixXd::Random returns uniform random numbers in (-1, 1))

	A_init_T = A_init.transpose();
	B_init_T = B_init.transpose();
	C_init_T = C_init.transpose();

	A_T = A.transpose();
	B_T = B.transpose();
	C_T = C.transpose();

	//	<----------------------		Write A_nit, B_init, C_init, X_C in one file each		------------------------------------->	//
	Write_to_File(I, F, A_init_T, "Data_cpp/A_init.bin");

	Write_to_File(J, F, B_init_T, "Data_cpp/B_init.bin");

	Write_to_File(K, F, C_init_T, "Data_cpp/C_init.bin");

	Write_to_File(I, F, A_T, "Data_cpp/A.bin");

	Write_to_File(J, F, B_T, "Data_cpp/B.bin");

	Write_to_File(K, F, C_T, "Data_cpp/C.bin");

	return 0;
}
