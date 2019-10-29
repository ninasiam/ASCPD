/*--------------------------------------------------------------------------------------------------*/
/* 					Function for the Computation of the Khatri-Rao-Product 							*/
/*    							of matrices U2 and U1											   	*/
/*                                                                           						*/
/* A. P. Liavas																						*/
/* Georgios Lourakis																				*/
/* Georgios Kostoulas																				*/
/* 15/11/2017                                              											*/
/*--------------------------------------------------------------------------------------------------*/
#include "brascpd_functions.h"

using namespace Eigen;
using namespace std;

void Khatri_Rao_Product(const Ref<const MatrixXd> U2, const Ref<const MatrixXd> U1, Ref<MatrixXd> Kr){
	int i,j;
	VectorXd temp = VectorXd::Zero(U1.rows());
	for (j = 0; j < U2.cols(); j++){
		temp = U1.col(j);
		for (i = 0; i < U2.rows(); i++)
			Kr.block(i*U1.rows(), j, U1.rows(), 1).noalias() =  U2(i,j) * temp;

	}
}