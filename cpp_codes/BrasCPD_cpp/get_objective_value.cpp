/*--------------------------------------------------------------------------------------------------*/
/* 						Function for the computation of the Objective Value		 					*/
/*    									of the NTF problem										   	*/
/*              	(make use of results already computed in the update of Factor C)          		*/
/*                        || X_0 - X || = sqrt(<X_0,X_0> + <X,X> - 2 <X_0,X>)		 				*/
/*                		where:   	<X_0,X_0> =  frob_X,											*/
/*                       			<X,X> =  sum(Z),												*/
/*                       			<X_0,X> = global_sum											*/
/*                                                                           						*/
/* A. P. Liavas																						*/
/* Georgios Lourakis																				*/
/* Georgios Kostoulas																				*/
/* 15/11/2017                                              											*/
/*--------------------------------------------------------------------------------------------------*/
#include "brascpd_functions.h"

using namespace Eigen;
//the <X_0,X>  is the sum of the product of theri entries (see KOLDA)
double Get_Objective_Value(const Ref<const MatrixXd> C, const Ref<const MatrixXd> X_C_Kr, const Ref<const MatrixXd> A_T_A,
							const Ref<const MatrixXd> B_T_B, const Ref<const MatrixXd> C_T_C, double frob_X){
	double global_sum;
	int R = X_C_Kr.cols();

	MatrixXd Z(R, R);
	Z.noalias() = A_T_A.cwiseProduct(B_T_B.cwiseProduct(C_T_C));					// (A^T*A) .* (B^T*B) .* (C^T*C)

	global_sum = (X_C_Kr.cwiseProduct(C)).sum();										// sum(sum((X_C * KhatriRao_BA) .* C))

	return sqrt(frob_X + Z.sum() - 2*global_sum );
}
