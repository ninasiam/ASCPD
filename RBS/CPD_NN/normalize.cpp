/*--------------------------------------------------------------------------------------------------*/
/* 				Function for the Normalization of the columns of the Factors B, C 					*/
/*    					All the weight goes in the columns of Factor A							   	*/
/*       			 				(Used throughout the Algorithm) 								*/
/*   																								*/
/*                                                                           						*/
/* A. P. Liavas																						*/
/* Georgios Lourakis																				*/ 
/* Georgios Kostoulas																				*/
/* 15/11/2017                                              											*/
/*--------------------------------------------------------------------------------------------------*/
#include "ntf_functions.h"

using namespace Eigen;

void Normalize(Ref<MatrixXd> A, Ref<MatrixXd> B, Ref<MatrixXd> C, Ref<MatrixXd> A_T_A, Ref<MatrixXd> B_T_B, Ref<MatrixXd> C_T_C){
	int R = A.cols();
	int i;
	double b, c;
	VectorXd lambda_B(R);
	VectorXd lambda_C(R);
	MatrixXd norm_A_matrix(R, R);
	MatrixXd norm_B_matrix(R, R);
	MatrixXd norm_C_matrix(R, R);
	
	lambda_B = B_T_B.diagonal();
	lambda_C = C_T_C.diagonal();

	for (i=0; i<R; i++){
		b = sqrt(lambda_B(i));
		c = sqrt(lambda_C(i));
		if ((b>0) && (c>0)){
			B.col(i) *= 1/b;
			C.col(i) *= 1/c;
			A.col(i) *= b*c;
			lambda_B(i) = b;
			lambda_C(i) = c;
		}
		else{
			lambda_B(i) = 1;
			lambda_C(i) = 1;
		}
	}
	
	norm_B_matrix.noalias() = lambda_B * lambda_B.transpose();
	norm_C_matrix.noalias() = lambda_C * lambda_C.transpose();
	norm_A_matrix.noalias() = norm_B_matrix.cwiseProduct(norm_C_matrix);
	
	B_T_B = B_T_B.cwiseQuotient(norm_B_matrix);
	C_T_C = C_T_C.cwiseQuotient(norm_C_matrix);
	A_T_A = A_T_A.cwiseProduct(norm_A_matrix);
}
