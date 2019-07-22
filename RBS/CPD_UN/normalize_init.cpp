/*--------------------------------------------------------------------------------------------------*/
/* 				Function for the Normalization of the columns of the Factors B, C 					*/
/*    					All the weight goes in the columns of Factor A							   	*/
/*                (Used only one time before the beginning of the Algorithm)	                	*/
/*   																								*/
/*                                                                           						*/
/* A. P. Liavas																						*/
/* Georgios Lourakis																				*/ 
/* Georgios Kostoulas																				*/
/* 15/11/2017                                              											*/
/*--------------------------------------------------------------------------------------------------*/
#include "cpd_functions.h"

using namespace Eigen;

void Normalize_Init(Ref<MatrixXd> A, Ref<MatrixXd> B, Ref<MatrixXd> C){
	int R = A.cols();
	int i;
	double b, c;
	VectorXd lambda_B(R);
	VectorXd lambda_C(R);

	for (i=0; i<R; i++){
		lambda_B(i) = B.col(i).squaredNorm();
		lambda_C(i) = C.col(i).squaredNorm();
	}

	for (i=0; i<R; i++){
		b = sqrt(lambda_B(i));
		c = sqrt(lambda_C(i));
		if ((b>0) && (c>0)){
			B.col(i) *= 1/b;
			C.col(i) *= 1/c;
			A.col(i) *= b*c;
		}
	}
}
