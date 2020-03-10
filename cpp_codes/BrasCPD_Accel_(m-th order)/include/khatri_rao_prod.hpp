#ifndef KHATRI_RAO_PROD_HPP
#define KHATRI_RAO_PROD_HPP

#include "master_library.hpp"

inline void Khatri_Rao_Product(const Ref<const MatrixXd> U2, const Ref<const MatrixXd> U1, Ref<MatrixXd> Kr)
{	
	//
	int i,j;
	VectorXd temp = VectorXd::Zero(U1.rows()); //implementation according the example of stack overflow
	for (j = 0; j < U2.cols(); j++){ 		   // for the columns of U2 (Rank)
		temp = U1.col(j); 			           //the column of U1
		for (i = 0; i < U2.rows(); i++)		   //for each element of the j column of U2
			Kr.block(i*U1.rows(), j, U1.rows(), 1).noalias() =  U2(i,j) * temp;

	}
}
#endif