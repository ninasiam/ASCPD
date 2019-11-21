#ifndef KHATRI_RAO_PROD_HPP
#define KHATRI_RAO_PROD_HPP

#include "master_library.hpp"

inline void Khatri_Rao_Product(const Ref<const MatrixXd> U2, const Ref<const MatrixXd> U1, Ref<MatrixXd> Kr){
	int i,j;
	VectorXd temp = VectorXd::Zero(U1.rows());
	for (j = 0; j < U2.cols(); j++){
		temp = U1.col(j);
		for (i = 0; i < U2.rows(); i++)
			Kr.block(i*U1.rows(), j, U1.rows(), 1).noalias() =  U2(i,j) * temp;

	}

#endif