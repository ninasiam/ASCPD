#ifndef GET_OBJECTIVE_VALUE_HPP
#define GET_OBJECTIVE_VALUE_HPP

#include "master_library.hpp"

inline void Get_Objective_Value(const Ref<const MatrixXd> C, const Ref<const MatrixXd> X_C_Kr, const Ref<const MatrixXd> A_T_A,
							const Ref<const MatrixXd> B_T_B, const Ref<const MatrixXd> C_T_C, double frob_X, double &f_value){
	double global_sum;
	int R = X_C_Kr.cols();

	MatrixXd Z(R, R);
	Z.noalias() = A_T_A.cwiseProduct(B_T_B.cwiseProduct(C_T_C));				    	// (A^T*A) .* (B^T*B) .* (C^T*C)

	global_sum = (X_C_Kr.cwiseProduct(C)).sum();										// sum(sum((X_C * KhatriRao_BA) .* C))

	f_value = sqrt(frob_X + Z.sum() - 2*global_sum );
}

#endif