/*--------------------------------------------------------------------------------------------------*/
/* 						Function that implements the acceleration step								*/
/*    			(calls Khatri_Rao_Product and Get_Objective_Value_Accel functions)				   	*/
/*    																							   	*/
/*                                                                           						*/
/* A. P. Liavas																						*/
/* Georgios Lourakis																				*/
/* Georgios Kostoulas																				*/
/* 15/11/2017                                              											*/
/*--------------------------------------------------------------------------------------------------*/
#include "cpd_functions.h"

using namespace Eigen;

void Line_Search_Accel(omp_obj omp_var, const Ref<const MatrixXd> A_old_N, const Ref<const MatrixXd> B_old_N, const Ref<const MatrixXd> C_old_N, Ref<MatrixXd> A,
					   Ref<MatrixXd> B, Ref<MatrixXd> C, Ref<MatrixXd> A_T_A, Ref<MatrixXd> B_T_B, Ref<MatrixXd> C_T_C, Ref<MatrixXd> KhatriRao_BA,
					   const Ref<const MatrixXd> X_C, int *acc_fail, int *acc_coeff, int iter, double f_value, double frob_X)
{

	int R = A.cols();
	MatrixXd A_accel, B_accel, C_accel;												// Factors A_accel, B_accel, C_accel
	MatrixXd A_T_A_accel(R, R), B_T_B_accel(R, R), C_T_C_accel(R, R);
	double f_accel;																	// Objective Value after the acceleration step
	double acc_step;

	acc_step = pow(iter+1,(1.0/(*acc_coeff)));
	A_accel.noalias() = A_old_N + acc_step * (A - A_old_N);
	B_accel.noalias() = B_old_N + acc_step * (B - B_old_N);
	C_accel.noalias() = C_old_N + acc_step * (C - C_old_N);

	A_T_A_accel.noalias() = A_accel.transpose() * A_accel;
	B_T_B_accel.noalias() = B_accel.transpose() * B_accel;
	C_T_C_accel.noalias() = C_accel.transpose() * C_accel;

	Khatri_Rao_Product(omp_var, B_accel, A_accel, KhatriRao_BA);
	f_accel = Get_Objective_Value_Accel(omp_var, size_t(A.rows()), size_t(B.rows()), C_accel, X_C, KhatriRao_BA, A_T_A_accel, B_T_B_accel, C_T_C_accel, frob_X);

	if (f_value>f_accel){
		A = A_accel;
		B = B_accel;
		C = C_accel;
		A_T_A = A_T_A_accel;
		B_T_B = B_T_B_accel;
		C_T_C = C_T_C_accel;
	}
	else
		(*acc_fail)++;

	if (*acc_fail==5){
		*acc_fail=0;
		(*acc_coeff)++;
	}

}
