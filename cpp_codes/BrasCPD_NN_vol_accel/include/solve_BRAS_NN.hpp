#ifndef SOLVE_BRAS_NN_HPP
#define SOLVE_BRAS_NN_HPP

#include "master_library.hpp"
#include "khatri_rao_prod.hpp"
#include "mttkrp.hpp"
#include "get_objective_value.hpp"
#include "sampling_funs.hpp"
#include "omp_lib.hpp"
#include "calc_gradient.hpp"

inline void Solve_brasNN(MatrixXd &A, MatrixXd &B, MatrixXd &C, MatrixXd &X_A, MatrixXd &X_B, MatrixXd &X_C, MatrixXd &KhatriRao_CB, MatrixXd &KhatriRao_CA, MatrixXd &KhatriRao_BA,
						 const int I, const int J, const int K, const int R, VectorXi &block_size, const double AO_tol, const double MAX_ITERS, 
						 double &f_value, int &AO_iter)
{
    double frob_X;				                		    // Frobenius norm of Tensor
    VectorXi F_n(block_size(0),1);							// | fibers to be selected
	int factor;	                                            // |
    int init_mode;
	int J_n;												// |
    VectorXi dims(3,1);
    dims(0) = I, dims(1) = J, dims(2) = K;

    double alpha;
    int order = 3;
    const unsigned int threads_num = get_num_threads();

	//------------------------------> Matrix Initializations <-------------------------------------
    MatrixXd A_T_A(R,R), B_T_B(R,R), C_T_C(R,R);
    MatrixXd W_A = A;                                       // | MTTKRP X_A*kr(C,B) 
	MatrixXd W_B = B;                                       // | MTTKRP X_B*kr(C,A)
	MatrixXd W_C = C;										// | MTTKRP X_C*kr(B,A) 

	MatrixXd Y_A = A;										//| Y sequence for A
	MatrixXd Y_B = B;										//|	Y sequence for B
	MatrixXd Y_C = C;										//|	Y sequence for C

	MatrixXd A_next = A;
	MatrixXd B_next = B;
	MatrixXd C_next = C;

	MatrixXd X_A_sub(I, block_size(0));						// |
	MatrixXd X_B_sub(J, block_size(1));						// | Tensor Matricization (sub)
	MatrixXd X_C_sub(K, block_size(2));						// |
	

	MatrixXd KhatriRao_CB_sub(block_size(0), R);			// |
	MatrixXd KhatriRao_CA_sub(block_size(1), R);			// | Khatri Rao products (sub)
	MatrixXd KhatriRao_BA_sub(block_size(2), R);			// |
	MatrixXd Hessian(R,R);

    MatrixXd Grad_A(I,R);									//| 
	MatrixXd Grad_B(J,R);									//| Gradients 
	MatrixXd Grad_C(K,R);									//|

	MatrixXd Zero_Matrix_A = MatrixXd::Zero(I, R);			//|
	MatrixXd Zero_Matrix_B = MatrixXd::Zero(J, R);			//| For Projection
	MatrixXd Zero_Matrix_C = MatrixXd::Zero(K, R);			//|

	double L, beta_accel, lambda;							// NAG parameters

	A_T_A.noalias() = A.transpose()*A;
	B_T_B.noalias() = B.transpose()*B;
	C_T_C.noalias() = C.transpose()*C;
    //----------------------------> F_value Computation <------------------------------------------

	frob_X = X_C.squaredNorm();								// Frob norm of tensor

	// MatrixXd KhatriRao_BA(size_t(I * J),R);
    init_mode = 2; 
	mttkrp(X_C, KhatriRao_BA, dims, init_mode, threads_num,  W_C);
	

	Get_Objective_Value(C, W_C, A_T_A, B_T_B, C_T_C, frob_X, f_value);

    //--------------------------> Begin Algorithm <-----------------------------------------------
    cout << " BEGIN ALGORITHM " << endl;
	high_resolution_clock::time_point t1 = high_resolution_clock::now();
	while (1)
	{	

		v1::Sampling_Operator(order, block_size, dims, F_n, factor);
		
		if(factor == 0)										// Factor A
		{	
			v1::Sampling_Sub_Matrices(F_n, X_A, C, B, KhatriRao_CB, KhatriRao_CB_sub, X_A_sub);

			Hessian = KhatriRao_CB_sub.transpose()*KhatriRao_CB_sub;
			Compute_NAG_parameters(Hessian, L, beta_accel, lambda);
			Calc_gradient(dims, factor, threads_num, lambda, A, Y_A, Hessian, KhatriRao_CB_sub, X_A_sub, Grad_A);	
			A_next = Y_A - Grad_A / (L + lambda);
			A_next = A_next.cwiseMax(Zero_Matrix_A);
			Y_A = A_next + beta_accel*(A_next - A); 
	
			A_T_A.noalias() = A_next.transpose()*A_next;
			if( int(AO_iter % (J*K/block_size(factor))) == 0)
			{
				mttkrp(X_A, KhatriRao_CB, dims, factor, threads_num,  W_A);
				Get_Objective_Value(A_next, W_A, A_T_A, B_T_B, C_T_C, frob_X, f_value);
				cout << AO_iter << " -- " << f_value/sqrt(frob_X) << " -- " << f_value << " -- " << frob_X << " -- " <<  endl;
			}
			A = A_next;
		}
		if(factor == 1)										// Factor B
		{				
			v1::Sampling_Sub_Matrices(F_n, X_B, C, A, KhatriRao_CA, KhatriRao_CA_sub, X_B_sub);

			Hessian = KhatriRao_CA_sub.transpose()*KhatriRao_CA_sub;
			Compute_NAG_parameters(Hessian, L, beta_accel, lambda);
			Calc_gradient(dims, factor, threads_num, lambda, B, Y_B, Hessian, KhatriRao_CA_sub, X_B_sub, Grad_B);	
			B_next = Y_B - Grad_B / (L + lambda);
			B_next = B_next.cwiseMax(Zero_Matrix_B);
			Y_B = B_next + beta_accel * (B_next - B);

			B_T_B.noalias() = B.transpose()*B;
			if( int(AO_iter % (I*K/block_size(factor))) == 0)
			{
				mttkrp(X_B, KhatriRao_CA, dims, factor, threads_num,  W_B);
				Get_Objective_Value(B_next, W_B, A_T_A, B_T_B, C_T_C, frob_X, f_value);
				cout << AO_iter << " -- " << f_value/sqrt(frob_X) << " -- " << f_value << " -- " << frob_X << " -- " <<  endl;
			}

			B = B_next;
		}
		if(factor == 2)										// Factor C
		{
			
			v1::Sampling_Sub_Matrices(F_n, X_C, B, A, KhatriRao_BA, KhatriRao_BA_sub, X_C_sub);

			Hessian = KhatriRao_BA_sub.transpose()*KhatriRao_BA_sub;
			Compute_NAG_parameters(Hessian, L, beta_accel, lambda);
			Calc_gradient( dims, factor, threads_num, lambda, C, Y_C, Hessian, KhatriRao_BA_sub, X_C_sub, Grad_C);	
			
			C_next = Y_C - Grad_C / (L + lambda);
			C_next = C_next.cwiseMax(Zero_Matrix_C);
			Y_C = C_next + beta_accel * (C_next - C);
			
			C_T_C.noalias() = C.transpose()*C;
			if( int(AO_iter % (I*J/block_size(factor))) == 0)
			{
				mttkrp(X_C, KhatriRao_BA, dims, factor, threads_num,  W_C);
				Get_Objective_Value(C_next, W_C, A_T_A, B_T_B, C_T_C, frob_X, f_value);
				cout << AO_iter << " -- " << f_value/sqrt(frob_X) << " -- " << f_value << " -- " << frob_X << " -- " <<  endl;
			}

			C = C_next;
		}

		

		if(f_value/sqrt(frob_X)  < AO_tol || AO_iter >= MAX_ITERS)
		{
			cout << "Exit Algorithm" << endl;
			break; 
		}
		AO_iter++;
		
	}

	high_resolution_clock::time_point t2 = high_resolution_clock::now();

	duration<double> stop_t = duration_cast<duration<double>>(t2-t1);
	cout << " CPU time = " << stop_t.count() << endl; 
	cout << " AO_iter = " << AO_iter << endl;
	cout << " relative f_value = " << f_value/sqrt(frob_X) << endl;
	cout << " number of threads = " << threads_num << endl << endl;

}
#endif