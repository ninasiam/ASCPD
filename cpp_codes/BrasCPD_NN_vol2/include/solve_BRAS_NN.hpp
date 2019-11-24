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
						 const double alpha_init, const double beta , double &f_value, int &AO_iter)
{
    double frob_X;				                		    // Frobenius norm of Tensor
    VectorXi F_n(block_size(0),1);							// | fibers to be selected
	int factor;	                                            // |
    int init_mode;											// |
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

	MatrixXd X_A_sub(I, block_size(0));						// |
	MatrixXd X_B_sub(J, block_size(1));						// | Tensor Matricization (sub)
	MatrixXd X_C_sub(K, block_size(2));						// |
	

	MatrixXd KhatriRao_CB_sub(R, block_size(0));			// |
	MatrixXd KhatriRao_CA_sub(R, block_size(1));			// | Khatri Rao products (sub)
	MatrixXd KhatriRao_BA_sub(R, block_size(2));			// |

    MatrixXd Grad_A(I,R);									//| 
	MatrixXd Grad_B(J,R);									//| Gradients 
	MatrixXd Grad_C(K,R);									//|

	MatrixXd Zero_Matrix_A = MatrixXd::Zero(I, R);			//|
	MatrixXd Zero_Matrix_B = MatrixXd::Zero(J, R);			//| For Projection
	MatrixXd Zero_Matrix_C = MatrixXd::Zero(K, R);			//|


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
		alpha = alpha_init/(pow(AO_iter,beta));
		// cout << alpha << endl;
		
		Sampling_Operator(order, block_size, dims, F_n, factor);
		// cout << factor << endl;
		if(factor == 0)										// Factor A
		{	
			Sampling_Sub_Matrices(F_n, KhatriRao_CB, X_A, C, B, KhatriRao_CB_sub, X_A_sub);
			// cout << KhatriRao_CB_sub << endl;
			Calc_gradient(block_size(0), dims, factor, threads_num, A, KhatriRao_CB_sub, X_A_sub, Grad_A);	
			// cout << X_A_sub << endl;
			A.noalias() -= alpha*Grad_A;
			A = A.cwiseMax(Zero_Matrix_A);
			// cout << A.norm() << endl;
			A_T_A.noalias() = A.transpose()*A;
			if( int(AO_iter % (J*K/block_size(factor))) == 0)
			{
				mttkrp(X_A, KhatriRao_CB, dims, factor, threads_num,  W_A);
				Get_Objective_Value(A, W_A, A_T_A, B_T_B, C_T_C, frob_X, f_value);
				cout << AO_iter << " -- " << f_value/sqrt(frob_X) << " -- " << f_value << " -- " << frob_X << " -- " <<  endl;
			}
		}
		if(factor == 1)										// Factor B
		{				
			Sampling_Sub_Matrices(F_n, KhatriRao_CA, X_B, C, A, KhatriRao_CA_sub, X_B_sub);
			// cout << KhatriRao_CA_sub << endl;
			Calc_gradient(block_size(1), dims, factor, threads_num, B, KhatriRao_CA_sub, X_B_sub, Grad_B);	
			// cout << X_B_sub << endl;
			B.noalias() -= alpha*Grad_B;
			B = B.cwiseMax(Zero_Matrix_B);
			// cout << B.norm() << endl;
			B_T_B.noalias() = B.transpose()*B;
			if( int(AO_iter % (I*K/block_size(factor))) == 0)
			{
				mttkrp(X_B, KhatriRao_CA, dims, factor, threads_num,  W_B);
				Get_Objective_Value(B, W_B, A_T_A, B_T_B, C_T_C, frob_X, f_value);
				cout << AO_iter << " -- " << f_value/sqrt(frob_X) << " -- " << f_value << " -- " << frob_X << " -- " <<  endl;
			}
		}
		if(factor == 2)										// Factor C
		{
			
			Sampling_Sub_Matrices(F_n, KhatriRao_BA, X_C, B, A, KhatriRao_BA_sub, X_C_sub);
			// cout << KhatriRao_BA_sub << endl;
			Calc_gradient(block_size(2), dims, factor, threads_num, C, KhatriRao_BA_sub, X_C_sub, Grad_C);	
			// cout << X_C_sub << endl;
			C.noalias() -= alpha*Grad_C;
			C = C.cwiseMax(Zero_Matrix_C);
			// cout << C.norm() << endl;
			C_T_C.noalias() = C.transpose()*C;
			if( int(AO_iter % (I*J/block_size(factor))) == 0)
			{
				mttkrp(X_C, KhatriRao_BA, dims, factor, threads_num,  W_C);
				Get_Objective_Value(C, W_C, A_T_A, B_T_B, C_T_C, frob_X, f_value);
				cout << AO_iter << " -- " << f_value/sqrt(frob_X) << " -- " << f_value << " -- " << frob_X << " -- " <<  endl;
			}
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