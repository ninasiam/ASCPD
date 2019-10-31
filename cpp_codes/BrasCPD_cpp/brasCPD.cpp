#include <iomanip>
#include <fstream>
#include <time.h>
#include <math.h>
#include <string>
#include "brascpd_functions.h"
#include <limits>
#include <ctime>
#include <random>

using namespace std;
using namespace Eigen;

int main(int argc, char **argv){

	VectorXi dims(3,1);
	int order = 3;											// Tensor Order
	int I, J, K;										    // Tensor Dimensions
	int R;												    // Rank of factorization
	time_t start_t, stop_t;									// Timer
	start_t = time(0);
	srand(time(NULL));

	const double AO_tol = 1e-3;							    // Tolerance for AO Algorithm
	int max_iter = 20;										// Maximum Number of iterations
	int AO_iter  = 1;										// Iterations counter

	VectorXi block_size(3,1);								// |
	block_size.setConstant(5);								// | Parameters for stochastic
	const double alpha_init = 0.1;							// | gradient
	const double beta = 1e-4;								// | 
	double alpha;											// |
	VectorXi F_n(block_size(0),1);							// | fibers to be selected
	int factor;												// |
	// VectorXi kr_idx(2,1);								// |


	double frob_X;				                		    // Frobenius norm of Tensor
	double f_value;										    // Objective Value

    Set_Info(&R, &I, &J, &K, "Data_cpp/info.bin");			// Initialize tensor size and rank from file

    cout << "R=" << R << ", I=" << I << ", J=" << J << ", K=" << K << endl;
	dims(0) = I, dims(1) = J, dims(2) = K;
	//------------------------------> Matrix Initializations <-------------------------------------

	MatrixXd A(I,R), B(J,R), C(K,R);						// Factors A,B,C
	MatrixXd A_T_A(R,R), B_T_B(R,R), C_T_C(R,R); 

	MatrixXd X_A(I, size_t(K * J));							// |
	MatrixXd X_B(J, size_t(K * I));							// | Tensor Matricization 
	MatrixXd X_C(K, size_t(I * J));							// |
	MatrixXd W_C(K, R);										// | MTTKRP X_C*kr(B,A) 

	MatrixXd X_A_sub(I, block_size(0));						// |
	MatrixXd X_B_sub(J, block_size(1));						// | Tensor Matricization (sub)
	MatrixXd X_C_sub(K, block_size(2));						// |
	

	MatrixXd KhatriRao_CB(size_t(J * K), R);				// |
	MatrixXd KhatriRao_CA(size_t(I * K), R);				// | Khatri Rao products
	MatrixXd KhatriRao_BA(size_t(I * J), R);				// |

	MatrixXd KhatriRao_CB_sub(R, block_size(0));			// |
	MatrixXd KhatriRao_CA_sub(R, block_size(1));			// | Khatri Rao products (sub)
	MatrixXd KhatriRao_BA_sub(R, block_size(2));			// |

	MatrixXd Grad_A(I,R);									//| 
	MatrixXd Grad_B(J,R);									//| Gradients 
	MatrixXd Grad_C(K,R);									//|


	Read_Data(A, B, C, X_A, X_B, X_C, I, J, K, R);			// Read Factors and matricization from file

	//----------------------------> F_value Computation <------------------------------------------

	frob_X = X_C.squaredNorm();								// Frob norm of tensor
	A_T_A.noalias() = A.transpose()*A;
	B_T_B.noalias() = B.transpose()*B;
	C_T_C.noalias() = C.transpose()*C;
	// MatrixXd KhatriRao_BA(size_t(I * J),R);
	Khatri_Rao_Product(B, A, KhatriRao_BA);
	W_C = X_C*KhatriRao_BA;

	f_value = Get_Objective_Value(C, X_C, A_T_A, B_T_B, C_T_C, frob_X);

	cout << " BEGIN ALGORITHM " << endl;
	;
	while (1)
	{	
		alpha = alpha_init/(pow(AO_iter,beta));
		cout << alpha << endl;
		cout << AO_iter << " -- " << f_value/sqrt(frob_X) << " -- " << f_value << " -- " << frob_X << " -- " <<  endl;
		Sampling_Operator(order, block_size, dims, F_n, factor);
		
		if(factor == 0)										// Factor A
		{	
			Sampling_Sub_Matrices(F_n, KhatriRao_CB, X_A, C, B, KhatriRao_CB_sub, X_A_sub);
			Calculate_Batch_Gradient(block_size(0), A, KhatriRao_CB_sub, X_A_sub, Grad_A);	
			A.noalias() = A - alpha*Grad_A;
			A_T_A.noalias() = A.transpose()*A;
		}
		if(factor == 1)										// Factor B
		{	
			
			Sampling_Sub_Matrices(F_n, KhatriRao_CA, X_B, C, A, KhatriRao_CA_sub, X_B_sub);
			Calculate_Batch_Gradient(block_size(1), B, KhatriRao_CA_sub, X_B_sub, Grad_B);
			B.noalias() = B - alpha*Grad_B;
			B_T_B.noalias() = B.transpose()*B;
		}
		if(factor == 2)										// Factor C
		{
			
			Sampling_Sub_Matrices(F_n, KhatriRao_BA, X_C, B, A, KhatriRao_BA_sub, X_C_sub);
			Calculate_Batch_Gradient(block_size(2), C, KhatriRao_BA_sub, X_C_sub, Grad_C);
			C.noalias() = C - alpha*Grad_C;
			C_T_C.noalias() = C.transpose()*C;
		}

		f_value = Get_Objective_Value(C, X_C, A_T_A, B_T_B, C_T_C, frob_X);
		

		if(f_value < AO_tol || AO_iter + 1 > max_iter)
		{
			cout << "Exit Algorithm" << endl;
			break; 
		}
		AO_iter++;
		
	}

	stop_t = time(0);

	cout << " CPU time = " << stop_t << endl; 
	cout << " AO_iter = " << AO_iter << endl;
	cout << " relative f_value = " << f_value/sqrt(frob_X) << endl << endl;

}