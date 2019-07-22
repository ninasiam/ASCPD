/*--------------------------------------------------------------------------------------------------------------*/
/* 							Solves a unconstrained CPD/PARAFAC problem in C++				    			    */
/*                    (The implementation uses the Eigen Library for Linear Algebra)                         	*/
/*                                                                           									*/
/* 1 - Create Data: make create_data																			*/
/*              	./create_data	                                             								*/
/*                                                                           									*/
/* 2 - Compile: make							 																*/
/*                                                                           									*/
/* 3 - Execute: ./cpd               			 	                    		          						*/
/*																												*/
/*																												*/
/* A. P. Liavas																									*/
/* Georgios Lourakis																							*/
/* Georgios Kostoulas																							*/
/* Paris Karakasis																							    */
/* 24/11/2018                                              														*/
/*--------------------------------------------------------------------------------------------------------------*/
#include <iomanip>
#include <fstream>
#include <time.h>
#include <math.h>
#include <string>
#include "cpd_functions.h"
#include "additional_func.h"
#include <limits>

using namespace std;
using namespace Eigen;

/* <----------------------------------------------------------------------------------------------------------> */
/* < ---------------------------------------		MAIN		----------------------------------------------> */
int main(int argc, char **argv){
	int I, J, K;										// Tensor Dimensions
	int R;												// Rank of factorization

	const double AO_tol = 1e-3;							// Tolerance for AO Algorithm
	int max_iter = 500;									// Maximum Number of iterations

	double frob_X;				                		// Frobenius norm of Tensor
	double f_value;										// Objective Value

	int acc_coeff = 3;									// | Variables
	int acc_fail = 0;									// | for
	int k_0 = 1;										// | Acceleration step

	double total_t;										// |
	double start_t, stop_t;     						// | Time variables
	int AO_iter;										// |
	double start_t_ATA, stop_t_ATA = 0, start_t_BTB, stop_t_BTB = 0, start_t_CTC, stop_t_CTC = 0;
	double start_t_KCB, stop_t_KCB = 0, start_t_KCA, stop_t_KCA = 0, start_t_KBA, stop_t_KBA = 0;
	double start_t_Wj1, stop_t_Wj1 = 0, start_t_Wj2, stop_t_Wj2 = 0, start_t_Wj3, stop_t_Wj3 = 0;
	double start_t_GOV, stop_t_GOV = 0;
	double start_t_Nor, stop_t_Nor = 0;
	double start_t_acc, stop_t_acc = 0;

	omp_obj omp_var;
	omp_var.sockets = numa_num_configured_nodes();
	omp_var.threads = Eigen::nbThreads();
	omp_var.threads_per_socket = omp_var.threads / omp_var.sockets;

	cout << "#sockets = " << omp_var.sockets << endl; //numa_num_configured_nodes() << endl;
	// cout << "#cores = " << numa_num_configured_cpus() << endl;
	cout << "#threads = " << omp_var.threads << endl;

	Set_Info(&R, &I, &J, &K, "Data_cpp/info.bin");			// Initialize tensor size and rank from file

    cout << "R=" << R << ", I=" << I << ", J=" << J << ", K=" << K << endl;

	VectorXi BZ(3,1);
	BZ.setConstant(min(2*R, R+10));
	// BZ.setConstant(I);
	cout << "Blocksize=\n" <<  BZ << endl;
	MatrixXi rand_indices(3, I);
	MatrixXi B_cal(3, BZ(1));

	//	<----------------------------		Matrix Initializations		--------------------------------------->	//
	MatrixXd A(I, R), B(J, R), C(K, R);								// Factors A, B, C
	MatrixXd A_sub(BZ(0), R), B_sub(BZ(1),R), C_sub(BZ(2),R);
	
	MatrixXd A_N(I,R), B_N(J,R), C_N(K,R);				// Normalized Factors A_N, B_N, C_N
	MatrixXd A_old_N(I,R), B_old_N(J,R), C_old_N(K,R);	// Normalized Factors A_old_N, B_old_N, C_old_N
	
	MatrixXd X_A(I, size_t(J * K));						// |
	MatrixXd X_B(size_t(J * K), I);						// | Matricized Tensors
	MatrixXd X_C(K, size_t(I * J));						// |
	MatrixXd X_A_sub(BZ(0), BZ(1) * BZ(2));				// |
	MatrixXd X_B_sub(BZ(1) * BZ(2), BZ(0));				// | Matricized Sub-Tensors
	MatrixXd X_C_sub(BZ(2), BZ(0) * BZ(1));				// |
	
	MatrixXd A_T_A(R, R), B_T_B(R, R), C_T_C(R, R);		// A^T*A, B^T*B, C^T*C
	MatrixXd W_C(K, R);	    							// MTTKRP for the computation of cost function
	MatrixXd W_C_sub(BZ(2), R);							// MTTKRP for the computation of cost function
	// MatrixXd KhatriRao_CB(size_t(J * K), R);				// |
	// MatrixXd KhatriRao_CA(size_t(I * K), R);					// |Khatri Rao products
	// MatrixXd KhatriRao_BA(size_t(I * J), R);						// |

	MatrixXd KhatriRao_CB(BZ(1) * BZ(2), R); // |
	MatrixXd KhatriRao_CA(BZ(0) * BZ(2), R);	// |Khatri Rao products
	MatrixXd KhatriRao_BA(BZ(0) * BZ(1), R);		// |

	MatrixXd KhatriRao_BA_full(size_t(I * J), R);

	KhatriRao_CB.setZero();
	KhatriRao_CA.setZero();
	KhatriRao_BA.setZero();

	KhatriRao_BA_full.setZero();
	//	<----------------------		Read Initial Factors and Tensor from file		--------------------------->	//
	Read_Data(omp_var, A, B, C, X_A, X_B, X_C, I, J, K, R);
	

	//	<-----------------------		Frobenious Squared Norm of Tensor		------------------------------->	//
	frob_X = X_C.squaredNorm();

	//	<-----------------------	Normalize and Calculate A^T*A, B^T*B, C^T*C		-------------------------->	//
	// Normalize_Init(A, B, C);

	// std::cout << rand_indices << "\n";
	// std::cout << A << "\n\n";
	// std::cout << A_sub << "\n";
	// std::cout << B_cal << "\n";
	// std::cout << X_A << "\n\n\n\n";
	// std::cout << X_A_sub << "\n\n\n";

	// create_subfactors(B, B_sub, X_B, X_B_sub, BZ, rand_indices, B_cal, 1, 2);
	// std::cout << B_cal << "\n";

	// // std::cout << rand_indices << "\n";
	// // // std::cout << A << "\n\n";
	// // // std::cout << A_sub << "\n";
	// // std::cout << B_cal << "\n\n";
	// std::cout << X_B << "\n\n\n\n";
	// std::cout << X_B_sub << "\n\n\n";

	// create_subfactors(C, C_sub, X_C, X_C_sub, BZ, rand_indices, B_cal, 1, 3);
	// std::cout << B_cal << "\n";
	// std::cout << X_C << "\n\n\n\n";
	// std::cout << X_C_sub << "\n\n\n";

	AO_iter = 1;

	//	<------------------------		Sample A,B,C		------------------------------->	//
	create_subfactors(A, A_sub, X_A, X_A_sub, BZ, rand_indices, B_cal, AO_iter, 1);
	// std::cout << rand_indices << "\n";
	// std::cout << B_cal << "\n";
	// std::cout << A_sub << "\n";
	// std::cout << "X_A = \n" << X_A << "\n\nX_A_sub = \n" << X_A_sub << "\n\n\n";
	create_subfactors(B, B_sub, X_B, X_B_sub, BZ, rand_indices, B_cal, AO_iter, 2);
	// std::cout << "X_B = \n" << X_B << "\n\nX_B_sub = \n" << X_B_sub << "\n\n\n";
	create_subfactors(C, C_sub, X_C, X_C_sub, BZ, rand_indices, B_cal, AO_iter, 3);
	// std::cout << "X_C = \n" << X_C << "\n\nX_C_sub = \n" << X_C_sub << "\n\n\n";

	cout << " BEGIN ALGORITHM " << endl;
	

	//	<-----------------------	Start Timers		--------------------------->	//
	start_t = tic();

	//	<-----------------------	Cost Function Computation		---------------------->	//
	// Khatri_Rao_Product(B, A, KhatriRao_BA);
	// double start_WC = tic();
	// W_C = X_C * KhatriRao_BA;
	// double stop_WC = toc(start_WC);
	// cout << "time W_C = " <<  stop_WC << endl;
	
	// f_value = Get_Objective_Value(C, W_C, A_T_A, B_T_B, C_T_C, frob_X);
	f_value = 0.0;
	// double stop_t_init = toc(start_t);
	while(1){	//	<-------------------------------------------	Begin AO Algorithm		------------------------------------------>	//
		cout << AO_iter << " -- " << f_value/sqrt(frob_X) << " -- " << f_value << " -- " << frob_X << " -- " <<  endl;
		
		A_T_A.noalias() = A_sub.transpose() * A_sub;
		B_T_B.noalias() = B_sub.transpose() * B_sub;
		C_T_C.noalias() = C_sub.transpose() * C_sub;
		// std::cout << rand_indices << "\n";
		std::cout << B_cal << "\n";
		//	<------------------------		 Update A		------------------------------->	//

		start_t_KCB = tic();
		Khatri_Rao_Product(omp_var, C_sub, B_sub, KhatriRao_CB); // C kr B
		stop_t_KCB += toc(start_t_KCB);
		start_t_Wj1 = tic();
		Workerjob(omp_var, BZ(1), BZ(2), A_sub, C_T_C, B_T_B, X_A_sub, KhatriRao_CB, 1);
		stop_t_Wj1 += toc(start_t_Wj1);
		start_t_ATA = tic();
		A_T_A.noalias() = A_sub.transpose() * A_sub;
		stop_t_ATA += toc(start_t_ATA);

		//	<------------------------		 Update B		------------------------------->	//
		start_t_KCA = tic();
		Khatri_Rao_Product(omp_var, C_sub, A_sub, KhatriRao_CA); // C kr A
		stop_t_KCA += toc(start_t_KCA);
		start_t_Wj2 = tic();
		Workerjob(omp_var, BZ(0), BZ(2), B_sub, C_T_C, A_T_A, X_B_sub, KhatriRao_CA, 2);
		stop_t_Wj2 += toc(start_t_Wj2);
		start_t_BTB = tic();
		B_T_B.noalias() = B_sub.transpose() * B_sub;
		stop_t_BTB += toc(start_t_BTB);

		//	<------------------------		 Update C		------------------------------->	//
		start_t_KBA = tic();
		Khatri_Rao_Product(omp_var, B_sub, A_sub, KhatriRao_BA); // B kr A
		stop_t_KBA += toc(start_t_KBA);
		start_t_Wj3 = tic();
		Workerjob_C(omp_var, BZ(0), BZ(1), C_sub, W_C_sub, B_T_B, A_T_A, X_C_sub, KhatriRao_BA);
		stop_t_Wj3 += toc(start_t_Wj3);
		start_t_CTC = tic();
		C_T_C.noalias() = C_sub.transpose() * C_sub;
		stop_t_CTC += toc(start_t_CTC);

		//	<------------------------	Merge sub factors	------------------------------->	//

		merge_Factors(A, A_sub, BZ, B_cal, 1);
		merge_Factors(B, B_sub, BZ, B_cal, 2);
		merge_Factors(C, C_sub, BZ, B_cal, 3);

		//	<-----------------------	Cost Function Computation		---------------------->	//
		Khatri_Rao_Product(omp_var, B, A, KhatriRao_BA_full);
		start_t_GOV = tic();
		W_C = X_C * KhatriRao_BA_full;
		A_T_A.noalias() = A.transpose() * A;
		B_T_B.noalias() = B.transpose() * B;
		C_T_C.noalias() = C.transpose() * C;

		f_value = Get_Objective_Value(C, W_C, A_T_A, B_T_B, C_T_C, frob_X);
		stop_t_GOV += toc(start_t_GOV);

		//	<-----------------------	Normalization step		--------------------------->	//
		start_t_Nor = tic();
		// Normalize(A, B, C, A_T_A, B_T_B, C_T_C);
		stop_t_Nor += toc(start_t_Nor);


		// A_N = A;
		// B_N = B;
		// C_N = C;

		//	<-----------------------	Terminating condition		----------------------->	//
		if (f_value/sqrt(frob_X) < AO_tol || AO_iter > max_iter)
			break;

		//	<-----------------------	Acceleration step		--------------------------->	//
		// if (AO_iter>k_0){
		// 	start_t_acc = tic();
		// 	Line_Search_Accel(omp_var, A_old_N, B_old_N, C_old_N, A, B, C, A_T_A, B_T_B, C_T_C, KhatriRao_BA, X_C, &acc_fail, &acc_coeff, AO_iter, f_value, frob_X);
		// 	stop_t_acc += toc(start_t_acc);
		// }
		// A_old_N = A_N;
		// B_old_N = B_N;
		// C_old_N = C_N;

		AO_iter++;

		//	<------------------------		Resample A,B,C		------------------------------->	//
		create_subfactors(A, A_sub, X_A, X_A_sub, BZ, rand_indices, B_cal, AO_iter, 1);
		// std::cout << "\n\nX_A_sub = \n" << X_A_sub << "\n\n\n";
		create_subfactors(B, B_sub, X_B, X_B_sub, BZ, rand_indices, B_cal, AO_iter, 2);
		// std::cout << "\n\nX_B_sub = \n" << X_B_sub << "\n\n\n";
		create_subfactors(C, C_sub, X_C, X_C_sub, BZ, rand_indices, B_cal, AO_iter, 3);
		// std::cout << "\n\nX_C_sub = \n" << X_C_sub << "\n\n\n";

		std::cout << "\n\nnorm(X_A_sub) = \n" << X_A_sub.norm() << "\n\n\n";

	}	//	<-------------------------------------------	End of AO Algorithm		------------------------------------------->	//

	//	<-----------------------	End of timers		--------------------------->	//
	// end_t = clock();
	// total_t = (end_t - start_t);
	stop_t = toc(start_t);

	cout.precision(15);

	//	<----------------	Printing some results in the terminal	------------------------->	//
	cout << " CPU time = " << stop_t << endl; //((float)total_t) / CLOCKS_PER_SEC << endl;
	cout << " AO_iter = " << AO_iter << endl;
	cout << " relative f_value = " << f_value/sqrt(frob_X) << endl << endl;
	// cout << " Elapsed Time until while()\t|\tCPU time = " << std::setprecision(5) << stop_t_init << "\t|\t" << std::setprecision(5) << 100 * stop_t_init / stop_t << " % (Total)" << endl;
	cout << " <Khatri_Rao_Product(CB)>\t|\tCPU time = " << std::setprecision(5) << stop_t_KCB << "\t|\t" << std::setprecision(5) << 100 * stop_t_KCB / stop_t << " % (Total)" << endl;
	cout << " <Workerjob(...,1)>      \t|\tCPU time = " << std::setprecision(5) << stop_t_Wj1 << "\t|\t" << std::setprecision(5) << 100 * stop_t_Wj1 / stop_t << " % (Total)" << endl;
	cout << " <A_T * A>               \t|\tCPU time = " << std::setprecision(5) << stop_t_ATA << "\t|\t" << std::setprecision(5) << 100 * stop_t_ATA / stop_t << " % (Total)" << endl;
	cout << " <Khatri_Rao_Product(CA)>\t|\tCPU time = " << std::setprecision(5) << stop_t_KCA << "\t|\t" << std::setprecision(5) << 100 * stop_t_KCA / stop_t << " % (Total)" << endl;
	cout << " <Workerjob(...,2)>      \t|\tCPU time = " << std::setprecision(5) << stop_t_Wj2 << "\t|\t" << std::setprecision(5) << 100 * stop_t_Wj2 / stop_t << " % (Total)" << endl;
	cout << " <B_T * B>               \t|\tCPU time = " << std::setprecision(5) << stop_t_BTB << "\t|\t" << std::setprecision(5) << 100 * stop_t_BTB / stop_t << " % (Total)" << endl;
	cout << " <Khatri_Rao_Product(BA)>\t|\tCPU time = " << std::setprecision(5) << stop_t_KBA << "\t|\t" << std::setprecision(5) << 100 * stop_t_KBA / stop_t << " % (Total)" << endl;
	cout << " <Workerjob_C()>         \t|\tCPU time = " << std::setprecision(5) << stop_t_Wj3 << "\t|\t" << std::setprecision(5) << 100 * stop_t_Wj3 / stop_t << " % (Total)" << endl;
	cout << " <C_T * C>               \t|\tCPU time = " << std::setprecision(5) << stop_t_CTC << "\t|\t" << std::setprecision(5) << 100 * stop_t_CTC / stop_t << " % (Total)" << endl;
	cout << " <Get_Objective_Value()> \t|\tCPU time = " << std::setprecision(5) << stop_t_GOV << "\t|\t" << std::setprecision(5) << 100 * stop_t_GOV / stop_t << " % (Total)" << endl;
	cout << " <Normalize()>           \t|\tCPU time = " << std::setprecision(5) << stop_t_Nor << "\t|\t" << std::setprecision(5) << 100 * stop_t_Nor / stop_t << " % (Total)" << endl;
	cout << " <Line_Search_Accel()>   \t|\tCPU time = " << std::setprecision(5) << stop_t_acc << "\t|\t" << std::setprecision(5) << 100 * stop_t_acc / stop_t << " % (Total)" << endl;

	double leftover_time = stop_t_KCB + stop_t_ATA + stop_t_Wj1 + stop_t_KCA + stop_t_BTB + stop_t_Wj2 + stop_t_KBA + stop_t_CTC + stop_t_Wj3 + stop_t_GOV + stop_t_Nor + stop_t_acc; // + stop_t_init;
	cout << "Leftovers...            \t|\tmixed time = " << std::setprecision(5) << stop_t - leftover_time << "\t|\t" << std::setprecision(5) << 100 * (stop_t - leftover_time) / stop_t << " % (Total)" << endl;
	cout << "#threads used = " << Eigen::nbThreads() << endl;
	//	<----------------	Writing some results in Results.txt	------------------------->	//
/*		ofstream my_file("Data_cpp/Results.txt", ios::out | ios::app);
		if (my_file.is_open()){
			my_file << comm_sz << " " << I << " " << J << " " << K << " " << R << " " << t_end_max << " " << ((float)total_t)/CLOCKS_PER_SEC << " " << AO_iter << " " << (f_value/sqrt(frob_X)) << " " << p_A << " " << p_B << " " << p_C;
			my_file << endl;
			my_file.close();
		}
		else
			cout << "Unable to open file";
		cout << "Machine Epsilon is: " << numeric_limits<double>::epsilon() << endl;
*/
	return 0;
}														// End of main
