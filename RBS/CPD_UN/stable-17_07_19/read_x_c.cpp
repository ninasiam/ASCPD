/*--------------------------------------------------------------------------------------------------*/
/* 				Function for reading the Matricized Tensor from one binary file						*/
/*    							and save it as Eigen matrix										   	*/
/*                                                                           						*/
/* A. P. Liavas																						*/
/* Georgios Lourakis																				*/ 
/* Georgios Kostoulas																				*/
/* 15/11/2017                                              											*/
/*--------------------------------------------------------------------------------------------------*/
#include <iostream>
#include <fstream>
#include "ntf_functions.h"

using namespace std;

void Read_X_C(int n_I, int n_J, int n_K, int skip_I, int skip_J, int skip_K, int I, int J, Ref<MatrixXd> X_C_one_file_T, const char *file_name){	
	int i, j;	
	int elements_read = 0;
	int elements_left_row = 0;
	int offset_I = skip_I;	
	int offset_J = skip_J * I;
	int offset_K = skip_K * I * J;

	ifstream my_file(file_name, ios::in | ios::binary);
	if (my_file.is_open()){
		my_file.ignore(offset_K*sizeof(double));
		for (i=0; i<n_K; i++){
			my_file.ignore((offset_I+offset_J)*sizeof(double));
			my_file.read((char *) X_C_one_file_T.data()+elements_read, n_I*sizeof(double));
			elements_read += n_I*sizeof(double);
			for (j=0; j<n_J-1; j++){
				my_file.ignore((I - n_I)*sizeof(double));
				my_file.read((char *) X_C_one_file_T.data()+elements_read, n_I*sizeof(double));
				elements_read += n_I*sizeof(double);
			}
			elements_left_row = (I * J) - (offset_I + offset_J + n_J*n_I + (n_J-1)*(I - n_I));
			my_file.ignore(elements_left_row*sizeof(double));
		}
		my_file.close();
	}
	else
		cout << "Unable to open file \n";
}
