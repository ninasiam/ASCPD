/*--------------------------------------------------------------------------------------------------*/
/* 						Function for reading a matrix from a binary file  							*/
/*    							and save it as Eigen matrix										   	*/
/*                                                                           						*/
/* A. P. Liavas																						*/
/* Georgios Lourakis																				*/ 
/* Georgios Kostoulas																				*/
/* 15/11/2017                                              											*/
/*--------------------------------------------------------------------------------------------------*/
#include <iostream>
#include <fstream>
#include "brascpd_functions.h"

using namespace std;

void Read_From_File(int nrows, int ncols, Ref<MatrixXd> Mat, const char *file_name, int skip){
	ifstream my_file(file_name, ios::in | ios::binary);
	if (my_file.is_open()){
		my_file.ignore(skip*sizeof(double));
		my_file.read((char *) Mat.data(), nrows*ncols*sizeof(double));
		my_file.close();
	}
	else
		cout << "Unable to open file \n";
}
