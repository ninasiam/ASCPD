/*--------------------------------------------------------------------------------------------------*/
/* 								Function for reading the dimensions of   							*/
/*    								the problem from a binary file									*/
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

void Set_Info(int* R, int* I, int* J, int* K, const char *file_name){
	ifstream my_file(file_name, ios::in | ios::binary);
	if (my_file.is_open()){
		my_file.read((char *) R, sizeof(int));
		my_file.read((char *) I, sizeof(int));
		my_file.read((char *) J, sizeof(int));
		my_file.read((char *) K, sizeof(int));
		my_file.close();
	}
	else
		cout << "Unable to open file \n";
}
