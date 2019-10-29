#include <iostream>
#include <random>
#include <cstdlib>
#include "kat_library.h"

using namespace std;

int Random_int_generator(int max){

    int random_int;

    random_int = rand() % max;
    // int min = 0;
    // // int random_int;

    // random_device rd;
    // mt19937 rng(rd());
    // uniform_int_distribution<int> uni(min,max);

    // auto random_int = uni(rng);
    
    return random_int;
}