#include <iostream>
#include <random>
#include <cstdlib>
#include "kat_library.h"

int Random_int_generator(int max){

    int min = 0;
    int random_int;

    std::random_device rd;
    std::mt19937 rng(rd());
    std::uniform_int_distribution<int> uni(min,max);

    auto random_int = uni(rng);
    
    return random_int;
}