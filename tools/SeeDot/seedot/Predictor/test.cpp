#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <cstring>
#include <cmath>
#include <cstdlib>
#include <thread>
#include <algorithm>
#include <random>

#include "datatypes.h"
#include "library_fixed.h"

using namespace std;

void test_AddOrSubCir2D(){
    // random_device rand_div;
    // mt19937 generator(rand_div());
    mt19937 generator(0);

    MYINT H, W, shrA, shrB, shrC, demote;
    bool add = true;

    H = 10; // Height of Array
    W = 2; // Width of Array

    shrA = 4;// Scale down 1
    shrB = 2;// Scale down 2
    shrC = 8;// Scale down final
    demote = 2;

    MYINT *A = new MYINT[H*W];
    MYINT *B = new MYINT[W];
    MYINT *X = new MYINT[H*W];
    MYINT *X_corr = new MYINT[H*W];

    for(int i = 0;i<H*W;i++){
        A[i] = generator();
    }
    for(int i=0;i<W;i++){
        B[i] = generator();
    }
    AddOrSubCir2D(A, B, X, H, W, shrA, shrB, shrC, add);
    for(int i=0;i<H*W;i++){
        cout<<X[i]<<endl;
    }

    AddOrSubCir2D<MYINT, MYINT, MYINT, MYINT>(A, B, X, H, W, shrA, shrB, shrC, add, demote);
    for(int i=0;i<H*W;i++){
        cout<<X[i]<<endl;
    }

    add = false;

    AddOrSubCir2D(A, B, X, H, W, shrA, shrB, shrC, add);
    for(int i=0;i<H*W;i++){
        cout<<X[i]<<endl;
    }

    AddOrSubCir2D<MYINT, MYINT, MYINT, MYINT>(A, B, X, H, W, shrA, shrB, shrC, add, demote);
    for(int i=0;i<H*W;i++){
        cout<<X[i]<<endl;
    }
}

void test_AddOrSubCir4D(){
    // random_device rand_div;
    // mt19937 generator(rand_div());
    mt19937 generator(0);

    MYINT N, H, W, C, shrA, shrB, shrC, demote;
    bool add = true;

    N = 15;
    H = 10; // Height of Array
    W = 2; // Width of Array
    C = 5;

    shrA = 4;// Scale down 1
    shrB = 2;// Scale down 2
    shrC = 8;// Scale down final
    demote = 2;

    MYINT *A = new MYINT[N*C*H*W];
    MYINT *B = new MYINT[C];
    MYINT *X = new MYINT[N*C*H*W];
    MYINT *X_corr = new MYINT[H*W];

    for(int i = 0;i<N*C*H*W;i++){
        A[i] = generator();
    }
    for(int i=0;i<C;i++){
        B[i] = generator();
    }
    AddOrSubCir4D(A, B, X, N, H, W, C, shrA, shrB, shrC, add);
    for(int i=0;i<N*H*W;i++){
        for(int j=0;j<C;j++)
        {
            cout<<X[i*C+j]<<endl;
        }
    }

    AddOrSubCir4D<MYINT, MYINT, MYINT, MYINT>(A, B, X, N, H, W, C, shrA, shrB, shrC, add, demote);
    for(int i=0;i<N*C*H*W;i++){
        cout<<X[i]<<endl;
    }

    add = false;

    AddOrSubCir4D(A, B, X, N, H, W, C, shrA, shrB, shrC, add);
    for(int i=0;i<N*C*H*W;i++){
        cout<<X[i]<<endl;
    }

    AddOrSubCir4D<MYINT, MYINT, MYINT, MYINT>(A, B, X,N, H, W, C, shrA, shrB, shrC, add, demote);
    for(int i=0;i<N*C*H*W;i++){
        cout<<X[i]<<endl;
    }
}


int main(){

    // test_AddOrSubCir2D();
    test_AddOrSubCir4D();
    
    return 0;
}