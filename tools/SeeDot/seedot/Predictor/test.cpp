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

    N = 4;
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
    // AddOrSubCir4D(A, B, X, N, H, W, C, shrA, shrB, shrC, add);
    // for(int i=0;i<N*H*W;i++){
    //     for(int j=0;j<C;j++)
    //     {
    //         cout<<X[i*C+j]<<endl;
    //     }
    // }

    AddOrSubCir4D<MYINT, MYINT, MYINT, MYINT>(A, B, X, N, H, W, C, shrA, shrB, shrC, add, demote);
    for(int i=0;i<N*C*H*W;i++){
        cout<<X[i]<<endl;
    }

    add = false;

    // AddOrSubCir4D(A, B, X, N, H, W, C, shrA, shrB, shrC, add);
    // for(int i=0;i<N*C*H*W;i++){
    //     cout<<X[i]<<endl;
    // }

    // AddOrSubCir4D<MYINT, MYINT, MYINT, MYINT>(A, B, X,N, H, W, C, shrA, shrB, shrC, add, demote);
    // for(int i=0;i<N*C*H*W;i++){
    //     cout<<X[i]<<endl;
    // }
}

void test_MatAdd(){
    mt19937 generator(0);

    MYINT I, J, shrA, shrB, shrC, demote;


    I = 10;
    J = 5;
    shrA = 4;
    shrB = 8;
    shrC = 16;
    demote = 2;

    MYINT* A = new MYINT[I*J];
    MYINT* B = new MYINT[I*J];
    MYINT* C = new MYINT[I*J];

    int16_t *A_s = new int16_t[I*J];
    int16_t *B_s = new int16_t[I*J];
    int16_t *C_s = new int16_t[I*J];

    for(int i=0;i<I*J;i++)
    {
        A[i] = generator();
        B[i] = generator();
        A_s[i] = generator();
        B_s[i] = generator();
    }

    MatAddNN(A, B, C, I, J, shrA, shrB, shrC);
    for(int i=0;i<I*J;i++)
    {
        cout<<C[i]<<endl;
    }
    
    MatAddNN<int16_t, int16_t, int32_t, int16_t>(A_s, B_s, C_s, I, J, shrA, shrB, shrC, demote);
    for(int i=0;i<I*J;i++)
    {
        cout<<int(C_s[i])<<endl;
    }

    MatAddCN(A, B, C, I, J, shrA, shrB, shrC);
    for(int i=0;i<I*J;i++)
    {
        cout<<C[i]<<endl;
    }
    
    MatAddCN<int16_t, int16_t, int32_t, int16_t>(A_s, B_s, C_s, I, J, shrA, shrB, shrC, demote);
    for(int i=0;i<I*J;i++)
    {
        cout<<int(C_s[i])<<endl;
    }

    MatAddNC(A, B, C, I, J, shrA, shrB, shrC);
    for(int i=0;i<I*J;i++)
    {
        cout<<C[i]<<endl;
    }
    
    MatAddNC<int16_t, int16_t, int32_t, int16_t>(A_s, B_s, C_s, I, J, shrA, shrB, shrC, demote);
    for(int i=0;i<I*J;i++)
    {
        cout<<int(C_s[i])<<endl;
    }
    
    MatAddCC(A, B, C, I, J, shrA, shrB, shrC);
    for(int i=0;i<I*J;i++)
    {
        cout<<C[i]<<endl;
    }
    
    MatAddCC<int16_t, int16_t, int32_t, int16_t>(A_s, B_s, C_s, I, J, shrA, shrB, shrC, demote);
    for(int i=0;i<I*J;i++)
    {
        cout<<int(C_s[i])<<endl;
    }

    MatAddBroadCastA(A, B, C, I, J, shrA, shrB, shrC);
    for(int i=0;i<I*J;i++)
    {
        cout<<C[i]<<endl;
    }
    
    MatAddBroadCastA<int16_t, int16_t, int32_t, int16_t>(A_s, B_s, C_s, I, J, shrA, shrB, shrC, demote);
    for(int i=0;i<I*J;i++)
    {
        cout<<int(C_s[i])<<endl;
    }

    MatAddBroadCastB(A, B, C, I, J, shrA, shrB, shrC);
    for(int i=0;i<I*J;i++)
    {
        cout<<C[i]<<endl;
    }
    
    MatAddBroadCastB<int16_t, int16_t, int32_t, int16_t>(A_s, B_s, C_s, I, J, shrA, shrB, shrC, demote);
    for(int i=0;i<I*J;i++)
    {
        cout<<int(C_s[i])<<endl;
    }

    MYINT N,H,W,X;
    N = 2;
    H = 4;
    W = 3;
    X = 4;

    int8_t *A4_s = new int8_t[N*H*W*X]; 
    int8_t *B4_s = new int8_t[N*H*W*X]; 
    int8_t *C4_s = new int8_t[N*H*W*X]; 

    for(int i=0;i<N*H*W*X;i++){
        A4_s[i] = generator();
        B4_s[i] = generator();
    }

    MatAdd4<int8_t, int8_t, int16_t, int8_t>(A4_s, B4_s, C4_s, N,H,W,X, shrA, shrB, shrC, demote);

    delete[] A;
    delete[] B;
    delete[] C;
    delete[] A_s;
    delete[] B_s;
    delete[] C_s;
    
}

void test_MatSub()
{
    mt19937 generator(0);

    MYINT I, J, shrA, shrB, shrC, demote;


    I = 10;
    J = 5;
    shrA = 4;
    shrB = 8;
    shrC = 16;
    demote = 2;

    MYINT* A = new MYINT[I*J];
    MYINT* B = new MYINT[I*J];
    MYINT* C = new MYINT[I*J];

    int8_t *A_s = new int8_t[I*J];
    int8_t *B_s = new int8_t[I*J];
    int8_t *C_s = new int8_t[I*J];

    for(int i=0;i<I*J;i++)
    {
        A[i] = generator();
        B[i] = generator();
        A_s[i] = generator();
        B_s[i] = generator();
    }

    MatSub(A, B, C, I, J, shrA, shrB, shrC);
    for(int i=0;i<I*J;i++){
        cout<<C[i]<<endl;
    }

    MatSub<int8_t, int8_t, int16_t, int8_t>(A_s, B_s, C_s, I, J, shrA, shrB, shrC, demote);
    for(int i=0;i<I*J;i++){
        cout<<int(C_s[i])<<endl;
    }

    MatSubBroadCastA(A, B, C, I, J, shrA, shrB, shrC);
    for(int i=0;i<I*J;i++){
        cout<<C[i]<<endl;
    }

    MatSubBroadCastA<int8_t, int8_t, int16_t, int8_t>(A_s, B_s, C_s, I, J, shrA, shrB, shrC, demote);
    for(int i=0;i<I*J;i++){
        cout<<int(C_s[i])<<endl;
    }

    MatSubBroadCastB(A, B, C, I, J, shrA, shrB, shrC);
    for(int i=0;i<I*J;i++){
        cout<<C[i]<<endl;
    }

    MatSubBroadCastB<int8_t, int8_t, int16_t, int8_t>(A_s, B_s, C_s, I, J, shrA, shrB, shrC, demote);
    for(int i=0;i<I*J;i++){
        cout<<int(C_s[i])<<endl;
    }
    
}

int main(){

    // test_AddOrSubCir2D();
    // test_AddOrSubCir4D();
    test_MatAdd();
    test_MatSub();
    
    return 0;
}