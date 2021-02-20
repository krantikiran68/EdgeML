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
    
    delete[] A;
    delete[] B;
    delete[] X;
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

    AddOrSubCir4D(A, B, X, N, H, W, C, shrA, shrB, shrC, add);
    for(int i=0;i<N*C*H*W;i++){
        cout<<X[i]<<endl;
    }

    AddOrSubCir4D<MYINT, MYINT, MYINT, MYINT>(A, B, X,N, H, W, C, shrA, shrB, shrC, add, demote);
    for(int i=0;i<N*C*H*W;i++){
        cout<<X[i]<<endl;
    }

    delete[] A;
    delete[] B;
    delete[] X;
    
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
    
    delete[] A;
    delete[] B;
    delete[] C;
    delete[] A_s;
    delete[] B_s;
    delete[] C_s;
    
}

void test_MatMul(){

    mt19937 generator(0);

    MYINT I, J, K, shrA, shrB, H1, H2, demote;


    I = 10;
    J = 5;
    K = 8;

    shrA = 4;
    shrB = 8;
    H1 = 2;
    H2 = 4;
    demote = 1;

    MYINT* A = new MYINT[I*K];
    MYINT* B = new MYINT[K*J];
    MYINT* C = new MYINT[I*J];
    MYINT *tmp = new MYINT[K];

    int16_t *A_s = new int16_t[I*K];
    int16_t *B_s = new int16_t[K*J];
    int16_t *C_s = new int16_t[I*J];
    int32_t *tmp_s = new int32_t[K];

    for(int i=0;i<I*K;i++)
    {
        A[i] = generator();
        A_s[i] = generator();
    }
    for(int i=0;i<K*J;i++)
    {
        B[i] = generator();
        B_s[i] = generator();
    }

    MatMulNN(A, B, C, tmp, I, K, J, shrA, shrB, H1, H2);
    for(int i=0;i<I*J;i++)
    {
        cout<<C[i]<<endl;
    }

    MatMulNN<int16_t, int16_t, int32_t, int16_t>(A_s, B_s, C_s, tmp_s, I, K, J, shrA, shrB, H1, H2, demote);
    for(int i=0;i<I*J;i++)
    {
        cout<<int(C_s[i])<<endl;
    }

    MatMulNC(A, B, C, tmp, I, K, J, shrA, shrB, H1, H2);
    for(int i=0;i<I*J;i++)
    {
        cout<<C[i]<<endl;
    }

    MatMulNC<int16_t, int16_t, int32_t, int16_t>(A_s, B_s, C_s, tmp_s, I, K, J, shrA, shrB, H1, H2, demote);
    for(int i=0;i<I*J;i++)
    {
        cout<<int(C_s[i])<<endl;
    }
    
    MatMulCN(A, B, C, tmp, I, K, J, shrA, shrB, H1, H2);
    for(int i=0;i<I*J;i++)
    {
        cout<<C[i]<<endl;
    }

    MatMulCN<int16_t, int16_t, int32_t, int16_t>(A_s, B_s, C_s, tmp_s, I, K, J, shrA, shrB, H1, H2, demote);
    for(int i=0;i<I*J;i++)
    {
        cout<<int(C_s[i])<<endl;
    }
    
    MatMulCC(A, B, C, tmp, I, K, J, shrA, shrB, H1, H2);
    for(int i=0;i<I*J;i++)
    {
        cout<<C[i]<<endl;
    }

    MatMulCC<int16_t, int16_t, int32_t, int16_t>(A_s, B_s, C_s, tmp_s, I, K, J, shrA, shrB, H1, H2, demote);
    for(int i=0;i<I*J;i++)
    {
        cout<<int(C_s[i])<<endl;
    }
    
    delete[] A;
    delete[] B;
    delete[] C;
    delete[] tmp;
    delete[] A_s;
    delete[] B_s;
    delete[] C_s;
    delete[] tmp_s;
    
}

void test_SparseMatMul(){

}

void test_MulCir(){
    mt19937 generator(0);

    MYINT I, J, shrA, shrB, demote;

    I = 10;
    J = 5;

    shrA = 4;
    shrB = 8;
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
        A_s[i] = generator();
    }
    for(int i=0;i<I*J;i++)
    {
        B[i] = generator();
        B_s[i] = generator();
    }

    MulCir(A, B, C, I, J, shrA, shrB);
    for(int i=0;i<I*J;i++)
    {
        cout<<C[i]<<endl;
    }

    MulCir<int8_t, int8_t, int16_t, int8_t>(A_s, B_s, C_s, I, J, shrA, shrB, demote);
    for(int i=0;i<I*J;i++)
    {
        cout<<int(C_s[i])<<endl;
    }

    delete[] A;
    delete[] B;
    delete[] C;
    
    delete[] A_s;
    delete[] B_s;
    delete[] C_s;
}

typedef int32_t TypeA;
typedef int16_t TypeF1;
typedef int16_t TypeB1W;
typedef int16_t TypeB1B;
typedef int16_t TypeF2;
typedef int16_t TypeB2W;
typedef int16_t TypeB2B;
typedef int16_t TypeF3;
typedef int16_t TypeB3W;
typedef int16_t TypeB3B;
typedef int32_t TypeC;
typedef int32_t TypeX;
typedef int32_t TypeT;
typedef int64_t TypeU;
typedef int64_t TypeUB1W;
typedef int64_t TypeUB2W;
typedef int64_t TypeUB3W;

void test_MBConv() {
    // std::random_device rand_div;
    // std::mt19937 generator(rand_div());
    std::mt19937 generator(0);
    
    MYINT N, H, W, Cin, Ct, HF, WF, Cout, Hout, Wout, WPADL, WPADR, HPADL, HPADR, HSTR, WSTR, D1, D2, D3;
    MYINT shr1, shr2, shr3, shr4, shr5, shr6, shr7, shr8, shr9; 
    MYINT shl1, shl2, shl3, shl4, shl5, shl6, shl7, shl8, shl9; 
    TypeUB1W SIX_1 = 1572864L;
    TypeUB2W SIX_2 = 393216L;

    N = 1;
    H = 15;
    W = 20;
    Cin = 96;
    Ct = 192;
    HF = 3;
    WF = 3;
    Cout = 96;
    Hout = 15;
    Wout = 20;
    HPADL = 1;
    HPADR = 1;
    WPADL = 1;
    WPADR = 1;
    HSTR = 1;
    WSTR = 1;
    D1 = 7;
    D2 = 4;
    D3 = 8;
    shr1 = 1;
    shr2 = 1;
    shr3 = 64;
    shr4 = 8;
    shr5 = 1;
    shr6 = 16;
    shr7 = 1;
    shr8 = 1;
    shr9 = 64;
    shl1 = 1;
    shl2 = 8;
    shl3 = 1;
    shl4 = 1;
    shl5 = 16;
    shl6 = 1;
    shl7 = 2;
    shl8 = 64;
    shl9 = 1;

    TypeA* A = new TypeA[N*H*W*Cin];
    TypeF1* F1 = new TypeF1[Cin*Ct];
    TypeB1W* BN1W = new TypeB1W[Ct];
    TypeB1B* BN1B = new TypeB1B[Ct];
    TypeF2* F2 = new TypeF2[Ct*HF*WF];
    TypeB2W* BN2W = new TypeB2W[Ct];
    TypeB2B* BN2B = new TypeB2B[Ct];
    TypeF3* F3 = new TypeF3[Ct*Cout];
    TypeB3W* BN3W = new TypeB3W[Cout];
    TypeB3B* BN3B = new TypeB3B[Cout];
    TypeC* C = new TypeC[N*Hout*Wout*Cout];
    TypeX* X = new TypeX[N*H*W*Ct];
    TypeT* T = new TypeT[N*Hout*Wout*Ct];
    TypeU* U = new TypeU[N*H*W*max(Ct, max(Cout, Cin))];

    for (int i = 0; i < N*H*W*Cin; i++) {
        A[i] = generator();
    }
    for (int i = 0; i < Cin*Ct; i++) {
        F1[i] = generator();
    }
    for (int i = 0; i < Ct*HF*WF; i++) {
        F2[i] = generator();
    }
    for (int i = 0; i < Ct*Cout; i++) {
        F3[i] = generator();
    }
    for (int i = 0; i < Ct; i++) {
        BN1W[i] = generator();
        BN1B[i] = generator();
        BN2W[i] = generator();
        BN2B[i] = generator();
    }
    for (int i = 0; i < Cout; i++) {
        BN3W[i] = generator();
        BN3B[i] = generator();
    }
    
    MBConv<TypeA, TypeF1, TypeB1W, TypeB1B, TypeF2, TypeB2W, TypeB2B, TypeF3, TypeB3W, TypeB3B, TypeC, TypeX, TypeT, TypeU, TypeUB1W, TypeUB2W, TypeUB3W>(A, F1, BN1W, BN1B, F2, BN2W, BN2B, F3, BN3W, BN3B, C, X, T, U, N, H, W, Cin, Ct, HF, WF, Cout, Hout, Wout, HPADL, HPADR, WPADL, WPADR, HSTR, WSTR, D1, D2, D3, SIX_1, SIX_2, shr1, shr2, shr3, shr4, shr5, shr6, shr7, shr8, shr9, shl1, shl2, shl3, shl4, shl5, shl6, shl7, shl8, shl9);
    for(int i = 0;i<N*Hout*Wout*Cout;i++)
    {
        cout<<(int64_t)C[i]<<endl;
    }

    delete[] A;
    delete[] F1;
    delete[] BN1W;
    delete[] BN1B;
    delete[] F2;
    delete[] BN2W;
    delete[] BN2B;
    delete[] F3;
    delete[] BN3W;
    delete[] BN3B;
    delete[] C;
    delete[] X;
    delete[] T;
    delete[] U;
}

// typedef int16_t TypeA;
typedef int32_t TypeB;
typedef int64_t TypeTemp;
// typedef int16_t TypeC;

void test_Conv(){
    // std::random_device rand_div;
    // std::mt19937 generator(rand_div());
    std::mt19937 generator(0);

    MYINT N, H, W, CI, HF, WF, CO, shrA, shrB, H1, H2, demote;

    N = 4;
    H = 8;
    W = 16;
    CI = 8;
    HF = 2;
    WF = 4;
    CO = 32;
    shrA = 4;
    shrB = 8;
    H1 = 1;
    H2 = 5;
    demote = 2;
    
    MYINT Asize, Bsize, Csize, tmpSize;

    Asize = N*H*W*CI;
    Bsize = HF*WF*CI*CO;
    Csize = N*H*W*CO;
    tmpSize = HF*WF*CI;

    MYINT* A = new MYINT[Asize];
    MYINT* B = new MYINT[Bsize];
    MYINT* C = new MYINT[Csize];
    MYINT* tmp = new MYINT[tmpSize];
    
    TypeA *A_s = new TypeA[Asize];
    TypeB *B_s = new TypeB[Bsize];
    TypeC *C_s = new TypeC[Csize];
    TypeTemp *tmp_s = new TypeTemp[tmpSize];

    for(int i=0;i<N*H*W*CI;i++){
        A[i] = generator();
        A_s[i] = generator();
    }
    
    for(int i=0;i<HF*WF*CI*CO;i++){
        B[i] = generator();
        B_s[i] = generator();
    }


    Conv(A, B, C, tmp, N, H, W, CI, HF, WF, CO, shrA, shrB, H1, H2);
    for(int i=0;i<N*H*W*CO;i++)
    {
        cout<<int(C[i])<<endl;
    }

    Conv<TypeA, TypeB, TypeTemp, TypeC>(A_s, B_s, C_s, tmp_s, N, H, W, CI, HF, WF, CO, shrA, shrB, H1, H2, demote);
    for(int i=0;i<N*H*W*CO;i++)
    {
        cout<<int(C_s[i])<<endl;
    }

    delete[] A;
    delete[] B;
    delete[] C;
    delete[] tmp;
    
    delete[] A_s;
    delete[] B_s;
    delete[] C_s;
    delete[] tmp_s;
}

void test_Convolution()
{
    // std::random_device rand_div;
    // std::mt19937 generator(rand_div());
    std::mt19937 generator(0);

    MYINT N, H, W, CIN, HF, WF, CINF, COUTF, HOUT, WOUT, HPADL, HPADR, WPADL;
    MYINT WPADR, HSTR, WSTR, HDL, WDL, G, shrA, shrB, H1, H2, demote;

    N = 1;
    H = 230;
    W = 230;
    CIN = 3;
    HF = 7;
    WF = 7;
    CINF = 3;
    COUTF = 64;
    HOUT = 112;
    WOUT = 112;
    HPADL = 0;
    HPADR = 0;
    WPADL = 0;
    WPADR = 0;
    HSTR = 2;
    WSTR = 2;
    HDL = 1;
    WDL = 1;
    G = 1;
    shrA = 1;
    shrB = 1;
    H1 = 18;
    H2 = 0;
    demote = 1;
    
    MYINT Asize, Bsize, Csize, tmpSize;

    Asize = N*H*W*CIN;
    Bsize = G*HF*WF*CINF*COUTF;
    Csize = N*HOUT*WOUT*COUTF*G;
    tmpSize = HF*WF*CINF;

    MYINT* A = new MYINT[Asize];
    MYINT* B = new MYINT[Bsize];
    MYINT* C = new MYINT[Csize];
    MYINT* tmp = new MYINT[tmpSize];
    
    TypeA* A_s = new TypeA[Asize];
    TypeB* B_s = new TypeB[Bsize];
    TypeC* C_s = new TypeC[Csize];
    TypeTemp* tmp_s = new TypeTemp[tmpSize];
    
    for (int i=0;i<Asize;i++) {
        A[i] = generator();
        A_s[i] = generator();
    }

    for (int i=0; i<Bsize;i++) {
        B[i] = generator();
        B_s[i] = generator();
    }

    // Convolution(A, B, C, tmp, N, H, W, CIN, HF, WF, CINF, COUTF, HOUT, WOUT, HPADL, HPADR, WPADL, WPADR, 
    //             HSTR, WSTR, HDL, WDL, G, shrA, shrB, H1, H2);
    
    // for(int i=0;i<Csize;i++)
    // {
    //     cout<<int(C[i])<<endl;
    // }

    Convolution<TypeA, TypeB, TypeTemp, TypeC>(A_s, B_s, C_s, tmp_s, N, H, W, CIN, HF, WF, CINF, COUTF, HOUT, WOUT, HPADL, HPADR, WPADL, WPADR, 
                HSTR, WSTR, HDL, WDL, G, shrA, shrB, H1, H2, demote);
    
    for(int i=0;i<Csize;i++)
    {
        cout<<int(C_s[i])<<endl;
    }

    delete[] A;
    delete[] B;
    delete[] C;
    delete[] tmp;

    delete[] A_s;
    delete[] B_s;
    delete[] C_s;
    delete[] tmp_s;
}

int main(){

    // test_AddOrSubCir2D();
    // test_AddOrSubCir4D();
    // test_MatAdd();
    // test_MatSub();
    // test_MatMul();
    // test_SparseMatMul();
    // test_MulCir();
    // test_MBConv();
    // test_Conv();
    test_Convolution();
    return 0;
}