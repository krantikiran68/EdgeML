// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.
#include "SoftPosit/source/include/softposit.h"
#include <fstream>
#include <iostream>

#pragma once
// This file contains declarations for floating point versions of all operators supported by SeeDot.
// Please refer to library_fixed.h for a description of each operator.

void MatAddNN(posit8_t* A, posit8_t* B, posit8_t* C, MYITE I, MYITE J, MYINT shrA, MYINT shrB, MYINT shrC);
void MatAddCN(const posit8_t* A, posit8_t* B, posit8_t* C, MYITE I, MYITE J, MYINT shrA, MYINT shrB, MYINT shrC);
void MatAddNC(posit8_t* A, const posit8_t* B, posit8_t* C, MYITE I, MYITE J, MYINT shrA, MYINT shrB, MYINT shrC);
void MatAddCC(const posit8_t* A, const posit8_t* B, posit8_t* C, MYITE I, MYITE J, MYINT shrA, MYINT shrB, MYINT shrC);

void MatAddBroadCastA(posit8_t* A, posit8_t* B, posit8_t* C, MYITE I, MYITE J, MYINT shrA, MYINT shrB, MYINT shrC);
void MatAddBroadCastB(posit8_t* A, posit8_t* B, posit8_t* C, MYITE I, MYITE J, MYINT shrA, MYINT shrB, MYINT shrC);

void MatAdd4(posit8_t* A, posit8_t* B, posit8_t* X, MYITE N, MYITE H, MYITE W, MYITE C, MYINT shrA, MYINT shrB, MYINT shrC);

void MatSub(posit8_t* A, const posit8_t* B, posit8_t* C, MYITE I, MYITE J, MYINT shrA, MYINT shrB, MYINT shrC);
void MatSubBroadCastA(posit8_t* A, posit8_t* B, posit8_t* C, MYITE I, MYITE J, MYINT shrA, MYINT shrB, MYINT shrC);
void MatSubBroadCastB(posit8_t* A, posit8_t* B, posit8_t* C, MYITE I, MYITE J, MYINT shrA, MYINT shrB, MYINT shrC);

void MatMulNN(posit8_t* A, posit8_t* B, posit8_t* C, MYITE I, MYITE K, MYITE J, MYINT shrA, MYINT shrB, MYITE H1, MYITE H2);
void MatMulCN(const posit8_t* A, posit8_t* B, posit8_t* C, MYITE I, MYITE K, MYITE J, MYINT shrA, MYINT shrB, MYITE H1, MYITE H2);
void MatMulNC(posit8_t* A, const posit8_t* B, posit8_t* C, MYITE I, MYITE K, MYITE J, MYINT shrA, MYINT shrB, MYITE H1, MYITE H2);
void MatMulCC(const posit8_t* A, const posit8_t* B, posit8_t* C, MYITE I, MYITE K, MYITE J, MYINT shrA, MYINT shrB, MYITE H1, MYITE H2);

void SparseMatMulX(const MYINT* Aidx, const posit8_t* Aval, posit8_t** B, posit8_t* C, int16_t K, MYINT shrA, MYINT shrB, MYINT shrC);
void SparseMatMul(const MYINT* Aidx, const posit8_t* Aval, posit8_t* B, posit8_t* C, int16_t K, MYINT shrA, MYINT shrB, MYINT shrC);

void MulCir(posit8_t* A, posit8_t* B, posit8_t* C, MYITE I, MYITE J, MYINT shrA, MYINT shrB);

void TanH(posit8_t* A, MYITE I, MYITE J, float scale_in, float scale_out, posit8_t* B);

void ArgMax(posit8_t* A, MYITE I, MYITE J, int* index);

void Transpose(posit8_t* A, posit8_t* B, MYITE I, MYITE J);

void ScalarMul(posit8_t* A, posit8_t* B, posit8_t* C, MYITE I, MYITE J, MYINT shrA, MYINT shrB);

void MBConv(posit8_t* A, const posit8_t* F1, const posit8_t* BN1W, const posit8_t* BN1B, const posit8_t* F2, const posit8_t* BN2W, const posit8_t* BN2B, const posit8_t* F3, const posit8_t* BN3W, const posit8_t* BN3B, posit8_t* C, posit8_t* X, posit8_t* T, posit8_t* U, MYITE N, MYITE H, MYITE W, MYITE Cin, MYITE Ct, MYITE HF, MYITE WF, MYITE Cout, MYITE Hout, MYITE Wout, MYITE HPADL, MYITE HPADR, MYITE WPADL, MYITE WPADR, MYITE HSTR, MYITE WSTR, MYITE D1, MYITE D2, MYITE D3, MYINT SIX_1, MYINT SIX_2, MYINT shr1, MYINT shr2, MYINT shr3, MYINT shr4, MYINT shr5, MYINT shr6, MYINT shr7, MYINT shr8, MYINT shr9, MYINT shl1, MYINT shl2, MYINT shl3, MYINT shl4, MYINT shl5, MYINT shl6, MYINT shl7, MYINT shl8, MYINT shl9, std::string name);

void Conv(posit8_t* A, const posit8_t* B, posit8_t* C, posit8_t* tmp, MYITE N, MYITE H, MYITE W, MYITE CI, MYITE HF, MYITE WF, MYITE CO, MYINT shrA, MYINT shrB, MYITE H1, MYITE H2);

void Convolution(posit8_t* A, const posit8_t* B, posit8_t* C, posit8_t* tmp, MYITE N, MYITE H, MYITE W, MYITE CIN, MYITE HF, MYITE WF, MYITE CINF, MYITE COUTF, MYITE HOUT, MYITE WOUT, MYITE HPADL, MYITE HPADR, MYITE WPADL, MYITE WPADR, MYITE HSTR, MYITE WSTR, MYITE HDL, MYITE WDL, MYITE G, MYINT shrA, MYINT shrB, MYITE H1, MYITE H2);

void AddOrSubCir4D(posit8_t* A, const posit8_t* B, posit8_t* X, MYITE N, MYITE H, MYITE W, MYITE C, MYINT shrA, MYINT shrB, MYINT shrC, bool add);

void AddOrSubCir2D(posit8_t* A, const posit8_t* B, posit8_t* X, MYITE H, MYITE W, MYINT shrA, MYINT shrB, MYINT shrC, bool add);

void Relu4D(posit8_t* A, MYITE N, MYITE H, MYITE W, MYITE C);

void Relu2D(posit8_t* A, MYITE H, MYITE W);

void Relu6(posit8_t* A, posit8_t* B, MYITE N, MYITE H, MYITE W, MYITE C, MYINT six, MYINT div);

void Maxpool(posit8_t* A, posit8_t* B, MYITE N, MYITE H, MYITE W, MYITE C, MYITE FH, MYITE FW, MYITE strideH, MYITE strideW, MYITE HPADL, MYITE HPADR, MYITE WPADL, MYITE WPADR);

void Exp(posit8_t* A, MYITE I, MYITE J, MYINT shrA, MYINT shrB, posit8_t* B);

void Sigmoid(posit8_t* A, MYITE I, MYITE J, float div, float add, float sigmoid_limit, MYINT scale_in, MYINT scale_out, posit8_t* B);

void AdjustScaleShr(posit8_t* A, MYITE I, MYITE J, MYINT scale);
void AdjustScaleShl(posit8_t* A, MYITE I, MYITE J, MYINT scale);

void Reverse2(posit8_t* A, MYITE axis, MYITE I, MYITE J, posit8_t* B);

void NormaliseL2(posit8_t* A, posit8_t* B, MYITE N, MYITE H, MYITE W, MYITE C, MYINT scaleA, MYINT shrA);
posit8_t operator-(const posit8_t& a);
bool operator>(const posit8_t& a, const int& b);
void AddInplace(posit8_t* A, posit8_t* B, MYITE I, MYITE J);


void MatAddNN(posit16_t* A, posit16_t* B, posit16_t* C, MYITE I, MYITE J, MYINT shrA, MYINT shrB, MYINT shrC);
void MatAddCN(const posit16_t* A, posit16_t* B, posit16_t* C, MYITE I, MYITE J, MYINT shrA, MYINT shrB, MYINT shrC);
void MatAddNC(posit16_t* A, const posit16_t* B, posit16_t* C, MYITE I, MYITE J, MYINT shrA, MYINT shrB, MYINT shrC);
void MatAddCC(const posit16_t* A, const posit16_t* B, posit16_t* C, MYITE I, MYITE J, MYINT shrA, MYINT shrB, MYINT shrC);

void MatAddBroadCastA(posit16_t* A, posit16_t* B, posit16_t* C, MYITE I, MYITE J, MYINT shrA, MYINT shrB, MYINT shrC);
void MatAddBroadCastB(posit16_t* A, posit16_t* B, posit16_t* C, MYITE I, MYITE J, MYINT shrA, MYINT shrB, MYINT shrC);

void MatAdd4(posit16_t* A, posit16_t* B, posit16_t* X, MYITE N, MYITE H, MYITE W, MYITE C, MYINT shrA, MYINT shrB, MYINT shrC);

void MatSub(posit16_t* A, const posit16_t* B, posit16_t* C, MYITE I, MYITE J, MYINT shrA, MYINT shrB, MYINT shrC);
void MatSubBroadCastA(posit16_t* A, posit16_t* B, posit16_t* C, MYITE I, MYITE J, MYINT shrA, MYINT shrB, MYINT shrC);
void MatSubBroadCastB(posit16_t* A, posit16_t* B, posit16_t* C, MYITE I, MYITE J, MYINT shrA, MYINT shrB, MYINT shrC);

void MatMulNN(posit16_t* A, posit16_t* B, posit16_t* C, MYITE I, MYITE K, MYITE J, MYINT shrA, MYINT shrB, MYITE H1, MYITE H2);
void MatMulCN(const posit16_t* A, posit16_t* B, posit16_t* C, MYITE I, MYITE K, MYITE J, MYINT shrA, MYINT shrB, MYITE H1, MYITE H2);
void MatMulNC(posit16_t* A, const posit16_t* B, posit16_t* C, MYITE I, MYITE K, MYITE J, MYINT shrA, MYINT shrB, MYITE H1, MYITE H2);
void MatMulCC(const posit16_t* A, const posit16_t* B, posit16_t* C, MYITE I, MYITE K, MYITE J, MYINT shrA, MYINT shrB, MYITE H1, MYITE H2);

void SparseMatMulX(const MYINT* Aidx, const posit16_t* Aval, posit16_t** B, posit16_t* C, int16_t K, MYINT shrA, MYINT shrB, MYINT shrC);
void SparseMatMul(const MYINT* Aidx, const posit16_t* Aval, posit16_t* B, posit16_t* C, int16_t K, MYINT shrA, MYINT shrB, MYINT shrC);

void MulCir(posit16_t* A, posit16_t* B, posit16_t* C, MYITE I, MYITE J, MYINT shrA, MYINT shrB);

void TanH(posit16_t* A, MYITE I, MYITE J, float scale_in, float scale_out, posit16_t* B);

void ArgMax(posit16_t* A, MYITE I, MYITE J, int* index);

void Transpose(posit16_t* A, posit16_t* B, MYITE I, MYITE J);

void ScalarMul(posit16_t* A, posit16_t* B, posit16_t* C, MYITE I, MYITE J, MYINT shrA, MYINT shrB);

void MBConv(posit16_t* A, const posit16_t* F1, const posit16_t* BN1W, const posit16_t* BN1B, const posit16_t* F2, const posit16_t* BN2W, const posit16_t* BN2B, const posit16_t* F3, const posit16_t* BN3W, const posit16_t* BN3B, posit16_t* C, posit16_t* X, posit16_t* T, posit16_t* U, MYITE N, MYITE H, MYITE W, MYITE Cin, MYITE Ct, MYITE HF, MYITE WF, MYITE Cout, MYITE Hout, MYITE Wout, MYITE HPADL, MYITE HPADR, MYITE WPADL, MYITE WPADR, MYITE HSTR, MYITE WSTR, MYITE D1, MYITE D2, MYITE D3, MYINT SIX_1, MYINT SIX_2, MYINT shr1, MYINT shr2, MYINT shr3, MYINT shr4, MYINT shr5, MYINT shr6, MYINT shr7, MYINT shr8, MYINT shr9, MYINT shl1, MYINT shl2, MYINT shl3, MYINT shl4, MYINT shl5, MYINT shl6, MYINT shl7, MYINT shl8, MYINT shl9, std::string name);

void Conv(posit16_t* A, const posit16_t* B, posit16_t* C, posit16_t* tmp, MYITE N, MYITE H, MYITE W, MYITE CI, MYITE HF, MYITE WF, MYITE CO, MYINT shrA, MYINT shrB, MYITE H1, MYITE H2);

void Convolution(posit16_t* A, const posit16_t* B, posit16_t* C, posit16_t* tmp, MYITE N, MYITE H, MYITE W, MYITE CIN, MYITE HF, MYITE WF, MYITE CINF, MYITE COUTF, MYITE HOUT, MYITE WOUT, MYITE HPADL, MYITE HPADR, MYITE WPADL, MYITE WPADR, MYITE HSTR, MYITE WSTR, MYITE HDL, MYITE WDL, MYITE G, MYINT shrA, MYINT shrB, MYITE H1, MYITE H2);

void AddOrSubCir4D(posit16_t* A, const posit16_t* B, posit16_t* X, MYITE N, MYITE H, MYITE W, MYITE C, MYINT shrA, MYINT shrB, MYINT shrC, bool add);

void AddOrSubCir2D(posit16_t* A, const posit16_t* B, posit16_t* X, MYITE H, MYITE W, MYINT shrA, MYINT shrB, MYINT shrC, bool add);

void Relu4D(posit16_t* A, MYITE N, MYITE H, MYITE W, MYITE C);

void Relu2D(posit16_t* A, MYITE H, MYITE W);

void Relu6(posit16_t* A, posit16_t* B, MYITE N, MYITE H, MYITE W, MYITE C, MYINT six, MYINT div);

void Maxpool(posit16_t* A, posit16_t* B, MYITE N, MYITE H, MYITE W, MYITE C, MYITE FH, MYITE FW, MYITE strideH, MYITE strideW, MYITE HPADL, MYITE HPADR, MYITE WPADL, MYITE WPADR);

void Exp(posit16_t* A, MYITE I, MYITE J, MYINT shrA, MYINT shrB, posit16_t* B);

void Sigmoid(posit16_t* A, MYITE I, MYITE J, float div, float add, float sigmoid_limit, MYINT scale_in, MYINT scale_out, posit16_t* B);

void AdjustScaleShr(posit16_t* A, MYITE I, MYITE J, MYINT scale);
void AdjustScaleShl(posit16_t* A, MYITE I, MYITE J, MYINT scale);

void Reverse2(posit16_t* A, MYITE axis, MYITE I, MYITE J, posit16_t* B);

void NormaliseL2(posit16_t* A, posit16_t* B, MYITE N, MYITE H, MYITE W, MYITE C, MYINT scaleA, MYINT shrA);
posit16_t operator-(const posit16_t& a);
bool operator>(const posit16_t& a, const int& b);
void AddInplace(posit16_t* A, posit16_t* B, MYITE I, MYITE J);

void MatAddNN(posit32_t* A, posit32_t* B, posit32_t* C, MYITE I, MYITE J, MYINT shrA, MYINT shrB, MYINT shrC);
void MatAddCN(const posit32_t* A, posit32_t* B, posit32_t* C, MYITE I, MYITE J, MYINT shrA, MYINT shrB, MYINT shrC);
void MatAddNC(posit32_t* A, const posit32_t* B, posit32_t* C, MYITE I, MYITE J, MYINT shrA, MYINT shrB, MYINT shrC);
void MatAddCC(const posit32_t* A, const posit32_t* B, posit32_t* C, MYITE I, MYITE J, MYINT shrA, MYINT shrB, MYINT shrC);

void MatAddBroadCastA(posit32_t* A, posit32_t* B, posit32_t* C, MYITE I, MYITE J, MYINT shrA, MYINT shrB, MYINT shrC);
void MatAddBroadCastB(posit32_t* A, posit32_t* B, posit32_t* C, MYITE I, MYITE J, MYINT shrA, MYINT shrB, MYINT shrC);

void MatAdd4(posit32_t* A, posit32_t* B, posit32_t* X, MYITE N, MYITE H, MYITE W, MYITE C, MYINT shrA, MYINT shrB, MYINT shrC);

void MatSub(posit32_t* A, const posit32_t* B, posit32_t* C, MYITE I, MYITE J, MYINT shrA, MYINT shrB, MYINT shrC);
void MatSubBroadCastA(posit32_t* A, posit32_t* B, posit32_t* C, MYITE I, MYITE J, MYINT shrA, MYINT shrB, MYINT shrC);
void MatSubBroadCastB(posit32_t* A, posit32_t* B, posit32_t* C, MYITE I, MYITE J, MYINT shrA, MYINT shrB, MYINT shrC);

void MatMulNN(posit32_t* A, posit32_t* B, posit32_t* C, MYITE I, MYITE K, MYITE J, MYINT shrA, MYINT shrB, MYITE H1, MYITE H2);
void MatMulCN(const posit32_t* A, posit32_t* B, posit32_t* C, MYITE I, MYITE K, MYITE J, MYINT shrA, MYINT shrB, MYITE H1, MYITE H2);
void MatMulNC(posit32_t* A, const posit32_t* B, posit32_t* C, MYITE I, MYITE K, MYITE J, MYINT shrA, MYINT shrB, MYITE H1, MYITE H2);
void MatMulCC(const posit32_t* A, const posit32_t* B, posit32_t* C, MYITE I, MYITE K, MYITE J, MYINT shrA, MYINT shrB, MYITE H1, MYITE H2);

void SparseMatMulX(const MYINT* Aidx, const posit32_t* Aval, posit32_t** B, posit32_t* C, int16_t K, MYINT shrA, MYINT shrB, MYINT shrC);
void SparseMatMul(const MYINT* Aidx, const posit32_t* Aval, posit32_t* B, posit32_t* C, int16_t K, MYINT shrA, MYINT shrB, MYINT shrC);

void MulCir(posit32_t* A, posit32_t* B, posit32_t* C, MYITE I, MYITE J, MYINT shrA, MYINT shrB);

void TanH(posit32_t* A, MYITE I, MYITE J, float scale_in, float scale_out, posit32_t* B);

void ArgMax(posit32_t* A, MYITE I, MYITE J, int* index);

void Transpose(posit32_t* A, posit32_t* B, MYITE I, MYITE J);

void ScalarMul(posit32_t* A, posit32_t* B, posit32_t* C, MYITE I, MYITE J, MYINT shrA, MYINT shrB);

void MBConv(posit32_t* A, const posit32_t* F1, const posit32_t* BN1W, const posit32_t* BN1B, const posit32_t* F2, const posit32_t* BN2W, const posit32_t* BN2B, const posit32_t* F3, const posit32_t* BN3W, const posit32_t* BN3B, posit32_t* C, posit32_t* X, posit32_t* T, posit32_t* U, MYITE N, MYITE H, MYITE W, MYITE Cin, MYITE Ct, MYITE HF, MYITE WF, MYITE Cout, MYITE Hout, MYITE Wout, MYITE HPADL, MYITE HPADR, MYITE WPADL, MYITE WPADR, MYITE HSTR, MYITE WSTR, MYITE D1, MYITE D2, MYITE D3, MYINT SIX_1, MYINT SIX_2, MYINT shr1, MYINT shr2, MYINT shr3, MYINT shr4, MYINT shr5, MYINT shr6, MYINT shr7, MYINT shr8, MYINT shr9, MYINT shl1, MYINT shl2, MYINT shl3, MYINT shl4, MYINT shl5, MYINT shl6, MYINT shl7, MYINT shl8, MYINT shl9, std::string name);

void Conv(posit32_t* A, const posit32_t* B, posit32_t* C, posit32_t* tmp, MYITE N, MYITE H, MYITE W, MYITE CI, MYITE HF, MYITE WF, MYITE CO, MYINT shrA, MYINT shrB, MYITE H1, MYITE H2);

void Convolution(posit32_t* A, const posit32_t* B, posit32_t* C, posit32_t* tmp, MYITE N, MYITE H, MYITE W, MYITE CIN, MYITE HF, MYITE WF, MYITE CINF, MYITE COUTF, MYITE HOUT, MYITE WOUT, MYITE HPADL, MYITE HPADR, MYITE WPADL, MYITE WPADR, MYITE HSTR, MYITE WSTR, MYITE HDL, MYITE WDL, MYITE G, MYINT shrA, MYINT shrB, MYITE H1, MYITE H2);

void AddOrSubCir4D(posit32_t* A, const posit32_t* B, posit32_t* X, MYITE N, MYITE H, MYITE W, MYITE C, MYINT shrA, MYINT shrB, MYINT shrC, bool add);

void AddOrSubCir2D(posit32_t* A, const posit32_t* B, posit32_t* X, MYITE H, MYITE W, MYINT shrA, MYINT shrB, MYINT shrC, bool add);

void Relu4D(posit32_t* A, MYITE N, MYITE H, MYITE W, MYITE C);

void Relu2D(posit32_t* A, MYITE H, MYITE W);

void Relu6(posit32_t* A, posit32_t* B, MYITE N, MYITE H, MYITE W, MYITE C, MYINT six, MYINT div);

void Maxpool(posit32_t* A, posit32_t* B, MYITE N, MYITE H, MYITE W, MYITE C, MYITE FH, MYITE FW, MYITE strideH, MYITE strideW, MYITE HPADL, MYITE HPADR, MYITE WPADL, MYITE WPADR);

void Exp(posit32_t* A, MYITE I, MYITE J, MYINT shrA, MYINT shrB, posit32_t* B);

void Sigmoid(posit32_t* A, MYITE I, MYITE J, float div, float add, float sigmoid_limit, MYINT scale_in, MYINT scale_out, posit32_t* B);

void AdjustScaleShr(posit32_t* A, MYITE I, MYITE J, MYINT scale);
void AdjustScaleShl(posit32_t* A, MYITE I, MYITE J, MYINT scale);

void Reverse2(posit32_t* A, MYITE axis, MYITE I, MYITE J, posit32_t* B);

void NormaliseL2(posit32_t* A, posit32_t* B, MYITE N, MYITE H, MYITE W, MYITE C, MYINT scaleA, MYINT shrA);
posit32_t operator-(const posit32_t& a);
bool operator>(const posit32_t& a, const int& b);
void AddInplace(posit32_t* A, posit32_t* B, MYITE I, MYITE J);



void MatAddNN(posit_2_t* A, posit_2_t* B, posit_2_t* C, MYITE I, MYITE J, MYINT shrA, MYINT shrB, MYINT shrC, int bitwidth);
void MatAddCN(const posit_2_t* A, posit_2_t* B, posit_2_t* C, MYITE I, MYITE J, MYINT shrA, MYINT shrB, MYINT shrC, int bitwidth);
void MatAddNC(posit_2_t* A, const posit_2_t* B, posit_2_t* C, MYITE I, MYITE J, MYINT shrA, MYINT shrB, MYINT shrC, int bitwidth);
void MatAddCC(const posit_2_t* A, const posit_2_t* B, posit_2_t* C, MYITE I, MYITE J, MYINT shrA, MYINT shrB, MYINT shrC, int bitwidth);

void MatAddBroadCastA(posit_2_t* A, posit_2_t* B, posit_2_t* C, MYITE I, MYITE J, MYINT shrA, MYINT shrB, MYINT shrC, int bitwidth);
void MatAddBroadCastB(posit_2_t* A, posit_2_t* B, posit_2_t* C, MYITE I, MYITE J, MYINT shrA, MYINT shrB, MYINT shrC, int bitwidth);

void MatAdd4(posit_2_t* A, posit_2_t* B, posit_2_t* X, MYITE N, MYITE H, MYITE W, MYITE C, MYINT shrA, MYINT shrB, MYINT shrC, int bitwidth);

void MatSub(posit_2_t* A, const posit_2_t* B, posit_2_t* C, MYITE I, MYITE J, MYINT shrA, MYINT shrB, MYINT shrC, int bitwidth);
void MatSubBroadCastA(posit_2_t* A, posit_2_t* B, posit_2_t* C, MYITE I, MYITE J, MYINT shrA, MYINT shrB, MYINT shrC, int bitwidth);
void MatSubBroadCastB(posit_2_t* A, posit_2_t* B, posit_2_t* C, MYITE I, MYITE J, MYINT shrA, MYINT shrB, MYINT shrC, int bitwidth);

void MatMulNN(posit_2_t* A, posit_2_t* B, posit_2_t* C, MYITE I, MYITE K, MYITE J, MYINT shrA, MYINT shrB, MYITE H1, MYITE H2, int bitwidth);
void MatMulCN(const posit_2_t* A, posit_2_t* B, posit_2_t* C, MYITE I, MYITE K, MYITE J, MYINT shrA, MYINT shrB, MYITE H1, MYITE H2, int bitwidth);
void MatMulNC(posit_2_t* A, const posit_2_t* B, posit_2_t* C, MYITE I, MYITE K, MYITE J, MYINT shrA, MYINT shrB, MYITE H1, MYITE H2, int bitwidth);
void MatMulCC(const posit_2_t* A, const posit_2_t* B, posit_2_t* C, MYITE I, MYITE K, MYITE J, MYINT shrA, MYINT shrB, MYITE H1, MYITE H2, int bitwidth);

void SparseMatMulX(const MYINT* Aidx, const posit_2_t* Aval, posit_2_t** B, posit_2_t* C, int16_t K, MYINT shrA, MYINT shrB, MYINT shrC, int bitwidth);
void SparseMatMul(const MYINT* Aidx, const posit_2_t* Aval, posit_2_t* B, posit_2_t* C, int16_t K, MYINT shrA, MYINT shrB, MYINT shrC, int bitwidth);

void MulCir(posit_2_t* A, posit_2_t* B, posit_2_t* C, MYITE I, MYITE J, MYINT shrA, MYINT shrB, int bitwidth);

void TanH(posit_2_t* A, MYITE I, MYITE J, float scale_in, float scale_out, posit_2_t* B, int bitwidth);

void ArgMax(posit_2_t* A, MYITE I, MYITE J, int* index, int bitwidth);

void Transpose(posit_2_t* A, posit_2_t* B, MYITE I, MYITE J, int bitwidth);

void ScalarMul(posit_2_t* A, posit_2_t* B, posit_2_t* C, MYITE I, MYITE J, MYINT shrA, MYINT shrB, int bitwidth);

void MBConv(posit_2_t* A, const posit_2_t* F1, const posit_2_t* BN1W, const posit_2_t* BN1B, const posit_2_t* F2, const posit_2_t* BN2W, const posit_2_t* BN2B, const posit_2_t* F3, const posit_2_t* BN3W, const posit_2_t* BN3B, posit_2_t* C, posit_2_t* X, posit_2_t* T, posit_2_t* U, MYITE N, MYITE H, MYITE W, MYITE Cin, MYITE Ct, MYITE HF, MYITE WF, MYITE Cout, MYITE Hout, MYITE Wout, MYITE HPADL, MYITE HPADR, MYITE WPADL, MYITE WPADR, MYITE HSTR, MYITE WSTR, MYITE D1, MYITE D2, MYITE D3, MYINT SIX_1, MYINT SIX_2, MYINT shr1, MYINT shr2, MYINT shr3, MYINT shr4, MYINT shr5, MYINT shr6, MYINT shr7, MYINT shr8, MYINT shr9, MYINT shl1, MYINT shl2, MYINT shl3, MYINT shl4, MYINT shl5, MYINT shl6, MYINT shl7, MYINT shl8, MYINT shl9, std::string name, int bitwidth);

void Conv(posit_2_t* A, const posit_2_t* B, posit_2_t* C, posit_2_t* tmp, MYITE N, MYITE H, MYITE W, MYITE CI, MYITE HF, MYITE WF, MYITE CO, MYINT shrA, MYINT shrB, MYITE H1, MYITE H2, int bitwidth);

void Convolution(posit_2_t* A, const posit_2_t* B, posit_2_t* C, posit_2_t* tmp, MYITE N, MYITE H, MYITE W, MYITE CIN, MYITE HF, MYITE WF, MYITE CINF, MYITE COUTF, MYITE HOUT, MYITE WOUT, MYITE HPADL, MYITE HPADR, MYITE WPADL, MYITE WPADR, MYITE HSTR, MYITE WSTR, MYITE HDL, MYITE WDL, MYITE G, MYINT shrA, MYINT shrB, MYITE H1, MYITE H2, int bitwidth);

void AddOrSubCir4D(posit_2_t* A, const posit_2_t* B, posit_2_t* X, MYITE N, MYITE H, MYITE W, MYITE C, MYINT shrA, MYINT shrB, MYINT shrC, bool add, int bitwidth);

void AddOrSubCir2D(posit_2_t* A, const posit_2_t* B, posit_2_t* X, MYITE H, MYITE W, MYINT shrA, MYINT shrB, MYINT shrC, bool add, int bitwidth);

void Relu4D(posit_2_t* A, MYITE N, MYITE H, MYITE W, MYITE C, int bitwidth);

void Relu2D(posit_2_t* A, MYITE H, MYITE W, int bitwidth);

void Relu6(posit_2_t* A, posit_2_t* B, MYITE N, MYITE H, MYITE W, MYITE C, MYINT six, MYINT div, int bitwidth);

void Maxpool(posit_2_t* A, posit_2_t* B, MYITE N, MYITE H, MYITE W, MYITE C, MYITE FH, MYITE FW, MYITE strideH, MYITE strideW, MYITE HPADL, MYITE HPADR, MYITE WPADL, MYITE WPADR, int bitwidth);

void Exp(posit_2_t* A, MYITE I, MYITE J, MYINT shrA, MYINT shrB, posit_2_t* B, int bitwidth);

void Sigmoid(posit_2_t* A, MYITE I, MYITE J, float div, float add, float sigmoid_limit, MYINT scale_in, MYINT scale_out, posit_2_t* B, int bitwidth);

void AdjustScaleShr(posit_2_t* A, MYITE I, MYITE J, MYINT scale, int bitwidth);
void AdjustScaleShl(posit_2_t* A, MYITE I, MYITE J, MYINT scale, int bitwidth);

void Reverse2(posit_2_t* A, MYITE axis, MYITE I, MYITE J, posit_2_t* B, int bitwidth);

void NormaliseL2(posit_2_t* A, posit_2_t* B, MYITE N, MYITE H, MYITE W, MYITE C, MYINT scaleA, MYINT shrA, int bitwidth);
void AddInplace(posit_2_t* A, posit_2_t* B, MYITE I, MYITE J, int bw);

void convertPosit(posit8_t* a, posit8_t *b, int bwA, int bwB);
void convertPosit(posit8_t* a, posit16_t *b, int bwA, int bwB);
void convertPosit(posit8_t* a, posit32_t *b, int bwA, int bwB);
void convertPosit(posit16_t* a, posit8_t *b, int bwA, int bwB);
void convertPosit(posit16_t* a, posit16_t *b, int bwA, int bwB);
void convertPosit(posit16_t* a, posit32_t *b, int bwA, int bwB);
void convertPosit(posit32_t* a, posit8_t *b, int bwA, int bwB);
void convertPosit(posit32_t* a, posit16_t *b, int bwA, int bwB);
void convertPosit(posit32_t* a, posit32_t *b, int bwA, int bwB);
void convertPosit(posit_1_t* a, posit8_t *b, int bwA, int bwB);
void convertPosit(posit_1_t* a, posit16_t *b, int bwA, int bwB);
void convertPosit(posit_1_t* a, posit32_t *b, int bwA, int bwB);
void convertPosit(posit_2_t* a, posit8_t *b, int bwA, int bwB);
void convertPosit(posit_2_t* a, posit16_t *b, int bwA, int bwB);
void convertPosit(posit_2_t* a, posit32_t *b, int bwA, int bwB);
void convertPosit(posit8_t* a, posit_2_t *b, int bwA, int bwB);
void convertPosit(posit16_t* a, posit_2_t *b, int bwA, int bwB);
void convertPosit(posit32_t* a, posit_2_t *b, int bwA, int bwB);
void convertPosit(posit8_t* a, posit_1_t *b, int bwA, int bwB);
void convertPosit(posit16_t* a, posit_1_t *b, int bwA, int bwB);
void convertPosit(posit32_t* a, posit_1_t *b, int bwA, int bwB);
void convertPosit(posit_1_t* a, posit_1_t *b, int bwA, int bwB);
void convertPosit(posit_1_t* a, posit_2_t *b, int bwA, int bwB);
void convertPosit(posit_2_t* a, posit_1_t *b, int bwA, int bwB);
void convertPosit(posit_2_t* a, posit_2_t *b, int bwA, int bwB);

posit8_t positAdd(posit8_t a, posit8_t b, int bw);
posit16_t positAdd(posit16_t a, posit16_t b, int bw);
posit32_t positAdd(posit32_t a, posit32_t b, int bw);

posit8_t positSub(posit8_t a, posit8_t b, int bw);
posit16_t positSub(posit16_t a, posit16_t b, int bw);
posit32_t positSub(posit32_t a, posit32_t b, int bw);

posit8_t positMul(posit8_t a, posit8_t b, int bw);
posit16_t positMul(posit16_t a, posit16_t b, int bw);
posit32_t positMul(posit32_t a, posit32_t b, int bw);

posit8_t positDiv(posit8_t a, posit8_t b, int bw);
posit16_t positDiv(posit16_t a, posit16_t b, int bw);
posit32_t positDiv(posit32_t a, posit32_t b, int bw);

posit8_t positSqrt(posit8_t a, int bw);
posit16_t positSqrt(posit16_t a, int bw);
posit32_t positSqrt(posit32_t a, int bw);

bool positLT(posit8_t a, posit8_t b, int bw);
bool positLT(posit16_t a, posit16_t b, int bw);
bool positLT(posit32_t a, posit32_t b, int bw);

bool positEQ(posit8_t a, posit8_t b, int bw);
bool positEQ(posit16_t a, posit16_t b, int bw);
bool positEQ(posit32_t a, posit32_t b, int bw);

double convertPositToDouble(posit8_t a, int bw);
double convertPositToDouble(posit16_t a, int bw);
double convertPositToDouble(posit32_t a, int bw);

void convertDoubleToPosit(double a, posit8_t *b, int bw);
void convertDoubleToPosit(double a, posit16_t *b, int bw);
void convertDoubleToPosit(double a, posit32_t *b, int bw);

posit_1_t positAdd(posit_1_t a, posit_1_t b, int bw);
posit_2_t positAdd(posit_2_t a, posit_2_t b, int bw);
posit_1_t positSub(posit_1_t a, posit_1_t b, int bw);
posit_2_t positSub(posit_2_t a, posit_2_t b, int bw);
posit_1_t positMul(posit_1_t a, posit_1_t b, int bw);
posit_2_t positMul(posit_2_t a, posit_2_t b, int bw);
posit_1_t positDiv(posit_1_t a, posit_1_t b, int bw);
posit_2_t positDiv(posit_2_t a, posit_2_t b, int bw);
posit_1_t positSqrt(posit_1_t a, int bw);
posit_2_t positSqrt(posit_2_t a, int bw);
posit_1_t convertQuireToPosit(quire_1_t q, int bw);
posit_2_t convertQuireToPosit(quire_2_t q, int bw);
quire_1_t positFMA(quire_1_t q, posit_1_t a, posit_1_t b, int bw);
quire_2_t positFMA(quire_2_t q, posit_2_t a, posit_2_t b, int bw);
bool positLT(posit_1_t a, posit_1_t b, int bw);
bool positLT(posit_2_t a, posit_2_t b, int bw);
bool positEQ(posit_1_t a, posit_1_t b, int bw);
bool positEQ(posit_2_t a, posit_2_t b, int bw);
quire_1_t clearQuire(quire_1_t q, int bw);
quire_2_t clearQuire(quire_2_t q, int bw);

double convertPositToDouble(posit_1_t a, int bw);
double convertPositToDouble(posit_2_t a, int bw);
void convertDoubleToPosit(double a, posit_1_t *b, int bw);
void convertDoubleToPosit(double a, posit_2_t *b, int bw);

quire8_t clearQuire(quire8_t q, int bw);
quire16_t clearQuire(quire16_t q, int bw);
quire32_t clearQuire(quire32_t q, int bw);

posit8_t convertQuireToPosit(quire8_t q, int bw);
posit16_t convertQuireToPosit(quire16_t q, int bw);
posit32_t convertQuireToPosit(quire32_t q, int bw);

quire8_t positFMA(quire8_t q, posit8_t a, posit8_t b, int bw);
quire16_t positFMA(quire16_t q, posit16_t a, posit16_t b, int bw);
quire32_t positFMA(quire32_t q, posit32_t a, posit32_t b, int bw);

void debugPrint(posit8_t* A, int I, int J, std::string varName, int bw=8);
void debugPrint(posit16_t* A, int I, int J, std::string varName, int bw=16);
void debugPrint(posit32_t* A, int I, int J, std::string varName, int bw=32);
void debugPrint(posit_2_t* A, int I, int J, std::string varName, int bw=16);

void debugPrint(posit8_t* A, int I, int J, int K, int L, std::string varName, int bw=8);
void debugPrint(posit16_t* A, int I, int J, int K, int L, std::string varName, int bw=16);
void debugPrint(posit32_t* A, int I, int J, int K, int L, std::string varName, int bw=32);
void debugPrint(posit_2_t* A, int I, int J, int K, int L, std::string varName, int bw=16);

void debugPrint(std::string str);

template<class TypeA, class TypeB, class TypeTemp, class TypeC>
void MatAdd(TypeA* A, TypeB* B, TypeC* C, MYITE I, MYITE J, int bwA, int bwB, int bwTemp, int bwC) {
	for (MYITE i = 0; i < I; i++) {
		for (MYITE j = 0; j < J; j++) {
			TypeTemp a;
            convertPosit(&A[i * J + j], &a, bwA, bwTemp);
			TypeTemp b;
            convertPosit(&B[i * J + j], &b, bwB, bwTemp);

			TypeTemp c = positAdd(a, b, bwTemp);

			convertPosit(&c, &C[i * J + j], bwTemp, bwC);
		}
	}
	return;
}

template<class TypeA, class TypeB, class TypeTemp, class TypeC>
void MatAddBroadCastA(TypeA* A, TypeB* B, TypeC* C, MYITE I, MYITE J, int bwA, int bwB, int bwTemp, int bwC) {
    TypeTemp a;
    convertPosit(A, &a, bwA, bwTemp);
	for (MYITE i = 0; i < I; i++) {
		for (MYITE j = 0; j < J; j++) {
			TypeTemp b;
            convertPosit(&B[i * J + j], &b, bwB, bwTemp);
			
            TypeTemp c = positAdd(a, b, bwTemp);

			convertPosit(&c, &C[i * J + j], bwTemp, bwC);

		}
	}
	return;
}

template<class TypeA, class TypeB, class TypeTemp, class TypeC>
void MatAddBroadCastB(TypeA* A, TypeB* B, TypeC* C, MYITE I, MYITE J, int bwA, int bwB, int bwTemp, int bwC) {
    TypeTemp b;
    convertPosit(B, &b, bwB, bwTemp);
	for (MYITE i = 0; i < I; i++) {
		for (MYITE j = 0; j < J; j++) {
			TypeTemp a;
            convertPosit(&A[i * J + j], &a, bwA, bwTemp);

			TypeTemp c = positAdd(a, b, bwTemp);

			convertPosit(&c, &C[i * J + j], bwTemp, bwC);
		}
	}
	return;
}

template<class TypeA, class TypeB, class TypeTemp, class TypeC>
// TODO: shrB is int32_t because in 8-bit/16-bit code, shrB is usually very high and int8_t/int16_t will overflow.
void MatSub(TypeA* A, TypeB* B, TypeC* C, MYITE I, MYITE J, int bwA, int bwB, int bwTemp, int bwC) {
	for (MYITE i = 0; i < I; i++) {
		for (MYITE j = 0; j < J; j++) {
			TypeTemp a;
            convertPosit(&A[i * J + j], &a, bwA, bwTemp);
			TypeTemp b;
            convertPosit(&B[i * J + j], &b, bwB, bwTemp);

			TypeTemp c = positSub(a, b, bwTemp);

			convertPosit(&c, &C[i * J + j], bwTemp, bwC);
		}
	}
	return;
}

template<class TypeA, class TypeB, class TypeTemp, class TypeC>
void MatSubBroadCastA(TypeA* A, TypeB* B, TypeC* C, MYITE I, MYITE J, int bwA, int bwB, int bwTemp, int bwC) {
    TypeTemp a;
    convertPosit(A, &a, bwA, bwTemp);
	for (MYITE i = 0; i < I; i++) {
		for (MYITE j = 0; j < J; j++) {
			TypeTemp b;
            convertPosit(&B[i * J + j], &b, bwB, bwTemp);

			TypeTemp c = positSub(a, b, bwTemp);

			convertPosit(&c, &C[i * J + j], bwTemp, bwC);
		}
	}
	return;
}

template<class TypeA, class TypeB, class TypeTemp, class TypeC>
void MatSubBroadCastB(TypeA* A, TypeB* B, TypeC* C, MYITE I, MYITE J, int bwA, int bwB, int bwTemp, int bwC) {
    TypeTemp b;
    convertPosit(B, &b, bwB, bwTemp);
	for (MYITE i = 0; i < I; i++) {
		for (MYITE j = 0; j < J; j++) {
			TypeTemp a;
            convertPosit(&A[i * J + j], &a, bwA, bwTemp);

			TypeTemp c = positSub(a, b, bwTemp);

			convertPosit(&c, &C[i * J + j], bwTemp, bwC);
		}
	}
	return;
}


template<class TypeA, class TypeB, class TypeTemp, class QuireType, class TypeC>
void MatMul(TypeA* A, TypeB* B, TypeC* C, MYITE I, MYITE K, MYITE J, int bwA, int bwB, int bwTemp, int bwC) {
	QuireType q;
    int positSize = 8 * sizeof(TypeTemp);
    for (MYITE i = 0; i < I; i++) {
		for (MYITE j = 0; j < J; j++) {
            q = clearQuire(q, bwTemp);
			for (MYITE k = 0; k < K; k++) {
				TypeTemp a;
                convertPosit(&A[i * K + k], &a, bwA, bwTemp);
                TypeTemp b;
                convertPosit(&B[k * J + j], &b, bwB, bwTemp);

				q = positFMA(q, a, b, bwTemp);

			}
            TypeTemp c_temp = convertQuireToPosit(q, bwTemp);
			convertPosit(&c_temp, &C[i * J + j], bwTemp, bwC);
		}
	}
	return;
}

template<class TypeA, class TypeB, class TypeTemp, class TypeC>
void MulCir(TypeA* A, TypeB* B, TypeC* C, MYITE I, MYITE J, int bwA, int bwB, int bwTemp, int bwC) {
	for (MYITE i = 0; i < I; i++) {
		for (MYITE j = 0; j < J; j++) {
            TypeTemp a;
            convertPosit(&A[i * J + j], &a, bwA, bwTemp);
			TypeTemp b;
            convertPosit(&B[i * J + j], &b, bwB, bwTemp);
			
            TypeTemp prod = positMul(a, b, bwTemp);
			convertPosit(&prod, &C[i * J + j], bwTemp, bwC);
		}
	}
	return;
}

template<class TypeA>
void ArgMax(TypeA* A, MYITE I, MYITE J, int* index, int bwA) {
	double max = convertPositToDouble(A[0], bwA);
	MYITE maxIndex = 0, counter = 0;
	for (MYITE i = 0; i < I; i++) {
		for (MYITE j = 0; j < J; j++) {
			double x = convertPositToDouble(A[i * J + j], bwA);

			if (max < x) {
				maxIndex = counter;
				max = x;
			}

			counter++;
		}
	}

	*index = maxIndex;
	return;
}

template<class TypeA>
void Transpose(TypeA* A, TypeA* B, MYITE I, MYITE J) {
	for (MYITE i = 0; i < I; i++) {
		for (MYITE j = 0; j < J; j++) {
			B[i * J + j] = A[j * I + i];
		}
	}
	return;
}

template<class TypeA, class TypeB, class TypeTemp, class TypeC>
void ScalarMul(TypeA* A, TypeB* B, TypeC* C, MYITE I, MYITE J, int bwA, int bwB, int bwTemp, int bwC) {
    TypeTemp a;
    convertPosit(A, &a, bwA, bwTemp);
	for (MYITE i = 0; i < I; i++) {
		for (MYITE j = 0; j < J; j++) {
			TypeTemp b;
            convertPosit(&B[i * J + j], &b, bwB, bwTemp);

			TypeTemp prod = positMul(a, b, bwTemp);
			convertPosit(&prod, &C[i * J + j], bwTemp, bwC);
		}
	}
	return;
}


template<class TypeA, class TypeB>
void Sigmoid(TypeA* A,  TypeA* B, MYITE I, MYITE J, int bwA, int bwTemp, int bwB) {

	for (MYITE i = 0; i < I; i++) {
		for (MYITE j = 0; j < J; j++) {
				double x = convertPositToDouble(A[i * J + j], bwA);
				double y = 1 / (1 + exp(-x));
				convertDoubleToPosit(y, &B[i *J + j], bwB);
        }
	}
	return;
}

template<class TypeA, class TypeB>
void TanH(TypeA* A, TypeB* B, MYITE I, MYITE J, int bwA, int bwTemp, int bwB) {
	for (MYITE i = 0; i < I; i++) {
		for (MYITE j = 0; j < J; j++) {
            double x = convertPositToDouble(A[i * J + j], bwA);         
			double y = tanh(x);
			convertDoubleToPosit(y, &B[i *J + j], bwB);
		}
	}
	return;
}

template<typename TypeA, typename TypeTemp, typename TypeB>
void Exp(TypeA* A, TypeB *B, MYITE I, MYITE J, int bwA, int bwTemp, int bwB) {
	for (MYITE i = 0; i < I; i++) {
		for (MYITE j = 0; j < J; j++) {
			TypeA x = A[i * J + j];
			convertDoubleToPosit(exp(convertPositToDouble(x, bwA)), &B[i * J + j], bwB);
		}
	}
	return;
}

template<class TypeA, class TypeAidx, class TypeB, class TypeTemp, class QuireType, class TypeC>
void SparseMatMulX(const TypeAidx* Aidx, TypeA* Aval, TypeB** B, TypeC* C, int16_t K, int16_t P, int bwA, int bwB, int bwTemp, int bwC) {
	MYITE ite_idx = 0, ite_val = 0;
	QuireType *tmp = new QuireType[P];
	for(int p=0;p<P;p++)
	{
		tmp[p] = clearQuire(tmp[p], bwTemp);
	}

	for (MYITE k = 0; k < K; k++) {
		TypeTemp b;
		convertPosit(&B[k * 1][0], &b, bwB, bwTemp);

		TypeAidx idx = Aidx[ite_idx];
		while (idx != 0) {
			TypeTemp a;
			convertPosit((&Aval[ite_val]), &a, bwA, bwTemp);

			tmp[idx - 1] = positFMA(tmp[idx - 1], a, b, bwTemp);
			
			ite_idx++;
			ite_val++;

			idx = Aidx[ite_idx];
		}
		ite_idx++;
	}

	for(int p=0;p<P;p++)
	{
		TypeTemp tmp_p = convertQuireToPosit(tmp[p], bwTemp);
		convertPosit(&tmp_p, &C[p], bwTemp, bwC);
	}

	return;
}

// C = A |*| B
template<class TypeA, class TypeAidx, class TypeB, class TypeTemp, class QuireType, class TypeC>
void SparseMatMul(const TypeAidx* Aidx, TypeA* Aval, TypeB* B, TypeC* C, int16_t K, int16_t P, int bwA, int bwB, int bwTemp, int bwC) {
	MYITE ite_idx = 0, ite_val = 0;
	QuireType *tmp = new QuireType[P];
	for(int p=0;p<P;p++)
	{
		tmp[p] = clearQuire(tmp[p], bwTemp);
	}

	for (MYITE k = 0; k < K; k++) {
		TypeTemp b;
		convertPosit(&B[k], &b, bwB, bwTemp);

		TypeAidx idx = Aidx[ite_idx];
		while (idx != 0) {
			TypeTemp a;
			convertPosit((&Aval[ite_val]), &a, bwA, bwTemp);

			tmp[idx - 1] = positFMA(tmp[idx - 1], a, b, bwTemp);
			
			ite_idx++;
			ite_val++;

			idx = Aidx[ite_idx];
		}
		ite_idx++;
	}

	for(int p=0;p<P;p++)
	{
		TypeTemp tmp_p = convertQuireToPosit(tmp[p], bwTemp);
		convertPosit(&tmp_p, &C[p], bwTemp, bwC);
	}

	return;
}

template<typename TypeA, typename TypeB, typename TypeTemp>
void AddInplace(TypeA* A, TypeB* B, MYITE I, MYITE J, int bwA, int bwTemp, int bwB) {
	for (MYITE i = 0; i < I; i++) {
		for (MYITE j = 0; j < J; j++) {
			TypeTemp a, b;
			convertPosit(&A[i * J + j], &a, bwA, bwTemp);
			convertPosit(&B[i * J + j], &b, bwB, bwTemp);

			TypeTemp c = positAdd(a, b, bwTemp);

			convertPosit(&c, &A[i * J + j], bwTemp, bwA);
		}
	}
	return;
}

// B = reverse(A, axis)
template<typename TypeA>
void Reverse2(TypeA* A, MYITE axis, MYITE I, MYITE J, TypeA* B) {
	for (MYITE i = 0; i < I; i++) {
		for (MYITE j = 0; j < J; j++) {
			MYITE i_prime = (axis == 0 ? (I - 1 - i) : i);
			MYITE j_prime = (axis == 1 ? (J - 1 - j) : j);

			B[i * J + j] = A[i_prime*J + j_prime];
		}
	}
	return;
}

template<typename TypeA, typename TypeB>
void ConvertPosit(TypeA* A, TypeB* B, MYITE I, MYITE J)
{
	for(int i=0;i<I;i++)
	{
		for(int j=0;j<J;j++)
		{
			convertPosit(&A[i*J + j], &B[i*J + j]);
		}
	}
}

template<class TypeA, class TypeB, class TypeTemp, class QuireType, class TypeC>
void Convolution(TypeA* A, TypeB* B, TypeC* C, MYITE N, MYITE H, MYITE W, MYITE CIN, MYITE HF, MYITE WF, MYITE CINF, MYITE COUTF, MYITE HOUT, MYITE WOUT, MYITE HPADL, MYITE HPADR, MYITE WPADL, MYITE WPADR, MYITE HSTR, MYITE WSTR, MYITE HDL, MYITE WDL, MYITE G, int bwA, int bwB, int bwTemp, int bwC) {
	MYITE HOffsetL = HDL*(HF/2) - HPADL;
	MYITE WOffsetL = WDL*(WF/2) - WPADL;
	MYITE HOffsetR = HDL*(HF/2) - HPADR;
	MYITE WOffsetR = WDL*(WF/2) - WPADR;

	TypeTemp zero;
	convertDoubleToPosit(0.0, &zero, bwTemp);
	QuireType quire;
	for (MYITE n = 0; n < N; n++) {
		for (MYITE h = HOffsetL, hout = 0; h < H - HOffsetR; h += HSTR, hout++) {
			for (MYITE w = WOffsetL, wout = 0; w < W - WOffsetR; w += WSTR, wout++) {
				for (MYITE g = 0; g < G; g++) {
					for (MYITE co = 0; co < COUTF; co++) {
						quire = clearQuire(quire, bwTemp);
						MYITE counter = 0;
						for (MYITE hf = -(HF / 2); hf <= HF / 2; hf++) {
							for (MYITE wf = -(WF / 2); wf <= WF / 2; wf++) {
								for (MYITE ci = 0; ci < CINF; ci++) {
									TypeTemp a, b;
									if(((h + HDL * hf) < 0) || ((h + HDL * hf) >= H) || ((w + WDL * wf) < 0) || ((w + WDL * wf) >= W))
									{
										a = zero;
									}
									else{
										convertPosit(&A[n * H * W * CIN + (h + HDL * hf) * W * CIN + (w + WDL * wf) * CIN + (ci + g * CINF)], &a, bwA, bwTemp);
									}

									convertPosit((&B[g * HF * WF * CINF * COUTF + (hf + HF / 2) * WF * CINF * COUTF + (wf + WF / 2) * CINF * COUTF + ci * COUTF + co]), &b, bwB, bwTemp);

									quire = positFMA(quire, a, b, bwTemp);
									counter++;
								}
							}
						}	
						TypeTemp c_tmp = convertQuireToPosit(quire, bwTemp);
						convertPosit(&c_tmp, &C[n * HOUT * WOUT * (COUTF * G) + hout * WOUT * (COUTF * G) + wout * (COUTF * G) + (co + g * COUTF)], bwTemp, bwC);
					}
				}
			}
		}
	}
}

template<class TypeA, class TypeB, class TypeTemp, class QuireType, class TypeC>
void Conv(TypeA* A, TypeB* B, TypeC* C, MYITE N, MYITE H, MYITE W, MYITE CI, MYITE HF, MYITE WF, MYITE CO, int bwA, int bwB, int bwTemp, int bwC) {
	MYITE padH = (HF - 1) / 2;
	MYITE padW = (WF - 1) / 2;

	TypeTemp zero;
	convertDoubleToPosit(0.0, &zero, bwTemp);
	QuireType quire;

	for (MYITE n = 0; n < N; n++) {
		for (MYITE h = 0; h < H; h++) {
			for (MYITE w = 0; w < W; w++) {
				for (MYITE co = 0; co < CO; co++) {
					quire = clearQuire(quire, bwTemp);
					MYITE counter = 0;
					for (MYITE hf = 0; hf < HF; hf++) {
						for (MYITE wf = 0; wf < WF; wf++) {
							for (MYITE ci = 0; ci < CI; ci++) {
								TypeTemp a,b;
								if((((h + hf) < padH) || ((h + hf) >= (H + padH))) || (((w + wf) < padW) || ((w + wf) >= (W + padW))))
								{
									a = zero;
								}
								else{
									convertPosit(&A[n * H * W * CI + ((h + hf) - padH) * W * CI + ((w + wf) - padW) * CI + ci], &a, bwA, bwTemp);
								}

								convertPosit((&B[hf * WF * CI * CO + wf * CI * CO + ci * CO + co]), &b, bwB, bwTemp);

								quire = positFMA(quire, a, b, bwTemp);
								counter++;
							}
						}
					}

					TypeTemp c_tmp = convertQuireToPosit(quire, bwTemp);
					convertPosit(&c_tmp, &C[n * H * W * CO + h * W * CO + w * CO + co], bwTemp, bwC);
				}
			}
		}
	}
	return;
}

template<class TypeA, class TypeF1, class TypeB1W, class TypeB1B, class TypeF2, class TypeB2W, class TypeB2B, class TypeF3, class TypeB3W, class TypeB3B, class TypeC, class TypeX, class TypeT, class TypeU, class QuireU, class TypeUB1W, class TypeUB2W, class TypeUB3W>
void MBConv(TypeA* A,
    TypeF1* F1,
    TypeB1W* BN1W,
    TypeB1B* BN1B,
    TypeF2* F2,
    TypeB2W* BN2W,
    TypeB2B* BN2B,
    TypeF3* F3,
    TypeB3W* BN3W,
    TypeB3B* BN3B,
    TypeC* C,
    TypeX* X,
    TypeT* T,
    MYITE N,
    MYITE H,
    MYITE W,
    MYITE Cin,
    MYITE Ct,
    MYITE HF,
    MYITE WF,
    MYITE Cout,
    MYITE Hout,
    MYITE Wout,
    MYITE HPADL,
    MYITE HPADR,
    MYITE WPADL,
    MYITE WPADR,
    MYITE HSTR,
    MYITE WSTR,
    MYITE bwA,
    MYITE bwF1,
    MYITE bwB1W,
    MYITE bwB1B,
    MYITE bwF2,
    MYITE bwB2W,
    MYITE bwB2B,
    MYITE bwF3,
    MYITE bwB3W,
    MYITE bwB3B,
    MYITE bwC,
    MYITE bwX,
    MYITE bwT,
    MYITE bwU,
    MYITE bwUB1W,
    MYITE bwUB2W,
    MYITE bwUB3W) 
{
	MYITE HOffsetL = (HF / 2) - HPADL;
	MYITE WOffsetL = (WF / 2) - WPADL;
	MYITE HOffsetR = (HF / 2) - HPADR;
	MYITE WOffsetR = (WF / 2) - WPADR;

	QuireU qU;

	for (MYITE n = 0; n < N; n++) {
		MYITE margin = HOffsetL + (HF / 2 + 1) - HSTR > 0 ? HOffsetL + (HF / 2 + 1) - HSTR : 0;
		MYITE nstart = HOffsetL - (HF / 2) < 0 ? 0 : HOffsetL - (HF / 2);
		for (MYITE i = nstart; i < margin; i++) {
			for (MYITE j = 0; j < W; j++) {
				for (MYITE k = 0; k < Ct; k++) {
					qU = clearQuire(qU, bwU);
					for (MYITE l = 0; l < Cin; l++) {
						TypeU a, f1;
						convertPosit(&A[n * H * W * Cin + i * W * Cin + j * Cin + l], &a, bwA, bwU);
						convertPosit(&F1[l * Ct + k], &f1, bwF1, bwU);
						qU = positFMA(qU, a, f1, bwU);
					}

					TypeU u_tmp = convertQuireToPosit(qU, bwU);
					TypeUB1W u0_tmp, bn1b, bn1w;
					convertPosit(&u_tmp, &u0_tmp, bwU, bwUB1W);
					convertPosit(&BN1B[k], &bn1b, bwB1B, bwUB1W);
					convertPosit(&BN1W[k], &bn1w, bwB1W, bwUB1W);
					TypeUB1W x;
					x = positMul(positAdd(u0_tmp, bn1b, bwUB1W), bn1w, bwUB1W);
					
					TypeUB1W zero, six;
					convertDoubleToPosit(0.0, &zero, bwUB1W);
					convertDoubleToPosit(6.0, &six, bwUB1W);

					x = positLT(x, zero, bwUB1W) ? zero : x;
					x = positLT(six, x, bwUB1W) ? six: x;

					convertPosit(&x, &X[i * W * Ct + j * Ct + k], bwUB1W, bwX);
				}
			}
		}

		for (MYITE h = HOffsetL, hout = 0; h < H - HOffsetR; hout++, h += HSTR) {

			for (MYITE i = 0; i < HSTR; i++) {
				for (MYITE j = 0; j < W; j++) {
					for (MYITE k = 0; k < Ct; k++) {
						MYITE iRed = (i + margin + hout * HSTR) % HF, iFull = i + margin + hout * HSTR;
						TypeX zero_x;
						convertDoubleToPosit(0.0, &zero_x, bwX);
						X[iRed * W * Ct + j * Ct + k] = zero_x;
						qU = clearQuire(qU, bwU);
						for (MYITE l = 0; l < Cin; l++) {
							TypeA a_a;
							convertDoubleToPosit(0.0, &a_a, bwA);
							a_a = iFull < H ? A[n * H * W * Cin + iFull * W * Cin + j * Cin + l] : a_a;
							TypeU a_u, f1_u;
							convertPosit(&a_a, &a_u, bwA, bwU);
							convertPosit(&F1[l * Ct + k], &f1_u, bwF1, bwU);
							qU = positFMA(qU, a_u, f1_u, bwU);
						}
						
						TypeU u_tmp = convertQuireToPosit(qU, bwU);
						TypeUB1W u0_tmp, bn1b, bn1w;
						convertPosit(&u_tmp, &u0_tmp, bwU, bwUB1W);
						convertPosit(&BN1B[k], &bn1b, bwB1B, bwUB1W);
						convertPosit(&BN1W[k], &bn1w, bwB1W, bwUB1W);
						TypeUB1W x;
						x = positMul(positAdd(u0_tmp, bn1b, bwUB1W), bn1w, bwUB1W);
						
						TypeUB1W zero, six;
						convertDoubleToPosit(0.0, &zero, bwUB1W);
						convertDoubleToPosit(6.0, &six, bwUB1W);

						x = positLT(x, zero, bwUB1W) ? zero : x;
						x = positLT(six, x, bwUB1W) ? six: x;

						convertPosit(&x, &X[iRed * W * Ct + j * Ct + k], bwUB1W, bwX);
					}
				}
			}

			for (MYITE w = WOffsetL, wout = 0; w < W - WOffsetR; w += WSTR, wout++) {
				for (MYITE g = 0; g < Ct; g++) {
					MYITE counter = 0;
					qU = clearQuire(qU, bwU);
					for (MYITE hf = -(HF / 2); hf <= (HF / 2); hf++) {
						for (MYITE wf = -(WF / 2); wf <= (WF / 2); wf++) {
							TypeX zero_x;
							convertDoubleToPosit(0.0, &zero_x, bwX);
							TypeX x_x = (((h + hf) < 0) || ((h + hf) >= H) || ((w + wf) < 0) || ((w + wf) >= W)) ? zero_x : X[((h + hf) % HF) * W * Ct + (w + wf) * Ct + g];
							TypeF2 b_f2 = F2[g * HF * WF + (hf + HF / 2) * WF + (wf + WF / 2)];
							TypeU x_u, b_u;
							convertPosit(&x_x, &x_u, bwX, bwU);
							convertPosit(&b_f2, &b_u, bwF2, bwU);
							qU = positFMA(qU, x_u, b_u, bwU);
							counter++;
						}
					}
					TypeU u_u = convertQuireToPosit(qU, bwU);
					TypeUB2W u_ub2w, bn2b_ub2w, bn2w_ub2w;
					convertPosit(&u_u, &u_ub2w, bwU, bwUB2W);
					convertPosit(&BN2B[g], &bn2b_ub2w, bwB2B, bwUB2W);
					convertPosit(&BN2W[g], &bn2w_ub2w, bwB2W, bwUB2W);

					TypeUB2W x_ub2w = positMul(positAdd(u_ub2w, bn2b_ub2w, bwUB2W), bn2w_ub2w, bwUB2W), zero_ub2w, six_ub2w;
					convertDoubleToPosit(0.0, &zero_ub2w, bwUB2W);
					convertDoubleToPosit(6.0, &six_ub2w, bwUB2W);
					x_ub2w = positLT(x_ub2w, zero_ub2w, bwUB2W) ? zero_ub2w : x_ub2w;
					x_ub2w = positLT(six_ub2w, x_ub2w, bwUB2W) ? six_ub2w: x_ub2w;
					
					convertPosit(&x_ub2w, &T[g], bwUB2W, bwT);
				}

				for (MYITE i = 0; i < Cout; i++) {
					qU = clearQuire(qU, bwU);
					for (MYITE g = 0; g < Ct; g++) {
						TypeU t_u, f3_u;
						convertPosit(&T[g], &t_u, bwT, bwU);
						convertPosit(&F3[g*Cout + i], &f3_u, bwF3, bwU);
						qU = positFMA(qU, t_u, f3_u, bwU);
					}

					TypeU u_u = convertQuireToPosit(qU, bwU);
					TypeUB3W u_ub3w, bn3b_ub3w, bn3w_ub3w;
					convertPosit(&u_u, &u_ub3w, bwU, bwUB3W);
					convertPosit(&BN3B[i], &bn3b_ub3w, bwB3B, bwUB3W);
					convertPosit(&BN3W[i], &bn3w_ub3w, bwB3W, bwUB3W);

					TypeUB3W c_ub3w = positMul(positAdd(u_ub3w, bn3b_ub3w, bwUB3W), bn3w_ub3w, bwUB3W);
					convertPosit(&c_ub3w, &C[n * Hout * Wout * Cout + hout * Wout * Cout + wout * Cout + i], bwUB3W, bwC);
				}
			}
		}
	}
}

// A = A <+> B
// A[N][H][W][C], B[C]
template<class TypeA, class TypeB, class TypeTemp, class TypeC>
void AddOrSubCir4D(TypeA* A, TypeB* B, TypeC* X, MYITE N, MYITE H, MYITE W, MYITE C, bool add, int bwA, int bwB, int bwTemp, int bwC) {
	for (MYITE n = 0; n < N; n++) {
		for (MYITE h = 0; h < H; h++) {
			for (MYITE w = 0; w < W; w++) {
				for (MYITE c = 0; c < C; c++) {
					TypeTemp a, b;
					convertPosit(&A[n * H * W * C + h * W * C + w * C + c], &a, bwA, bwTemp);
					convertPosit(&B[c], bwB, bwTemp);

					TypeTemp res;
					if (add) {
						res = positAdd(a, b, bwTemp);
					} else {
						res = positSub(a, b, bwTemp);
					}

					convertPosit(&res, &X[n * H * W * C + h * W * C + w * C + c], bwTemp, bwC);
				}
			}
		}
	}
	return;
}

template<class TypeA, class TypeB, class TypeTemp, class TypeC>
void AddOrSubCir2D(TypeA* A, TypeB* B, TypeC* X, MYITE H, MYITE W, bool add, int bwA, int bwB, int bwTemp, int bwC) {
	for (MYITE h = 0; h < H; h++) {
		for (MYITE w = 0; w < W; w++) {
			TypeTemp a, b;
			convertPosit(&A[h * W + w], &a, bwA, bwTemp);
			convertPosit(&B[w], &b, bwB, bwTemp);

			TypeTemp res;
			if (add) {
				res = positAdd(a, b, bwTemp);
			} else {
				res = positSub(a, b, bwTemp);
			}

			convertPosit(&res, &X[h * W + w], bwTemp, bwC);
		}
	}
	return;
}

template<class TypeA>
void Relu4D(TypeA* A, MYITE N, MYITE H, MYITE W, MYITE C, int bwA) {
	for (MYITE n = 0; n < N; n++) {
		for (MYITE h = 0; h < H; h++) {
			for (MYITE w = 0; w < W; w++) {
				for (MYITE c = 0; c < C; c++) {
					TypeA a = A[n * H * W * C + h * W * C + w * C + c];
					TypeA zero;
					convertDoubleToPosit(0.0, &zero, bwA);

					if (positLT(a, zero, bwA)) {
						a = zero;
					}

					A[n * H * W * C + h * W * C + w * C + c] = a;
				}
			}
		}
	}
	return;
}

template<class TypeA, class TypeB>
void Relu6(TypeA* A, TypeB* B, MYITE N, MYITE H, MYITE W, MYITE C, int bwA, int bwB) {
	TypeA zero, six_a;
	convertDoubleToPosit(6.0, &six_a, bwA);
	convertDoubleToPosit(0.0, &zero, bwA);
	for (MYITE n = 0; n < N; n++) {
		for (MYITE h = 0; h < H; h++) {
			for (MYITE w = 0; w < W; w++) {
				for (MYITE c = 0; c < C; c++) {
					TypeA a = A[n * H * W * C + h * W * C + w * C + c];
					if (positLT(a, zero, bwA)) {
						a = zero;
					}
					if (positLT(six_a, a, bwA)) {
						a = six_a;
					}
					convertPosit(&a, &B[n * H * W * C + h * W * C + w * C + c], bwA, bwB);
				}
			}
		}
	}
	return;
}

template<class TypeA>
void Relu2D(TypeA* A, MYITE H, MYITE W, int bwA) {
	for (MYITE h = 0; h < H; h++) {
		for (MYITE w = 0; w < W; w++) {
			TypeA a = A[h * W + w];
			TypeA zero;
			convertDoubleToPosit(0.0, &zero, bwA);
			
			if (positLT(a, zero, bwA)) {
				a = zero;
			}

			A[h * W + w] = a;
		}
	}
	return;
}

template<class TypeA, class TypeB>
void Maxpool(TypeA* A, TypeB* B, MYITE N, MYITE H, MYITE W, MYITE C, MYITE FH, MYITE FW, MYITE strideH, MYITE strideW, MYITE HPADL, MYITE HPADR, MYITE WPADL, MYITE WPADR, int bwA, int bwB) {
	MYITE HO = H / strideH;
	MYITE WO = W / strideW;

	for (MYITE n = 0; n < N; n++) {
		for (MYITE ho = 0; ho < HO; ho++) {
			for (MYITE wo = 0; wo < WO; wo++) {
				for (MYITE c = 0; c < C; c++) {

					TypeA max = A[n * H * W * C + (strideH * ho) * W * C + (strideW * wo) * C + c];
					for (MYITE hs = 0; hs < FH; hs++) {
						for (MYITE ws = 0; ws < FW; ws++) {
							TypeA a = A[n * H * W * C + ((strideH * ho) + hs) * W * C + ((strideW * wo) + ws) * C + c];
							if (positLT(max, a, bwA)) {
								max = a;
							}
						}
					}
					convertPosit(&max, &B[n * HO * WO * C + ho * WO * C + wo * C + c], bwA, bwB);
				}
			}
		}
	}
	return;
}

template<class TypeA>
void NormaliseL2(TypeA* A, TypeA* B, MYITE N, MYITE H, MYITE W, MYITE C, int bwA) {
	TypeA zero, one;
	convertDoubleToPosit(0, &zero, bwA);
	convertDoubleToPosit(1, &one, bwA);
	for (MYITE n = 0; n < N; n++) {
		for (MYITE h = 0; h < H; h++) {
			for (MYITE w = 0; w < W; w++) {

				// Calculate the sum square.
				TypeA sumSquare = zero;
				for (MYITE c = 0; c < C; c++) {
					TypeA tmp = A[n * H * W * C + h * W * C + w * C + c];
					sumSquare = positAdd(sumSquare, positMul(tmp, tmp, bwA), bwA);
				}

				// Calculate the inverse square root of sumSquare.
				if (positEQ(sumSquare,  zero, bwA)) {
					convertDoubleToPosit(1e-5, &sumSquare, bwA);
				}

				TypeA inverseNorm = positDiv(one, positSqrt(sumSquare, bwA), bwA);

				// Multiply all elements by the 1 / sqrt(sumSquare).
				for (MYITE c = 0; c < C; c++) {
					B[n * H * W * C + h * W * C + w * C + c]  = positMul(A[n * H * W * C + h * W * C + w * C + c], inverseNorm, bwA);
				}
			}
		}
	}
	return;
}

template<class TypeA, class TypeB, class TypeTemp, class TypeX>
void MatAdd4(TypeA* A, TypeB* B, TypeX* X, MYITE N, MYITE H, MYITE W, MYITE C, int bwA, int bwB, int bwTemp, int bwC) {
	for (MYITE n = 0; n < N; n++) {
		for (MYITE h = 0; h < H; h++) {
			for (MYITE w = 0; w < W; w++) {
				for (MYITE c = 0; c < C; c++) {
					TypeTemp a,b;
					convertPosit(&A[n * H * W * C + h * W * C + w * C + c], &a, bwA, bwTemp);
					convertPosit(&B[n * H * W * C + h * W * C + w * C + c], &b, bwB, bwTemp);

					TypeTemp x = positAdd(a, b, bwTemp);

					convertPosit(&x, &X[n * H * W * C + h * W * C + w * C + c], bwTemp, bwC);
				}
			}
		}
	}
	return;
}

template<typename TypeA>
TypeA UnaryNegate(TypeA A, int bw) {
	float a = convertPositToDouble(A, bw);
	convertDoubleToPosit(-1*a, &A, bw);
	return A;
}