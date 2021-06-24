// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#pragma once

#include "datatypes.h"

// This file contains declarations for floating point versions of all operators supported by SeeDot.
// Please refer to library_fixed.h for a description of each operator.

void MatAddNN(FP_TYPE* A, FP_TYPE* B, FP_TYPE* C, MYINT I, MYINT J, MYINT shrA, MYINT shrB, MYINT shrC);
void MatAddCN(const FP_TYPE* A, FP_TYPE* B, FP_TYPE* C, MYINT I, MYINT J, MYINT shrA, MYINT shrB, MYINT shrC);
void MatAddNC(FP_TYPE* A, const FP_TYPE* B, FP_TYPE* C, MYINT I, MYINT J, MYINT shrA, MYINT shrB, MYINT shrC);
void MatAddCC(const FP_TYPE* A, const FP_TYPE* B, FP_TYPE* C, MYINT I, MYINT J, MYINT shrA, MYINT shrB, MYINT shrC);

void MatAddBroadCastA(FP_TYPE* A, FP_TYPE* B, FP_TYPE* C, MYINT I, MYINT J, MYINT shrA, MYINT shrB, MYINT shrC);
void MatAddBroadCastB(FP_TYPE* A, FP_TYPE* B, FP_TYPE* C, MYINT I, MYINT J, MYINT shrA, MYINT shrB, MYINT shrC);

void MatAdd4(FP_TYPE* A, FP_TYPE* B, FP_TYPE* X, MYINT N, MYINT H, MYINT W, MYINT C, MYINT shrA, MYINT shrB, MYINT shrC);

void MatSub(FP_TYPE* A, const FP_TYPE* B, FP_TYPE* C, MYINT I, MYINT J, MYINT shrA, MYINT shrB, MYINT shrC);
void MatSubBroadCastA(FP_TYPE* A, FP_TYPE* B, FP_TYPE* C, MYINT I, MYINT J, MYINT shrA, MYINT shrB, MYINT shrC);
void MatSubBroadCastB(FP_TYPE* A, FP_TYPE* B, FP_TYPE* C, MYINT I, MYINT J, MYINT shrA, MYINT shrB, MYINT shrC);

void MatMulNN(FP_TYPE* A, FP_TYPE* B, FP_TYPE* C, FP_TYPE* tmp, MYINT I, MYINT K, MYINT J, MYINT shrA, MYINT shrB, MYINT H1, MYINT H2);
void MatMulCN(const FP_TYPE* A, FP_TYPE* B, FP_TYPE* C, FP_TYPE* tmp, MYINT I, MYINT K, MYINT J, MYINT shrA, MYINT shrB, MYINT H1, MYINT H2);
void MatMulNC(FP_TYPE* A, const FP_TYPE* B, FP_TYPE* C, FP_TYPE* tmp, MYINT I, MYINT K, MYINT J, MYINT shrA, MYINT shrB, MYINT H1, MYINT H2);
void MatMulCC(const FP_TYPE* A, const FP_TYPE* B, FP_TYPE* C, FP_TYPE* tmp, MYINT I, MYINT K, MYINT J, MYINT shrA, MYINT shrB, MYINT H1, MYINT H2);

void SparseMatMulX(const MYINT* Aidx, const FP_TYPE* Aval, FP_TYPE** B, FP_TYPE* C, int16_t K, MYINT shrA, MYINT shrB, MYINT shrC);
void SparseMatMul(const MYINT* Aidx, const FP_TYPE* Aval, FP_TYPE* B, FP_TYPE* C, int16_t K, MYINT shrA, MYINT shrB, MYINT shrC);

void MulCir(FP_TYPE* A, FP_TYPE* B, FP_TYPE* C, MYINT I, MYINT J, MYINT shrA, MYINT shrB);

void TanH(FP_TYPE* A, MYINT I, MYINT J, float scale_in, float scale_out, FP_TYPE* B);

void ArgMax(FP_TYPE* A, MYINT I, MYINT J, int* index);

void Transpose(FP_TYPE* A, FP_TYPE* B, MYINT I, MYINT J);

void ScalarMul(FP_TYPE* A, FP_TYPE* B, FP_TYPE* C, MYINT I, MYINT J, MYINT shrA, MYINT shrB);

void MBConv(FP_TYPE* A, const FP_TYPE* F1, const FP_TYPE* BN1W, const FP_TYPE* BN1B, const FP_TYPE* F2, const FP_TYPE* BN2W, const FP_TYPE* BN2B, const FP_TYPE* F3, const FP_TYPE* BN3W, const FP_TYPE* BN3B, FP_TYPE* C, FP_TYPE* X, FP_TYPE* T, FP_TYPE* U, MYITE N, MYITE H, MYITE W, MYITE Cin, MYITE Ct, MYITE HF, MYITE WF, MYITE Cout, MYITE Hout, MYITE Wout, MYITE HPADL, MYITE HPADR, MYITE WPADL, MYITE WPADR, MYITE HSTR, MYITE WSTR, MYITE D1, MYITE D2, MYITE D3, MYINT SIX_1, MYINT SIX_2, MYINT shr1, MYINT shr2, MYINT shr3, MYINT shr4, MYINT shr5, MYINT shr6, MYINT shr7, MYINT shr8, MYINT shr9, MYINT shl1, MYINT shl2, MYINT shl3, MYINT shl4, MYINT shl5, MYINT shl6, MYINT shl7, MYINT shl8, MYINT shl9, std::string name);

void Conv(FP_TYPE* A, const FP_TYPE* B, FP_TYPE* C, FP_TYPE* tmp, MYINT N, MYINT H, MYINT W, MYINT CI, MYINT HF, MYINT WF, MYINT CO, MYINT shrA, MYINT shrB, MYINT H1, MYINT H2);

void Convolution(FP_TYPE* A, const FP_TYPE* B, FP_TYPE* C, FP_TYPE* tmp, MYINT N, MYINT H, MYINT W, MYINT CIN, MYINT HF, MYINT WF, MYINT CINF, MYINT COUTF, MYINT HOUT, MYINT WOUT, MYINT HPADL, MYINT HPADR, MYINT WPADL, MYINT WPADR, MYINT HSTR, MYINT WSTR, MYINT HDL, MYINT WDL, MYINT G, MYINT shrA, MYINT shrB, MYINT H1, MYINT H2);

void AddOrSubCir4D(FP_TYPE* A, const FP_TYPE* B, FP_TYPE* X, MYINT N, MYINT H, MYINT W, MYINT C, MYINT shrA, MYINT shrB, MYINT shrC, bool add);

void AddOrSubCir2D(FP_TYPE* A, const FP_TYPE* B, FP_TYPE* X, MYINT H, MYINT W, MYINT shrA, MYINT shrB, MYINT shrC, bool add);

void Relu4D(FP_TYPE* A, MYINT N, MYINT H, MYINT W, MYINT C);

void Relu2D(FP_TYPE* A, MYINT H, MYINT W);

void Relu6(FP_TYPE* A, FP_TYPE* B, MYINT N, MYINT H, MYINT W, MYINT C, MYINT six, MYINT div);

void Maxpool(FP_TYPE* A, FP_TYPE* B, MYINT N, MYINT H, MYINT W, MYINT C, MYINT FH, MYINT FW, MYINT strideH, MYINT strideW, MYINT HPADL, MYINT HPADR, MYINT WPADL, MYINT WPADR);

void Exp(FP_TYPE* A, MYINT I, MYINT J, MYINT shrA, MYINT shrB, FP_TYPE* B);

void Sigmoid(FP_TYPE* A, MYINT I, MYINT J, float div, float add, float sigmoid_limit, MYINT scale_in, MYINT scale_out, FP_TYPE* B);

void AdjustScaleShr(FP_TYPE* A, MYINT I, MYINT J, MYINT scale);
void AdjustScaleShl(FP_TYPE* A, MYINT I, MYINT J, MYINT scale);

void Reverse2(FP_TYPE* A, MYINT axis, MYINT I, MYINT J, FP_TYPE* B);

void NormaliseL2(FP_TYPE* A, FP_TYPE* B, MYINT N, MYINT H, MYINT W, MYINT C, MYINT scaleA, MYINT shrA);
