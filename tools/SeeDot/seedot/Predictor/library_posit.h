// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.
#include "softposit.h"

#pragma once
// This file contains declarations for floating point versions of all operators supported by SeeDot.
// Please refer to library_fixed.h for a description of each operator.

void MatAddNN(posit8_t* A, posit8_t* B, posit8_t* C, MYINT I, MYINT J, MYINT shrA, MYINT shrB, MYINT shrC);
void MatAddCN(const posit8_t* A, posit8_t* B, posit8_t* C, MYINT I, MYINT J, MYINT shrA, MYINT shrB, MYINT shrC);
void MatAddNC(posit8_t* A, const posit8_t* B, posit8_t* C, MYINT I, MYINT J, MYINT shrA, MYINT shrB, MYINT shrC);
void MatAddCC(const posit8_t* A, const posit8_t* B, posit8_t* C, MYINT I, MYINT J, MYINT shrA, MYINT shrB, MYINT shrC);

void MatAddBroadCastA(posit8_t* A, posit8_t* B, posit8_t* C, MYINT I, MYINT J, MYINT shrA, MYINT shrB, MYINT shrC);
void MatAddBroadCastB(posit8_t* A, posit8_t* B, posit8_t* C, MYINT I, MYINT J, MYINT shrA, MYINT shrB, MYINT shrC);

void MatAdd4(posit8_t* A, posit8_t* B, posit8_t* X, MYINT N, MYINT H, MYINT W, MYINT C, MYINT shrA, MYINT shrB, MYINT shrC);

void MatSub(posit8_t* A, const posit8_t* B, posit8_t* C, MYINT I, MYINT J, MYINT shrA, MYINT shrB, MYINT shrC);
void MatSubBroadCastA(posit8_t* A, posit8_t* B, posit8_t* C, MYINT I, MYINT J, MYINT shrA, MYINT shrB, MYINT shrC);
void MatSubBroadCastB(posit8_t* A, posit8_t* B, posit8_t* C, MYINT I, MYINT J, MYINT shrA, MYINT shrB, MYINT shrC);

void MatMulNN(posit8_t* A, posit8_t* B, posit8_t* C, posit8_t* tmp, MYINT I, MYINT K, MYINT J, MYINT shrA, MYINT shrB, MYINT H1, MYINT H2);
void MatMulCN(const posit8_t* A, posit8_t* B, posit8_t* C, posit8_t* tmp, MYINT I, MYINT K, MYINT J, MYINT shrA, MYINT shrB, MYINT H1, MYINT H2);
void MatMulNC(posit8_t* A, const posit8_t* B, posit8_t* C, posit8_t* tmp, MYINT I, MYINT K, MYINT J, MYINT shrA, MYINT shrB, MYINT H1, MYINT H2);
void MatMulCC(const posit8_t* A, const posit8_t* B, posit8_t* C, posit8_t* tmp, MYINT I, MYINT K, MYINT J, MYINT shrA, MYINT shrB, MYINT H1, MYINT H2);

void SparseMatMulX(const MYINT* Aidx, const posit8_t* Aval, posit8_t** B, posit8_t* C, int16_t K, MYINT shrA, MYINT shrB, MYINT shrC);
void SparseMatMul(const MYINT* Aidx, const posit8_t* Aval, posit8_t* B, posit8_t* C, int16_t K, MYINT shrA, MYINT shrB, MYINT shrC);

void MulCir(posit8_t* A, posit8_t* B, posit8_t* C, MYINT I, MYINT J, MYINT shrA, MYINT shrB);

void TanH(posit8_t* A, MYINT I, MYINT J, float scale_in, float scale_out, posit8_t* B);

void ArgMax(posit8_t* A, MYINT I, MYINT J, int* index);

void Transpose(posit8_t* A, posit8_t* B, MYINT I, MYINT J);

void ScalarMul(posit8_t* A, posit8_t* B, posit8_t* C, MYINT I, MYINT J, MYINT shrA, MYINT shrB);

void MBConv(posit8_t* A, const posit8_t* F1, const posit8_t* BN1W, const posit8_t* BN1B, const posit8_t* F2, const posit8_t* BN2W, const posit8_t* BN2B, const posit8_t* F3, const posit8_t* BN3W, const posit8_t* BN3B, posit8_t* C, posit8_t* X, posit8_t* T, posit8_t* U, MYITE N, MYITE H, MYITE W, MYITE Cin, MYITE Ct, MYITE HF, MYITE WF, MYITE Cout, MYITE Hout, MYITE Wout, MYITE HPADL, MYITE HPADR, MYITE WPADL, MYITE WPADR, MYITE HSTR, MYITE WSTR, MYITE D1, MYITE D2, MYITE D3, MYINT SIX_1, MYINT SIX_2, MYINT shr1, MYINT shr2, MYINT shr3, MYINT shr4, MYINT shr5, MYINT shr6, MYINT shr7, MYINT shr8, MYINT shr9, MYINT shl1, MYINT shl2, MYINT shl3, MYINT shl4, MYINT shl5, MYINT shl6, MYINT shl7, MYINT shl8, MYINT shl9, std::string name);

void Conv(posit8_t* A, const posit8_t* B, posit8_t* C, posit8_t* tmp, MYINT N, MYINT H, MYINT W, MYINT CI, MYINT HF, MYINT WF, MYINT CO, MYINT shrA, MYINT shrB, MYINT H1, MYINT H2);

void Convolution(posit8_t* A, const posit8_t* B, posit8_t* C, posit8_t* tmp, MYINT N, MYINT H, MYINT W, MYINT CIN, MYINT HF, MYINT WF, MYINT CINF, MYINT COUTF, MYINT HOUT, MYINT WOUT, MYINT HPADL, MYINT HPADR, MYINT WPADL, MYINT WPADR, MYINT HSTR, MYINT WSTR, MYINT HDL, MYINT WDL, MYINT G, MYINT shrA, MYINT shrB, MYINT H1, MYINT H2);

void AddOrSubCir4D(posit8_t* A, const posit8_t* B, posit8_t* X, MYINT N, MYINT H, MYINT W, MYINT C, MYINT shrA, MYINT shrB, MYINT shrC, bool add);

void AddOrSubCir2D(posit8_t* A, const posit8_t* B, posit8_t* X, MYINT H, MYINT W, MYINT shrA, MYINT shrB, MYINT shrC, bool add);

void Relu4D(posit8_t* A, MYINT N, MYINT H, MYINT W, MYINT C);

void Relu2D(posit8_t* A, MYINT H, MYINT W);

void Relu6(posit8_t* A, posit8_t* B, MYINT N, MYINT H, MYINT W, MYINT C, MYINT six, MYINT div);

void Maxpool(posit8_t* A, posit8_t* B, MYINT N, MYINT H, MYINT W, MYINT C, MYINT FH, MYINT FW, MYINT strideH, MYINT strideW, MYINT HPADL, MYINT HPADR, MYINT WPADL, MYINT WPADR);

void Exp(posit8_t* A, MYINT I, MYINT J, MYINT shrA, MYINT shrB, posit8_t* B);

void Sigmoid(posit8_t* A, MYINT I, MYINT J, float div, float add, float sigmoid_limit, MYINT scale_in, MYINT scale_out, posit8_t* B);

void AdjustScaleShr(posit8_t* A, MYINT I, MYINT J, MYINT scale);
void AdjustScaleShl(posit8_t* A, MYINT I, MYINT J, MYINT scale);

void Reverse2(posit8_t* A, MYINT axis, MYINT I, MYINT J, posit8_t* B);

void NormaliseL2(posit8_t* A, posit8_t* B, MYINT N, MYINT H, MYINT W, MYINT C, MYINT scaleA, MYINT shrA);


void MatAddNN(posit16_t* A, posit16_t* B, posit16_t* C, MYINT I, MYINT J, MYINT shrA, MYINT shrB, MYINT shrC);
void MatAddCN(const posit16_t* A, posit16_t* B, posit16_t* C, MYINT I, MYINT J, MYINT shrA, MYINT shrB, MYINT shrC);
void MatAddNC(posit16_t* A, const posit16_t* B, posit16_t* C, MYINT I, MYINT J, MYINT shrA, MYINT shrB, MYINT shrC);
void MatAddCC(const posit16_t* A, const posit16_t* B, posit16_t* C, MYINT I, MYINT J, MYINT shrA, MYINT shrB, MYINT shrC);

void MatAddBroadCastA(posit16_t* A, posit16_t* B, posit16_t* C, MYINT I, MYINT J, MYINT shrA, MYINT shrB, MYINT shrC);
void MatAddBroadCastB(posit16_t* A, posit16_t* B, posit16_t* C, MYINT I, MYINT J, MYINT shrA, MYINT shrB, MYINT shrC);

void MatAdd4(posit16_t* A, posit16_t* B, posit16_t* X, MYINT N, MYINT H, MYINT W, MYINT C, MYINT shrA, MYINT shrB, MYINT shrC);

void MatSub(posit16_t* A, const posit16_t* B, posit16_t* C, MYINT I, MYINT J, MYINT shrA, MYINT shrB, MYINT shrC);
void MatSubBroadCastA(posit16_t* A, posit16_t* B, posit16_t* C, MYINT I, MYINT J, MYINT shrA, MYINT shrB, MYINT shrC);
void MatSubBroadCastB(posit16_t* A, posit16_t* B, posit16_t* C, MYINT I, MYINT J, MYINT shrA, MYINT shrB, MYINT shrC);

void MatMulNN(posit16_t* A, posit16_t* B, posit16_t* C, posit16_t* tmp, MYINT I, MYINT K, MYINT J, MYINT shrA, MYINT shrB, MYINT H1, MYINT H2);
void MatMulCN(const posit16_t* A, posit16_t* B, posit16_t* C, posit16_t* tmp, MYINT I, MYINT K, MYINT J, MYINT shrA, MYINT shrB, MYINT H1, MYINT H2);
void MatMulNC(posit16_t* A, const posit16_t* B, posit16_t* C, posit16_t* tmp, MYINT I, MYINT K, MYINT J, MYINT shrA, MYINT shrB, MYINT H1, MYINT H2);
void MatMulCC(const posit16_t* A, const posit16_t* B, posit16_t* C, posit16_t* tmp, MYINT I, MYINT K, MYINT J, MYINT shrA, MYINT shrB, MYINT H1, MYINT H2);

void SparseMatMulX(const MYINT* Aidx, const posit16_t* Aval, posit16_t** B, posit16_t* C, int16_t K, MYINT shrA, MYINT shrB, MYINT shrC);
void SparseMatMul(const MYINT* Aidx, const posit16_t* Aval, posit16_t* B, posit16_t* C, int16_t K, MYINT shrA, MYINT shrB, MYINT shrC);

void MulCir(posit16_t* A, posit16_t* B, posit16_t* C, MYINT I, MYINT J, MYINT shrA, MYINT shrB);

void TanH(posit16_t* A, MYINT I, MYINT J, float scale_in, float scale_out, posit16_t* B);

void ArgMax(posit16_t* A, MYINT I, MYINT J, int* index);

void Transpose(posit16_t* A, posit16_t* B, MYINT I, MYINT J);

void ScalarMul(posit16_t* A, posit16_t* B, posit16_t* C, MYINT I, MYINT J, MYINT shrA, MYINT shrB);

void MBConv(posit16_t* A, const posit16_t* F1, const posit16_t* BN1W, const posit16_t* BN1B, const posit16_t* F2, const posit16_t* BN2W, const posit16_t* BN2B, const posit16_t* F3, const posit16_t* BN3W, const posit16_t* BN3B, posit16_t* C, posit16_t* X, posit16_t* T, posit16_t* U, MYITE N, MYITE H, MYITE W, MYITE Cin, MYITE Ct, MYITE HF, MYITE WF, MYITE Cout, MYITE Hout, MYITE Wout, MYITE HPADL, MYITE HPADR, MYITE WPADL, MYITE WPADR, MYITE HSTR, MYITE WSTR, MYITE D1, MYITE D2, MYITE D3, MYINT SIX_1, MYINT SIX_2, MYINT shr1, MYINT shr2, MYINT shr3, MYINT shr4, MYINT shr5, MYINT shr6, MYINT shr7, MYINT shr8, MYINT shr9, MYINT shl1, MYINT shl2, MYINT shl3, MYINT shl4, MYINT shl5, MYINT shl6, MYINT shl7, MYINT shl8, MYINT shl9, std::string name);

void Conv(posit16_t* A, const posit16_t* B, posit16_t* C, posit16_t* tmp, MYINT N, MYINT H, MYINT W, MYINT CI, MYINT HF, MYINT WF, MYINT CO, MYINT shrA, MYINT shrB, MYINT H1, MYINT H2);

void Convolution(posit16_t* A, const posit16_t* B, posit16_t* C, posit16_t* tmp, MYINT N, MYINT H, MYINT W, MYINT CIN, MYINT HF, MYINT WF, MYINT CINF, MYINT COUTF, MYINT HOUT, MYINT WOUT, MYINT HPADL, MYINT HPADR, MYINT WPADL, MYINT WPADR, MYINT HSTR, MYINT WSTR, MYINT HDL, MYINT WDL, MYINT G, MYINT shrA, MYINT shrB, MYINT H1, MYINT H2);

void AddOrSubCir4D(posit16_t* A, const posit16_t* B, posit16_t* X, MYINT N, MYINT H, MYINT W, MYINT C, MYINT shrA, MYINT shrB, MYINT shrC, bool add);

void AddOrSubCir2D(posit16_t* A, const posit16_t* B, posit16_t* X, MYINT H, MYINT W, MYINT shrA, MYINT shrB, MYINT shrC, bool add);

void Relu4D(posit16_t* A, MYINT N, MYINT H, MYINT W, MYINT C);

void Relu2D(posit16_t* A, MYINT H, MYINT W);

void Relu6(posit16_t* A, posit16_t* B, MYINT N, MYINT H, MYINT W, MYINT C, MYINT six, MYINT div);

void Maxpool(posit16_t* A, posit16_t* B, MYINT N, MYINT H, MYINT W, MYINT C, MYINT FH, MYINT FW, MYINT strideH, MYINT strideW, MYINT HPADL, MYINT HPADR, MYINT WPADL, MYINT WPADR);

void Exp(posit16_t* A, MYINT I, MYINT J, MYINT shrA, MYINT shrB, posit16_t* B);

void Sigmoid(posit16_t* A, MYINT I, MYINT J, float div, float add, float sigmoid_limit, MYINT scale_in, MYINT scale_out, posit16_t* B);

void AdjustScaleShr(posit16_t* A, MYINT I, MYINT J, MYINT scale);
void AdjustScaleShl(posit16_t* A, MYINT I, MYINT J, MYINT scale);

void Reverse2(posit16_t* A, MYINT axis, MYINT I, MYINT J, posit16_t* B);

void NormaliseL2(posit16_t* A, posit16_t* B, MYINT N, MYINT H, MYINT W, MYINT C, MYINT scaleA, MYINT shrA);


void MatAddNN(posit32_t* A, posit32_t* B, posit32_t* C, MYINT I, MYINT J, MYINT shrA, MYINT shrB, MYINT shrC);
void MatAddCN(const posit32_t* A, posit32_t* B, posit32_t* C, MYINT I, MYINT J, MYINT shrA, MYINT shrB, MYINT shrC);
void MatAddNC(posit32_t* A, const posit32_t* B, posit32_t* C, MYINT I, MYINT J, MYINT shrA, MYINT shrB, MYINT shrC);
void MatAddCC(const posit32_t* A, const posit32_t* B, posit32_t* C, MYINT I, MYINT J, MYINT shrA, MYINT shrB, MYINT shrC);

void MatAddBroadCastA(posit32_t* A, posit32_t* B, posit32_t* C, MYINT I, MYINT J, MYINT shrA, MYINT shrB, MYINT shrC);
void MatAddBroadCastB(posit32_t* A, posit32_t* B, posit32_t* C, MYINT I, MYINT J, MYINT shrA, MYINT shrB, MYINT shrC);

void MatAdd4(posit32_t* A, posit32_t* B, posit32_t* X, MYINT N, MYINT H, MYINT W, MYINT C, MYINT shrA, MYINT shrB, MYINT shrC);

void MatSub(posit32_t* A, const posit32_t* B, posit32_t* C, MYINT I, MYINT J, MYINT shrA, MYINT shrB, MYINT shrC);
void MatSubBroadCastA(posit32_t* A, posit32_t* B, posit32_t* C, MYINT I, MYINT J, MYINT shrA, MYINT shrB, MYINT shrC);
void MatSubBroadCastB(posit32_t* A, posit32_t* B, posit32_t* C, MYINT I, MYINT J, MYINT shrA, MYINT shrB, MYINT shrC);

void MatMulNN(posit32_t* A, posit32_t* B, posit32_t* C, posit32_t* tmp, MYINT I, MYINT K, MYINT J, MYINT shrA, MYINT shrB, MYINT H1, MYINT H2);
void MatMulCN(const posit32_t* A, posit32_t* B, posit32_t* C, posit32_t* tmp, MYINT I, MYINT K, MYINT J, MYINT shrA, MYINT shrB, MYINT H1, MYINT H2);
void MatMulNC(posit32_t* A, const posit32_t* B, posit32_t* C, posit32_t* tmp, MYINT I, MYINT K, MYINT J, MYINT shrA, MYINT shrB, MYINT H1, MYINT H2);
void MatMulCC(const posit32_t* A, const posit32_t* B, posit32_t* C, posit32_t* tmp, MYINT I, MYINT K, MYINT J, MYINT shrA, MYINT shrB, MYINT H1, MYINT H2);

void SparseMatMulX(const MYINT* Aidx, const posit32_t* Aval, posit32_t** B, posit32_t* C, int16_t K, MYINT shrA, MYINT shrB, MYINT shrC);
void SparseMatMul(const MYINT* Aidx, const posit32_t* Aval, posit32_t* B, posit32_t* C, int16_t K, MYINT shrA, MYINT shrB, MYINT shrC);

void MulCir(posit32_t* A, posit32_t* B, posit32_t* C, MYINT I, MYINT J, MYINT shrA, MYINT shrB);

void TanH(posit32_t* A, MYINT I, MYINT J, float scale_in, float scale_out, posit32_t* B);

void ArgMax(posit32_t* A, MYINT I, MYINT J, int* index);

void Transpose(posit32_t* A, posit32_t* B, MYINT I, MYINT J);

void ScalarMul(posit32_t* A, posit32_t* B, posit32_t* C, MYINT I, MYINT J, MYINT shrA, MYINT shrB);

void MBConv(posit32_t* A, const posit32_t* F1, const posit32_t* BN1W, const posit32_t* BN1B, const posit32_t* F2, const posit32_t* BN2W, const posit32_t* BN2B, const posit32_t* F3, const posit32_t* BN3W, const posit32_t* BN3B, posit32_t* C, posit32_t* X, posit32_t* T, posit32_t* U, MYITE N, MYITE H, MYITE W, MYITE Cin, MYITE Ct, MYITE HF, MYITE WF, MYITE Cout, MYITE Hout, MYITE Wout, MYITE HPADL, MYITE HPADR, MYITE WPADL, MYITE WPADR, MYITE HSTR, MYITE WSTR, MYITE D1, MYITE D2, MYITE D3, MYINT SIX_1, MYINT SIX_2, MYINT shr1, MYINT shr2, MYINT shr3, MYINT shr4, MYINT shr5, MYINT shr6, MYINT shr7, MYINT shr8, MYINT shr9, MYINT shl1, MYINT shl2, MYINT shl3, MYINT shl4, MYINT shl5, MYINT shl6, MYINT shl7, MYINT shl8, MYINT shl9, std::string name);

void Conv(posit32_t* A, const posit32_t* B, posit32_t* C, posit32_t* tmp, MYINT N, MYINT H, MYINT W, MYINT CI, MYINT HF, MYINT WF, MYINT CO, MYINT shrA, MYINT shrB, MYINT H1, MYINT H2);

void Convolution(posit32_t* A, const posit32_t* B, posit32_t* C, posit32_t* tmp, MYINT N, MYINT H, MYINT W, MYINT CIN, MYINT HF, MYINT WF, MYINT CINF, MYINT COUTF, MYINT HOUT, MYINT WOUT, MYINT HPADL, MYINT HPADR, MYINT WPADL, MYINT WPADR, MYINT HSTR, MYINT WSTR, MYINT HDL, MYINT WDL, MYINT G, MYINT shrA, MYINT shrB, MYINT H1, MYINT H2);

void AddOrSubCir4D(posit32_t* A, const posit32_t* B, posit32_t* X, MYINT N, MYINT H, MYINT W, MYINT C, MYINT shrA, MYINT shrB, MYINT shrC, bool add);

void AddOrSubCir2D(posit32_t* A, const posit32_t* B, posit32_t* X, MYINT H, MYINT W, MYINT shrA, MYINT shrB, MYINT shrC, bool add);

void Relu4D(posit32_t* A, MYINT N, MYINT H, MYINT W, MYINT C);

void Relu2D(posit32_t* A, MYINT H, MYINT W);

void Relu6(posit32_t* A, posit32_t* B, MYINT N, MYINT H, MYINT W, MYINT C, MYINT six, MYINT div);

void Maxpool(posit32_t* A, posit32_t* B, MYINT N, MYINT H, MYINT W, MYINT C, MYINT FH, MYINT FW, MYINT strideH, MYINT strideW, MYINT HPADL, MYINT HPADR, MYINT WPADL, MYINT WPADR);

void Exp(posit32_t* A, MYINT I, MYINT J, MYINT shrA, MYINT shrB, posit32_t* B);

void Sigmoid(posit32_t* A, MYINT I, MYINT J, float div, float add, float sigmoid_limit, MYINT scale_in, MYINT scale_out, posit32_t* B);

void AdjustScaleShr(posit32_t* A, MYINT I, MYINT J, MYINT scale);
void AdjustScaleShl(posit32_t* A, MYINT I, MYINT J, MYINT scale);

void Reverse2(posit32_t* A, MYINT axis, MYINT I, MYINT J, posit32_t* B);

void NormaliseL2(posit32_t* A, posit32_t* B, MYINT N, MYINT H, MYINT W, MYINT C, MYINT scaleA, MYINT shrA);



void MatAddNN(posit_2_t* A, posit_2_t* B, posit_2_t* C, MYINT I, MYINT J, MYINT shrA, MYINT shrB, MYINT shrC, int bitwidth);
void MatAddCN(const posit_2_t* A, posit_2_t* B, posit_2_t* C, MYINT I, MYINT J, MYINT shrA, MYINT shrB, MYINT shrC, int bitwidth);
void MatAddNC(posit_2_t* A, const posit_2_t* B, posit_2_t* C, MYINT I, MYINT J, MYINT shrA, MYINT shrB, MYINT shrC, int bitwidth);
void MatAddCC(const posit_2_t* A, const posit_2_t* B, posit_2_t* C, MYINT I, MYINT J, MYINT shrA, MYINT shrB, MYINT shrC, int bitwidth);

void MatAddBroadCastA(posit_2_t* A, posit_2_t* B, posit_2_t* C, MYINT I, MYINT J, MYINT shrA, MYINT shrB, MYINT shrC, int bitwidth);
void MatAddBroadCastB(posit_2_t* A, posit_2_t* B, posit_2_t* C, MYINT I, MYINT J, MYINT shrA, MYINT shrB, MYINT shrC, int bitwidth);

void MatAdd4(posit_2_t* A, posit_2_t* B, posit_2_t* X, MYINT N, MYINT H, MYINT W, MYINT C, MYINT shrA, MYINT shrB, MYINT shrC, int bitwidth);

void MatSub(posit_2_t* A, const posit_2_t* B, posit_2_t* C, MYINT I, MYINT J, MYINT shrA, MYINT shrB, MYINT shrC, int bitwidth);
void MatSubBroadCastA(posit_2_t* A, posit_2_t* B, posit_2_t* C, MYINT I, MYINT J, MYINT shrA, MYINT shrB, MYINT shrC, int bitwidth);
void MatSubBroadCastB(posit_2_t* A, posit_2_t* B, posit_2_t* C, MYINT I, MYINT J, MYINT shrA, MYINT shrB, MYINT shrC, int bitwidth);

void MatMulNN(posit_2_t* A, posit_2_t* B, posit_2_t* C, posit_2_t* tmp, MYINT I, MYINT K, MYINT J, MYINT shrA, MYINT shrB, MYINT H1, MYINT H2, int bitwidth);
void MatMulCN(const posit_2_t* A, posit_2_t* B, posit_2_t* C, posit_2_t* tmp, MYINT I, MYINT K, MYINT J, MYINT shrA, MYINT shrB, MYINT H1, MYINT H2, int bitwidth);
void MatMulNC(posit_2_t* A, const posit_2_t* B, posit_2_t* C, posit_2_t* tmp, MYINT I, MYINT K, MYINT J, MYINT shrA, MYINT shrB, MYINT H1, MYINT H2, int bitwidth);
void MatMulCC(const posit_2_t* A, const posit_2_t* B, posit_2_t* C, posit_2_t* tmp, MYINT I, MYINT K, MYINT J, MYINT shrA, MYINT shrB, MYINT H1, MYINT H2, int bitwidth);

void SparseMatMulX(const MYINT* Aidx, const posit_2_t* Aval, posit_2_t** B, posit_2_t* C, int16_t K, MYINT shrA, MYINT shrB, MYINT shrC, int bitwidth);
void SparseMatMul(const MYINT* Aidx, const posit_2_t* Aval, posit_2_t* B, posit_2_t* C, int16_t K, MYINT shrA, MYINT shrB, MYINT shrC, int bitwidth);

void MulCir(posit_2_t* A, posit_2_t* B, posit_2_t* C, MYINT I, MYINT J, MYINT shrA, MYINT shrB, int bitwidth);

void TanH(posit_2_t* A, MYINT I, MYINT J, float scale_in, float scale_out, posit_2_t* B, int bitwidth);

void ArgMax(posit_2_t* A, MYINT I, MYINT J, int* index, int bitwidth);

void Transpose(posit_2_t* A, posit_2_t* B, MYINT I, MYINT J, int bitwidth);

void ScalarMul(posit_2_t* A, posit_2_t* B, posit_2_t* C, MYINT I, MYINT J, MYINT shrA, MYINT shrB, int bitwidth);

void MBConv(posit_2_t* A, const posit_2_t* F1, const posit_2_t* BN1W, const posit_2_t* BN1B, const posit_2_t* F2, const posit_2_t* BN2W, const posit_2_t* BN2B, const posit_2_t* F3, const posit_2_t* BN3W, const posit_2_t* BN3B, posit_2_t* C, posit_2_t* X, posit_2_t* T, posit_2_t* U, MYITE N, MYITE H, MYITE W, MYITE Cin, MYITE Ct, MYITE HF, MYITE WF, MYITE Cout, MYITE Hout, MYITE Wout, MYITE HPADL, MYITE HPADR, MYITE WPADL, MYITE WPADR, MYITE HSTR, MYITE WSTR, MYITE D1, MYITE D2, MYITE D3, MYINT SIX_1, MYINT SIX_2, MYINT shr1, MYINT shr2, MYINT shr3, MYINT shr4, MYINT shr5, MYINT shr6, MYINT shr7, MYINT shr8, MYINT shr9, MYINT shl1, MYINT shl2, MYINT shl3, MYINT shl4, MYINT shl5, MYINT shl6, MYINT shl7, MYINT shl8, MYINT shl9, std::string name, int bitwidth);

void Conv(posit_2_t* A, const posit_2_t* B, posit_2_t* C, posit_2_t* tmp, MYINT N, MYINT H, MYINT W, MYINT CI, MYINT HF, MYINT WF, MYINT CO, MYINT shrA, MYINT shrB, MYINT H1, MYINT H2, int bitwidth);

void Convolution(posit_2_t* A, const posit_2_t* B, posit_2_t* C, posit_2_t* tmp, MYINT N, MYINT H, MYINT W, MYINT CIN, MYINT HF, MYINT WF, MYINT CINF, MYINT COUTF, MYINT HOUT, MYINT WOUT, MYINT HPADL, MYINT HPADR, MYINT WPADL, MYINT WPADR, MYINT HSTR, MYINT WSTR, MYINT HDL, MYINT WDL, MYINT G, MYINT shrA, MYINT shrB, MYINT H1, MYINT H2, int bitwidth);

void AddOrSubCir4D(posit_2_t* A, const posit_2_t* B, posit_2_t* X, MYINT N, MYINT H, MYINT W, MYINT C, MYINT shrA, MYINT shrB, MYINT shrC, bool add, int bitwidth);

void AddOrSubCir2D(posit_2_t* A, const posit_2_t* B, posit_2_t* X, MYINT H, MYINT W, MYINT shrA, MYINT shrB, MYINT shrC, bool add, int bitwidth);

void Relu4D(posit_2_t* A, MYINT N, MYINT H, MYINT W, MYINT C, int bitwidth);

void Relu2D(posit_2_t* A, MYINT H, MYINT W, int bitwidth);

void Relu6(posit_2_t* A, posit_2_t* B, MYINT N, MYINT H, MYINT W, MYINT C, MYINT six, MYINT div, int bitwidth);

void Maxpool(posit_2_t* A, posit_2_t* B, MYINT N, MYINT H, MYINT W, MYINT C, MYINT FH, MYINT FW, MYINT strideH, MYINT strideW, MYINT HPADL, MYINT HPADR, MYINT WPADL, MYINT WPADR, int bitwidth);

void Exp(posit_2_t* A, MYINT I, MYINT J, MYINT shrA, MYINT shrB, posit_2_t* B, int bitwidth);

void Sigmoid(posit_2_t* A, MYINT I, MYINT J, float div, float add, float sigmoid_limit, MYINT scale_in, MYINT scale_out, posit_2_t* B, int bitwidth);

void AdjustScaleShr(posit_2_t* A, MYINT I, MYINT J, MYINT scale, int bitwidth);
void AdjustScaleShl(posit_2_t* A, MYINT I, MYINT J, MYINT scale, int bitwidth);

void Reverse2(posit_2_t* A, MYINT axis, MYINT I, MYINT J, posit_2_t* B, int bitwidth);

void NormaliseL2(posit_2_t* A, posit_2_t* B, MYINT N, MYINT H, MYINT W, MYINT C, MYINT scaleA, MYINT shrA, int bitwidth);

