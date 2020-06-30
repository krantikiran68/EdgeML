// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#pragma once

void MatAddNN(float *A, float *B, float *C, MYINT I, MYINT J, MYINT shrA, MYINT shrB, MYINT shrC);
void MatAddCN(const float *A, float *B, float *C, MYINT I, MYINT J, MYINT shrA, MYINT shrB, MYINT shrC);
void MatAddNC(float *A, const float *B, float *C, MYINT I, MYINT J, MYINT shrA, MYINT shrB, MYINT shrC);
void MatAddCC(const float *A, const float *B, float *C, MYINT I, MYINT J, MYINT shrA, MYINT shrB, MYINT shrC);

void MatAddBroadCastA(float *A, float *B, float *C, MYINT I, MYINT J, MYINT shrA, MYINT shrB, MYINT shrC);
void MatAddBroadCastB(float *A, float *B, float *C, MYINT I, MYINT J, MYINT shrA, MYINT shrB, MYINT shrC);

void MatSub(float *A, const float *B, float *C, MYINT I, MYINT J, MYINT shrA, MYINT shrB, MYINT shrC);
void MatSubBroadCastA(float *A, float *B, float *C, MYINT I, MYINT J, MYINT shrA, MYINT shrB, MYINT shrC);
void MatSubBroadCastB(float *A, float *B, float *C, MYINT I, MYINT J, MYINT shrA, MYINT shrB, MYINT shrC);

void MatMulNN(float *A, float *B, float *C, float *tmp, MYINT I, MYINT K, MYINT J, MYINT shrA, MYINT shrB, MYINT H1, MYINT H2);
void MatMulCN(const float *A, float *B, float *C, float *tmp, MYINT I, MYINT K, MYINT J, MYINT shrA, MYINT shrB, MYINT H1, MYINT H2);
void MatMulNC(float *A, const float *B, float *C, float *tmp, MYINT I, MYINT K, MYINT J, MYINT shrA, MYINT shrB, MYINT H1, MYINT H2);
void MatMulCC(const float *A, const float *B, float *C, float *tmp, MYINT I, MYINT K, MYINT J, MYINT shrA, MYINT shrB, MYINT H1, MYINT H2);

void SparseMatMul(const MYINT *Aidx, const float *Aval, float **B, float *C, int16_t K, MYINT shrA, MYINT shrB, MYINT shrC);

void MulCir(float *A, float *B, float *C, MYINT I, MYINT J, MYINT shrA, MYINT shrB);

void TanH(float *A, MYINT I, MYINT J, float scale_in, float scale_out, float *B);

void ArgMax(float *A, MYINT I, MYINT J, MYINT *index);

void Transpose(float *A, float *B, MYINT I, MYINT J);

void ScalarMul(float *A, float *B, float *C, MYINT I, MYINT J, MYINT shrA, MYINT shrB);

void Conv(float *A, const float *B, float *C, float *tmp, MYINT N, MYINT H, MYINT W, MYINT CI, MYINT HF, MYINT WF, MYINT CO, MYINT shrA, MYINT shrB, MYINT H1, MYINT H2);

void Convolution(float *A, const float *B, float *C, float *tmp, MYINT N, MYINT H, MYINT W, MYINT CIN, MYINT HF, MYINT WF, MYINT CINF, MYINT COUTF, MYINT HOUT, MYINT WOUT, MYINT HPADL, MYINT HPADR, MYINT WPADL, MYINT WPADR, MYINT HSTR, MYINT WSTR, MYINT HDL, MYINT WDL, MYINT G, MYINT shrA, MYINT shrB, MYINT H1, MYINT H2);

void AddOrSubCir4D(float *A, const float *B, float *X, MYINT N, MYINT H, MYINT W, MYINT C, MYINT shrA, MYINT shrB, MYINT shrC, bool add);

void AddOrSubCir2D(float *A, const float *B, float *X, MYINT H, MYINT W, MYINT shrA, MYINT shrB, MYINT shrC, bool add);

void Relu4D(float *A, MYINT N, MYINT H, MYINT W, MYINT C);

void Relu2D(float *A, MYINT H, MYINT W);

void Maxpool(float *A, float *B, MYINT N, MYINT H, MYINT W, MYINT C, MYINT FH, MYINT FW, MYINT strideH, MYINT strideW, MYINT HPADL, MYINT HPADR, MYINT WPADL, MYINT WPADR);

void Exp(float *A, MYINT I, MYINT J, MYINT shrA, MYINT shrB, float *B);

void Sigmoid(float *A, MYINT I, MYINT J, float div, float add, float sigmoid_limit, MYINT scale_in, MYINT scale_out, float *B);

void AdjustScaleShr(float *A, MYINT I, MYINT J, MYINT scale);
void AdjustScaleShl(float *A, MYINT I, MYINT J, MYINT scale);

void Reverse2(float *A, MYINT axis, MYINT I, MYINT J, float *B);

void NormaliseL2(float* A, MYINT N, MYINT H, MYINT W, MYINT C, MYINT scaleA, MYINT shrA);