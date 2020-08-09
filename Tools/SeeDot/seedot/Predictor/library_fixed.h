// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#pragma once

#include "datatypes.h"

void MatAddNN(MYINT *A, MYINT *B, MYINT *C, MYINT I, MYINT J, MYINT shrA, MYINT shrB, MYINT shrC);
void MatAddCN(const MYINT *A, MYINT *B, MYINT *C, MYINT I, MYINT J, MYINT shrA, MYINT shrB, MYINT shrC);
void MatAddNC(MYINT *A, const MYINT *B, MYINT *C, MYINT I, MYINT J, MYINT shrA, MYINT shrB, MYINT shrC);
void MatAddCC(const MYINT *A, const MYINT *B, MYINT *C, MYINT I, MYINT J, MYINT shrA, MYINT shrB, MYINT shrC);

void MatAddBroadCastA(MYINT *A, MYINT *B, MYINT *C, MYINT I, MYINT J, MYINT shrA, MYINT shrB, MYINT shrC);
void MatAddBroadCastB(MYINT *A, MYINT *B, MYINT *C, MYINT I, MYINT J, MYINT shrA, MYINT shrB, MYINT shrC);

void MatSub(MYINT *A, const MYINT *B, MYINT *C, MYINT I, MYINT J, MYINT shrA, int32_t shrB, MYINT shrC);
void MatSubBroadCastA(MYINT *A, MYINT *B, MYINT *C, MYINT I, MYINT J, MYINT shrA, int32_t shrB, MYINT shrC);
void MatSubBroadCastB(MYINT *A, MYINT *B, MYINT *C, MYINT I, MYINT J, MYINT shrA, int32_t shrB, MYINT shrC);

void MatMulNN(MYINT *A, MYINT *B, MYINT *C, MYINT *tmp, MYINT I, MYINT K, MYINT J, MYINT shrA, MYINT shrB, MYINT H1, MYINT H2);

void MatMulCN(const MYINT *A, MYINT *B, MYINT *C, MYINT *tmp, MYINT I, MYINT K, MYINT J, MYINT shrA, MYINT shrB, MYINT H1, MYINT H2);

void MatMulNC(MYINT *A, const MYINT *B, MYINT *C, MYINT *tmp, MYINT I, MYINT K, MYINT J, MYINT shrA, MYINT shrB, MYINT H1, MYINT H2);

void MatMulCC(const MYINT *A, const MYINT *B, MYINT *C, MYINT *tmp, MYINT I, MYINT K, MYINT J, MYINT shrA, MYINT shrB, MYINT H1, MYINT H2);

void SparseMatMul(const MYINT *Aidx, const MYINT *Aval, MYINT **B, MYINT *C, int16_t K, MYINT shrA, MYINT shrB, MYINT shrC);

void MulCir(MYINT *A, MYINT *B, MYINT *C, MYINT I, MYINT J, MYINT shrA, MYINT shrB);

void TanH(MYINT *A, MYINT I, MYINT J, MYINT scale_in, MYINT scale_out, MYINT *B);

void ArgMax(MYINT *A, MYINT I, MYINT J, MYINT *index);

void Transpose(MYINT *A, MYINT *B, MYINT I, MYINT J);

void ScalarMul(MYINT *A, MYINT *B, MYINT *C, MYINT I, MYINT J, MYINT shrA, MYINT shrB);

void AddOrSubCir4D(MYINT *A, const MYINT *B, MYINT *X, MYINT N, MYINT H, MYINT W, MYINT C, MYINT shrA, MYINT shrB, MYINT shrC, bool add);

void AddOrSubCir2D(MYINT *A, const MYINT *B, MYINT *X, MYINT H, MYINT W, MYINT shrA, MYINT shrB, MYINT shrC, bool add);

void Relu4D(MYINT *A, MYINT N, MYINT H, MYINT W, MYINT C);

void Relu2D(MYINT *A, MYINT H, MYINT W);

void Maxpool(MYINT *A, MYINT *B, MYINT N, MYINT H, MYINT W, MYINT C, MYINT stride);

void Exp(MYINT *A, MYINT I, MYINT J, MYINT shrA, MYINT shrB, MYINT *B);

void Sigmoid(MYINT *A, MYINT I, MYINT J, MYINT div, MYINT add, MYINT sigmoid_limit, MYINT scale_in, MYINT scale_out, MYINT *B);

void AdjustScaleShr(MYINT *A, MYINT I, MYINT J, MYINT scale);
void AdjustScaleShl(MYINT *A, MYINT I, MYINT J, MYINT scale);

//Templated Operations: For cases when Variable BitWidth is enabled

template<class TypeA>
inline TypeA Saturate(int32_t inp) {
	return (TypeA)inp;
}

template<>
inline int16_t Saturate(int32_t inp) {
#ifdef SATURATE
	inp = inp > 32767 ? 32767 : inp;
	return (int16_t)(inp < -32768 ? -32768 : inp);
#else
	return (int16_t)inp;
#endif
}

template<>
inline int8_t Saturate(int32_t inp) {
#ifdef SATURATE
	inp = inp > 127 ? 127 : inp;
	return (int8_t)(inp < -128 ? -128 : inp);
#else
	return (int8_t)inp;
#endif
}

template<class TypeA, class TypeB, class TypeTemp, class TypeC>
void MatAddNN(TypeA* A, TypeB* B, TypeC* C, MYINT I, MYINT J, MYINT shrA, MYINT shrB, MYINT shrC, MYINT demote) {
	for (MYITE i = 0; i < I; i++) {
		for (MYITE j = 0; j < J; j++) {
			TypeTemp a = (TypeTemp)A[i * J + j];
			TypeTemp b = (TypeTemp)B[i * J + j];

			a = a / shrA;
			b = b / shrB;

			TypeTemp c = a / shrC + b / shrC;

			C[i * J + j] = Saturate<TypeC>(c / demote);
		}
	}
	return;
}
template<class TypeA, class TypeB, class TypeTemp, class TypeC>
void MatAddCN(const TypeA* A, TypeB* B, TypeC* C, MYINT I, MYINT J, MYINT shrA, MYINT shrB, MYINT shrC, MYINT demote) {
	for (MYITE i = 0; i < I; i++) {
		for (MYITE j = 0; j < J; j++) {
			TypeTemp a = (TypeTemp)A[i * J + j];
			TypeTemp b = (TypeTemp)B[i * J + j];

			a = a / shrA;
			b = b / shrB;

			TypeTemp c = a / shrC + b / shrC;

			C[i * J + j] = Saturate<TypeC>(c / demote);
		}
	}
	return;
}
template<class TypeA, class TypeB, class TypeTemp, class TypeC>
void MatAddNC(TypeA* A, const TypeB* B, TypeC* C, MYINT I, MYINT J, MYINT shrA, MYINT shrB, MYINT shrC, MYINT demote) {
	for (MYITE i = 0; i < I; i++) {
		for (MYITE j = 0; j < J; j++) {
			TypeTemp a = (TypeTemp)A[i * J + j];
			TypeTemp b = (TypeTemp)B[i * J + j];

			a = a / shrA;
			b = b / shrB;

			TypeTemp c = a / shrC + b / shrC;

			C[i * J + j] = Saturate<TypeC>(c / demote);
		}
	}
	return;
}
template<class TypeA, class TypeB, class TypeTemp, class TypeC>
void MatAddCC(const TypeA* A, const TypeB* B, TypeC* C, MYINT I, MYINT J, MYINT shrA, MYINT shrB, MYINT shrC, MYINT demote) {
	for (MYITE i = 0; i < I; i++) {
		for (MYITE j = 0; j < J; j++) {
			TypeTemp a = (TypeTemp)A[i * J + j];
			TypeTemp b = (TypeTemp)B[i * J + j];

			a = a / shrA;
			b = b / shrB;

			TypeTemp c = a / shrC + b / shrC;

			C[i * J + j] = Saturate<TypeC>(c / demote);
		}
	}
	return;
}

template<class TypeA, class TypeB, class TypeTemp, class TypeC>
void MatAddBroadCastA(TypeA* A, TypeB* B, TypeC* C, MYINT I, MYINT J, MYINT shrA, MYINT shrB, MYINT shrC, MYINT demote) {
	for (MYITE i = 0; i < I; i++) {
		for (MYITE j = 0; j < J; j++) {
			TypeTemp a = (TypeTemp)* A;
			TypeTemp b = (TypeTemp)B[i * J + j];

			a = a / shrA;
			b = b / shrB;

			TypeTemp c = a / shrC + b / shrC;

			C[i * J + j] = Saturate<TypeC>(c / demote);
		}
	}
	return;
}
template<class TypeA, class TypeB, class TypeTemp, class TypeC>
void MatAddBroadCastB(TypeA* A, TypeB* B, TypeC* C, MYINT I, MYINT J, MYINT shrA, MYINT shrB, MYINT shrC, MYINT demote) {
	for (MYITE i = 0; i < I; i++) {
		for (MYITE j = 0; j < J; j++) {
			TypeTemp a = (TypeTemp)A[i * J + j];
			TypeTemp b = (TypeTemp)* B;

			a = a / shrA;
			b = b / shrB;

			TypeTemp c = a / shrC + b / shrC;

			C[i * J + j] = Saturate<TypeC>(c / demote);
		}
	}
	return;
}

template<class TypeA, class TypeB, class TypeTemp, class TypeC>
// TODO: shrB is int32_t because in 8-bit/16-bit code, shrB is usually very high and int8_t/int16_t will overflow.
void MatSub(TypeA* A, const TypeB* B, TypeC* C, MYINT I, MYINT J, MYINT shrA, int32_t shrB, MYINT shrC, MYINT demote) {
	for (MYITE i = 0; i < I; i++) {
		for (MYITE j = 0; j < J; j++) {
			TypeTemp a = (TypeTemp)A[i * J + j];
			TypeTemp b = (TypeTemp)B[i * J + j];

			a = a / shrA;
			b = b / shrB;

			TypeTemp c = a / shrC - b / shrC;

			C[i * J + j] = Saturate<TypeC>(c / demote);
		}
	}
	return;
}
template<class TypeA, class TypeB, class TypeTemp, class TypeC>
void MatSubBroadCastA(TypeA* A, TypeB* B, TypeC* C, MYINT I, MYINT J, MYINT shrA, MYINT shrB, MYINT shrC, MYINT demote) {
	for (MYITE i = 0; i < I; i++) {
		for (MYITE j = 0; j < J; j++) {
			TypeTemp a = (TypeTemp)* A;
			TypeTemp b = (TypeTemp)B[i * J + j];

			a = a / shrA;
			b = b / shrB;

			TypeTemp c = a / shrC - b / shrC;

			C[i * J + j] = Saturate<TypeC>(c / demote);
		}
	}
	return;
}
template<class TypeA, class TypeB, class TypeTemp, class TypeC>
void MatSubBroadCastB(TypeA* A, TypeB* B, TypeC* C, MYINT I, MYINT J, MYINT shrA, MYINT shrB, MYINT shrC, MYINT demote) {
	for (MYITE i = 0; i < I; i++) {
		for (MYITE j = 0; j < J; j++) {
			TypeTemp a = (TypeTemp)A[i * J + j];
			TypeTemp b = (TypeTemp)* B;

			a = a / shrA;
			b = b / shrB;

			TypeTemp c = a / shrC - b / shrC;

			C[i * J + j] = Saturate<TypeC>(c / demote);
		}
	}
	return;
}

template<class TypeA, class TypeB, class TypeTemp, class TypeC>
void MatMulNN(TypeA* A, TypeB* B, TypeC* C, TypeTemp* tmp, MYINT I, MYINT K, MYINT J, MYINT shrA, MYINT shrB, MYINT H1, MYINT H2, MYINT demote) {
	for (MYITE i = 0; i < I; i++) {
		for (MYITE j = 0; j < J; j++) {
			for (MYITE k = 0; k < K; k++) {
				TypeTemp a = (TypeTemp)A[i * K + k];
				TypeTemp b = (TypeTemp)B[k * J + j];

				TypeTemp prod = a * b;

				tmp[k] = prod;
			}

			MYITE count = K, depth = 0;
			bool shr = true;

			while (depth < (H1 + H2)) {
				if (depth >= H1)
					shr = false;

				for (MYITE p = 0; p < (K / 2 + 1); p++) {
					TypeTemp sum;
					if (p < (count >> 1)) {
						if (shr)
							sum = tmp[2 * p] / 2 + tmp[(2 * p) + 1] / 2;
						else
							sum = tmp[2 * p] + tmp[(2 * p) + 1];
					}
					else if ((p == (count >> 1)) && ((count & 1) == 1)) {
						if (shr)
							sum = tmp[2 * p] / 2;
						else
							sum = tmp[2 * p];
					}
					else
						sum = 0;

					tmp[p] = sum;
				}
				count = (count + 1) >> 1;

				depth++;
			}

			C[i * J + j] = Saturate<TypeC>(((tmp[0] / shrA) / shrB) / demote);
		}
	}
	return;
}
template<class TypeA, class TypeB, class TypeTemp, class TypeC>
void MatMulCN(const TypeA* A, TypeB* B, TypeC* C, TypeTemp* tmp, MYINT I, MYINT K, MYINT J, MYINT shrA, MYINT shrB, MYINT H1, MYINT H2, MYINT demote) {
	for (MYITE i = 0; i < I; i++) {
		for (MYITE j = 0; j < J; j++) {
			for (MYITE k = 0; k < K; k++) {
				TypeTemp a = (TypeTemp)A[i * K + k];
				TypeTemp b = (TypeTemp)B[k * J + j];

				TypeTemp prod = a * b;

				tmp[k] = prod;
			}

			MYITE count = K, depth = 0;
			bool shr = true;

			while (depth < (H1 + H2)) {
				if (depth >= H1)
					shr = false;

				for (MYITE p = 0; p < (K / 2 + 1); p++) {
					TypeTemp sum;
					if (p < (count >> 1)) {
						if (shr)
							sum = tmp[2 * p] / 2 + tmp[(2 * p) + 1] / 2;
						else
							sum = tmp[2 * p] + tmp[(2 * p) + 1];
					}
					else if ((p == (count >> 1)) && ((count & 1) == 1)) {
						if (shr)
							sum = tmp[2 * p] / 2;
						else
							sum = tmp[2 * p];
					}
					else
						sum = 0;

					tmp[p] = sum;
				}
				count = (count + 1) >> 1;

				depth++;
			}

			C[i * J + j] = Saturate<TypeC>(((tmp[0] / shrA) / shrB) / demote);
		}
	}
	return;
}
template<class TypeA, class TypeB, class TypeTemp, class TypeC>
void MatMulNC(TypeA* A, const TypeB* B, TypeC* C, TypeTemp* tmp, MYINT I, MYINT K, MYINT J, MYINT shrA, MYINT shrB, MYINT H1, MYINT H2, MYINT demote) {
	for (MYITE i = 0; i < I; i++) {
		for (MYITE j = 0; j < J; j++) {
			for (MYITE k = 0; k < K; k++) {
				TypeTemp a = (TypeTemp)A[i * K + k];
				TypeTemp b = (TypeTemp)B[k * J + j];

				TypeTemp prod = a * b;

				tmp[k] = prod;
			}

			MYITE count = K, depth = 0;
			bool shr = true;

			while (depth < (H1 + H2)) {
				if (depth >= H1)
					shr = false;

				for (MYITE p = 0; p < (K / 2 + 1); p++) {
					TypeTemp sum;
					if (p < (count >> 1)) {
						if (shr)
							sum = tmp[2 * p] / 2 + tmp[(2 * p) + 1] / 2;
						else
							sum = tmp[2 * p] + tmp[(2 * p) + 1];
					}
					else if ((p == (count >> 1)) && ((count & 1) == 1)) {
						if (shr)
							sum = tmp[2 * p] / 2;
						else
							sum = tmp[2 * p];
					}
					else
						sum = 0;

					tmp[p] = sum;
				}
				count = (count + 1) >> 1;

				depth++;
			}

			C[i * J + j] = Saturate<TypeC>(((tmp[0] / shrA) / shrB) / demote);
		}
	}
	return;
}
template<class TypeA, class TypeB, class TypeTemp, class TypeC>
void MatMulCC(const TypeA* A, const TypeB* B, TypeC* C, TypeTemp* tmp, MYINT I, MYINT K, MYINT J, MYINT shrA, MYINT shrB, MYINT H1, MYINT H2, MYINT demote) {
	for (MYITE i = 0; i < I; i++) {
		for (MYITE j = 0; j < J; j++) {
			for (MYITE k = 0; k < K; k++) {
				TypeTemp a = (TypeTemp)A[i * K + k];
				TypeTemp b = (TypeTemp)B[k * J + j];

				TypeTemp prod = a * b;

				tmp[k] = prod;
			}

			MYITE count = K, depth = 0;
			bool shr = true;

			while (depth < (H1 + H2)) {
				if (depth >= H1)
					shr = false;

				for (MYITE p = 0; p < (K / 2 + 1); p++) {
					TypeTemp sum;
					if (p < (count >> 1)) {
						if (shr)
							sum = tmp[2 * p] / 2 + tmp[(2 * p) + 1] / 2;
						else
							sum = tmp[2 * p] + tmp[(2 * p) + 1];
					}
					else if ((p == (count >> 1)) && ((count & 1) == 1)) {
						if (shr)
							sum = tmp[2 * p] / 2;
						else
							sum = tmp[2 * p];
					}
					else
						sum = 0;

					tmp[p] = sum;
				}
				count = (count + 1) >> 1;

				depth++;
			}

			C[i * J + j] = Saturate<TypeC>(((tmp[0] / shrA) / shrB) / demote);
		}
	}
	return;
}

template<class TypeA, class TypeAidx, class TypeB, class TypeTemp, class TypeC>
void SparseMatMul(const TypeAidx* Aidx, const TypeA* Aval, TypeB** B, TypeC* C, int16_t K, MYINT shrA, MYINT shrB, MYINT shrC, MYINT demote) {

	MYITE ite_idx = 0, ite_val = 0;
	for (MYITE k = 0; k < K; k++) {
		// MYINT b = getIntFeature(k);
		TypeTemp b = (TypeTemp)B[k * 1][0];
		//b = b / shrB;

		MYITE idx = Aidx[ite_idx];
		while (idx != 0) {
			TypeTemp a = (TypeTemp)Aval[ite_val];
			//a = a / shrA;
			TypeTemp c = (TypeTemp)(a * b);
			//c = c / shrC;

			C[idx - 1] += Saturate<TypeC>((((c / shrA) / shrB) / shrC) / demote);

			ite_idx++;
			ite_val++;

			idx = Aidx[ite_idx];
		}
		ite_idx++;
	}

	return;
}
template<class TypeA, class TypeB, class TypeTemp, class TypeC>
void MulCir(TypeA* A, TypeB* B, TypeC* C, MYINT I, MYINT J, MYINT shrA, MYINT shrB, MYINT demote) {
	for (MYITE i = 0; i < I; i++) {
		for (MYITE j = 0; j < J; j++) {
			TypeTemp a = (TypeTemp)A[i * J + j];
			TypeTemp b = (TypeTemp)B[i * J + j];

			TypeTemp prod = a * b;
			C[i * J + j] = Saturate<TypeC>(((prod / shrA) / shrB) / demote);
		}
	}
	return;
}

template<class TypeA>
void Confidence(TypeA* A, float* confidence) {
	*confidence = *A;
	if (*confidence < 0)
		* confidence = -(*confidence);
}

template<class TypeA>
void Confidence(TypeA* A, MYINT I, MYINT J, MYITE* index, float* confidence) {
	TypeA max = A[0];
	TypeA min = A[0];
	MYITE maxIndex = 0, counter = 0;
	float sum = 0;
	for (MYITE i = 0; i < I; i++) {
		for (MYITE j = 0; j < J; j++) {
			TypeA x = A[i * J + j];
			//sum += x;
			if (max < x) {
				maxIndex = counter;
				max = x;
			}
			if (min > x) {
				min = x;
			}
			counter++;
		}
	}

	for (MYITE i = 0; i < I; i++) {
		for (MYITE j = 0; j < J; j++) {
			sum += (A[i * J + j] - min);
		}
	}

	*index = maxIndex;
	if (sum < 0.0001 && sum > -0.0001)
		* confidence = ((float)1) / (I * J); //Maybe could penalise more as this is a underflow
	else
		*confidence = (float)(A[*index]-min) / (sum);
	return;
}


template<class TypeA>
void ArgMax(TypeA* A, MYINT I, MYINT J, MYITE* index) {
	TypeA max = A[0];
	MYITE maxIndex = 0, counter = 0;
	for (MYITE i = 0; i < I; i++) {
		for (MYITE j = 0; j < J; j++) {
			TypeA x = A[i * J + j];

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
void Transpose(TypeA* A, TypeA* B, MYINT I, MYINT J) {
	for (MYITE i = 0; i < I; i++) {
		for (MYITE j = 0; j < J; j++) {
			B[i * J + j] = A[j * I + i];
		}
	}
	return;
}

template<class TypeA, class TypeB, class TypeTemp, class TypeC>
void ScalarMul(TypeA* A, TypeB* B, TypeC* C, MYINT I, MYINT J, MYINT shrA, MYINT shrB, int demote) {
	TypeTemp a = (TypeTemp)* A;
	for (MYITE i = 0; i < I; i++) {
		for (MYITE j = 0; j < J; j++) {
			TypeTemp b = (TypeTemp)B[i * J + j];

			TypeTemp prod = a * b;
			C[i * J + j] = Saturate<TypeC>(((prod / shrA) / shrB) / demote);
		}
	}

	return;
}

template<class TypeA, class TypeB, class TypeTemp, class TypeC>
void AddOrSubCir4D(TypeA* A, const TypeB* B, TypeC* X, MYINT N, MYINT H, MYINT W, MYINT C, MYINT shrA, MYINT shrB, MYINT shrC, bool add, MYINT demote) {
	for (MYITE n = 0; n < N; n++) {
		for (MYITE h = 0; h < H; h++) {
			for (MYITE w = 0; w < W; w++) {
				for (MYITE c = 0; c < C; c++) {
					TypeTemp a = (TypeTemp)A[n * H * W * C + h * W * C + w * C + c];
					a = a / shrA;

					TypeTemp b = (TypeTemp)B[c];
					b = b / shrB;

					TypeTemp res;
					if (add)
						res = a / shrC + b / shrC;
					else
						res = a / shrC - b / shrC;
					X[n * H * W * C + h * W * C + w * C + c] = Saturate<TypeC>(res / demote);
				}
			}
		}
	}
	return;
}
template<class TypeA, class TypeB, class TypeTemp, class TypeC>
void AddOrSubCir2D(TypeA* A, const TypeB* B, TypeC* X, MYINT H, MYINT W, MYINT shrA, MYINT shrB, MYINT shrC, bool add, MYINT demote) {
	for (MYITE h = 0; h < H; h++) {
		for (MYITE w = 0; w < W; w++) {
			TypeTemp a = (TypeTemp)A[h * W + w];
			a = a / shrA;

			TypeTemp b = (TypeTemp)B[w];
			b = b / shrB;

			TypeTemp res;
			if (add)
				res = a / shrC + b / shrC;
			else
				res = a / shrC - b / shrC;
			X[h * W + w] = Saturate<TypeC>(res / demote);
		}
	}
	return;
}

template<class TypeA>
void Relu4D(TypeA* A, MYINT N, MYINT H, MYINT W, MYINT C) {
	for (MYITE n = 0; n < N; n++) {
		for (MYITE h = 0; h < H; h++) {
			for (MYITE w = 0; w < W; w++) {
				for (MYITE c = 0; c < C; c++) {
					TypeA a = A[n * H * W * C + h * W * C + w * C + c];
					if (a < 0)
						a = 0;
					A[n * H * W * C + h * W * C + w * C + c] = a;
				}
			}
		}
	}
	return;
}
template<class TypeA>
void Relu2D(TypeA* A, MYINT H, MYINT W) {
	for (MYITE h = 0; h < H; h++) {
		for (MYITE w = 0; w < W; w++) {
			TypeA a = A[h * W + w];
			if (a < 0)
				a = 0;
			A[h * W + w] = a;
		}
	}
	return;
}
template<class TypeA, class TypeB>
void Maxpool(TypeA* A, TypeB* B, MYINT N, MYINT H, MYINT W, MYINT C, MYINT stride, MYINT demote) {
	MYITE HO = H / stride;
	MYITE WO = W / stride;

	for (MYITE n = 0; n < N; n++) {
		for (MYITE ho = 0; ho < HO; ho++) {
			for (MYITE wo = 0; wo < WO; wo++) {
				for (MYITE c = 0; c < C; c++) {

					TypeA max = A[n * H * W * C + (stride * ho) * W * C + (stride * wo) * C + c];
					for (MYITE hs = 0; hs < stride; hs++) {
						for (MYITE ws = 0; ws < stride; ws++) {
							TypeA a = A[n * H * W * C + ((stride * ho) + hs) * W * C + ((stride * wo) + ws) * C + c];
							if (a > max)
								max = a;
						}
					}

					B[n * HO * WO * C + ho * WO * C + wo * C + c] = (TypeB)(max / demote);
				}
			}
		}
	}
	return;
}

//shrB overflows int16_t
template<class TypeA, class TypeB>
void Exp(TypeA* A, MYINT I, MYINT J, MYINT shrA, int32_t shrB, TypeB* B, MYINT demote) {
	for (MYITE i = 0; i < I; i++) {
		for (MYITE j = 0; j < J; j++) {
			B[i * J + j] = (TypeB)((exp(((float)A[i * J + j]) / shrA) * shrB) / demote);
		}
	}
	return;
}

const int8_t expTable8[128] = {64, 60, 56, 53, 50, 47, 44, 41, 39, 36, 34, 32, 30, 28, 27, 25, 24, 22, 21, 20, 18, 17, 16, 15, 14, 13, 13, 12, 11, 10, 10, 9, 9, 8, 8, 7, 7, 6, 6, 6, 5, 5, 5, 4, 4, 4, 4, 3, 3, 3, 3, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
template<class TypeB>
inline TypeB expBase8(int8_t A, MYINT adjust) {
	int8_t val = (A == -128) ? 127 : -A;
	if(val < 0) {
		val = 127;
	}
	return (TypeB) (expTable8[val] * adjust);
}

const int16_t expTable16A[256] = {16384, 15391, 14459, 13583, 12760, 11987, 11261, 10578, 9937, 9335, 8770, 8238, 7739, 7270, 6830, 6416, 6027, 5662, 5319, 4997, 4694, 4410, 4143, 3892, 3656, 3434, 3226, 3031, 2847, 2675, 2513, 2360, 2217, 2083, 1957, 1838, 1727, 1622, 1524, 1432, 1345, 1263, 1187, 1115, 1047, 984, 924, 868, 816, 766, 720, 676, 635, 597, 561, 527, 495, 465, 437, 410, 385, 362, 340, 319, 300, 282, 265, 249, 234, 220, 206, 194, 182, 171, 161, 151, 142, 133, 125, 118, 110, 104, 97, 92, 86, 81, 76, 71, 67, 63, 59, 56, 52, 49, 46, 43, 41, 38, 36, 34, 32, 30, 28, 26, 25, 23, 22, 20, 19, 18, 17, 16, 15, 14, 13, 12, 12, 11, 10, 10, 9, 9, 8, 8, 7, 7, 6, 6, 5, 5, 5, 5, 4, 4, 4, 4, 3, 3, 3, 3, 3, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
const int16_t expTable16B[128] = {16384, 16376, 16368, 16360, 16352, 16344, 16336, 16328, 16320, 16312, 16304, 16296, 16288, 16280, 16272, 16264, 16256, 16249, 16241, 16233, 16225, 16217, 16209, 16201, 16193, 16185, 16177, 16169, 16162, 16154, 16146, 16138, 16130, 16122, 16114, 16106, 16099, 16091, 16083, 16075, 16067, 16059, 16051, 16044, 16036, 16028, 16020, 16012, 16004, 15997, 15989, 15981, 15973, 15965, 15958, 15950, 15942, 15934, 15927, 15919, 15911, 15903, 15895, 15888, 15880, 15872, 15864, 15857, 15849, 15841, 15833, 15826, 15818, 15810, 15803, 15795, 15787, 15779, 15772, 15764, 15756, 15749, 15741, 15733, 15726, 15718, 15710, 15703, 15695, 15687, 15680, 15672, 15664, 15657, 15649, 15641, 15634, 15626, 15618, 15611, 15603, 15596, 15588, 15580, 15573, 15565, 15558, 15550, 15542, 15535, 15527, 15520, 15512, 15504, 15497, 15489, 15482, 15474, 15467, 15459, 15452, 15444, 15437, 15429, 15421, 15414, 15406, 15399};
template<class TypeB>
inline TypeB expBase16(int16_t A, MYINT adjust) {
	int16_t val = (A == -32768) ? 32767 : -A;
	int16_t val1 = val % 128;
	int16_t val2 = val / 128;
	int32_t res = expTable16A[val2] * expTable16B[val1];
	return (TypeB) (res / (16384 * adjust));
}

template<class TypeB>
void ExpNew8(int8_t *A, MYINT I, MYINT J, MYINT adjust, TypeB *B) {
	for (MYITE i = 0; i < I; i++) {
		for (MYITE j = 0; j < J; j++) {
			B[i * J + j] = expBase8<TypeB>(A[i * J + j], adjust);
		}
	}
	return;
}

template<class TypeB>
void ExpNew16(int16_t *A, MYINT I, MYINT J, MYINT adjust, TypeB *B) {
	for (MYITE i = 0; i < I; i++) {
		for (MYITE j = 0; j < J; j++) {
			B[i * J + j] = expBase16<TypeB>(A[i * J + j], adjust);
		}
	}
	return;
}


template<class TypeA>
void Sigmoid(TypeA* A, MYINT I, MYINT J, MYINT div, MYINT add, MYINT sigmoid_limit, MYINT scale_in, MYINT scale_out, TypeA* B) {
	TypeA scale_diff = scale_out / scale_in;
	for (MYITE i = 0; i < I; i++) {
		for (MYITE j = 0; j < J; j++) {
		#ifdef FLOATEXP
			float x = float(A[i * J + j]) / scale_in;

			float y = 1 / (1 + exp(-x));

			TypeA z = (TypeA)(y * scale_out);

			B[i * J + j] = z;
		#else
			TypeA x = A[i * J + j];
			x = (x / div) + add;
			TypeA y;
			if (x >= sigmoid_limit)
				y = sigmoid_limit;
			else if (x <= 0)
				y = 0;
			else
				y = x;
			y = y * scale_diff;
			B[i * J + j] = y;
		#endif
		}
	}

	return;
}
// Integer sigmoid using new table exponentiation
template<int dummy>
void SigmoidNew8(int8_t* A, MYINT I, MYINT J, int8_t* B) {
	for (MYITE i = 0; i < I; i++) {
		for (MYITE j = 0; j < J; j++) {
			int8_t a = A[i * J + j];
			if (a <= 0) {
				int8_t b = expBase8<int8_t>(a, 1);
				B[i * J + j] = (int8_t)((64 * (int16_t)b) / ((int16_t)b + (int16_t)64));
			} else {
				B[i * J + j] = (int8_t)(((int16_t)4096) / ((int16_t)64 + (int16_t)expBase8<int8_t>(-a, 1)));
			}
			
		}
	}
	return;
}
template<int dummy>
void SigmoidNew16(int16_t* A, MYINT I, MYINT J, int16_t* B) {
	for (MYITE i = 0; i < I; i++) {
		for (MYITE j = 0; j < J; j++) {
			int16_t a = A[i * J + j];
			if (a <= 0) {
				int16_t b = expBase16<int16_t>(a, 1);
				B[i * J + j] = (int16_t)((16384 * (int32_t)b) / ((int32_t)b + (int32_t)16384));
			} else {
				B[i * J + j] = (int16_t)(((int32_t)267943936L) / ((int32_t)16384 + (int32_t)expBase16<int16_t>(-a, 1)));
			}
			
		}
	}
	return;
}

template<class TypeA>
void TanH(TypeA* A, MYINT I, MYINT J, TypeA scale_in, TypeA scale_out, TypeA* B) {
	for (MYITE i = 0; i < I; i++) {
		for (MYITE j = 0; j < J; j++) {
		#ifdef FLOATEXP
			float x = float(A[i * J + j]) / scale_in;

			float y = tanh(x);

			MYINT z = (TypeA)(y * scale_out);

			B[i * J + j] = z;
		#else
			TypeA x = A[i * J + j], y;
			if (x >= scale_in)
				y = scale_in;
			else if (x <= -scale_in)
				y = -scale_in;
			else
				y = x;
			TypeA scale_diff = scale_out / scale_in;
			y *= scale_diff;
			B[i * J + j] = y;
		#endif
		}
	}
	return;
}
// Integer TanH using new table exponentiation
template<int dummy>
void TanHNew8(int8_t* A, MYINT I, MYINT J, int8_t* B) {
	for (MYITE i = 0; i < I; i++) {
		for (MYITE j = 0; j < J; j++) {
			int8_t a = A[i * J + j];
			if (a <= 0) {
				int16_t b = expBase8<int8_t>(2*a, 1);
				B[i * J + j] = (int8_t)( (((int16_t)64)*(b - 64)) / (b + 64));
			} else {
				int16_t b = expBase8<int8_t>(-2*a, 1);
				B[i * J + j] = (int8_t)( (((int16_t)64)*(64 - b)) / (b + 64));
			}
			
		}
	}
	return;
}
template<int dummy>
void TanHNew16(int16_t* A, MYINT I, MYINT J, int16_t* B) {
	for (MYITE i = 0; i < I; i++) {
		for (MYITE j = 0; j < J; j++) {
			int16_t a = A[i * J + j];
			if (a <= 0) {
				int32_t b = expBase16<int16_t>(2*a, 1);
				B[i * J + j] = (int16_t)( (((int32_t)16384)*(b - 16384)) / (b + 16384));
			} else {
				int32_t b = expBase16<int16_t>(-2*a, 1);
				B[i * J + j] = (int16_t)( (((int32_t)16384)*(16384 - b)) / (b + 16384));
			}
			
		}
	}
	return;
}


template<class TypeA>
void AdjustScaleShr(TypeA* A, MYINT I, MYINT J, MYINT scale) {
	for (MYITE i = 0; i < I; i++) {
		for (MYITE j = 0; j < J; j++) {
			TypeA a = A[i * J + j];
			A[i * J + j] = a / scale;
		}
	}
	return;
}
template<class TypeA>
void AdjustScaleShl(TypeA* A, MYINT I, MYINT J, MYINT scale) {
	for (MYITE i = 0; i < I; i++) {
		for (MYITE j = 0; j < J; j++) {
			TypeA a = A[i * J + j];
			A[i * J + j] = a * scale;
		}
	}
	return;
}
template<class TypeA>
void AdjustScaleShlSaturate(TypeA* A, MYINT I, MYINT J, MYINT scale, MYINT saturate) {
	for (MYITE i = 0; i < I; i++) {
		for (MYITE j = 0; j < J; j++) {
			TypeA a = A[i * J + j];
			a = (a < saturate && a > -saturate) ? a : (a > 0 ? saturate : -saturate);
			A[i * J + j] = a * scale;
		}
	}
	return;
}
