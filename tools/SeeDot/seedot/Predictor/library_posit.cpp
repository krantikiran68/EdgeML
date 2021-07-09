// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include <iostream>
#include <cmath>

#include "datatypes.h"
#include "library_posit.h"
#include "profile.h"


// This file contains floating point implementations of operations supported by SeeDot.

// C = A + B
void MatAddNN(posit8_t* A, posit8_t* B, posit8_t* C, MYITE I, MYITE J, MYINT shrA, MYINT shrB, MYINT shrC) {
	for (MYITE i = 0; i < I; i++) {
		for (MYITE j = 0; j < J; j++) {
			posit8_t a = A[i * J + j];
			posit8_t b = B[i * J + j];

			posit8_t c = p8_add(a, b);

			C[i * J + j] = c;
		}
	}
	return;
}

// C = A + B
void MatAddCN(const posit8_t* A, posit8_t* B, posit8_t* C, MYITE I, MYITE J, MYINT shrA, MYINT shrB, MYINT shrC) {
	for (MYITE i = 0; i < I; i++) {
		for (MYITE j = 0; j < J; j++) {
			posit8_t a = A[i * J + j];
			posit8_t b = B[i * J + j];

			posit8_t c = p8_add(a, b);

			C[i * J + j] = c;
		}
	}
	return;
}

// C = A + B
void MatAddNC(posit8_t* A, const posit8_t* B, posit8_t* C, MYITE I, MYITE J, MYINT shrA, MYINT shrB, MYINT shrC) {
	for (MYITE i = 0; i < I; i++) {
		for (MYITE j = 0; j < J; j++) {
			posit8_t a = A[i * J + j];
			posit8_t b = B[i * J + j];

			posit8_t c = p8_add(a, b);

			C[i * J + j] = c;
		}
	}
	return;
}

// C = A + B
void MatAddCC(const posit8_t* A, const posit8_t* B, posit8_t* C, MYITE I, MYITE J, MYINT shrA, MYINT shrB, MYINT shrC) {
	for (MYITE i = 0; i < I; i++) {
		for (MYITE j = 0; j < J; j++) {
			posit8_t a = A[i * J + j];
			posit8_t b = B[i * J + j];

			posit8_t c = p8_add(a, b);

			C[i * J + j] = c;
		}
	}
	return;
}

// C = a + B
void MatAddBroadCastA(posit8_t* A, posit8_t* B, posit8_t* C, MYITE I, MYITE J, MYINT shrA, MYINT shrB, MYINT shrC) {
	for (MYITE i = 0; i < I; i++) {
		for (MYITE j = 0; j < J; j++) {
			posit8_t a = *A;
			posit8_t b = B[i * J + j];

			posit8_t c = p8_add(a, b);

			C[i * J + j] = c;
		}
	}
	return;
}

// C = A + b
void MatAddBroadCastB(posit8_t* A, posit8_t* B, posit8_t* C, MYITE I, MYITE J, MYINT shrA, MYINT shrB, MYINT shrC) {
	for (MYITE i = 0; i < I; i++) {
		for (MYITE j = 0; j < J; j++) {
			posit8_t a = A[i * J + j];
			posit8_t b = *B;

			posit8_t c = p8_add(a, b);

			C[i * J + j] = c;
		}
	}
	return;
}

// C = A + B
void MatAdd4(posit8_t* A, posit8_t* B, posit8_t* X, MYITE N, MYITE H, MYITE W, MYITE C, MYINT shrA, MYINT shrB, MYINT shrC) {
	for (MYITE n = 0; n < N; n++) {
		for (MYITE h = 0; h < H; h++) {
			for (MYITE w = 0; w < W; w++) {
				for (MYITE c = 0; c < C; c++) {
					posit8_t a = A[n * H * W * C + h * W * C + w * C + c];
					posit8_t b = B[n * H * W * C + h * W * C + w * C + c];

					posit8_t x = p8_add(a, b);

					X[n * H * W * C + h * W * C + w * C + c] = x;
				}
			}
		}
	}
	return;
}

// C = A - B
void MatSub(posit8_t* A, const posit8_t* B, posit8_t* C, MYITE I, MYITE J, MYINT shrA, MYINT shrB, MYINT shrC) {
	for (MYITE i = 0; i < I; i++) {
		for (MYITE j = 0; j < J; j++) {
			posit8_t a = A[i * J + j];
			posit8_t b = B[i * J + j];

			posit8_t c = p8_sub(a, b);

			C[i * J + j] = c;
		}
	}
	return;
}

// C = a - B
void MatSubBroadCastA(posit8_t* A, posit8_t* B, posit8_t* C, MYITE I, MYITE J, MYINT shrA, MYINT shrB, MYINT shrC) {
	for (MYITE i = 0; i < I; i++) {
		for (MYITE j = 0; j < J; j++) {
			posit8_t a = *A;
			posit8_t b = B[i * J + j];

			posit8_t c = p8_sub(a, b);

			C[i * J + j] = c;
		}
	}
	return;
}

// C = A - b
void MatSubBroadCastB(posit8_t* A, posit8_t* B, posit8_t* C, MYITE I, MYITE J, MYINT shrA, MYINT shrB, MYINT shrC) {
	for (MYITE i = 0; i < I; i++) {
		for (MYITE j = 0; j < J; j++) {
			posit8_t a = A[i * J + j];
			posit8_t b = *B;

			posit8_t c = p8_sub(a, b);

			C[i * J + j] = c;
		}
	}
	return;
}

// C = A * B
void MatMulNN(posit8_t* A, posit8_t* B, posit8_t* C, MYITE I, MYITE K, MYITE J, MYINT shrA, MYINT shrB, MYITE H1, MYITE H2) {
	quire8_t qz;
	for (MYITE i = 0; i < I; i++) {
		for (MYITE j = 0; j < J; j++) {
			qz = q8_clr(qz);
			for (MYITE k = 0; k < K; k++) {
				posit8_t a = A[i * K + k];
				posit8_t b = B[k * J + j];
				qz = q8_fdp_add(qz, a, b);
				// tmp[k] = p8_mul(a, b);
			}

			// MYITE count = K, depth = 0;
			// bool shr = true;

			// while (depth < (H1 + H2)) {
			// 	if (depth >= H1) {
			// 		shr = false;
			// 	}

			// 	for (MYITE p = 0; p < (K / 2 + 1); p++) {
			// 		posit8_t sum;
			// 		if (p < (count >> 1)) {
			// 			sum = p8_add(tmp[2 * p], tmp[(2 * p) + 1]);
			// 		} else if ((p == (count >> 1)) && ((count & 1) == 1)) {
			// 			sum = tmp[2 * p];
			// 		} else {
			// 			sum = convertDoubleToP8(0.0);
			// 		}

			// 		if (shr) {
			// 			tmp[p] = sum;
			// 		} else {
			// 			tmp[p] = sum;
			// 		}
			// 	}

			// 	count = (count + 1) >> 1;
			// 	depth++;
			// }

			C[i * J + j] = q8_to_p8(qz);
		}
	}
	return;
}

// C = A * B
void MatMulCN(const posit8_t* A, posit8_t* B, posit8_t* C, MYITE I, MYITE K, MYITE J, MYINT shrA, MYINT shrB, MYITE H1, MYITE H2) {
	quire8_t qz;
	for (MYITE i = 0; i < I; i++) {
		for (MYITE j = 0; j < J; j++) {
			qz = q8_clr(qz);
			for (MYITE k = 0; k < K; k++) {
				posit8_t a = A[i * K + k];
				posit8_t b = B[k * J + j];
				qz = q8_fdp_add(qz, a, b);
				// tmp[k] = p8_mul(a, b);
			}

			// MYITE count = K, depth = 0;
			// bool shr = true;

			// while (depth < (H1 + H2)) {
			// 	if (depth >= H1) {
			// 		shr = false;
			// 	}

			// 	for (MYITE p = 0; p < (K / 2 + 1); p++) {
			// 		posit8_t sum;
			// 		if (p < (count >> 1)) {
			// 			sum = p8_add(tmp[2 * p], tmp[(2 * p) + 1]);
			// 		} else if ((p == (count >> 1)) && ((count & 1) == 1)) {
			// 			sum = tmp[2 * p];
			// 		} else {
			// 			sum = convertDoubleToP8(0.0);
			// 		}

			// 		if (shr) {
			// 			tmp[p] = sum;
			// 		} else {
			// 			tmp[p] = sum;
			// 		}
			// 	}

			// 	count = (count + 1) >> 1;
			// 	depth++;
			// }

			C[i * J + j] = q8_to_p8(qz);
		}
	}
	return;
}

// C = A * B
void MatMulNC(posit8_t* A, const posit8_t* B, posit8_t* C, MYITE I, MYITE K, MYITE J, MYINT shrA, MYINT shrB, MYITE H1, MYITE H2) {
	quire8_t qz;
	for (MYITE i = 0; i < I; i++) {
		for (MYITE j = 0; j < J; j++) {
			qz = q8_clr(qz);
			for (MYITE k = 0; k < K; k++) {
				posit8_t a = A[i * K + k];
				posit8_t b = B[k * J + j];
				qz = q8_fdp_add(qz, a, b);
				// tmp[k] = p8_mul(a, b);
			}

			// MYITE count = K, depth = 0;
			// bool shr = true;

			// while (depth < (H1 + H2)) {
			// 	if (depth >= H1) {
			// 		shr = false;
			// 	}

			// 	for (MYITE p = 0; p < (K / 2 + 1); p++) {
			// 		posit8_t sum;
			// 		if (p < (count >> 1)) {
			// 			sum = p8_add(tmp[2 * p], tmp[(2 * p) + 1]);
			// 		} else if ((p == (count >> 1)) && ((count & 1) == 1)) {
			// 			sum = tmp[2 * p];
			// 		} else {
			// 			sum = convertDoubleToP8(0.0);
			// 		}

			// 		if (shr) {
			// 			tmp[p] = sum;
			// 		} else {
			// 			tmp[p] = sum;
			// 		}
			// 	}

			// 	count = (count + 1) >> 1;
			// 	depth++;
			// }

			C[i * J + j] = q8_to_p8(qz);
		}
	}
	return;
}

// C = A * B
void MatMulCC(const posit8_t* A, const posit8_t* B, posit8_t* C, MYITE I, MYITE K, MYITE J, MYINT shrA, MYINT shrB, MYITE H1, MYITE H2) {
	quire8_t qz;
	for (MYITE i = 0; i < I; i++) {
		for (MYITE j = 0; j < J; j++) {
			qz = q8_clr(qz);
			for (MYITE k = 0; k < K; k++) {
				posit8_t a = A[i * K + k];
				posit8_t b = B[k * J + j];
				qz = q8_fdp_add(qz, a, b);
				// tmp[k] = p8_mul(a, b);
			}

			// MYITE count = K, depth = 0;
			// bool shr = true;

			// while (depth < (H1 + H2)) {
			// 	if (depth >= H1) {
			// 		shr = false;
			// 	}

			// 	for (MYITE p = 0; p < (K / 2 + 1); p++) {
			// 		posit8_t sum;
			// 		if (p < (count >> 1)) {
			// 			sum = p8_add(tmp[2 * p], tmp[(2 * p) + 1]);
			// 		} else if ((p == (count >> 1)) && ((count & 1) == 1)) {
			// 			sum = tmp[2 * p];
			// 		} else {
			// 			sum = convertDoubleToP8(0.0);
			// 		}

			// 		if (shr) {
			// 			tmp[p] = sum;
			// 		} else {
			// 			tmp[p] = sum;
			// 		}
			// 	}

			// 	count = (count + 1) >> 1;
			// 	depth++;
			// }

			C[i * J + j] = q8_to_p8(qz);
		}
	}
	return;
}

// C = A |*| B
void SparseMatMulX(const MYINT* Aidx, const posit8_t* Aval, posit8_t** B, posit8_t* C, int16_t K, MYINT shrA, MYINT shrB, MYINT shrC) {
	MYITE ite_idx = 0, ite_val = 0;
	for (MYITE k = 0; k < K; k++) {
		posit8_t b = B[k * 1][0];

		MYINT idx = Aidx[ite_idx];
		while (idx != 0) {
			posit8_t a = Aval[ite_val];

			posit8_t c = p8_mul(a, b);

			C[idx - 1] = p8_add(C[idx - 1], c);

			ite_idx++;
			ite_val++;

			idx = Aidx[ite_idx];
		}
		ite_idx++;
	}
	return;
}

// C = A |*| B
void SparseMatMul(const MYINT* Aidx, const posit8_t* Aval, posit8_t* B, posit8_t* C, int16_t K, MYINT shrA, MYINT shrB, MYINT shrC) {
	MYITE ite_idx = 0, ite_val = 0;
	for (MYITE k = 0; k < K; k++) {
		posit8_t b = B[k];

		MYINT idx = Aidx[ite_idx];
		while (idx != 0) {
			posit8_t a = Aval[ite_val];

			posit8_t c = p8_mul(a, b);

			C[idx - 1] = p8_add(C[idx - 1], c);

			ite_idx++;
			ite_val++;

			idx = Aidx[ite_idx];
		}
		ite_idx++;
	}

	return;
}

// C = A <*> B
void MulCir(posit8_t* A, posit8_t* B, posit8_t* C, MYITE I, MYITE J, MYINT shrA, MYINT shrB) {
	for (MYITE i = 0; i < I; i++) {
		for (MYITE j = 0; j < J; j++) {
			posit8_t a = A[i * J + j];
			posit8_t b = B[i * J + j];

			C[i * J + j] = p8_mul(a, b);
		}
	}
	return;
}

// A = tanh(A)
void TanH(posit8_t* A, MYITE I, MYITE J, float scale_in, float scale_out, posit8_t* B) {
	for (MYITE i = 0; i < I; i++) {
		for (MYITE j = 0; j < J; j++) {
			double x = convertP8ToDouble(A[i * J + j]), y;

			#ifdef FLOATEXP
				y = tanh(x);
			#else
				y = x > -1 ? x : -1;
				y = y < 1 ? y : 1;
			#endif

			B[i * J + j] = convertDoubleToP8(y);
		}
	}
	return;
}

// B = reverse(A, axis)
void Reverse2(posit8_t* A, MYITE axis, MYITE I, MYITE J, posit8_t* B) {
	for (MYITE i = 0; i < I; i++) {
		for (MYITE j = 0; j < J; j++) {
			MYITE i_prime = (axis == 0 ? (I - 1 - i) : i);
			MYITE j_prime = (axis == 1 ? (J - 1 - j) : j);

			B[i * J + j] = A[i_prime*J + j_prime];
		}
	}
	return;
}

// index = argmax(A)
void ArgMax(posit8_t* A, MYITE I, MYITE J, int* index) {
	posit8_t max = A[0];
	MYITE maxIndex = 0, counter = 0;
	for (MYITE i = 0; i < I; i++) {
		for (MYITE j = 0; j < J; j++) {
			posit8_t x = A[i * J + j];

			if (p8_lt(max, x)) {
				maxIndex = counter;
				max = x;
			}
			counter++;
		}
	}
	*index = maxIndex;
	return;
}

// A = A^T
void Transpose(posit8_t* A, posit8_t* B, MYITE I, MYITE J) {
	for (MYITE i = 0; i < I; i++) {
		for (MYITE j = 0; j < J; j++) {
			B[i * J + j] = A[j * I + i];
		}
	}
	return;
}

// C = a * B
void ScalarMul(posit8_t* A, posit8_t* B, posit8_t* C, MYITE I, MYITE J, MYINT shrA, MYINT shrB) {
	posit8_t a = *A;
	for (MYITE i = 0; i < I; i++) {
		for (MYITE j = 0; j < J; j++) {
			posit8_t b = B[i * J + j];
			C[i * J + j] = p8_mul(a, b);
		}
	}
	return;
}

// C = MBConv(A, params)
// A[N][H][W][Cin], C[N][Hout][Wout][Cout]
// X[HF][W][Ct], T[Ct], U[max(Ct, Cin, HF*WF)]
// F1[1][1][1][Cin][Ct], BN1W[Ct], BN1B[Ct]
// F2[Ct][HF][WF][1][1], BN2W[Ct], BN2B[Ct]
// F3[1][1][1][Ct][Cout], BN3W[Cout], BN3B[Cout]
void MBConv(posit8_t* A, const posit8_t* F1, const posit8_t* BN1W, const posit8_t* BN1B, const posit8_t* F2, const posit8_t* BN2W, const posit8_t* BN2B, const posit8_t* F3, const posit8_t* BN3W, const posit8_t* BN3B, posit8_t* C, posit8_t* X, posit8_t* T, posit8_t* U, MYITE N, MYITE H, MYITE W, MYITE Cin, MYITE Ct, MYITE HF, MYITE WF, MYITE Cout, MYITE Hout, MYITE Wout, MYITE HPADL, MYITE HPADR, MYITE WPADL, MYITE WPADR, MYITE HSTR, MYITE WSTR, MYITE D1, MYITE D2, MYITE D3, MYINT SIX_1, MYINT SIX_2, MYINT shr1, MYINT shr2, MYINT shr3, MYINT shr4, MYINT shr5, MYINT shr6, MYINT shr7, MYINT shr8, MYINT shr9, MYINT shl1, MYINT shl2, MYINT shl3, MYINT shl4, MYINT shl5, MYINT shl6, MYINT shl7, MYINT shl8, MYINT shl9, std::string name) {
	MYITE HOffsetL = (HF / 2) - HPADL;
	MYITE WOffsetL = (WF / 2) - WPADL;
	MYITE HOffsetR = (HF / 2) - HPADR;
	MYITE WOffsetR = (WF / 2) - WPADR;

	for (MYITE n = 0; n < N; n++) {
		MYITE margin = HOffsetL + (HF / 2 + 1) - HSTR > 0 ? HOffsetL + (HF/2 + 1) - HSTR : 0;
		MYITE nstart = HOffsetL - (HF / 2) < 0 ? 0 : HOffsetL - (HF / 2);
		for (MYITE i = nstart; i < margin; i++) {
			for (MYITE j = 0; j < W; j++) {
				for (MYITE k = 0; k < Ct; k++) {
					for (MYITE l = 0; l < Cin; l++) {
						U[l] = p8_mul(A[n * H * W * Cin + i * W * Cin + j * Cin + l], F1[l * Ct + k]);
					}
					MYITE totalEle = Cin;
					MYITE count = Cin;
					MYITE depth = 0;

					while (depth < D1) {
						for (MYITE p = 0; p < (totalEle / 2 + 1); p++) {
							if (p < count / 2) {
								U[p] = p8_add(U[2 * p], U[(2 * p) + 1]);
							} else if ((p == (count / 2)) && ((count % 2) == 1)) {
								U[p] = U[2 * p];
							} else {
								U[p] =  convertDoubleToP8(0);
							}
						}

						count = (count + 1) / 2;
						depth++;
					}
	
					posit8_t ar = p8_add(U[0], BN1B[k]);
					X[i * W * Ct + j * Ct + k] = p8_mul((ar),  BN1W[k]);
					Profile2(&ar, 1, 1, name + "t1");
					X[i * W * Ct + j * Ct + k] = p8_lt(X[i * W * Ct + j * Ct + k], convertDoubleToP8(0.0)) ? convertDoubleToP8(0.0) : X[i * W * Ct + j * Ct + k];
					X[i * W * Ct + j * Ct + k] = p8_lt(convertDoubleToP8(6.0), X[i * W * Ct + j * Ct + k]) ? convertDoubleToP8(6.0) : X[i * W * Ct + j * Ct + k];
				}
			}
		}

		for (MYITE h = HOffsetL, hout = 0; h < H - HOffsetR; hout++, h += HSTR) {

			for (MYITE i = 0; i < HSTR; i++) {
				for (MYITE j = 0; j < W; j++) {
					for (MYITE k = 0; k < Ct; k++) {
						MYITE iRed = (i + margin + hout * HSTR) % HF, iFull = i + margin + hout * HSTR;
						X[iRed * W * Ct + j * Ct + k] = convertDoubleToP8(0.0);
						for (MYITE l = 0; l < Cin; l++) {
							posit8_t a = iFull < H ? A[n * H * W * Cin + iFull * W * Cin + j * Cin + l] : convertDoubleToP8(0.0);
							U[l] = p8_mul(a, F1[l * Ct + k]);
						}
						MYITE totalEle = Cin;
						MYITE count = Cin;
						MYITE depth = 0;

						while (depth < D1) {
							for (MYITE p = 0; p <(totalEle / 2 + 1); p++) {
								if (p < count / 2) {
									U[p] = p8_add(U[2 * p], U[(2 * p) + 1]);
								} else if ((p == (count / 2)) && ((count % 2) == 1)) {
									U[p] = U[2 * p];
								} else {
									U[p] = convertDoubleToP8(0);
								}
							}

							count = (count + 1) / 2;
							depth++;
						}

						posit8_t ar = p8_add(U[0], BN1B[k]);
						X[iRed * W * Ct + j * Ct + k] = p8_mul((ar), BN1W[k]);
						Profile2(&ar, 1, 1, name + "t1");
						X[iRed * W * Ct + j * Ct + k] = p8_lt(X[iRed * W * Ct + j * Ct + k], convertDoubleToP8(0.0)) ? convertDoubleToP8(0.0) : X[iRed * W * Ct + j * Ct + k];
						X[iRed * W * Ct + j * Ct + k] = p8_lt(convertDoubleToP8(6.0), X[iRed * W * Ct + j * Ct + k]) ? convertDoubleToP8(6.0) : X[iRed * W * Ct + j * Ct + k];
					}
				}
			}

			for (MYITE w = WOffsetL, wout = 0; w < W - WOffsetR; w += WSTR, wout++) {
				for (MYITE g = 0; g < Ct; g++) {
					MYITE counter = 0;
					for (MYITE hf = -(HF / 2); hf <= (HF / 2); hf++) {
						for (MYITE wf = -(WF / 2); wf <= (WF / 2); wf++) {
							posit8_t x = (((h + hf) < 0) || ((h + hf) >= H) || ((w + wf) < 0) || ((w + wf) >= W)) ? convertDoubleToP8(0.0) : X[((h + hf) % HF) * W * Ct + (w + wf) * Ct + g];
							posit8_t b = F2[g * HF * WF + (hf + HF / 2) * WF + (wf + WF / 2)];
							U[counter] = p8_mul(x, b);
							counter++;
						}
					}
					MYITE totalEle = HF * WF;
					MYITE count = HF * WF;
					MYITE depth = 0;

					while (depth < D2) {
						for (MYITE p = 0; p < (totalEle / 2 + 1); p++) {
							if (p < count / 2) {
								U[p] = p8_add(U[2 * p], U[(2 * p) + 1]);
							} else if ((p == (count / 2)) && ((count % 2) == 1)) {
								U[p] = U[2 * p];
							} else {
								U[p] = convertDoubleToP8(0.0);
							}
						}

						count = (count + 1) / 2;
						depth++;
					}

					posit8_t ar = p8_add(U[0], BN2B[g]);
					T[g] = p8_mul((ar), BN2W[g]);
					Profile2(&ar, 1, 1, name + "t3");
					T[g] = p8_lt(T[g], convertDoubleToP8(0.0)) ? convertDoubleToP8(0.0) : T[g];
					T[g] = p8_lt(convertDoubleToP8(6.0), T[g]) ? convertDoubleToP8(6.0) : T[g];
				}

				for (MYITE i = 0; i < Cout; i++) {
					for (MYITE g = 0; g < Ct; g++) {
						U[g] = p8_mul(T[g], F3[g * Cout + i]);
					}
					MYITE totalEle = Ct;
					MYITE count = Ct;
					MYITE depth = 0;

					while (depth < D3) {
						for (MYITE p = 0; p < (totalEle / 2 + 1); p++) {
							if (p < count / 2) {
								U[p] = p8_add(U[2 * p], U[(2 * p) + 1]);
							} else if ((p == (count / 2)) && ((count % 2) == 1)) {
								U[p] = U[2 * p];
							} else {
								U[p] = convertDoubleToP8(0);
							}
						}

						count = (count + 1) / 2;
						depth++;
					}

					posit8_t ar = p8_add(U[0], BN3B[i]);
					C[n * Hout * Wout * Cout + hout * Wout * Cout + wout * Cout + i] = p8_mul((ar), BN3W[i]);
					Profile2(&ar, 1, 1, name + "t5");
				}
			}
		}
	}
}

// C = conv(A, B, <params>)
// A[N][H][W][CIN], B[G][HF][WF][CINF][COUTF], C[N][HOUT][WOUT][COUTF*G]
void Convolution(posit8_t* A, const posit8_t* B, posit8_t* C, posit8_t* tmp, MYITE N, MYITE H, MYITE W, MYITE CIN, MYITE HF, MYITE WF, MYITE CINF, MYITE COUTF, MYITE HOUT, MYITE WOUT, MYITE HPADL, MYITE HPADR, MYITE WPADL, MYITE WPADR, MYITE HSTR, MYITE WSTR, MYITE HDL, MYITE WDL, MYITE G, MYINT shrA, MYINT shrB, MYITE H1, MYITE H2) {
	MYITE HOffsetL = HDL * (HF / 2) - HPADL;
	MYITE WOffsetL = WDL * (WF / 2) - WPADL;
	MYITE HOffsetR = HDL * (HF / 2) - HPADR;
	MYITE WOffsetR = WDL * (WF / 2) - WPADR;

	for (MYITE n = 0; n < N; n++) {
		for (MYITE h = HOffsetL, hout = 0; h < H - HOffsetR; h += HSTR, hout++) {
			for (MYITE w = WOffsetL, wout = 0; w < W - WOffsetR; w += WSTR, wout++) {
				for (MYITE g = 0; g < G; g++) {
					for (MYITE co = 0; co < COUTF; co++) {

						MYITE counter = 0;
						for (MYITE hf = -(HF / 2); hf <= HF / 2; hf++) {
							for (MYITE wf = -(WF / 2); wf <= WF / 2; wf++) {
								for (MYITE ci = 0; ci < CINF; ci++) {
									posit8_t a = (((h + HDL * hf) < 0) || ((h + HDL * hf) >= H) || ((w + WDL * wf) < 0) || ((w + WDL * wf) >= W)) ? convertDoubleToP8(0) : A[n * H * W * CIN + (h + HDL * hf) * W * CIN + (w + WDL * wf) * CIN + (ci + g * CINF)];
									posit8_t b = B[g * HF * WF * CINF * COUTF + (hf + HF / 2) * WF * CINF * COUTF + (wf + WF / 2) * CINF * COUTF + ci * COUTF + co];

									tmp[counter] = p8_mul(a, b);
									counter++;
								}
							}
						}

						MYITE totalEle = HF * WF * CINF;
						MYITE count = HF * WF * CINF, depth = 0;
						bool shr = true;

						while (depth < (H1 + H2)) {
							if (depth >= H1) {
								shr = false;
							}

							for (MYITE p = 0; p < (totalEle / 2 + 1); p++) {
								posit8_t sum;
								if (p < (count >> 1)) {
									sum = p8_add(tmp[2 * p], tmp[(2 * p) + 1]);
								} else if ((p == (count >> 1)) && ((count & 1) == 1)) {
									sum = tmp[2 * p];
								} else {
									sum = convertDoubleToP8(0);
								}

								if (shr) {
									tmp[p] = sum;
								} else {
									tmp[p] = sum;
								}
							}

							count = (count + 1) >> 1;
							depth++;
						}

						C[n * HOUT * WOUT * (COUTF * G) + hout * WOUT * (COUTF * G) + wout * (COUTF * G) + (co + g * COUTF)] = tmp[0];
					}
				}
			}
		}
	}
}

// C = A # B
// A[N][H][W][CI], B[HF][WF][CI][CO], C[N][H][W][CO]
void Conv(posit8_t* A, const posit8_t* B, posit8_t* C, posit8_t* tmp, MYITE N, MYITE H, MYITE W, MYITE CI, MYITE HF, MYITE WF, MYITE CO, MYINT shrA, MYINT shrB, MYITE H1, MYITE H2) {
	MYITE padH = (HF - 1) / 2;
	MYITE padW = (WF - 1) / 2;

	for (MYITE n = 0; n < N; n++) {
		for (MYITE h = 0; h < H; h++) {
			for (MYITE w = 0; w < W; w++) {
				for (MYITE co = 0; co < CO; co++) {

					MYITE counter = 0;
					for (MYITE hf = 0; hf < HF; hf++) {
						for (MYITE wf = 0; wf < WF; wf++) {
							for (MYITE ci = 0; ci < CI; ci++) {
								posit8_t a = (((((h + hf) < padH) || ((h + hf) >= (H + padH))) || (((w + wf) < padW) || ((w + wf) >= (W + padW)))) ? convertDoubleToP8(0) : A[n * H * W * CI + ((h + hf) - padH) * W * CI + ((w + wf) - padW) * CI + ci]);
								posit8_t b = B[hf * WF * CI * CO + wf * CI * CO + ci * CO + co];

								tmp[counter] = p8_mul(a, b);
								counter++;
							}
						}
					}

					MYITE totalEle = HF * WF * CI;
					MYITE count = HF * WF * CI, depth = 0;
					bool shr = true;

					while (depth < (H1 + H2)) {
						if (depth >= H1) {
							shr = false;
						}

						for (MYITE p = 0; p < (totalEle / 2 + 1); p++) {
							posit8_t sum;
							if (p < (count >> 1)) {
								sum = p8_add(tmp[2 * p], tmp[(2 * p) + 1]);
							} else if ((p == (count >> 1)) && ((count & 1) == 1)) {
								sum = tmp[2 * p];
							} else {
								sum = convertDoubleToP8(0);
							}

							if (shr) {
								tmp[p] = sum;
							} else {
								tmp[p] = sum;
							}
						}

						count = (count + 1) >> 1;
						depth++;
					}

					C[n * H * W * CO + h * W * CO + w * CO + co] = tmp[0];
				}
			}
		}
	}

	return;
}

// A = A <+> B
// A[N][H][W][C], B[C]
void AddOrSubCir4D(posit8_t* A, const posit8_t* B, posit8_t* X, MYITE N, MYITE H, MYITE W, MYITE C, MYINT shrA, MYINT shrB, MYINT shrC, bool add) {
	for (MYITE n = 0; n < N; n++) {
		for (MYITE h = 0; h < H; h++) {
			for (MYITE w = 0; w < W; w++) {
				for (MYITE c = 0; c < C; c++) {
					posit8_t a = A[n * H * W * C + h * W * C + w * C + c];
					posit8_t b = B[c];

					posit8_t res;
					if (add) {
						res = p8_add(a, b);
					} else {
						res = p8_sub(a, b);
					}

					X[n * H * W * C + h * W * C + w * C + c] = res;
				}
			}
		}
	}
	return;
}

// A = A <+> B
// A[N][H][W][C], B[C]
void AddOrSubCir2D(posit8_t* A, const posit8_t* B, posit8_t* X, MYITE H, MYITE W, MYINT shrA, MYINT shrB, MYINT shrC, bool add) {
	for (MYITE h = 0; h < H; h++) {
		for (MYITE w = 0; w < W; w++) {
			posit8_t a = A[h * W + w];
			posit8_t b = B[w];

			posit8_t res;
			if (add) {
				res = p8_add(a, b);
			} else {
				res = p8_sub(a, b);
			}

			X[h * W + w] = res;
		}
	}
	return;
}

// A = relu(A)
// A[N][H][W][C]
void Relu4D(posit8_t* A, MYITE N, MYITE H, MYITE W, MYITE C) {
	for (MYITE n = 0; n < N; n++) {
		for (MYITE h = 0; h < H; h++) {
			for (MYITE w = 0; w < W; w++) {
				for (MYITE c = 0; c < C; c++) {
					posit8_t a = A[n * H * W * C + h * W * C + w * C + c];
					posit8_t zero = convertDoubleToP8(0);
					if (p8_lt(a, zero)) {
						a = zero;
					}

					A[n * H * W * C + h * W * C + w * C + c] = a;
				}
			}
		}
	}
	return;
}

// B = relu6(A)
// A[N][H][W][C]
void Relu6(posit8_t* A, posit8_t* B, MYITE N, MYITE H, MYITE W, MYITE C, MYINT six, MYINT div) {
	for (MYITE n = 0; n < N; n++) {
		for (MYITE h = 0; h < H; h++) {
			for (MYITE w = 0; w < W; w++) {
				for (MYITE c = 0; c < C; c++) {
					posit8_t a = A[n * H * W * C + h * W * C + w * C + c];
					posit8_t zero = convertDoubleToP8(0);
					posit8_t six = convertDoubleToP8(6.0);
					if (p8_lt(a, zero)) {
						a = zero;
					}
					if (p8_lt(six, a)) {
						a = six;
					}

					B[n * H * W * C + h * W * C + w * C + c] = a;
				}
			}
		}
	}
	return;
}

// A = relu(A)
// A[N][H][W][C]
void Relu2D(posit8_t* A, MYITE H, MYITE W) {
	for (MYITE h = 0; h < H; h++) {
		for (MYITE w = 0; w < W; w++) {
			posit8_t a = A[h * W + w];
			posit8_t zero = convertDoubleToP8(0);
			if (p8_lt(a, zero)) {
				a = zero;
			}

			A[h * W + w] = a;
		}
	}
	return;
}

// B = maxpool(A)
// A[N][H][W][C], B[N][H][W][C]
void Maxpool(posit8_t* A, posit8_t* B, MYITE N, MYITE H, MYITE W, MYITE C, MYITE FH, MYITE FW, MYITE strideH, MYITE strideW, MYITE HPADL, MYITE HPADR, MYITE WPADL, MYITE WPADR) {
	MYITE HO = H / strideH;
	MYITE WO = W / strideW;

	for (MYITE n = 0; n < N; n++) {
		for (MYITE ho = 0; ho < HO; ho++) {
			for (MYITE wo = 0; wo < WO; wo++) {
				for (MYITE c = 0; c < C; c++) {

					posit8_t max = A[n * H * W * C + (strideH * ho) * W * C + (strideW * wo) * C + c];
					for (MYITE hs = 0; hs < FH; hs++) {
						for (MYITE ws = 0; ws < FW; ws++) {
							posit8_t a = A[n * H * W * C + ((strideH * ho) + hs) * W * C + ((strideW * wo) + ws) * C + c];
							if (p8_lt(max, a)) {
								max = a;
							}
						}
					}

					B[n * HO * WO * C + ho * WO * C + wo * C + c] = max;
				}
			}
		}
	}
	return;
}

void NormaliseL2(posit8_t* A, posit8_t* B, MYITE N, MYITE H, MYITE W, MYITE C, MYINT scaleA, MYINT shrA) {
	for (MYITE n = 0; n < N; n++) {
		for (MYITE h = 0; h < H; h++) {
			for (MYITE w = 0; w < W; w++) {

				// Calculate the sum square.
				posit8_t sumSquare = convertDoubleToP8(0);
				for (MYITE c = 0; c < C; c++) {
					posit8_t tmp = A[n * H * W * C + h * W * C + w * C + c];
					sumSquare = p8_add(sumSquare, p8_mul(tmp, tmp));
				}

				// Calculate the inverse square root of sumSquare.
				if (p8_eq(sumSquare,  convertDoubleToP8(0))) {
					sumSquare = convertDoubleToP8(1e-5);
				}

				posit8_t inverseNorm = p8_div(convertDoubleToP8(1), p8_sqrt(sumSquare));

				// Multiply all elements by the 1 / sqrt(sumSquare).
				for (MYITE c = 0; c < C; c++) {
					B[n * H * W * C + h * W * C + w * C + c]  = p8_mul(A[n * H * W * C + h * W * C + w * C + c], inverseNorm);
				}
			}
		}
	}
	return;
}

// B = exp(A)
void Exp(posit8_t* A, MYITE I, MYITE J, MYINT shrA, MYINT shrB, posit8_t* B) {
	for (MYITE i = 0; i < I; i++) {
		for (MYITE j = 0; j < J; j++) {
			posit8_t x = A[i * J + j];

			updateRangeOfExp(-1*convertP8ToDouble(x));

			B[i * J + j] = convertDoubleToP8(exp(convertP8ToDouble(x)));
		}
	}
	return;
}

// A = sigmoid(A)
void Sigmoid(posit8_t* A, MYITE I, MYITE J, float div, float add, float sigmoid_limit, MYINT scale_in, MYINT scale_out, posit8_t* B) {
	for (MYITE i = 0; i < I; i++) {
		for (MYITE j = 0; j < J; j++) {
			float x = convertP8ToDouble(A[i * J + j]), y;

#ifdef FLOATEXP
			y = 1 / (1 + exp(-x));
#else
			y = (x + 1) / 2;
			y = y > 0 ? y : 0;
			y = y < 1 ? y : 1;
#endif
			B[i * J + j] = convertDoubleToP8(y);
		}
	}
	return;
}

// A = AdjustScaleShr(A)
void AdjustScaleShr(posit8_t* A, MYITE I, MYITE J, MYINT scale) {
	return;
}

// A = AdjustScaleShl(A)
void AdjustScaleShl(posit8_t* A, MYITE I, MYITE J, MYINT scale) {
	return;
}

void MatAddNN(posit16_t* A, posit16_t* B, posit16_t* C, MYITE I, MYITE J, MYINT shrA, MYINT shrB, MYINT shrC) {
	for (MYITE i = 0; i < I; i++) {
		for (MYITE j = 0; j < J; j++) {
			posit16_t a = A[i * J + j];
			posit16_t b = B[i * J + j];

			posit16_t c = p16_add(a, b);

			C[i * J + j] = c;
		}
	}
	return;
}

// C = A + B
void MatAddCN(const posit16_t* A, posit16_t* B, posit16_t* C, MYITE I, MYITE J, MYINT shrA, MYINT shrB, MYINT shrC) {
	for (MYITE i = 0; i < I; i++) {
		for (MYITE j = 0; j < J; j++) {
			posit16_t a = A[i * J + j];
			posit16_t b = B[i * J + j];

			posit16_t c = p16_add(a, b);

			C[i * J + j] = c;
		}
	}
	return;
}

// C = A + B
void MatAddNC(posit16_t* A, const posit16_t* B, posit16_t* C, MYITE I, MYITE J, MYINT shrA, MYINT shrB, MYINT shrC) {
	for (MYITE i = 0; i < I; i++) {
		for (MYITE j = 0; j < J; j++) {
			posit16_t a = A[i * J + j];
			posit16_t b = B[i * J + j];

			posit16_t c = p16_add(a, b);

			C[i * J + j] = c;
		}
	}
	return;
}

// C = A + B
void MatAddCC(const posit16_t* A, const posit16_t* B, posit16_t* C, MYITE I, MYITE J, MYINT shrA, MYINT shrB, MYINT shrC) {
	for (MYITE i = 0; i < I; i++) {
		for (MYITE j = 0; j < J; j++) {
			posit16_t a = A[i * J + j];
			posit16_t b = B[i * J + j];

			posit16_t c = p16_add(a, b);

			C[i * J + j] = c;
		}
	}
	return;
}

// C = a + B
void MatAddBroadCastA(posit16_t* A, posit16_t* B, posit16_t* C, MYITE I, MYITE J, MYINT shrA, MYINT shrB, MYINT shrC) {
	for (MYITE i = 0; i < I; i++) {
		for (MYITE j = 0; j < J; j++) {
			posit16_t a = *A;
			posit16_t b = B[i * J + j];

			posit16_t c = p16_add(a, b);

			C[i * J + j] = c;
		}
	}
	return;
}

// C = A + b
void MatAddBroadCastB(posit16_t* A, posit16_t* B, posit16_t* C, MYITE I, MYITE J, MYINT shrA, MYINT shrB, MYINT shrC) {
	for (MYITE i = 0; i < I; i++) {
		for (MYITE j = 0; j < J; j++) {
			posit16_t a = A[i * J + j];
			posit16_t b = *B;

			posit16_t c = p16_add(a, b);

			C[i * J + j] = c;
		}
	}
	return;
}

// C = A + B
void MatAdd4(posit16_t* A, posit16_t* B, posit16_t* X, MYITE N, MYITE H, MYITE W, MYITE C, MYINT shrA, MYINT shrB, MYINT shrC) {
	for (MYITE n = 0; n < N; n++) {
		for (MYITE h = 0; h < H; h++) {
			for (MYITE w = 0; w < W; w++) {
				for (MYITE c = 0; c < C; c++) {
					posit16_t a = A[n * H * W * C + h * W * C + w * C + c];
					posit16_t b = B[n * H * W * C + h * W * C + w * C + c];

					posit16_t x = p16_add(a, b);

					X[n * H * W * C + h * W * C + w * C + c] = x;
				}
			}
		}
	}
	return;
}

// C = A - B
void MatSub(posit16_t* A, const posit16_t* B, posit16_t* C, MYITE I, MYITE J, MYINT shrA, MYINT shrB, MYINT shrC) {
	for (MYITE i = 0; i < I; i++) {
		for (MYITE j = 0; j < J; j++) {
			posit16_t a = A[i * J + j];
			posit16_t b = B[i * J + j];

			posit16_t c = p16_sub(a, b);

			C[i * J + j] = c;
		}
	}
	return;
}

// C = a - B
void MatSubBroadCastA(posit16_t* A, posit16_t* B, posit16_t* C, MYITE I, MYITE J, MYINT shrA, MYINT shrB, MYINT shrC) {
	for (MYITE i = 0; i < I; i++) {
		for (MYITE j = 0; j < J; j++) {
			posit16_t a = *A;
			posit16_t b = B[i * J + j];

			posit16_t c = p16_sub(a, b);

			C[i * J + j] = c;
		}
	}
	return;
}

// C = A - b
void MatSubBroadCastB(posit16_t* A, posit16_t* B, posit16_t* C, MYITE I, MYITE J, MYINT shrA, MYINT shrB, MYINT shrC) {
	for (MYITE i = 0; i < I; i++) {
		for (MYITE j = 0; j < J; j++) {
			posit16_t a = A[i * J + j];
			posit16_t b = *B;

			posit16_t c = p16_sub(a, b);

			C[i * J + j] = c;
		}
	}
	return;
}

// C = A * B
void MatMulNN(posit16_t* A, posit16_t* B, posit16_t* C, MYITE I, MYITE K, MYITE J, MYINT shrA, MYINT shrB, MYITE H1, MYITE H2) {
	quire16_t qz;
	for (MYITE i = 0; i < I; i++) {
		for (MYITE j = 0; j < J; j++) {
			qz = q16_clr(qz);
			for (MYITE k = 0; k < K; k++) {
				posit16_t a = A[i * K + k];
				posit16_t b = B[k * J + j];
				qz = q16_fdp_add(qz, a, b);
				// tmp[k] = p8_mul(a, b);
			}

			// MYITE count = K, depth = 0;
			// bool shr = true;

			// while (depth < (H1 + H2)) {
			// 	if (depth >= H1) {
			// 		shr = false;
			// 	}

			// 	for (MYITE p = 0; p < (K / 2 + 1); p++) {
			// 		posit8_t sum;
			// 		if (p < (count >> 1)) {
			// 			sum = p8_add(tmp[2 * p], tmp[(2 * p) + 1]);
			// 		} else if ((p == (count >> 1)) && ((count & 1) == 1)) {
			// 			sum = tmp[2 * p];
			// 		} else {
			// 			sum = convertDoubleToP8(0.0);
			// 		}

			// 		if (shr) {
			// 			tmp[p] = sum;
			// 		} else {
			// 			tmp[p] = sum;
			// 		}
			// 	}

			// 	count = (count + 1) >> 1;
			// 	depth++;
			// }

			C[i * J + j] = q16_to_p16(qz);
		}
	}
	return;
}

// C = A * B
void MatMulCN(const posit16_t* A, posit16_t* B, posit16_t* C, MYITE I, MYITE K, MYITE J, MYINT shrA, MYINT shrB, MYITE H1, MYITE H2) {
	quire16_t qz;
	for (MYITE i = 0; i < I; i++) {
		for (MYITE j = 0; j < J; j++) {
			qz = q16_clr(qz);
			for (MYITE k = 0; k < K; k++) {
				posit16_t a = A[i * K + k];
				posit16_t b = B[k * J + j];
				qz = q16_fdp_add(qz, a, b);
				// tmp[k] = p8_mul(a, b);
			}

			// MYITE count = K, depth = 0;
			// bool shr = true;

			// while (depth < (H1 + H2)) {
			// 	if (depth >= H1) {
			// 		shr = false;
			// 	}

			// 	for (MYITE p = 0; p < (K / 2 + 1); p++) {
			// 		posit8_t sum;
			// 		if (p < (count >> 1)) {
			// 			sum = p8_add(tmp[2 * p], tmp[(2 * p) + 1]);
			// 		} else if ((p == (count >> 1)) && ((count & 1) == 1)) {
			// 			sum = tmp[2 * p];
			// 		} else {
			// 			sum = convertDoubleToP8(0.0);
			// 		}

			// 		if (shr) {
			// 			tmp[p] = sum;
			// 		} else {
			// 			tmp[p] = sum;
			// 		}
			// 	}

			// 	count = (count + 1) >> 1;
			// 	depth++;
			// }

			C[i * J + j] = q16_to_p16(qz);
		}
	}
	return;
}

// C = A * B
void MatMulNC(posit16_t* A, const posit16_t* B, posit16_t* C, MYITE I, MYITE K, MYITE J, MYINT shrA, MYINT shrB, MYITE H1, MYITE H2) {
	quire16_t qz;
	for (MYITE i = 0; i < I; i++) {
		for (MYITE j = 0; j < J; j++) {
			qz = q16_clr(qz);
			for (MYITE k = 0; k < K; k++) {
				posit16_t a = A[i * K + k];
				posit16_t b = B[k * J + j];
				qz = q16_fdp_add(qz, a, b);
				// tmp[k] = p8_mul(a, b);
			}

			// MYITE count = K, depth = 0;
			// bool shr = true;

			// while (depth < (H1 + H2)) {
			// 	if (depth >= H1) {
			// 		shr = false;
			// 	}

			// 	for (MYITE p = 0; p < (K / 2 + 1); p++) {
			// 		posit8_t sum;
			// 		if (p < (count >> 1)) {
			// 			sum = p8_add(tmp[2 * p], tmp[(2 * p) + 1]);
			// 		} else if ((p == (count >> 1)) && ((count & 1) == 1)) {
			// 			sum = tmp[2 * p];
			// 		} else {
			// 			sum = convertDoubleToP8(0.0);
			// 		}

			// 		if (shr) {
			// 			tmp[p] = sum;
			// 		} else {
			// 			tmp[p] = sum;
			// 		}
			// 	}

			// 	count = (count + 1) >> 1;
			// 	depth++;
			// }

			C[i * J + j] = q16_to_p16(qz);
		}
	}
	return;
}

// C = A * B
void MatMulCC(const posit16_t* A, const posit16_t* B, posit16_t* C, MYITE I, MYITE K, MYITE J, MYINT shrA, MYINT shrB, MYITE H1, MYITE H2) {
	quire16_t qz;
	for (MYITE i = 0; i < I; i++) {
		for (MYITE j = 0; j < J; j++) {
			qz = q16_clr(qz);
			for (MYITE k = 0; k < K; k++) {
				posit16_t a = A[i * K + k];
				posit16_t b = B[k * J + j];
				qz = q16_fdp_add(qz, a, b);
				// tmp[k] = p8_mul(a, b);
			}

			// MYITE count = K, depth = 0;
			// bool shr = true;

			// while (depth < (H1 + H2)) {
			// 	if (depth >= H1) {
			// 		shr = false;
			// 	}

			// 	for (MYITE p = 0; p < (K / 2 + 1); p++) {
			// 		posit8_t sum;
			// 		if (p < (count >> 1)) {
			// 			sum = p8_add(tmp[2 * p], tmp[(2 * p) + 1]);
			// 		} else if ((p == (count >> 1)) && ((count & 1) == 1)) {
			// 			sum = tmp[2 * p];
			// 		} else {
			// 			sum = convertDoubleToP8(0.0);
			// 		}

			// 		if (shr) {
			// 			tmp[p] = sum;
			// 		} else {
			// 			tmp[p] = sum;
			// 		}
			// 	}

			// 	count = (count + 1) >> 1;
			// 	depth++;
			// }

			C[i * J + j] = q16_to_p16(qz);
		}
	}
	return;
}

// C = A |*| B
void SparseMatMulX(const MYINT* Aidx, const posit16_t* Aval, posit16_t** B, posit16_t* C, int16_t K, MYINT shrA, MYINT shrB, MYINT shrC) {
	MYITE ite_idx = 0, ite_val = 0;
	for (MYITE k = 0; k < K; k++) {
		posit16_t b = B[k * 1][0];

		MYINT idx = Aidx[ite_idx];
		while (idx != 0) {
			posit16_t a = Aval[ite_val];

			posit16_t c = p16_mul(a, b);

			C[idx - 1] = p16_add(C[idx - 1], c);

			ite_idx++;
			ite_val++;

			idx = Aidx[ite_idx];
		}
		ite_idx++;
	}
	return;
}

// C = A |*| B
void SparseMatMul(const MYINT* Aidx, const posit16_t* Aval, posit16_t* B, posit16_t* C, int16_t K, MYINT shrA, MYINT shrB, MYINT shrC) {
	MYITE ite_idx = 0, ite_val = 0;
	for (MYITE k = 0; k < K; k++) {
		posit16_t b = B[k];

		MYINT idx = Aidx[ite_idx];
		while (idx != 0) {
			posit16_t a = Aval[ite_val];

			posit16_t c = p16_mul(a, b);

			C[idx - 1] = p16_add(C[idx - 1], c);

			ite_idx++;
			ite_val++;

			idx = Aidx[ite_idx];
		}
		ite_idx++;
	}

	return;
}

// C = A <*> B
void MulCir(posit16_t* A, posit16_t* B, posit16_t* C, MYITE I, MYITE J, MYINT shrA, MYINT shrB) {
	for (MYITE i = 0; i < I; i++) {
		for (MYITE j = 0; j < J; j++) {
			posit16_t a = A[i * J + j];
			posit16_t b = B[i * J + j];

			C[i * J + j] = p16_mul(a, b);
		}
	}
	return;
}

// A = tanh(A)
void TanH(posit16_t* A, MYITE I, MYITE J, float scale_in, float scale_out, posit16_t* B) {
	for (MYITE i = 0; i < I; i++) {
		for (MYITE j = 0; j < J; j++) {
			double x = convertP16ToDouble(A[i * J + j]), y;

			#ifdef FLOATEXP
				y = tanh(x);
			#else
				y = x > -1 ? x : -1;
				y = y < 1 ? y : 1;
			#endif

			B[i * J + j] = convertDoubleToP16(y);
		}
	}
	return;
}

// B = reverse(A, axis)
void Reverse2(posit16_t* A, MYITE axis, MYITE I, MYITE J, posit16_t* B) {
	for (MYITE i = 0; i < I; i++) {
		for (MYITE j = 0; j < J; j++) {
			MYITE i_prime = (axis == 0 ? (I - 1 - i) : i);
			MYITE j_prime = (axis == 1 ? (J - 1 - j) : j);

			B[i * J + j] = A[i_prime*J + j_prime];
		}
	}
	return;
}

// index = argmax(A)
void ArgMax(posit16_t* A, MYITE I, MYITE J, int* index) {
	posit16_t max = A[0];
	MYITE maxIndex = 0, counter = 0;
	for (MYITE i = 0; i < I; i++) {
		for (MYITE j = 0; j < J; j++) {
			posit16_t x = A[i * J + j];

			if (p16_lt(max, x)) {
				maxIndex = counter;
				max = x;
			}
			counter++;
		}
	}
	*index = maxIndex;
	return;
}

// A = A^T
void Transpose(posit16_t* A, posit16_t* B, MYITE I, MYITE J) {
	for (MYITE i = 0; i < I; i++) {
		for (MYITE j = 0; j < J; j++) {
			B[i * J + j] = A[j * I + i];
		}
	}
	return;
}

// C = a * B
void ScalarMul(posit16_t* A, posit16_t* B, posit16_t* C, MYITE I, MYITE J, MYINT shrA, MYINT shrB) {
	posit16_t a = *A;
	for (MYITE i = 0; i < I; i++) {
		for (MYITE j = 0; j < J; j++) {
			posit16_t b = B[i * J + j];
			C[i * J + j] = p16_mul(a, b);
		}
	}
	return;
}

// C = MBConv(A, params)
// A[N][H][W][Cin], C[N][Hout][Wout][Cout]
// X[HF][W][Ct], T[Ct], U[max(Ct, Cin, HF*WF)]
// F1[1][1][1][Cin][Ct], BN1W[Ct], BN1B[Ct]
// F2[Ct][HF][WF][1][1], BN2W[Ct], BN2B[Ct]
// F3[1][1][1][Ct][Cout], BN3W[Cout], BN3B[Cout]
void MBConv(posit16_t* A, const posit16_t* F1, const posit16_t* BN1W, const posit16_t* BN1B, const posit16_t* F2, const posit16_t* BN2W, const posit16_t* BN2B, const posit16_t* F3, const posit16_t* BN3W, const posit16_t* BN3B, posit16_t* C, posit16_t* X, posit16_t* T, posit16_t* U, MYITE N, MYITE H, MYITE W, MYITE Cin, MYITE Ct, MYITE HF, MYITE WF, MYITE Cout, MYITE Hout, MYITE Wout, MYITE HPADL, MYITE HPADR, MYITE WPADL, MYITE WPADR, MYITE HSTR, MYITE WSTR, MYITE D1, MYITE D2, MYITE D3, MYINT SIX_1, MYINT SIX_2, MYINT shr1, MYINT shr2, MYINT shr3, MYINT shr4, MYINT shr5, MYINT shr6, MYINT shr7, MYINT shr8, MYINT shr9, MYINT shl1, MYINT shl2, MYINT shl3, MYINT shl4, MYINT shl5, MYINT shl6, MYINT shl7, MYINT shl8, MYINT shl9, std::string name) {
	MYITE HOffsetL = (HF / 2) - HPADL;
	MYITE WOffsetL = (WF / 2) - WPADL;
	MYITE HOffsetR = (HF / 2) - HPADR;
	MYITE WOffsetR = (WF / 2) - WPADR;

	for (MYITE n = 0; n < N; n++) {
		MYITE margin = HOffsetL + (HF / 2 + 1) - HSTR > 0 ? HOffsetL + (HF/2 + 1) - HSTR : 0;
		MYITE nstart = HOffsetL - (HF / 2) < 0 ? 0 : HOffsetL - (HF / 2);
		for (MYITE i = nstart; i < margin; i++) {
			for (MYITE j = 0; j < W; j++) {
				for (MYITE k = 0; k < Ct; k++) {
					for (MYITE l = 0; l < Cin; l++) {
						U[l] = p16_mul(A[n * H * W * Cin + i * W * Cin + j * Cin + l], F1[l * Ct + k]);
					}
					MYITE totalEle = Cin;
					MYITE count = Cin;
					MYITE depth = 0;

					while (depth < D1) {
						for (MYITE p = 0; p < (totalEle / 2 + 1); p++) {
							if (p < count / 2) {
								U[p] = p16_add(U[2 * p], U[(2 * p) + 1]);
							} else if ((p == (count / 2)) && ((count % 2) == 1)) {
								U[p] = U[2 * p];
							} else {
								U[p] =  convertDoubleToP16(0);
							}
						}

						count = (count + 1) / 2;
						depth++;
					}
	
					posit16_t ar = p16_add(U[0], BN1B[k]);
					X[i * W * Ct + j * Ct + k] = p16_mul((ar),  BN1W[k]);
					Profile2(&ar, 1, 1, name + "t1");
					X[i * W * Ct + j * Ct + k] = p16_lt(X[i * W * Ct + j * Ct + k], convertDoubleToP16(0.0)) ? convertDoubleToP16(0.0) : X[i * W * Ct + j * Ct + k];
					X[i * W * Ct + j * Ct + k] = p16_lt(convertDoubleToP16(6.0), X[i * W * Ct + j * Ct + k]) ? convertDoubleToP16(6.0) : X[i * W * Ct + j * Ct + k];
				}
			}
		}

		for (MYITE h = HOffsetL, hout = 0; h < H - HOffsetR; hout++, h += HSTR) {

			for (MYITE i = 0; i < HSTR; i++) {
				for (MYITE j = 0; j < W; j++) {
					for (MYITE k = 0; k < Ct; k++) {
						MYITE iRed = (i + margin + hout * HSTR) % HF, iFull = i + margin + hout * HSTR;
						X[iRed * W * Ct + j * Ct + k] = convertDoubleToP16(0.0);
						for (MYITE l = 0; l < Cin; l++) {
							posit16_t a = iFull < H ? A[n * H * W * Cin + iFull * W * Cin + j * Cin + l] : convertDoubleToP16(0.0);
							U[l] = p16_mul(a, F1[l * Ct + k]);
						}
						MYITE totalEle = Cin;
						MYITE count = Cin;
						MYITE depth = 0;

						while (depth < D1) {
							for (MYITE p = 0; p <(totalEle / 2 + 1); p++) {
								if (p < count / 2) {
									U[p] = p16_add(U[2 * p], U[(2 * p) + 1]);
								} else if ((p == (count / 2)) && ((count % 2) == 1)) {
									U[p] = U[2 * p];
								} else {
									U[p] = convertDoubleToP16(0);
								}
							}

							count = (count + 1) / 2;
							depth++;
						}

						posit16_t ar = p16_add(U[0], BN1B[k]);
						X[iRed * W * Ct + j * Ct + k] = p16_mul((ar), BN1W[k]);
						Profile2(&ar, 1, 1, name + "t1");
						X[iRed * W * Ct + j * Ct + k] = p16_lt(X[iRed * W * Ct + j * Ct + k], convertDoubleToP16(0.0)) ? convertDoubleToP16(0.0) : X[iRed * W * Ct + j * Ct + k];
						X[iRed * W * Ct + j * Ct + k] = p16_lt(convertDoubleToP16(6.0), X[iRed * W * Ct + j * Ct + k]) ? convertDoubleToP16(6.0) : X[iRed * W * Ct + j * Ct + k];
					}
				}
			}

			for (MYITE w = WOffsetL, wout = 0; w < W - WOffsetR; w += WSTR, wout++) {
				for (MYITE g = 0; g < Ct; g++) {
					MYITE counter = 0;
					for (MYITE hf = -(HF / 2); hf <= (HF / 2); hf++) {
						for (MYITE wf = -(WF / 2); wf <= (WF / 2); wf++) {
							posit16_t x = (((h + hf) < 0) || ((h + hf) >= H) || ((w + wf) < 0) || ((w + wf) >= W)) ? convertDoubleToP16(0.0) : X[((h + hf) % HF) * W * Ct + (w + wf) * Ct + g];
							posit16_t b = F2[g * HF * WF + (hf + HF / 2) * WF + (wf + WF / 2)];
							U[counter] = p16_mul(x, b);
							counter++;
						}
					}
					MYITE totalEle = HF * WF;
					MYITE count = HF * WF;
					MYITE depth = 0;

					while (depth < D2) {
						for (MYITE p = 0; p < (totalEle / 2 + 1); p++) {
							if (p < count / 2) {
								U[p] = p16_add(U[2 * p], U[(2 * p) + 1]);
							} else if ((p == (count / 2)) && ((count % 2) == 1)) {
								U[p] = U[2 * p];
							} else {
								U[p] = convertDoubleToP16(0.0);
							}
						}

						count = (count + 1) / 2;
						depth++;
					}

					posit16_t ar = p16_add(U[0], BN2B[g]);
					T[g] = p16_mul((ar), BN2W[g]);
					Profile2(&ar, 1, 1, name + "t3");
					T[g] = p16_lt(T[g], convertDoubleToP16(0.0)) ? convertDoubleToP16(0.0) : T[g];
					T[g] = p16_lt(convertDoubleToP16(6.0), T[g]) ? convertDoubleToP16(6.0) : T[g];
				}

				for (MYITE i = 0; i < Cout; i++) {
					for (MYITE g = 0; g < Ct; g++) {
						U[g] = p16_mul(T[g], F3[g * Cout + i]);
					}
					MYITE totalEle = Ct;
					MYITE count = Ct;
					MYITE depth = 0;

					while (depth < D3) {
						for (MYITE p = 0; p < (totalEle / 2 + 1); p++) {
							if (p < count / 2) {
								U[p] = p16_add(U[2 * p], U[(2 * p) + 1]);
							} else if ((p == (count / 2)) && ((count % 2) == 1)) {
								U[p] = U[2 * p];
							} else {
								U[p] = convertDoubleToP16(0);
							}
						}

						count = (count + 1) / 2;
						depth++;
					}

					posit16_t ar = p16_add(U[0], BN3B[i]);
					C[n * Hout * Wout * Cout + hout * Wout * Cout + wout * Cout + i] = p16_mul((ar), BN3W[i]);
					Profile2(&ar, 1, 1, name + "t5");
				}
			}
		}
	}
}

// C = conv(A, B, <params>)
// A[N][H][W][CIN], B[G][HF][WF][CINF][COUTF], C[N][HOUT][WOUT][COUTF*G]
void Convolution(posit16_t* A, const posit16_t* B, posit16_t* C, posit16_t* tmp, MYITE N, MYITE H, MYITE W, MYITE CIN, MYITE HF, MYITE WF, MYITE CINF, MYITE COUTF, MYITE HOUT, MYITE WOUT, MYITE HPADL, MYITE HPADR, MYITE WPADL, MYITE WPADR, MYITE HSTR, MYITE WSTR, MYITE HDL, MYITE WDL, MYITE G, MYINT shrA, MYINT shrB, MYITE H1, MYITE H2) {
	MYITE HOffsetL = HDL * (HF / 2) - HPADL;
	MYITE WOffsetL = WDL * (WF / 2) - WPADL;
	MYITE HOffsetR = HDL * (HF / 2) - HPADR;
	MYITE WOffsetR = WDL * (WF / 2) - WPADR;

	for (MYITE n = 0; n < N; n++) {
		for (MYITE h = HOffsetL, hout = 0; h < H - HOffsetR; h += HSTR, hout++) {
			for (MYITE w = WOffsetL, wout = 0; w < W - WOffsetR; w += WSTR, wout++) {
				for (MYITE g = 0; g < G; g++) {
					for (MYITE co = 0; co < COUTF; co++) {

						MYITE counter = 0;
						for (MYITE hf = -(HF / 2); hf <= HF / 2; hf++) {
							for (MYITE wf = -(WF / 2); wf <= WF / 2; wf++) {
								for (MYITE ci = 0; ci < CINF; ci++) {
									posit16_t a = (((h + HDL * hf) < 0) || ((h + HDL * hf) >= H) || ((w + WDL * wf) < 0) || ((w + WDL * wf) >= W)) ? convertDoubleToP16(0) : A[n * H * W * CIN + (h + HDL * hf) * W * CIN + (w + WDL * wf) * CIN + (ci + g * CINF)];
									posit16_t b = B[g * HF * WF * CINF * COUTF + (hf + HF / 2) * WF * CINF * COUTF + (wf + WF / 2) * CINF * COUTF + ci * COUTF + co];

									tmp[counter] = p16_mul(a, b);
									counter++;
								}
							}
						}

						MYITE totalEle = HF * WF * CINF;
						MYITE count = HF * WF * CINF, depth = 0;
						bool shr = true;

						while (depth < (H1 + H2)) {
							if (depth >= H1) {
								shr = false;
							}

							for (MYITE p = 0; p < (totalEle / 2 + 1); p++) {
								posit16_t sum;
								if (p < (count >> 1)) {
									sum = p16_add(tmp[2 * p], tmp[(2 * p) + 1]);
								} else if ((p == (count >> 1)) && ((count & 1) == 1)) {
									sum = tmp[2 * p];
								} else {
									sum = convertDoubleToP16(0);
								}

								if (shr) {
									tmp[p] = sum;
								} else {
									tmp[p] = sum;
								}
							}

							count = (count + 1) >> 1;
							depth++;
						}

						C[n * HOUT * WOUT * (COUTF * G) + hout * WOUT * (COUTF * G) + wout * (COUTF * G) + (co + g * COUTF)] = tmp[0];
					}
				}
			}
		}
	}
}

// C = A # B
// A[N][H][W][CI], B[HF][WF][CI][CO], C[N][H][W][CO]
void Conv(posit16_t* A, const posit16_t* B, posit16_t* C, posit16_t* tmp, MYITE N, MYITE H, MYITE W, MYITE CI, MYITE HF, MYITE WF, MYITE CO, MYINT shrA, MYINT shrB, MYITE H1, MYITE H2) {
	MYITE padH = (HF - 1) / 2;
	MYITE padW = (WF - 1) / 2;

	for (MYITE n = 0; n < N; n++) {
		for (MYITE h = 0; h < H; h++) {
			for (MYITE w = 0; w < W; w++) {
				for (MYITE co = 0; co < CO; co++) {

					MYITE counter = 0;
					for (MYITE hf = 0; hf < HF; hf++) {
						for (MYITE wf = 0; wf < WF; wf++) {
							for (MYITE ci = 0; ci < CI; ci++) {
								posit16_t a = (((((h + hf) < padH) || ((h + hf) >= (H + padH))) || (((w + wf) < padW) || ((w + wf) >= (W + padW)))) ? convertDoubleToP16(0) : A[n * H * W * CI + ((h + hf) - padH) * W * CI + ((w + wf) - padW) * CI + ci]);
								posit16_t b = B[hf * WF * CI * CO + wf * CI * CO + ci * CO + co];

								tmp[counter] = p16_mul(a, b);
								counter++;
							}
						}
					}

					MYITE totalEle = HF * WF * CI;
					MYITE count = HF * WF * CI, depth = 0;
					bool shr = true;

					while (depth < (H1 + H2)) {
						if (depth >= H1) {
							shr = false;
						}

						for (MYITE p = 0; p < (totalEle / 2 + 1); p++) {
							posit16_t sum;
							if (p < (count >> 1)) {
								sum = p16_add(tmp[2 * p], tmp[(2 * p) + 1]);
							} else if ((p == (count >> 1)) && ((count & 1) == 1)) {
								sum = tmp[2 * p];
							} else {
								sum = convertDoubleToP16(0);
							}

							if (shr) {
								tmp[p] = sum;
							} else {
								tmp[p] = sum;
							}
						}

						count = (count + 1) >> 1;
						depth++;
					}

					C[n * H * W * CO + h * W * CO + w * CO + co] = tmp[0];
				}
			}
		}
	}

	return;
}

// A = A <+> B
// A[N][H][W][C], B[C]
void AddOrSubCir4D(posit16_t* A, const posit16_t* B, posit16_t* X, MYITE N, MYITE H, MYITE W, MYITE C, MYINT shrA, MYINT shrB, MYINT shrC, bool add) {
	for (MYITE n = 0; n < N; n++) {
		for (MYITE h = 0; h < H; h++) {
			for (MYITE w = 0; w < W; w++) {
				for (MYITE c = 0; c < C; c++) {
					posit16_t a = A[n * H * W * C + h * W * C + w * C + c];
					posit16_t b = B[c];

					posit16_t res;
					if (add) {
						res = p16_add(a, b);
					} else {
						res = p16_sub(a, b);
					}

					X[n * H * W * C + h * W * C + w * C + c] = res;
				}
			}
		}
	}
	return;
}

// A = A <+> B
// A[N][H][W][C], B[C]
void AddOrSubCir2D(posit16_t* A, const posit16_t* B, posit16_t* X, MYITE H, MYITE W, MYINT shrA, MYINT shrB, MYINT shrC, bool add) {
	for (MYITE h = 0; h < H; h++) {
		for (MYITE w = 0; w < W; w++) {
			posit16_t a = A[h * W + w];
			posit16_t b = B[w];

			posit16_t res;
			if (add) {
				res = p16_add(a, b);
			} else {
				res = p16_sub(a, b);
			}

			X[h * W + w] = res;
		}
	}
	return;
}

// A = relu(A)
// A[N][H][W][C]
void Relu4D(posit16_t* A, MYITE N, MYITE H, MYITE W, MYITE C) {
	for (MYITE n = 0; n < N; n++) {
		for (MYITE h = 0; h < H; h++) {
			for (MYITE w = 0; w < W; w++) {
				for (MYITE c = 0; c < C; c++) {
					posit16_t a = A[n * H * W * C + h * W * C + w * C + c];
					posit16_t zero = convertDoubleToP16(0);
					if (p16_lt(a, zero)) {
						a = zero;
					}

					A[n * H * W * C + h * W * C + w * C + c] = a;
				}
			}
		}
	}
	return;
}

// B = relu6(A)
// A[N][H][W][C]
void Relu6(posit16_t* A, posit16_t* B, MYITE N, MYITE H, MYITE W, MYITE C, MYINT six, MYINT div) {
	for (MYITE n = 0; n < N; n++) {
		for (MYITE h = 0; h < H; h++) {
			for (MYITE w = 0; w < W; w++) {
				for (MYITE c = 0; c < C; c++) {
					posit16_t a = A[n * H * W * C + h * W * C + w * C + c];
					posit16_t zero = convertDoubleToP16(0);
					posit16_t six = convertDoubleToP16(6.0);
					if (p16_lt(a, zero)) {
						a = zero;
					}
					if (p16_lt(six, a)) {
						a = six;
					}

					B[n * H * W * C + h * W * C + w * C + c] = a;
				}
			}
		}
	}
	return;
}

// A = relu(A)
// A[N][H][W][C]
void Relu2D(posit16_t* A, MYITE H, MYITE W) {
	for (MYITE h = 0; h < H; h++) {
		for (MYITE w = 0; w < W; w++) {
			posit16_t a = A[h * W + w];
			posit16_t zero = convertDoubleToP16(0);
			if (p16_lt(a, zero)) {
				a = zero;
			}

			A[h * W + w] = a;
		}
	}
	return;
}

// B = maxpool(A)
// A[N][H][W][C], B[N][H][W][C]
void Maxpool(posit16_t* A, posit16_t* B, MYITE N, MYITE H, MYITE W, MYITE C, MYITE FH, MYITE FW, MYITE strideH, MYITE strideW, MYITE HPADL, MYITE HPADR, MYITE WPADL, MYITE WPADR) {
	MYITE HO = H / strideH;
	MYITE WO = W / strideW;

	for (MYITE n = 0; n < N; n++) {
		for (MYITE ho = 0; ho < HO; ho++) {
			for (MYITE wo = 0; wo < WO; wo++) {
				for (MYITE c = 0; c < C; c++) {

					posit16_t max = A[n * H * W * C + (strideH * ho) * W * C + (strideW * wo) * C + c];
					for (MYITE hs = 0; hs < FH; hs++) {
						for (MYITE ws = 0; ws < FW; ws++) {
							posit16_t a = A[n * H * W * C + ((strideH * ho) + hs) * W * C + ((strideW * wo) + ws) * C + c];
							if (p16_lt(max, a)) {
								max = a;
							}
						}
					}

					B[n * HO * WO * C + ho * WO * C + wo * C + c] = max;
				}
			}
		}
	}
	return;
}

void NormaliseL2(posit16_t* A, posit16_t* B, MYITE N, MYITE H, MYITE W, MYITE C, MYINT scaleA, MYINT shrA) {
	for (MYITE n = 0; n < N; n++) {
		for (MYITE h = 0; h < H; h++) {
			for (MYITE w = 0; w < W; w++) {

				// Calculate the sum square.
				posit16_t sumSquare = convertDoubleToP16(0);
				for (MYITE c = 0; c < C; c++) {
					posit16_t tmp = A[n * H * W * C + h * W * C + w * C + c];
					sumSquare = p16_add(sumSquare, p16_mul(tmp, tmp));
				}

				// Calculate the inverse square root of sumSquare.
				if (p16_eq(sumSquare,  convertDoubleToP16(0))) {
					sumSquare = convertDoubleToP16(1e-5);
				}

				posit16_t inverseNorm = p16_div(convertDoubleToP16(1), p16_sqrt(sumSquare));

				// Multiply all elements by the 1 / sqrt(sumSquare).
				for (MYITE c = 0; c < C; c++) {
					B[n * H * W * C + h * W * C + w * C + c]  = p16_mul(A[n * H * W * C + h * W * C + w * C + c], inverseNorm);
				}
			}
		}
	}
	return;
}

// B = exp(A)
void Exp(posit16_t* A, MYITE I, MYITE J, MYINT shrA, MYINT shrB, posit16_t* B) {
	for (MYITE i = 0; i < I; i++) {
		for (MYITE j = 0; j < J; j++) {
			posit16_t x = A[i * J + j];

			updateRangeOfExp(-1*convertP16ToDouble(x));

			B[i * J + j] = convertDoubleToP16(exp(convertP16ToDouble(x)));
		}
	}
	return;
}

// A = sigmoid(A)
void Sigmoid(posit16_t* A, MYITE I, MYITE J, float div, float add, float sigmoid_limit, MYINT scale_in, MYINT scale_out, posit16_t* B) {
	for (MYITE i = 0; i < I; i++) {
		for (MYITE j = 0; j < J; j++) {
			float x = convertP16ToDouble(A[i * J + j]), y;

#ifdef FLOATEXP
			y = 1 / (1 + exp(-x));
#else
			y = (x + 1) / 2;
			y = y > 0 ? y : 0;
			y = y < 1 ? y : 1;
#endif
			B[i * J + j] = convertDoubleToP16(y);
		}
	}
	return;
}

// A = AdjustScaleShr(A)
void AdjustScaleShr(posit16_t* A, MYITE I, MYITE J, MYINT scale) {
	return;
}

// A = AdjustScaleShl(A)
void AdjustScaleShl(posit16_t* A, MYITE I, MYITE J, MYINT scale) {
	return;
}


// C = A + B
void MatAddNN(posit32_t* A, posit32_t* B, posit32_t* C, MYITE I, MYITE J, MYINT shrA, MYINT shrB, MYINT shrC) {
	for (MYITE i = 0; i < I; i++) {
		for (MYITE j = 0; j < J; j++) {
			posit32_t a = A[i * J + j];
			posit32_t b = B[i * J + j];

			posit32_t c = p32_add(a, b);

			C[i * J + j] = c;
		}
	}
	return;
}

// C = A + B
void MatAddCN(const posit32_t* A, posit32_t* B, posit32_t* C, MYITE I, MYITE J, MYINT shrA, MYINT shrB, MYINT shrC) {
	for (MYITE i = 0; i < I; i++) {
		for (MYITE j = 0; j < J; j++) {
			posit32_t a = A[i * J + j];
			posit32_t b = B[i * J + j];

			posit32_t c = p32_add(a, b);

			C[i * J + j] = c;
		}
	}
	return;
}

// C = A + B
void MatAddNC(posit32_t* A, const posit32_t* B, posit32_t* C, MYITE I, MYITE J, MYINT shrA, MYINT shrB, MYINT shrC) {
	for (MYITE i = 0; i < I; i++) {
		for (MYITE j = 0; j < J; j++) {
			posit32_t a = A[i * J + j];
			posit32_t b = B[i * J + j];

			posit32_t c = p32_add(a, b);

			C[i * J + j] = c;
		}
	}
	return;
}

// C = A + B
void MatAddCC(const posit32_t* A, const posit32_t* B, posit32_t* C, MYITE I, MYITE J, MYINT shrA, MYINT shrB, MYINT shrC) {
	for (MYITE i = 0; i < I; i++) {
		for (MYITE j = 0; j < J; j++) {
			posit32_t a = A[i * J + j];
			posit32_t b = B[i * J + j];

			posit32_t c = p32_add(a, b);

			C[i * J + j] = c;
		}
	}
	return;
}

// C = a + B
void MatAddBroadCastA(posit32_t* A, posit32_t* B, posit32_t* C, MYITE I, MYITE J, MYINT shrA, MYINT shrB, MYINT shrC) {
	for (MYITE i = 0; i < I; i++) {
		for (MYITE j = 0; j < J; j++) {
			posit32_t a = *A;
			posit32_t b = B[i * J + j];

			posit32_t c = p32_add(a, b);

			C[i * J + j] = c;
		}
	}
	return;
}

// C = A + b
void MatAddBroadCastB(posit32_t* A, posit32_t* B, posit32_t* C, MYITE I, MYITE J, MYINT shrA, MYINT shrB, MYINT shrC) {
	for (MYITE i = 0; i < I; i++) {
		for (MYITE j = 0; j < J; j++) {
			posit32_t a = A[i * J + j];
			posit32_t b = *B;

			posit32_t c = p32_add(a, b);

			C[i * J + j] = c;
		}
	}
	return;
}

// C = A + B
void MatAdd4(posit32_t* A, posit32_t* B, posit32_t* X, MYITE N, MYITE H, MYITE W, MYITE C, MYINT shrA, MYINT shrB, MYINT shrC) {
	for (MYITE n = 0; n < N; n++) {
		for (MYITE h = 0; h < H; h++) {
			for (MYITE w = 0; w < W; w++) {
				for (MYITE c = 0; c < C; c++) {
					posit32_t a = A[n * H * W * C + h * W * C + w * C + c];
					posit32_t b = B[n * H * W * C + h * W * C + w * C + c];

					posit32_t x = p32_add(a, b);

					X[n * H * W * C + h * W * C + w * C + c] = x;
				}
			}
		}
	}
	return;
}

// C = A - B
void MatSub(posit32_t* A, const posit32_t* B, posit32_t* C, MYITE I, MYITE J, MYINT shrA, MYINT shrB, MYINT shrC) {
	for (MYITE i = 0; i < I; i++) {
		for (MYITE j = 0; j < J; j++) {
			posit32_t a = A[i * J + j];
			posit32_t b = B[i * J + j];

			posit32_t c = p32_sub(a, b);

			C[i * J + j] = c;
		}
	}
	return;
}

// C = a - B
void MatSubBroadCastA(posit32_t* A, posit32_t* B, posit32_t* C, MYITE I, MYITE J, MYINT shrA, MYINT shrB, MYINT shrC) {
	for (MYITE i = 0; i < I; i++) {
		for (MYITE j = 0; j < J; j++) {
			posit32_t a = *A;
			posit32_t b = B[i * J + j];

			posit32_t c = p32_sub(a, b);

			C[i * J + j] = c;
		}
	}
	return;
}

// C = A - b
void MatSubBroadCastB(posit32_t* A, posit32_t* B, posit32_t* C, MYITE I, MYITE J, MYINT shrA, MYINT shrB, MYINT shrC) {
	for (MYITE i = 0; i < I; i++) {
		for (MYITE j = 0; j < J; j++) {
			posit32_t a = A[i * J + j];
			posit32_t b = *B;

			posit32_t c = p32_sub(a, b);

			C[i * J + j] = c;
		}
	}
	return;
}

// C = A * B
void MatMulNN(posit32_t* A, posit32_t* B, posit32_t* C, MYITE I, MYITE K, MYITE J, MYINT shrA, MYINT shrB, MYITE H1, MYITE H2) {
	quire32_t qz;
	for (MYITE i = 0; i < I; i++) {
		for (MYITE j = 0; j < J; j++) {
			qz = q32_clr(qz);
			for (MYITE k = 0; k < K; k++) {
				posit32_t a = A[i * K + k];
				posit32_t b = B[k * J + j];
				qz = q32_fdp_add(qz, a, b);
				// tmp[k] = p8_mul(a, b);
			}

			// MYITE count = K, depth = 0;
			// bool shr = true;

			// while (depth < (H1 + H2)) {
			// 	if (depth >= H1) {
			// 		shr = false;
			// 	}

			// 	for (MYITE p = 0; p < (K / 2 + 1); p++) {
			// 		posit8_t sum;
			// 		if (p < (count >> 1)) {
			// 			sum = p8_add(tmp[2 * p], tmp[(2 * p) + 1]);
			// 		} else if ((p == (count >> 1)) && ((count & 1) == 1)) {
			// 			sum = tmp[2 * p];
			// 		} else {
			// 			sum = convertDoubleToP8(0.0);
			// 		}

			// 		if (shr) {
			// 			tmp[p] = sum;
			// 		} else {
			// 			tmp[p] = sum;
			// 		}
			// 	}

			// 	count = (count + 1) >> 1;
			// 	depth++;
			// }

			C[i * J + j] = q32_to_p32(qz);
		}
	}
	return;
}

// C = A * B
void MatMulCN(const posit32_t* A, posit32_t* B, posit32_t* C, MYITE I, MYITE K, MYITE J, MYINT shrA, MYINT shrB, MYITE H1, MYITE H2) {
	quire32_t qz;
	for (MYITE i = 0; i < I; i++) {
		for (MYITE j = 0; j < J; j++) {
			qz = q32_clr(qz);
			for (MYITE k = 0; k < K; k++) {
				posit32_t a = A[i * K + k];
				posit32_t b = B[k * J + j];
				qz = q32_fdp_add(qz, a, b);
				// tmp[k] = p8_mul(a, b);
			}

			// MYITE count = K, depth = 0;
			// bool shr = true;

			// while (depth < (H1 + H2)) {
			// 	if (depth >= H1) {
			// 		shr = false;
			// 	}

			// 	for (MYITE p = 0; p < (K / 2 + 1); p++) {
			// 		posit8_t sum;
			// 		if (p < (count >> 1)) {
			// 			sum = p8_add(tmp[2 * p], tmp[(2 * p) + 1]);
			// 		} else if ((p == (count >> 1)) && ((count & 1) == 1)) {
			// 			sum = tmp[2 * p];
			// 		} else {
			// 			sum = convertDoubleToP8(0.0);
			// 		}

			// 		if (shr) {
			// 			tmp[p] = sum;
			// 		} else {
			// 			tmp[p] = sum;
			// 		}
			// 	}

			// 	count = (count + 1) >> 1;
			// 	depth++;
			// }

			C[i * J + j] = q32_to_p32(qz);
		}
	}
	return;
}

// C = A * B
void MatMulNC(posit32_t* A, const posit32_t* B, posit32_t* C, MYITE I, MYITE K, MYITE J, MYINT shrA, MYINT shrB, MYITE H1, MYITE H2) {
	quire32_t qz;
	for (MYITE i = 0; i < I; i++) {
		for (MYITE j = 0; j < J; j++) {
			qz = q32_clr(qz);
			for (MYITE k = 0; k < K; k++) {
				posit32_t a = A[i * K + k];
				posit32_t b = B[k * J + j];
				qz = q32_fdp_add(qz, a, b);
				// tmp[k] = p8_mul(a, b);
			}

			// MYITE count = K, depth = 0;
			// bool shr = true;

			// while (depth < (H1 + H2)) {
			// 	if (depth >= H1) {
			// 		shr = false;
			// 	}

			// 	for (MYITE p = 0; p < (K / 2 + 1); p++) {
			// 		posit8_t sum;
			// 		if (p < (count >> 1)) {
			// 			sum = p8_add(tmp[2 * p], tmp[(2 * p) + 1]);
			// 		} else if ((p == (count >> 1)) && ((count & 1) == 1)) {
			// 			sum = tmp[2 * p];
			// 		} else {
			// 			sum = convertDoubleToP8(0.0);
			// 		}

			// 		if (shr) {
			// 			tmp[p] = sum;
			// 		} else {
			// 			tmp[p] = sum;
			// 		}
			// 	}

			// 	count = (count + 1) >> 1;
			// 	depth++;
			// }

			C[i * J + j] = q32_to_p32(qz);
		}
	}
	return;
}

// C = A * B
void MatMulCC(const posit32_t* A, const posit32_t* B, posit32_t* C, MYITE I, MYITE K, MYITE J, MYINT shrA, MYINT shrB, MYITE H1, MYITE H2) {
	quire32_t qz;
	for (MYITE i = 0; i < I; i++) {
		for (MYITE j = 0; j < J; j++) {
			qz = q32_clr(qz);
			for (MYITE k = 0; k < K; k++) {
				posit32_t a = A[i * K + k];
				posit32_t b = B[k * J + j];
				qz = q32_fdp_add(qz, a, b);
				// tmp[k] = p8_mul(a, b);
			}

			// MYITE count = K, depth = 0;
			// bool shr = true;

			// while (depth < (H1 + H2)) {
			// 	if (depth >= H1) {
			// 		shr = false;
			// 	}

			// 	for (MYITE p = 0; p < (K / 2 + 1); p++) {
			// 		posit8_t sum;
			// 		if (p < (count >> 1)) {
			// 			sum = p8_add(tmp[2 * p], tmp[(2 * p) + 1]);
			// 		} else if ((p == (count >> 1)) && ((count & 1) == 1)) {
			// 			sum = tmp[2 * p];
			// 		} else {
			// 			sum = convertDoubleToP8(0.0);
			// 		}

			// 		if (shr) {
			// 			tmp[p] = sum;
			// 		} else {
			// 			tmp[p] = sum;
			// 		}
			// 	}

			// 	count = (count + 1) >> 1;
			// 	depth++;
			// }

			C[i * J + j] = q32_to_p32(qz);
		}
	}
	return;
}

// C = A |*| B
void SparseMatMulX(const MYINT* Aidx, const posit32_t* Aval, posit32_t** B, posit32_t* C, int16_t K, MYINT shrA, MYINT shrB, MYINT shrC) {
	MYITE ite_idx = 0, ite_val = 0;
	for (MYITE k = 0; k < K; k++) {
		posit32_t b = B[k * 1][0];

		MYINT idx = Aidx[ite_idx];
		while (idx != 0) {
			posit32_t a = Aval[ite_val];

			posit32_t c = p32_mul(a, b);

			C[idx - 1] = p32_add(C[idx - 1], c);

			ite_idx++;
			ite_val++;

			idx = Aidx[ite_idx];
		}
		ite_idx++;
	}
	return;
}

// C = A |*| B
void SparseMatMul(const MYINT* Aidx, const posit32_t* Aval, posit32_t* B, posit32_t* C, int16_t K, MYINT shrA, MYINT shrB, MYINT shrC) {
	MYITE ite_idx = 0, ite_val = 0;
	for (MYITE k = 0; k < K; k++) {
		posit32_t b = B[k];

		MYINT idx = Aidx[ite_idx];
		while (idx != 0) {
			posit32_t a = Aval[ite_val];

			posit32_t c = p32_mul(a, b);

			C[idx - 1] = p32_add(C[idx - 1], c);

			ite_idx++;
			ite_val++;

			idx = Aidx[ite_idx];
		}
		ite_idx++;
	}

	return;
}

// C = A <*> B
void MulCir(posit32_t* A, posit32_t* B, posit32_t* C, MYITE I, MYITE J, MYINT shrA, MYINT shrB) {
	for (MYITE i = 0; i < I; i++) {
		for (MYITE j = 0; j < J; j++) {
			posit32_t a = A[i * J + j];
			posit32_t b = B[i * J + j];

			C[i * J + j] = p32_mul(a, b);
		}
	}
	return;
}

// A = tanh(A)
void TanH(posit32_t* A, MYITE I, MYITE J, float scale_in, float scale_out, posit32_t* B) {
	for (MYITE i = 0; i < I; i++) {
		for (MYITE j = 0; j < J; j++) {
			double x = convertP32ToDouble(A[i * J + j]), y;

			#ifdef FLOATEXP
				y = tanh(x);
			#else
				y = x > -1 ? x : -1;
				y = y < 1 ? y : 1;
			#endif

			B[i * J + j] = convertDoubleToP32(y);
		}
	}
	return;
}

// B = reverse(A, axis)
void Reverse2(posit32_t* A, MYITE axis, MYITE I, MYITE J, posit32_t* B) {
	for (MYITE i = 0; i < I; i++) {
		for (MYITE j = 0; j < J; j++) {
			MYITE i_prime = (axis == 0 ? (I - 1 - i) : i);
			MYITE j_prime = (axis == 1 ? (J - 1 - j) : j);

			B[i * J + j] = A[i_prime*J + j_prime];
		}
	}
	return;
}

// index = argmax(A)
void ArgMax(posit32_t* A, MYITE I, MYITE J, int* index) {
	posit32_t max = A[0];
	MYITE maxIndex = 0, counter = 0;
	for (MYITE i = 0; i < I; i++) {
		for (MYITE j = 0; j < J; j++) {
			posit32_t x = A[i * J + j];

			if (p32_lt(max, x)) {
				maxIndex = counter;
				max = x;
			}
			counter++;
		}
	}
	*index = maxIndex;
	return;
}

// A = A^T
void Transpose(posit32_t* A, posit32_t* B, MYITE I, MYITE J) {
	for (MYITE i = 0; i < I; i++) {
		for (MYITE j = 0; j < J; j++) {
			B[i * J + j] = A[j * I + i];
		}
	}
	return;
}

// C = a * B
void ScalarMul(posit32_t* A, posit32_t* B, posit32_t* C, MYITE I, MYITE J, MYINT shrA, MYINT shrB) {
	posit32_t a = *A;
	for (MYITE i = 0; i < I; i++) {
		for (MYITE j = 0; j < J; j++) {
			posit32_t b = B[i * J + j];
			C[i * J + j] = p32_mul(a, b);
		}
	}
	return;
}

// C = MBConv(A, params)
// A[N][H][W][Cin], C[N][Hout][Wout][Cout]
// X[HF][W][Ct], T[Ct], U[max(Ct, Cin, HF*WF)]
// F1[1][1][1][Cin][Ct], BN1W[Ct], BN1B[Ct]
// F2[Ct][HF][WF][1][1], BN2W[Ct], BN2B[Ct]
// F3[1][1][1][Ct][Cout], BN3W[Cout], BN3B[Cout]
void MBConv(posit32_t* A, const posit32_t* F1, const posit32_t* BN1W, const posit32_t* BN1B, const posit32_t* F2, const posit32_t* BN2W, const posit32_t* BN2B, const posit32_t* F3, const posit32_t* BN3W, const posit32_t* BN3B, posit32_t* C, posit32_t* X, posit32_t* T, posit32_t* U, MYITE N, MYITE H, MYITE W, MYITE Cin, MYITE Ct, MYITE HF, MYITE WF, MYITE Cout, MYITE Hout, MYITE Wout, MYITE HPADL, MYITE HPADR, MYITE WPADL, MYITE WPADR, MYITE HSTR, MYITE WSTR, MYITE D1, MYITE D2, MYITE D3, MYINT SIX_1, MYINT SIX_2, MYINT shr1, MYINT shr2, MYINT shr3, MYINT shr4, MYINT shr5, MYINT shr6, MYINT shr7, MYINT shr8, MYINT shr9, MYINT shl1, MYINT shl2, MYINT shl3, MYINT shl4, MYINT shl5, MYINT shl6, MYINT shl7, MYINT shl8, MYINT shl9, std::string name) {
	MYITE HOffsetL = (HF / 2) - HPADL;
	MYITE WOffsetL = (WF / 2) - WPADL;
	MYITE HOffsetR = (HF / 2) - HPADR;
	MYITE WOffsetR = (WF / 2) - WPADR;

	for (MYITE n = 0; n < N; n++) {
		MYITE margin = HOffsetL + (HF / 2 + 1) - HSTR > 0 ? HOffsetL + (HF/2 + 1) - HSTR : 0;
		MYITE nstart = HOffsetL - (HF / 2) < 0 ? 0 : HOffsetL - (HF / 2);
		for (MYITE i = nstart; i < margin; i++) {
			for (MYITE j = 0; j < W; j++) {
				for (MYITE k = 0; k < Ct; k++) {
					for (MYITE l = 0; l < Cin; l++) {
						U[l] = p32_mul(A[n * H * W * Cin + i * W * Cin + j * Cin + l], F1[l * Ct + k]);
					}
					MYITE totalEle = Cin;
					MYITE count = Cin;
					MYITE depth = 0;

					while (depth < D1) {
						for (MYITE p = 0; p < (totalEle / 2 + 1); p++) {
							if (p < count / 2) {
								U[p] = p32_add(U[2 * p], U[(2 * p) + 1]);
							} else if ((p == (count / 2)) && ((count % 2) == 1)) {
								U[p] = U[2 * p];
							} else {
								U[p] =  convertDoubleToP32(0);
							}
						}

						count = (count + 1) / 2;
						depth++;
					}
	
					posit32_t ar = p32_add(U[0], BN1B[k]);
					X[i * W * Ct + j * Ct + k] = p32_mul((ar),  BN1W[k]);
					Profile2(&ar, 1, 1, name + "t1");
					X[i * W * Ct + j * Ct + k] = p32_lt(X[i * W * Ct + j * Ct + k], convertDoubleToP32(0.0)) ? convertDoubleToP32(0.0) : X[i * W * Ct + j * Ct + k];
					X[i * W * Ct + j * Ct + k] = p32_lt(convertDoubleToP32(6.0), X[i * W * Ct + j * Ct + k]) ? convertDoubleToP32(6.0) : X[i * W * Ct + j * Ct + k];
				}
			}
		}

		for (MYITE h = HOffsetL, hout = 0; h < H - HOffsetR; hout++, h += HSTR) {

			for (MYITE i = 0; i < HSTR; i++) {
				for (MYITE j = 0; j < W; j++) {
					for (MYITE k = 0; k < Ct; k++) {
						MYITE iRed = (i + margin + hout * HSTR) % HF, iFull = i + margin + hout * HSTR;
						X[iRed * W * Ct + j * Ct + k] = convertDoubleToP32(0.0);
						for (MYITE l = 0; l < Cin; l++) {
							posit32_t a = iFull < H ? A[n * H * W * Cin + iFull * W * Cin + j * Cin + l] : convertDoubleToP32(0.0);
							U[l] = p32_mul(a, F1[l * Ct + k]);
						}
						MYITE totalEle = Cin;
						MYITE count = Cin;
						MYITE depth = 0;

						while (depth < D1) {
							for (MYITE p = 0; p <(totalEle / 2 + 1); p++) {
								if (p < count / 2) {
									U[p] = p32_add(U[2 * p], U[(2 * p) + 1]);
								} else if ((p == (count / 2)) && ((count % 2) == 1)) {
									U[p] = U[2 * p];
								} else {
									U[p] = convertDoubleToP32(0);
								}
							}

							count = (count + 1) / 2;
							depth++;
						}

						posit32_t ar = p32_add(U[0], BN1B[k]);
						X[iRed * W * Ct + j * Ct + k] = p32_mul((ar), BN1W[k]);
						Profile2(&ar, 1, 1, name + "t1");
						X[iRed * W * Ct + j * Ct + k] = p32_lt(X[iRed * W * Ct + j * Ct + k], convertDoubleToP32(0.0)) ? convertDoubleToP32(0.0) : X[iRed * W * Ct + j * Ct + k];
						X[iRed * W * Ct + j * Ct + k] = p32_lt(convertDoubleToP32(6.0), X[iRed * W * Ct + j * Ct + k]) ? convertDoubleToP32(6.0) : X[iRed * W * Ct + j * Ct + k];
					}
				}
			}

			for (MYITE w = WOffsetL, wout = 0; w < W - WOffsetR; w += WSTR, wout++) {
				for (MYITE g = 0; g < Ct; g++) {
					MYITE counter = 0;
					for (MYITE hf = -(HF / 2); hf <= (HF / 2); hf++) {
						for (MYITE wf = -(WF / 2); wf <= (WF / 2); wf++) {
							posit32_t x = (((h + hf) < 0) || ((h + hf) >= H) || ((w + wf) < 0) || ((w + wf) >= W)) ? convertDoubleToP32(0.0) : X[((h + hf) % HF) * W * Ct + (w + wf) * Ct + g];
							posit32_t b = F2[g * HF * WF + (hf + HF / 2) * WF + (wf + WF / 2)];
							U[counter] = p32_mul(x, b);
							counter++;
						}
					}
					MYITE totalEle = HF * WF;
					MYITE count = HF * WF;
					MYITE depth = 0;

					while (depth < D2) {
						for (MYITE p = 0; p < (totalEle / 2 + 1); p++) {
							if (p < count / 2) {
								U[p] = p32_add(U[2 * p], U[(2 * p) + 1]);
							} else if ((p == (count / 2)) && ((count % 2) == 1)) {
								U[p] = U[2 * p];
							} else {
								U[p] = convertDoubleToP32(0.0);
							}
						}

						count = (count + 1) / 2;
						depth++;
					}

					posit32_t ar = p32_add(U[0], BN2B[g]);
					T[g] = p32_mul((ar), BN2W[g]);
					Profile2(&ar, 1, 1, name + "t3");
					T[g] = p32_lt(T[g], convertDoubleToP32(0.0)) ? convertDoubleToP32(0.0) : T[g];
					T[g] = p32_lt(convertDoubleToP32(6.0), T[g]) ? convertDoubleToP32(6.0) : T[g];
				}

				for (MYITE i = 0; i < Cout; i++) {
					for (MYITE g = 0; g < Ct; g++) {
						U[g] = p32_mul(T[g], F3[g * Cout + i]);
					}
					MYITE totalEle = Ct;
					MYITE count = Ct;
					MYITE depth = 0;

					while (depth < D3) {
						for (MYITE p = 0; p < (totalEle / 2 + 1); p++) {
							if (p < count / 2) {
								U[p] = p32_add(U[2 * p], U[(2 * p) + 1]);
							} else if ((p == (count / 2)) && ((count % 2) == 1)) {
								U[p] = U[2 * p];
							} else {
								U[p] = convertDoubleToP32(0);
							}
						}

						count = (count + 1) / 2;
						depth++;
					}

					posit32_t ar = p32_add(U[0], BN3B[i]);
					C[n * Hout * Wout * Cout + hout * Wout * Cout + wout * Cout + i] = p32_mul((ar), BN3W[i]);
					Profile2(&ar, 1, 1, name + "t5");
				}
			}
		}
	}
}

// C = conv(A, B, <params>)
// A[N][H][W][CIN], B[G][HF][WF][CINF][COUTF], C[N][HOUT][WOUT][COUTF*G]
void Convolution(posit32_t* A, const posit32_t* B, posit32_t* C, posit32_t* tmp, MYITE N, MYITE H, MYITE W, MYITE CIN, MYITE HF, MYITE WF, MYITE CINF, MYITE COUTF, MYITE HOUT, MYITE WOUT, MYITE HPADL, MYITE HPADR, MYITE WPADL, MYITE WPADR, MYITE HSTR, MYITE WSTR, MYITE HDL, MYITE WDL, MYITE G, MYINT shrA, MYINT shrB, MYITE H1, MYITE H2) {
	MYITE HOffsetL = HDL * (HF / 2) - HPADL;
	MYITE WOffsetL = WDL * (WF / 2) - WPADL;
	MYITE HOffsetR = HDL * (HF / 2) - HPADR;
	MYITE WOffsetR = WDL * (WF / 2) - WPADR;

	for (MYITE n = 0; n < N; n++) {
		for (MYITE h = HOffsetL, hout = 0; h < H - HOffsetR; h += HSTR, hout++) {
			for (MYITE w = WOffsetL, wout = 0; w < W - WOffsetR; w += WSTR, wout++) {
				for (MYITE g = 0; g < G; g++) {
					for (MYITE co = 0; co < COUTF; co++) {

						MYITE counter = 0;
						for (MYITE hf = -(HF / 2); hf <= HF / 2; hf++) {
							for (MYITE wf = -(WF / 2); wf <= WF / 2; wf++) {
								for (MYITE ci = 0; ci < CINF; ci++) {
									posit32_t a = (((h + HDL * hf) < 0) || ((h + HDL * hf) >= H) || ((w + WDL * wf) < 0) || ((w + WDL * wf) >= W)) ? convertDoubleToP32(0) : A[n * H * W * CIN + (h + HDL * hf) * W * CIN + (w + WDL * wf) * CIN + (ci + g * CINF)];
									posit32_t b = B[g * HF * WF * CINF * COUTF + (hf + HF / 2) * WF * CINF * COUTF + (wf + WF / 2) * CINF * COUTF + ci * COUTF + co];

									tmp[counter] = p32_mul(a, b);
									counter++;
								}
							}
						}

						MYITE totalEle = HF * WF * CINF;
						MYITE count = HF * WF * CINF, depth = 0;
						bool shr = true;

						while (depth < (H1 + H2)) {
							if (depth >= H1) {
								shr = false;
							}

							for (MYITE p = 0; p < (totalEle / 2 + 1); p++) {
								posit32_t sum;
								if (p < (count >> 1)) {
									sum = p32_add(tmp[2 * p], tmp[(2 * p) + 1]);
								} else if ((p == (count >> 1)) && ((count & 1) == 1)) {
									sum = tmp[2 * p];
								} else {
									sum = convertDoubleToP32(0);
								}

								if (shr) {
									tmp[p] = sum;
								} else {
									tmp[p] = sum;
								}
							}

							count = (count + 1) >> 1;
							depth++;
						}

						C[n * HOUT * WOUT * (COUTF * G) + hout * WOUT * (COUTF * G) + wout * (COUTF * G) + (co + g * COUTF)] = tmp[0];
					}
				}
			}
		}
	}
}

// C = A # B
// A[N][H][W][CI], B[HF][WF][CI][CO], C[N][H][W][CO]
void Conv(posit32_t* A, const posit32_t* B, posit32_t* C, posit32_t* tmp, MYITE N, MYITE H, MYITE W, MYITE CI, MYITE HF, MYITE WF, MYITE CO, MYINT shrA, MYINT shrB, MYITE H1, MYITE H2) {
	MYITE padH = (HF - 1) / 2;
	MYITE padW = (WF - 1) / 2;

	for (MYITE n = 0; n < N; n++) {
		for (MYITE h = 0; h < H; h++) {
			for (MYITE w = 0; w < W; w++) {
				for (MYITE co = 0; co < CO; co++) {

					MYITE counter = 0;
					for (MYITE hf = 0; hf < HF; hf++) {
						for (MYITE wf = 0; wf < WF; wf++) {
							for (MYITE ci = 0; ci < CI; ci++) {
								posit32_t a = (((((h + hf) < padH) || ((h + hf) >= (H + padH))) || (((w + wf) < padW) || ((w + wf) >= (W + padW)))) ? convertDoubleToP32(0) : A[n * H * W * CI + ((h + hf) - padH) * W * CI + ((w + wf) - padW) * CI + ci]);
								posit32_t b = B[hf * WF * CI * CO + wf * CI * CO + ci * CO + co];

								tmp[counter] = p32_mul(a, b);
								counter++;
							}
						}
					}

					MYITE totalEle = HF * WF * CI;
					MYITE count = HF * WF * CI, depth = 0;
					bool shr = true;

					while (depth < (H1 + H2)) {
						if (depth >= H1) {
							shr = false;
						}

						for (MYITE p = 0; p < (totalEle / 2 + 1); p++) {
							posit32_t sum;
							if (p < (count >> 1)) {
								sum = p32_add(tmp[2 * p], tmp[(2 * p) + 1]);
							} else if ((p == (count >> 1)) && ((count & 1) == 1)) {
								sum = tmp[2 * p];
							} else {
								sum = convertDoubleToP32(0);
							}

							if (shr) {
								tmp[p] = sum;
							} else {
								tmp[p] = sum;
							}
						}

						count = (count + 1) >> 1;
						depth++;
					}

					C[n * H * W * CO + h * W * CO + w * CO + co] = tmp[0];
				}
			}
		}
	}

	return;
}

// A = A <+> B
// A[N][H][W][C], B[C]
void AddOrSubCir4D(posit32_t* A, const posit32_t* B, posit32_t* X, MYITE N, MYITE H, MYITE W, MYITE C, MYINT shrA, MYINT shrB, MYINT shrC, bool add) {
	for (MYITE n = 0; n < N; n++) {
		for (MYITE h = 0; h < H; h++) {
			for (MYITE w = 0; w < W; w++) {
				for (MYITE c = 0; c < C; c++) {
					posit32_t a = A[n * H * W * C + h * W * C + w * C + c];
					posit32_t b = B[c];

					posit32_t res;
					if (add) {
						res = p32_add(a, b);
					} else {
						res = p32_sub(a, b);
					}

					X[n * H * W * C + h * W * C + w * C + c] = res;
				}
			}
		}
	}
	return;
}

// A = A <+> B
// A[N][H][W][C], B[C]
void AddOrSubCir2D(posit32_t* A, const posit32_t* B, posit32_t* X, MYITE H, MYITE W, MYINT shrA, MYINT shrB, MYINT shrC, bool add) {
	for (MYITE h = 0; h < H; h++) {
		for (MYITE w = 0; w < W; w++) {
			posit32_t a = A[h * W + w];
			posit32_t b = B[w];

			posit32_t res;
			if (add) {
				res = p32_add(a, b);
			} else {
				res = p32_sub(a, b);
			}

			X[h * W + w] = res;
		}
	}
	return;
}

// A = relu(A)
// A[N][H][W][C]
void Relu4D(posit32_t* A, MYITE N, MYITE H, MYITE W, MYITE C) {
	for (MYITE n = 0; n < N; n++) {
		for (MYITE h = 0; h < H; h++) {
			for (MYITE w = 0; w < W; w++) {
				for (MYITE c = 0; c < C; c++) {
					posit32_t a = A[n * H * W * C + h * W * C + w * C + c];
					posit32_t zero = convertDoubleToP32(0);
					if (p32_lt(a, zero)) {
						a = zero;
					}

					A[n * H * W * C + h * W * C + w * C + c] = a;
				}
			}
		}
	}
	return;
}

// B = relu6(A)
// A[N][H][W][C]
void Relu6(posit32_t* A, posit32_t* B, MYITE N, MYITE H, MYITE W, MYITE C, MYINT six, MYINT div) {
	for (MYITE n = 0; n < N; n++) {
		for (MYITE h = 0; h < H; h++) {
			for (MYITE w = 0; w < W; w++) {
				for (MYITE c = 0; c < C; c++) {
					posit32_t a = A[n * H * W * C + h * W * C + w * C + c];
					posit32_t zero = convertDoubleToP32(0);
					posit32_t six = convertDoubleToP32(6.0);
					if (p32_lt(a, zero)) {
						a = zero;
					}
					if (p32_lt(six, a)) {
						a = six;
					}

					B[n * H * W * C + h * W * C + w * C + c] = a;
				}
			}
		}
	}
	return;
}

// A = relu(A)
// A[N][H][W][C]
void Relu2D(posit32_t* A, MYITE H, MYITE W) {
	for (MYITE h = 0; h < H; h++) {
		for (MYITE w = 0; w < W; w++) {
			posit32_t a = A[h * W + w];
			posit32_t zero = convertDoubleToP32(0);
			if (p32_lt(a, zero)) {
				a = zero;
			}

			A[h * W + w] = a;
		}
	}
	return;
}

// B = maxpool(A)
// A[N][H][W][C], B[N][H][W][C]
void Maxpool(posit32_t* A, posit32_t* B, MYITE N, MYITE H, MYITE W, MYITE C, MYITE FH, MYITE FW, MYITE strideH, MYITE strideW, MYITE HPADL, MYITE HPADR, MYITE WPADL, MYITE WPADR) {
	MYITE HO = H / strideH;
	MYITE WO = W / strideW;

	for (MYITE n = 0; n < N; n++) {
		for (MYITE ho = 0; ho < HO; ho++) {
			for (MYITE wo = 0; wo < WO; wo++) {
				for (MYITE c = 0; c < C; c++) {

					posit32_t max = A[n * H * W * C + (strideH * ho) * W * C + (strideW * wo) * C + c];
					for (MYITE hs = 0; hs < FH; hs++) {
						for (MYITE ws = 0; ws < FW; ws++) {
							posit32_t a = A[n * H * W * C + ((strideH * ho) + hs) * W * C + ((strideW * wo) + ws) * C + c];
							if (p32_lt(max, a)) {
								max = a;
							}
						}
					}

					B[n * HO * WO * C + ho * WO * C + wo * C + c] = max;
				}
			}
		}
	}
	return;
}

void NormaliseL2(posit32_t* A, posit32_t* B, MYITE N, MYITE H, MYITE W, MYITE C, MYINT scaleA, MYINT shrA) {
	for (MYITE n = 0; n < N; n++) {
		for (MYITE h = 0; h < H; h++) {
			for (MYITE w = 0; w < W; w++) {

				// Calculate the sum square.
				posit32_t sumSquare = convertDoubleToP32(0);
				for (MYITE c = 0; c < C; c++) {
					posit32_t tmp = A[n * H * W * C + h * W * C + w * C + c];
					sumSquare = p32_add(sumSquare, p32_mul(tmp, tmp));
				}

				// Calculate the inverse square root of sumSquare.
				if (p32_eq(sumSquare,  convertDoubleToP32(0))) {
					sumSquare = convertDoubleToP32(1e-5);
				}

				posit32_t inverseNorm = p32_div(convertDoubleToP32(1), p32_sqrt(sumSquare));

				// Multiply all elements by the 1 / sqrt(sumSquare).
				for (MYITE c = 0; c < C; c++) {
					B[n * H * W * C + h * W * C + w * C + c]  = p32_mul(A[n * H * W * C + h * W * C + w * C + c], inverseNorm);
				}
			}
		}
	}
	return;
}

// B = exp(A)
void Exp(posit32_t* A, MYITE I, MYITE J, MYINT shrA, MYINT shrB, posit32_t* B) {
	for (MYITE i = 0; i < I; i++) {
		for (MYITE j = 0; j < J; j++) {
			posit32_t x = A[i * J + j];

			updateRangeOfExp(-1*convertP32ToDouble(x));

			B[i * J + j] = convertDoubleToP32(exp(convertP32ToDouble(x)));
		}
	}
	return;
}

// A = sigmoid(A)
void Sigmoid(posit32_t* A, MYITE I, MYITE J, float div, float add, float sigmoid_limit, MYINT scale_in, MYINT scale_out, posit32_t* B) {
	for (MYITE i = 0; i < I; i++) {
		for (MYITE j = 0; j < J; j++) {
			float x = convertP32ToDouble(A[i * J + j]), y;

#ifdef FLOATEXP
			y = 1 / (1 + exp(-x));
#else
			y = (x + 1) / 2;
			y = y > 0 ? y : 0;
			y = y < 1 ? y : 1;
#endif
			B[i * J + j] = convertDoubleToP32(y);
		}
	}
	return;
}

// A = AdjustScaleShr(A)
void AdjustScaleShr(posit32_t* A, MYITE I, MYITE J, MYINT scale) {
	return;
}

// A = AdjustScaleShl(A)
void AdjustScaleShl(posit32_t* A, MYITE I, MYITE J, MYINT scale) {
	return;
}


// C = A + B
void MatAddNN(posit_2_t* A, posit_2_t* B, posit_2_t* C, MYITE I, MYITE J, MYINT shrA, MYINT shrB, MYINT shrC, int bitwidth) {
	for (MYITE i = 0; i < I; i++) {
		for (MYITE j = 0; j < J; j++) {
			posit_2_t a = A[i * J + j];
			posit_2_t b = B[i * J + j];

			posit_2_t c = pX2_add(a, b, bitwidth);

			C[i * J + j] = c;
		}
	}
	return;
}

// C = A + B
void MatAddCN(const posit_2_t* A, posit_2_t* B, posit_2_t* C, MYITE I, MYITE J, MYINT shrA, MYINT shrB, MYINT shrC, int bitwidth) {
	for (MYITE i = 0; i < I; i++) {
		for (MYITE j = 0; j < J; j++) {
			posit_2_t a = A[i * J + j];
			posit_2_t b = B[i * J + j];

			posit_2_t c = pX2_add(a, b, bitwidth);

			C[i * J + j] = c;
		}
	}
	return;
}

// C = A + B
void MatAddNC(posit_2_t* A, const posit_2_t* B, posit_2_t* C, MYITE I, MYITE J, MYINT shrA, MYINT shrB, MYINT shrC, int bitwidth) {
	for (MYITE i = 0; i < I; i++) {
		for (MYITE j = 0; j < J; j++) {
			posit_2_t a = A[i * J + j];
			posit_2_t b = B[i * J + j];

			posit_2_t c = pX2_add(a, b, bitwidth);

			C[i * J + j] = c;
		}
	}
	return;
}

// C = A + B
void MatAddCC(const posit_2_t* A, const posit_2_t* B, posit_2_t* C, MYITE I, MYITE J, MYINT shrA, MYINT shrB, MYINT shrC, int bitwidth) {
	for (MYITE i = 0; i < I; i++) {
		for (MYITE j = 0; j < J; j++) {
			posit_2_t a = A[i * J + j];
			posit_2_t b = B[i * J + j];

			posit_2_t c = pX2_add(a, b, bitwidth);

			C[i * J + j] = c;
		}
	}
	return;
}

// C = a + B
void MatAddBroadCastA(posit_2_t* A, posit_2_t* B, posit_2_t* C, MYITE I, MYITE J, MYINT shrA, MYINT shrB, MYINT shrC, int bitwidth) {
	for (MYITE i = 0; i < I; i++) {
		for (MYITE j = 0; j < J; j++) {
			posit_2_t a = *A;
			posit_2_t b = B[i * J + j];

			posit_2_t c = pX2_add(a, b, bitwidth);

			C[i * J + j] = c;
		}
	}
	return;
}

// C = A + b
void MatAddBroadCastB(posit_2_t* A, posit_2_t* B, posit_2_t* C, MYITE I, MYITE J, MYINT shrA, MYINT shrB, MYINT shrC, int bitwidth) {
	for (MYITE i = 0; i < I; i++) {
		for (MYITE j = 0; j < J; j++) {
			posit_2_t a = A[i * J + j];
			posit_2_t b = *B;

			posit_2_t c = pX2_add(a, b, bitwidth);

			C[i * J + j] = c;
		}
	}
	return;
}

// C = A + B
void MatAdd4(posit_2_t* A, posit_2_t* B, posit_2_t* X, MYITE N, MYITE H, MYITE W, MYITE C, MYINT shrA, MYINT shrB, MYINT shrC, int bitwidth) {
	for (MYITE n = 0; n < N; n++) {
		for (MYITE h = 0; h < H; h++) {
			for (MYITE w = 0; w < W; w++) {
				for (MYITE c = 0; c < C; c++) {
					posit_2_t a = A[n * H * W * C + h * W * C + w * C + c];
					posit_2_t b = B[n * H * W * C + h * W * C + w * C + c];

					posit_2_t x = pX2_add(a, b, bitwidth);

					X[n * H * W * C + h * W * C + w * C + c] = x;
				}
			}
		}
	}
	return;
}

// C = A - B
void MatSub(posit_2_t* A, const posit_2_t* B, posit_2_t* C, MYITE I, MYITE J, MYINT shrA, MYINT shrB, MYINT shrC, int bitwidth) {
	for (MYITE i = 0; i < I; i++) {
		for (MYITE j = 0; j < J; j++) {
			posit_2_t a = A[i * J + j];
			posit_2_t b = B[i * J + j];

			posit_2_t c = pX2_sub(a, b, bitwidth);

			C[i * J + j] = c;
		}
	}
	return;
}

// C = a - B
void MatSubBroadCastA(posit_2_t* A, posit_2_t* B, posit_2_t* C, MYITE I, MYITE J, MYINT shrA, MYINT shrB, MYINT shrC, int bitwidth) {
	for (MYITE i = 0; i < I; i++) {
		for (MYITE j = 0; j < J; j++) {
			posit_2_t a = *A;
			posit_2_t b = B[i * J + j];

			posit_2_t c = pX2_sub(a, b, bitwidth);

			C[i * J + j] = c;
		}
	}
	return;
}

// C = A - b
void MatSubBroadCastB(posit_2_t* A, posit_2_t* B, posit_2_t* C, MYITE I, MYITE J, MYINT shrA, MYINT shrB, MYINT shrC, int bitwidth) {
	for (MYITE i = 0; i < I; i++) {
		for (MYITE j = 0; j < J; j++) {
			posit_2_t a = A[i * J + j];
			posit_2_t b = *B;

			posit_2_t c = pX2_sub(a, b, bitwidth);

			C[i * J + j] = c;
		}
	}
	return;
}

// C = A * B
void MatMulNN(posit_2_t* A, posit_2_t* B, posit_2_t* C, MYITE I, MYITE K, MYITE J, MYINT shrA, MYINT shrB, MYITE H1, MYITE H2, int bitwidth) {
	quire_2_t qz;
	for (MYITE i = 0; i < I; i++) {
		for (MYITE j = 0; j < J; j++) {
			qz = qX2_clr(qz);
			for (MYITE k = 0; k < K; k++) {
				posit_2_t a = A[i * K + k];
				posit_2_t b = B[k * J + j];
				qz = qX2_fdp_add(qz, a, b);
				// tmp[k] = pX2_mul(a, b);
			}

			// MYITE count = K, depth = 0;
			// bool shr = true;

			// while (depth < (H1 + H2)) {
			// 	if (depth >= H1) {
			// 		shr = false;
			// 	}

			// 	for (MYITE p = 0; p < (K / 2 + 1); p++) {
			// 		posit_2_t sum;
			// 		if (p < (count >> 1)) {
			// 			sum = pX2_add(tmp[2 * p], tmp[(2 * p) + 1]);
			// 		} else if ((p == (count >> 1)) && ((count & 1) == 1)) {
			// 			sum = tmp[2 * p];
			// 		} else {
			// 			sum = convertDoubleToPX2(0.0);
			// 		}

			// 		if (shr) {
			// 			tmp[p] = sum;
			// 		} else {
			// 			tmp[p] = sum;
			// 		}
			// 	}

			// 	count = (count + 1) >> 1;
			// 	depth++;
			// }

			C[i * J + j] = qX2_to_pX2(qz, bitwidth);
		}
	}
	return;
}

// C = A * B
void MatMulCN(const posit_2_t* A, posit_2_t* B, posit_2_t* C, MYITE I, MYITE K, MYITE J, MYINT shrA, MYINT shrB, MYITE H1, MYITE H2, int bitwidth) {
	quire_2_t qz;
	for (MYITE i = 0; i < I; i++) {
		for (MYITE j = 0; j < J; j++) {
			qz = qX2_clr(qz);
			for (MYITE k = 0; k < K; k++) {
				posit_2_t a = A[i * K + k];
				posit_2_t b = B[k * J + j];
				qz = qX2_fdp_add(qz, a, b);
				// tmp[k] = pX2_mul(a, b);
			}

			// MYITE count = K, depth = 0;
			// bool shr = true;

			// while (depth < (H1 + H2)) {
			// 	if (depth >= H1) {
			// 		shr = false;
			// 	}

			// 	for (MYITE p = 0; p < (K / 2 + 1); p++) {
			// 		posit_2_t sum;
			// 		if (p < (count >> 1)) {
			// 			sum = pX2_add(tmp[2 * p], tmp[(2 * p) + 1]);
			// 		} else if ((p == (count >> 1)) && ((count & 1) == 1)) {
			// 			sum = tmp[2 * p];
			// 		} else {
			// 			sum = convertDoubleToPX2(0.0);
			// 		}

			// 		if (shr) {
			// 			tmp[p] = sum;
			// 		} else {
			// 			tmp[p] = sum;
			// 		}
			// 	}

			// 	count = (count + 1) >> 1;
			// 	depth++;
			// }

			C[i * J + j] = qX2_to_pX2(qz, bitwidth);
		}
	}
	return;
}

// C = A * B
void MatMulNC(posit_2_t* A, const posit_2_t* B, posit_2_t* C, MYITE I, MYITE K, MYITE J, MYINT shrA, MYINT shrB, MYITE H1, MYITE H2, int bitwidth) {
	quire_2_t qz;
	for (MYITE i = 0; i < I; i++) {
		for (MYITE j = 0; j < J; j++) {
			qz = qX2_clr(qz);
			for (MYITE k = 0; k < K; k++) {
				posit_2_t a = A[i * K + k];
				posit_2_t b = B[k * J + j];
				qz = qX2_fdp_add(qz, a, b);
				// tmp[k] = pX2_mul(a, b);
			}

			// MYITE count = K, depth = 0;
			// bool shr = true;

			// while (depth < (H1 + H2)) {
			// 	if (depth >= H1) {
			// 		shr = false;
			// 	}

			// 	for (MYITE p = 0; p < (K / 2 + 1); p++) {
			// 		posit_2_t sum;
			// 		if (p < (count >> 1)) {
			// 			sum = pX2_add(tmp[2 * p], tmp[(2 * p) + 1]);
			// 		} else if ((p == (count >> 1)) && ((count & 1) == 1)) {
			// 			sum = tmp[2 * p];
			// 		} else {
			// 			sum = convertDoubleToPX2(0.0);
			// 		}

			// 		if (shr) {
			// 			tmp[p] = sum;
			// 		} else {
			// 			tmp[p] = sum;
			// 		}
			// 	}

			// 	count = (count + 1) >> 1;
			// 	depth++;
			// }

			C[i * J + j] = qX2_to_pX2(qz, bitwidth);
		}
	}
	return;
}

// C = A * B
void MatMulCC(const posit_2_t* A, const posit_2_t* B, posit_2_t* C, MYITE I, MYITE K, MYITE J, MYINT shrA, MYINT shrB, MYITE H1, MYITE H2, int bitwidth) {
	quire_2_t qz;
	for (MYITE i = 0; i < I; i++) {
		for (MYITE j = 0; j < J; j++) {
			qz = qX2_clr(qz);
			for (MYITE k = 0; k < K; k++) {
				posit_2_t a = A[i * K + k];
				posit_2_t b = B[k * J + j];
				qz = qX2_fdp_add(qz, a, b);
				// tmp[k] = pX2_mul(a, b);
			}

			// MYITE count = K, depth = 0;
			// bool shr = true;

			// while (depth < (H1 + H2)) {
			// 	if (depth >= H1) {
			// 		shr = false;
			// 	}

			// 	for (MYITE p = 0; p < (K / 2 + 1); p++) {
			// 		posit_2_t sum;
			// 		if (p < (count >> 1)) {
			// 			sum = pX2_add(tmp[2 * p], tmp[(2 * p) + 1]);
			// 		} else if ((p == (count >> 1)) && ((count & 1) == 1)) {
			// 			sum = tmp[2 * p];
			// 		} else {
			// 			sum = convertDoubleToPX2(0.0);
			// 		}

			// 		if (shr) {
			// 			tmp[p] = sum;
			// 		} else {
			// 			tmp[p] = sum;
			// 		}
			// 	}

			// 	count = (count + 1) >> 1;
			// 	depth++;
			// }

			C[i * J + j] = qX2_to_pX2(qz, bitwidth);
		}
	}
	return;
}

// C = A |*| B
void SparseMatMulX(const MYINT* Aidx, const posit_2_t* Aval, posit_2_t** B, posit_2_t* C, int16_t K, MYINT shrA, MYINT shrB, MYINT shrC, int bitwidth) {
	MYITE ite_idx = 0, ite_val = 0;
	for (MYITE k = 0; k < K; k++) {
		posit_2_t b = B[k * 1][0];

		MYINT idx = Aidx[ite_idx];
		while (idx != 0) {
			posit_2_t a = Aval[ite_val];

			posit_2_t c = pX2_mul(a, b, bitwidth);

			C[idx - 1] = pX2_add(C[idx - 1], c, bitwidth);

			ite_idx++;
			ite_val++;

			idx = Aidx[ite_idx];
		}
		ite_idx++;
	}
	return;
}

// C = A |*| B
void SparseMatMul(const MYINT* Aidx, const posit_2_t* Aval, posit_2_t* B, posit_2_t* C, int16_t K, MYINT shrA, MYINT shrB, MYINT shrC, int bitwidth) {
	MYITE ite_idx = 0, ite_val = 0;
	for (MYITE k = 0; k < K; k++) {
		posit_2_t b = B[k];

		MYINT idx = Aidx[ite_idx];
		while (idx != 0) {
			posit_2_t a = Aval[ite_val];

			posit_2_t c = pX2_mul(a, b, bitwidth);

			C[idx - 1] = pX2_add(C[idx - 1], c, bitwidth);

			ite_idx++;
			ite_val++;

			idx = Aidx[ite_idx];
		}
		ite_idx++;
	}

	return;
}

// C = A <*> B
void MulCir(posit_2_t* A, posit_2_t* B, posit_2_t* C, MYITE I, MYITE J, MYINT shrA, MYINT shrB, int bitwidth) {
	for (MYITE i = 0; i < I; i++) {
		for (MYITE j = 0; j < J; j++) {
			posit_2_t a = A[i * J + j];
			posit_2_t b = B[i * J + j];

			C[i * J + j] = pX2_mul(a, b, bitwidth);
		}
	}
	return;
}

// A = tanh(A)
void TanH(posit_2_t* A, MYITE I, MYITE J, float scale_in, float scale_out, posit_2_t* B, int bitwidth) {
	for (MYITE i = 0; i < I; i++) {
		for (MYITE j = 0; j < J; j++) {
			double x = convertPX2ToDouble(A[i * J + j]), y;

			#ifdef FLOATEXP
				y = tanh(x);
			#else
				y = x > -1 ? x : -1;
				y = y < 1 ? y : 1;
			#endif

			B[i * J + j] = convertDoubleToPX2(y, bitwidth);
		}
	}
	return;
}

// B = reverse(A, axis)
void Reverse2(posit_2_t* A, MYITE axis, MYITE I, MYITE J, posit_2_t* B, int bitwidth) {
	for (MYITE i = 0; i < I; i++) {
		for (MYITE j = 0; j < J; j++) {
			MYITE i_prime = (axis == 0 ? (I - 1 - i) : i);
			MYITE j_prime = (axis == 1 ? (J - 1 - j) : j);

			B[i * J + j] = A[i_prime*J + j_prime];
		}
	}
	return;
}

// index = argmax(A)
void ArgMax(posit_2_t* A, MYITE I, MYITE J, int* index, int bitwidth) {
	posit_2_t max = A[0];
	MYITE maxIndex = 0, counter = 0;
	for (MYITE i = 0; i < I; i++) {
		for (MYITE j = 0; j < J; j++) {
			posit_2_t x = A[i * J + j];

			if (pX2_lt(max, x)) {
				maxIndex = counter;
				max = x;
			}
			counter++;
		}
	}
	*index = maxIndex;
	return;
}

// A = A^T
void Transpose(posit_2_t* A, posit_2_t* B, MYITE I, MYITE J) {
	for (MYITE i = 0; i < I; i++) {
		for (MYITE j = 0; j < J; j++) {
			B[i * J + j] = A[j * I + i];
		}
	}
	return;
}

// C = a * B
void ScalarMul(posit_2_t* A, posit_2_t* B, posit_2_t* C, MYITE I, MYITE J, MYINT shrA, MYINT shrB, int bitwidth) {
	posit_2_t a = *A;
	for (MYITE i = 0; i < I; i++) {
		for (MYITE j = 0; j < J; j++) {
			posit_2_t b = B[i * J + j];
			C[i * J + j] = pX2_mul(a, b, bitwidth);
		}
	}
	return;
}

// C = MBConv(A, params)
// A[N][H][W][Cin], C[N][Hout][Wout][Cout]
// X[HF][W][Ct], T[Ct], U[max(Ct, Cin, HF*WF)]
// F1[1][1][1][Cin][Ct], BN1W[Ct], BN1B[Ct]
// F2[Ct][HF][WF][1][1], BN2W[Ct], BN2B[Ct]
// F3[1][1][1][Ct][Cout], BN3W[Cout], BN3B[Cout]
void MBConv(posit_2_t* A, const posit_2_t* F1, const posit_2_t* BN1W, const posit_2_t* BN1B, const posit_2_t* F2, const posit_2_t* BN2W, const posit_2_t* BN2B, const posit_2_t* F3, const posit_2_t* BN3W, const posit_2_t* BN3B, posit_2_t* C, posit_2_t* X, posit_2_t* T, posit_2_t* U, MYITE N, MYITE H, MYITE W, MYITE Cin, MYITE Ct, MYITE HF, MYITE WF, MYITE Cout, MYITE Hout, MYITE Wout, MYITE HPADL, MYITE HPADR, MYITE WPADL, MYITE WPADR, MYITE HSTR, MYITE WSTR, MYITE D1, MYITE D2, MYITE D3, MYINT SIX_1, MYINT SIX_2, MYINT shr1, MYINT shr2, MYINT shr3, MYINT shr4, MYINT shr5, MYINT shr6, MYINT shr7, MYINT shr8, MYINT shr9, MYINT shl1, MYINT shl2, MYINT shl3, MYINT shl4, MYINT shl5, MYINT shl6, MYINT shl7, MYINT shl8, MYINT shl9, std::string name, int bitwidth) {
	MYITE HOffsetL = (HF / 2) - HPADL;
	MYITE WOffsetL = (WF / 2) - WPADL;
	MYITE HOffsetR = (HF / 2) - HPADR;
	MYITE WOffsetR = (WF / 2) - WPADR;

	for (MYITE n = 0; n < N; n++) {
		MYITE margin = HOffsetL + (HF / 2 + 1) - HSTR > 0 ? HOffsetL + (HF/2 + 1) - HSTR : 0;
		MYITE nstart = HOffsetL - (HF / 2) < 0 ? 0 : HOffsetL - (HF / 2);
		for (MYITE i = nstart; i < margin; i++) {
			for (MYITE j = 0; j < W; j++) {
				for (MYITE k = 0; k < Ct; k++) {
					for (MYITE l = 0; l < Cin; l++) {
						U[l] = pX2_mul(A[n * H * W * Cin + i * W * Cin + j * Cin + l], F1[l * Ct + k], bitwidth);
					}
					MYITE totalEle = Cin;
					MYITE count = Cin;
					MYITE depth = 0;

					while (depth < D1) {
						for (MYITE p = 0; p < (totalEle / 2 + 1); p++) {
							if (p < count / 2) {
								U[p] = pX2_add(U[2 * p], U[(2 * p) + 1], bitwidth);
							} else if ((p == (count / 2)) && ((count % 2) == 1)) {
								U[p] = U[2 * p];
							} else {
								U[p] =  convertDoubleToPX2(0, bitwidth);
							}
						}

						count = (count + 1) / 2;
						depth++;
					}
	
					posit_2_t ar = pX2_add(U[0], BN1B[k], bitwidth);
					X[i * W * Ct + j * Ct + k] = pX2_mul((ar),  BN1W[k], bitwidth);
					Profile2(&ar, 1, 1, name + "t1", bitwidth);
					X[i * W * Ct + j * Ct + k] = pX2_lt(X[i * W * Ct + j * Ct + k], convertDoubleToPX2(0.0, bitwidth)) ? convertDoubleToPX2(0.0, bitwidth) : X[i * W * Ct + j * Ct + k];
					X[i * W * Ct + j * Ct + k] = pX2_lt(convertDoubleToPX2(6.0, bitwidth), X[i * W * Ct + j * Ct + k]) ? convertDoubleToPX2(6.0, bitwidth) : X[i * W * Ct + j * Ct + k];
				}
			}
		}

		for (MYITE h = HOffsetL, hout = 0; h < H - HOffsetR; hout++, h += HSTR) {

			for (MYITE i = 0; i < HSTR; i++) {
				for (MYITE j = 0; j < W; j++) {
					for (MYITE k = 0; k < Ct; k++) {
						MYITE iRed = (i + margin + hout * HSTR) % HF, iFull = i + margin + hout * HSTR;
						X[iRed * W * Ct + j * Ct + k] = convertDoubleToPX2(0.0, bitwidth);
						for (MYITE l = 0; l < Cin; l++) {
							posit_2_t a = iFull < H ? A[n * H * W * Cin + iFull * W * Cin + j * Cin + l] : convertDoubleToPX2(0.0, bitwidth);
							U[l] = pX2_mul(a, F1[l * Ct + k], bitwidth);
						}
						MYITE totalEle = Cin;
						MYITE count = Cin;
						MYITE depth = 0;

						while (depth < D1) {
							for (MYITE p = 0; p <(totalEle / 2 + 1); p++) {
								if (p < count / 2) {
									U[p] = pX2_add(U[2 * p], U[(2 * p) + 1], bitwidth);
								} else if ((p == (count / 2)) && ((count % 2) == 1)) {
									U[p] = U[2 * p];
								} else {
									U[p] = convertDoubleToPX2(0, bitwidth);
								}
							}

							count = (count + 1) / 2;
							depth++;
						}

						posit_2_t ar = pX2_add(U[0], BN1B[k], bitwidth);
						X[iRed * W * Ct + j * Ct + k] = pX2_mul((ar), BN1W[k], bitwidth);
						Profile2(&ar, 1, 1, name + "t1", bitwidth);
						X[iRed * W * Ct + j * Ct + k] = pX2_lt(X[iRed * W * Ct + j * Ct + k], convertDoubleToPX2(0.0, bitwidth)) ? convertDoubleToPX2(0.0, bitwidth) : X[iRed * W * Ct + j * Ct + k];
						X[iRed * W * Ct + j * Ct + k] = pX2_lt(convertDoubleToPX2(6.0, bitwidth), X[iRed * W * Ct + j * Ct + k]) ? convertDoubleToPX2(6.0, bitwidth) : X[iRed * W * Ct + j * Ct + k];
					}
				}
			}

			for (MYITE w = WOffsetL, wout = 0; w < W - WOffsetR; w += WSTR, wout++) {
				for (MYITE g = 0; g < Ct; g++) {
					MYITE counter = 0;
					for (MYITE hf = -(HF / 2); hf <= (HF / 2); hf++) {
						for (MYITE wf = -(WF / 2); wf <= (WF / 2); wf++) {
							posit_2_t x = (((h + hf) < 0) || ((h + hf) >= H) || ((w + wf) < 0) || ((w + wf) >= W)) ? convertDoubleToPX2(0.0, bitwidth) : X[((h + hf) % HF) * W * Ct + (w + wf) * Ct + g];
							posit_2_t b = F2[g * HF * WF + (hf + HF / 2) * WF + (wf + WF / 2)];
							U[counter] = pX2_mul(x, b, bitwidth);
							counter++;
						}
					}
					MYITE totalEle = HF * WF;
					MYITE count = HF * WF;
					MYITE depth = 0;

					while (depth < D2) {
						for (MYITE p = 0; p < (totalEle / 2 + 1); p++) {
							if (p < count / 2) {
								U[p] = pX2_add(U[2 * p], U[(2 * p) + 1], bitwidth);
							} else if ((p == (count / 2)) && ((count % 2) == 1)) {
								U[p] = U[2 * p];
							} else {
								U[p] = convertDoubleToPX2(0.0, bitwidth);
							}
						}

						count = (count + 1) / 2;
						depth++;
					}

					posit_2_t ar = pX2_add(U[0], BN2B[g], bitwidth);
					T[g] = pX2_mul((ar), BN2W[g], bitwidth);
					Profile2(&ar, 1, 1, name + "t3", bitwidth);
					T[g] = pX2_lt(T[g], convertDoubleToPX2(0.0, bitwidth)) ? convertDoubleToPX2(0.0, bitwidth) : T[g];
					T[g] = pX2_lt(convertDoubleToPX2(6.0, bitwidth), T[g]) ? convertDoubleToPX2(6.0, bitwidth) : T[g];
				}

				for (MYITE i = 0; i < Cout; i++) {
					for (MYITE g = 0; g < Ct; g++) {
						U[g] = pX2_mul(T[g], F3[g * Cout + i], bitwidth);
					}
					MYITE totalEle = Ct;
					MYITE count = Ct;
					MYITE depth = 0;

					while (depth < D3) {
						for (MYITE p = 0; p < (totalEle / 2 + 1); p++) {
							if (p < count / 2) {
								U[p] = pX2_add(U[2 * p], U[(2 * p) + 1], bitwidth);
							} else if ((p == (count / 2)) && ((count % 2) == 1)) {
								U[p] = U[2 * p];
							} else {
								U[p] = convertDoubleToPX2(0, bitwidth);
							}
						}

						count = (count + 1) / 2;
						depth++;
					}

					posit_2_t ar = pX2_add(U[0], BN3B[i], bitwidth);
					C[n * Hout * Wout * Cout + hout * Wout * Cout + wout * Cout + i] = pX2_mul((ar), BN3W[i], bitwidth);
					Profile2(&ar, 1, 1, name + "t5", bitwidth);
				}
			}
		}
	}
}

// C = conv(A, B, <params>)
// A[N][H][W][CIN], B[G][HF][WF][CINF][COUTF], C[N][HOUT][WOUT][COUTF*G]
void Convolution(posit_2_t* A, const posit_2_t* B, posit_2_t* C, posit_2_t* tmp, MYITE N, MYITE H, MYITE W, MYITE CIN, MYITE HF, MYITE WF, MYITE CINF, MYITE COUTF, MYITE HOUT, MYITE WOUT, MYITE HPADL, MYITE HPADR, MYITE WPADL, MYITE WPADR, MYITE HSTR, MYITE WSTR, MYITE HDL, MYITE WDL, MYITE G, MYINT shrA, MYINT shrB, MYITE H1, MYITE H2, int bitwidth) {
	MYITE HOffsetL = HDL * (HF / 2) - HPADL;
	MYITE WOffsetL = WDL * (WF / 2) - WPADL;
	MYITE HOffsetR = HDL * (HF / 2) - HPADR;
	MYITE WOffsetR = WDL * (WF / 2) - WPADR;

	for (MYITE n = 0; n < N; n++) {
		for (MYITE h = HOffsetL, hout = 0; h < H - HOffsetR; h += HSTR, hout++) {
			for (MYITE w = WOffsetL, wout = 0; w < W - WOffsetR; w += WSTR, wout++) {
				for (MYITE g = 0; g < G; g++) {
					for (MYITE co = 0; co < COUTF; co++) {

						MYITE counter = 0;
						for (MYITE hf = -(HF / 2); hf <= HF / 2; hf++) {
							for (MYITE wf = -(WF / 2); wf <= WF / 2; wf++) {
								for (MYITE ci = 0; ci < CINF; ci++) {
									posit_2_t a = (((h + HDL * hf) < 0) || ((h + HDL * hf) >= H) || ((w + WDL * wf) < 0) || ((w + WDL * wf) >= W)) ? convertDoubleToPX2(0, bitwidth) : A[n * H * W * CIN + (h + HDL * hf) * W * CIN + (w + WDL * wf) * CIN + (ci + g * CINF)];
									posit_2_t b = B[g * HF * WF * CINF * COUTF + (hf + HF / 2) * WF * CINF * COUTF + (wf + WF / 2) * CINF * COUTF + ci * COUTF + co];

									tmp[counter] = pX2_mul(a, b, bitwidth);
									counter++;
								}
							}
						}

						MYITE totalEle = HF * WF * CINF;
						MYITE count = HF * WF * CINF, depth = 0;
						bool shr = true;

						while (depth < (H1 + H2)) {
							if (depth >= H1) {
								shr = false;
							}

							for (MYITE p = 0; p < (totalEle / 2 + 1); p++) {
								posit_2_t sum;
								if (p < (count >> 1)) {
									sum = pX2_add(tmp[2 * p], tmp[(2 * p) + 1], bitwidth);
								} else if ((p == (count >> 1)) && ((count & 1) == 1)) {
									sum = tmp[2 * p];
								} else {
									sum = convertDoubleToPX2(0, bitwidth);
								}

								if (shr) {
									tmp[p] = sum;
								} else {
									tmp[p] = sum;
								}
							}

							count = (count + 1) >> 1;
							depth++;
						}

						C[n * HOUT * WOUT * (COUTF * G) + hout * WOUT * (COUTF * G) + wout * (COUTF * G) + (co + g * COUTF)] = tmp[0];
					}
				}
			}
		}
	}
}

// C = A # B
// A[N][H][W][CI], B[HF][WF][CI][CO], C[N][H][W][CO]
void Conv(posit_2_t* A, const posit_2_t* B, posit_2_t* C, posit_2_t* tmp, MYITE N, MYITE H, MYITE W, MYITE CI, MYITE HF, MYITE WF, MYITE CO, MYINT shrA, MYINT shrB, MYITE H1, MYITE H2, int bitwidth) {
	MYITE padH = (HF - 1) / 2;
	MYITE padW = (WF - 1) / 2;

	for (MYITE n = 0; n < N; n++) {
		for (MYITE h = 0; h < H; h++) {
			for (MYITE w = 0; w < W; w++) {
				for (MYITE co = 0; co < CO; co++) {

					MYITE counter = 0;
					for (MYITE hf = 0; hf < HF; hf++) {
						for (MYITE wf = 0; wf < WF; wf++) {
							for (MYITE ci = 0; ci < CI; ci++) {
								posit_2_t a = (((((h + hf) < padH) || ((h + hf) >= (H + padH))) || (((w + wf) < padW) || ((w + wf) >= (W + padW)))) ? convertDoubleToPX2(0, bitwidth) : A[n * H * W * CI + ((h + hf) - padH) * W * CI + ((w + wf) - padW) * CI + ci]);
								posit_2_t b = B[hf * WF * CI * CO + wf * CI * CO + ci * CO + co];

								tmp[counter] = pX2_mul(a, b, bitwidth);
								counter++;
							}
						}
					}

					MYITE totalEle = HF * WF * CI;
					MYITE count = HF * WF * CI, depth = 0;
					bool shr = true;

					while (depth < (H1 + H2)) {
						if (depth >= H1) {
							shr = false;
						}

						for (MYITE p = 0; p < (totalEle / 2 + 1); p++) {
							posit_2_t sum;
							if (p < (count >> 1)) {
								sum = pX2_add(tmp[2 * p], tmp[(2 * p) + 1], bitwidth);
							} else if ((p == (count >> 1)) && ((count & 1) == 1)) {
								sum = tmp[2 * p];
							} else {
								sum = convertDoubleToPX2(0, bitwidth);
							}

							if (shr) {
								tmp[p] = sum;
							} else {
								tmp[p] = sum;
							}
						}

						count = (count + 1) >> 1;
						depth++;
					}

					C[n * H * W * CO + h * W * CO + w * CO + co] = tmp[0];
				}
			}
		}
	}

	return;
}

// A = A <+> B
// A[N][H][W][C], B[C]
void AddOrSubCir4D(posit_2_t* A, const posit_2_t* B, posit_2_t* X, MYITE N, MYITE H, MYITE W, MYITE C, MYINT shrA, MYINT shrB, MYINT shrC, bool add, int bitwidth) {
	for (MYITE n = 0; n < N; n++) {
		for (MYITE h = 0; h < H; h++) {
			for (MYITE w = 0; w < W; w++) {
				for (MYITE c = 0; c < C; c++) {
					posit_2_t a = A[n * H * W * C + h * W * C + w * C + c];
					posit_2_t b = B[c];

					posit_2_t res;
					if (add) {
						res = pX2_add(a, b, bitwidth);
					} else {
						res = pX2_sub(a, b, bitwidth);
					}

					X[n * H * W * C + h * W * C + w * C + c] = res;
				}
			}
		}
	}
	return;
}

// A = A <+> B
// A[N][H][W][C], B[C]
void AddOrSubCir2D(posit_2_t* A, const posit_2_t* B, posit_2_t* X, MYITE H, MYITE W, MYINT shrA, MYINT shrB, MYINT shrC, bool add, int bitwidth) {
	for (MYITE h = 0; h < H; h++) {
		for (MYITE w = 0; w < W; w++) {
			posit_2_t a = A[h * W + w];
			posit_2_t b = B[w];

			posit_2_t res;
			if (add) {
				res = pX2_add(a, b, bitwidth);
			} else {
				res = pX2_sub(a, b, bitwidth);
			}

			X[h * W + w] = res;
		}
	}
	return;
}

// A = relu(A)
// A[N][H][W][C]
void Relu4D(posit_2_t* A, MYITE N, MYITE H, MYITE W, MYITE C, int bitwidth) {
	for (MYITE n = 0; n < N; n++) {
		for (MYITE h = 0; h < H; h++) {
			for (MYITE w = 0; w < W; w++) {
				for (MYITE c = 0; c < C; c++) {
					posit_2_t a = A[n * H * W * C + h * W * C + w * C + c];
					posit_2_t zero = convertDoubleToPX2(0, bitwidth);
					if (pX2_lt(a, zero)) {
						a = zero;
					}

					A[n * H * W * C + h * W * C + w * C + c] = a;
				}
			}
		}
	}
	return;
}

// B = relu6(A)
// A[N][H][W][C]
void Relu6(posit_2_t* A, posit_2_t* B, MYITE N, MYITE H, MYITE W, MYITE C, MYINT six, MYINT div, int bitwidth) {
	for (MYITE n = 0; n < N; n++) {
		for (MYITE h = 0; h < H; h++) {
			for (MYITE w = 0; w < W; w++) {
				for (MYITE c = 0; c < C; c++) {
					posit_2_t a = A[n * H * W * C + h * W * C + w * C + c];
					posit_2_t zero = convertDoubleToPX2(0, bitwidth);
					posit_2_t six = convertDoubleToPX2(6.0, bitwidth);
					if (pX2_lt(a, zero)) {
						a = zero;
					}
					if (pX2_lt(six, a)) {
						a = six;
					}

					B[n * H * W * C + h * W * C + w * C + c] = a;
				}
			}
		}
	}
	return;
}

// A = relu(A)
// A[N][H][W][C]
void Relu2D(posit_2_t* A, MYITE H, MYITE W, int bitwidth) {
	for (MYITE h = 0; h < H; h++) {
		for (MYITE w = 0; w < W; w++) {
			posit_2_t a = A[h * W + w];
			posit_2_t zero = convertDoubleToPX2(0, bitwidth);
			if (pX2_lt(a, zero)) {
				a = zero;
			}

			A[h * W + w] = a;
		}
	}
	return;
}

// B = maxpool(A)
// A[N][H][W][C], B[N][H][W][C]
void Maxpool(posit_2_t* A, posit_2_t* B, MYITE N, MYITE H, MYITE W, MYITE C, MYITE FH, MYITE FW, MYITE strideH, MYITE strideW, MYITE HPADL, MYITE HPADR, MYITE WPADL, MYITE WPADR) {
	MYITE HO = H / strideH;
	MYITE WO = W / strideW;

	for (MYITE n = 0; n < N; n++) {
		for (MYITE ho = 0; ho < HO; ho++) {
			for (MYITE wo = 0; wo < WO; wo++) {
				for (MYITE c = 0; c < C; c++) {

					posit_2_t max = A[n * H * W * C + (strideH * ho) * W * C + (strideW * wo) * C + c];
					for (MYITE hs = 0; hs < FH; hs++) {
						for (MYITE ws = 0; ws < FW; ws++) {
							posit_2_t a = A[n * H * W * C + ((strideH * ho) + hs) * W * C + ((strideW * wo) + ws) * C + c];
							if (pX2_lt(max, a)) {
								max = a;
							}
						}
					}

					B[n * HO * WO * C + ho * WO * C + wo * C + c] = max;
				}
			}
		}
	}
	return;
}

void NormaliseL2(posit_2_t* A, posit_2_t* B, MYITE N, MYITE H, MYITE W, MYITE C, MYINT scaleA, MYINT shrA, int bitwidth) {
	for (MYITE n = 0; n < N; n++) {
		for (MYITE h = 0; h < H; h++) {
			for (MYITE w = 0; w < W; w++) {

				// Calculate the sum square.
				posit_2_t sumSquare = convertDoubleToPX2(0, bitwidth);
				for (MYITE c = 0; c < C; c++) {
					posit_2_t tmp = A[n * H * W * C + h * W * C + w * C + c];
					sumSquare = pX2_add(sumSquare, pX2_mul(tmp, tmp, bitwidth), bitwidth);
				}

				// Calculate the inverse square root of sumSquare.
				if (pX2_eq(sumSquare,  convertDoubleToPX2(0, bitwidth))) {
					sumSquare = convertDoubleToPX2(1e-5, bitwidth);
				}

				posit_2_t inverseNorm = pX2_div(convertDoubleToPX2(1, bitwidth), pX2_sqrt(sumSquare, bitwidth), bitwidth);

				// Multiply all elements by the 1 / sqrt(sumSquare).
				for (MYITE c = 0; c < C; c++) {
					B[n * H * W * C + h * W * C + w * C + c]  = pX2_mul(A[n * H * W * C + h * W * C + w * C + c], inverseNorm, bitwidth);
				}
			}
		}
	}
	return;
}

// B = exp(A)
void Exp(posit_2_t* A, MYITE I, MYITE J, MYINT shrA, MYINT shrB, posit_2_t* B, int bitwidth) {
	for (MYITE i = 0; i < I; i++) {
		for (MYITE j = 0; j < J; j++) {
			posit_2_t x = A[i * J + j];

			updateRangeOfExp(-1*convertPX2ToDouble(x));

			B[i * J + j] = convertDoubleToPX2(exp(convertPX2ToDouble(x)), bitwidth);
		}
	}
	return;
}

// A = sigmoid(A)
void Sigmoid(posit_2_t* A, MYITE I, MYITE J, float div, float add, float sigmoid_limit, MYINT scale_in, MYINT scale_out, posit_2_t* B, int bitwidth) {
	for (MYITE i = 0; i < I; i++) {
		for (MYITE j = 0; j < J; j++) {
			float x = convertPX2ToDouble(A[i * J + j]), y;

#ifdef FLOATEXP
			y = 1 / (1 + exp(-x));
#else
			y = (x + 1) / 2;
			y = y > 0 ? y : 0;
			y = y < 1 ? y : 1;
#endif
			B[i * J + j] = convertDoubleToPX2(y, bitwidth);
		}
	}
	return;
}

// A = AdjustScaleShr(A)
void AdjustScaleShr(posit_2_t* A, MYITE I, MYITE J, MYINT scale) {
	return;
}

// A = AdjustScaleShl(A)
void AdjustScaleShl(posit_2_t* A, MYITE I, MYITE J, MYINT scale) {
	return;
}

void convertPosit(posit8_t* a, posit8_t *b, int bwA, int bwB)
{
        *b=*a;
}


void convertPosit(posit8_t* a, posit16_t *b, int bwA, int bwB)
{
        *b = p8_to_p16(*a);
}


void convertPosit(posit8_t* a, posit32_t *b, int bwA, int bwB)
{
        *b = p8_to_p32(*a);
}


void convertPosit(posit16_t* a, posit8_t *b, int bwA, int bwB)
{
        *b = p16_to_p8(*a);
}


void convertPosit(posit16_t* a, posit16_t *b, int bwA, int bwB)
{
        *b=*a;
}


void convertPosit(posit16_t* a, posit32_t *b, int bwA, int bwB)
{
        *b = p16_to_p32(*a);
}


void convertPosit(posit32_t* a, posit8_t *b, int bwA, int bwB)
{
        *b = p32_to_p8(*a);
}


void convertPosit(posit32_t* a, posit16_t *b, int bwA, int bwB)
{
        *b = p32_to_p16(*a);
}


void convertPosit(posit32_t* a, posit32_t *b, int bwA, int bwB)
{
        *b=*a;
}

// void convertPosit(posit_1_t* a, posit8_t *b, int bwA, int bwB)
// {
//         *b = pX1_to_p8(*a);
// }

// void convertPosit(posit_1_t* a, posit16_t *b, int bwA, int bwB)
// {
//         *b = pX1_to_p16(*a);
// }


// void convertPosit(posit_1_t* a, posit32_t *b, int bwA, int bwB)
// {
//         *b = pX1_to_p32(*a);
// }

void convertPosit(posit_2_t* a, posit8_t *b, int bwA, int bwB)
{
        *b = pX2_to_p8(*a);
}

void convertPosit(posit_2_t* a, posit16_t *b, int bwA, int bwB)
{
        *b = pX2_to_p16(*a);
}

void convertPosit(posit_2_t* a, posit32_t *b, int bwA, int bwB)
{
        *b = pX2_to_p32(*a);
}

// void convertPosit(posit8_t* a, posit_1_t *b, int bwA, int bwB)
// {
//         *b = p8_to_pX1(*a, bwB);
// }


// void convertPosit(posit16_t* a, posit_1_t *b, int bwA, int bwB)
// {
//         *b = p16_to_pX1(*a, bwB);
// }


// void convertPosit(posit32_t* a, posit_1_t *b, int bwA, int bwB)
// {
//         *b = p32_to_pX1(*a, bwB);
// }

void convertPosit(posit8_t* a, posit_2_t *b, int bwA, int bwB)
{
        *b = p8_to_pX2(*a, bwB);
}


void convertPosit(posit16_t* a, posit_2_t *b, int bwA, int bwB)
{
        *b = p16_to_pX2(*a, bwB);
}


void convertPosit(posit32_t* a, posit_2_t *b, int bwA, int bwB)
{
        *b = p32_to_pX2(*a, bwB);
}

// void convertPosit(posit_1_t* a, posit_1_t *b, int bwA, int bwB)
// {
//     *b = pX1_to_pX1(*a, bwB);
// }

// void convertPosit(posit_1_t* a, posit_2_t *b, int bwA, int bwB)
// {
//     *b = pX1_to_pX2(*a, bwB);
// }

// void convertPosit(posit_2_t* a, posit_1_t *b, int bwA, int bwB)
// {
//     *b = pX2_to_pX1(*a, bwB);
// }

void convertPosit(posit_2_t* a, posit_2_t *b, int bwA, int bwB)
{
    *b = pX2_to_pX2(*a, bwB);
}

// double convertPositToDouble(posit_1_t a, int bw)
// {
// 	return convertPX1ToDouble(a);
// }

double convertPositToDouble(posit_2_t a, int bw)
{
	return convertPX2ToDouble(a);
}

// void convertDoubleToPosit(double a, posit_1_t *b, int bw)
// {
// 	*b = convertDoubleToPX1(a, bw);
// }

void convertDoubleToPosit(double a, posit_2_t *b, int bw)
{
	*b = convertDoubleToPX2(a, bw);
}

posit8_t positAdd(posit8_t a, posit8_t b, int bw)
{
	return p8_add(a, b);
}

posit16_t positAdd(posit16_t a, posit16_t b, int bw)
{
	return p16_add(a, b);
}


posit32_t positAdd(posit32_t a, posit32_t b, int bw)
{
	return p32_add(a, b);
}

posit8_t positSub(posit8_t a, posit8_t b, int bw)
{
	return p8_sub(a, b);
}

posit16_t positSub(posit16_t a, posit16_t b, int bw)
{
	return p16_sub(a, b);
}


posit32_t positSub(posit32_t a, posit32_t b, int bw)
{
	return p32_sub(a, b);
}

posit8_t positMul(posit8_t a, posit8_t b, int bw)
{
	return p8_mul(a, b);
}

posit16_t positMul(posit16_t a, posit16_t b, int bw)
{
	return p16_mul(a, b);
}


posit32_t positMul(posit32_t a, posit32_t b, int bw)
{
	return p32_mul(a, b);
}

double convertPositToDouble(posit8_t a, int bw)
{
	return convertP8ToDouble(a);
}

double convertPositToDouble(posit16_t a, int bw)
{
	return convertP16ToDouble(a);
}

double convertPositToDouble(posit32_t a, int bw)
{
	return convertP32ToDouble(a);
}


void convertDoubleToPosit(double a, posit8_t *b, int bw)
{
	*b = convertDoubleToP8(a);
}

void convertDoubleToPosit(double a, posit16_t *b, int bw)
{
	*b = convertDoubleToP16(a);
}

void convertDoubleToPosit(double a, posit32_t *b, int bw)
{
	*b = convertDoubleToP32(a);
}

quire8_t clearQuire(quire8_t q, int bw)
{
	return q8_clr(q);
}

quire16_t clearQuire(quire16_t q, int bw)
{
	return q16_clr(q);
}

quire32_t clearQuire(quire32_t q, int bw)
{
	return q32_clr(q);
}

posit8_t convertQuireToPosit(quire8_t q, int bw)
{
	return q8_to_p8(q);
}

posit16_t convertQuireToPosit(quire16_t q, int bw)
{
	return q16_to_p16(q);
}

posit32_t convertQuireToPosit(quire32_t q, int bw)
{
	return q32_to_p32(q);
}

quire8_t positFMA(quire8_t q, posit8_t a, posit8_t b, int bw)
{
	return q8_fdp_add(q, a, b);
}

quire16_t positFMA(quire16_t q, posit16_t a, posit16_t b, int bw)
{
	return q16_fdp_add(q, a, b);
}

quire32_t positFMA(quire32_t q, posit32_t a, posit32_t b, int bw)
{
	return q32_fdp_add(q, a, b);
}

posit8_t operator-(const posit8_t& a)
{
	posit8_t b;
	b = convertDoubleToP8(-1*convertP8ToDouble(a));
	return b;
}

posit16_t operator-(const posit16_t& a)
{
	posit16_t b;
	b = convertDoubleToP16(-1*convertP16ToDouble(a));
	return b;
}

posit32_t operator-(const posit32_t& a)
{
	posit32_t b;
	b = convertDoubleToP32(-1*convertP32ToDouble(a));
	return b;
}


void MatAddInplace(posit8_t* A, posit8_t* B, MYITE I, MYITE J) {
	for (MYITE i = 0; i < I; i++) {
		for (MYITE j = 0; j < J; j++) {
			posit8_t a = A[i * J + j];
			posit8_t b = B[i * J + j];

			posit8_t c = p8_add(a, b);

			A[i * J + j] = c;
		}
	}
	return;
}

void MatAddInplace(posit16_t* A, posit16_t* B, MYITE I, MYITE J) {
	for (MYITE i = 0; i < I; i++) {
		for (MYITE j = 0; j < J; j++) {
			posit16_t a = A[i * J + j];
			posit16_t b = B[i * J + j];

			posit16_t c = p16_add(a, b);

			A[i * J + j] = c;
		}
	}
	return;
}

void MatAddInplace(posit32_t* A, posit32_t* B, MYITE I, MYITE J) {
	for (MYITE i = 0; i < I; i++) {
		for (MYITE j = 0; j < J; j++) {
			posit32_t a = A[i * J + j];
			posit32_t b = B[i * J + j];

			posit32_t c = p32_add(a, b);

			A[i * J + j] = c;
		}
	}
	return;
}

void AddInplace(posit_2_t* A, posit_2_t* B, MYITE I, MYITE J, int bitwidth) {
	for (MYITE i = 0; i < I; i++) {
		for (MYITE j = 0; j < J; j++) {
			posit_2_t a = A[i * J + j];
			posit_2_t b = B[i * J + j];

			posit_2_t c = pX2_add(a, b, bitwidth);

			A[i * J + j] = c;
		}
	}
	return;
}

bool operator>(const posit8_t& a, const int& b)
{
	return p8_lt(convertDoubleToP8(double(b)), a);
}

bool operator>(const posit16_t& a, const int& b)
{
	return p16_lt(convertDoubleToP16(double(b)), a);
}

bool operator>(const posit32_t& a, const int& b)
{
	return p32_lt(convertDoubleToP32(double(b)), a);
}


bool positLT(posit8_t a, posit8_t b)
{
	return p8_lt(a, b);
}

bool positLT(posit16_t a, posit16_t b)
{
	return p16_lt(a, b);
}

bool positLT(posit32_t a, posit32_t b)
{
	return p32_lt(a, b);
}

bool positEQ(posit8_t a, posit8_t b)
{
	return p8_eq(a, b);
}

bool positEQ(posit16_t a, posit16_t b)
{
	return p16_eq(a, b);
}

bool positEQ(posit32_t a, posit32_t b)
{
	return p32_eq(a, b);
}

posit8_t positDiv(posit8_t a, posit8_t b){
	return p8_div(a, b);
}

posit16_t positDiv(posit16_t a, posit16_t b){
	return p16_div(a, b);
}

posit32_t positDiv(posit32_t a, posit32_t b){
	return p32_div(a, b);
}

posit8_t positSqrt(posit8_t a) {
	return p8_sqrt(a);
}

posit16_t positSqrt(posit16_t a) {
	return p16_sqrt(a);
}

posit32_t positSqrt(posit32_t a) {
	return p32_sqrt(a);
}

// posit_1_t positAdd(posit_1_t a, posit_1_t b, int bw)
// {
// 	return pX1_add(a, b, bw);
// }

posit_2_t positAdd(posit_2_t a, posit_2_t b, int bw)
{
	return pX2_add(a, b, bw);
}

// posit_1_t positSub(posit_1_t a, posit_1_t b, int bw)
// {
// 	return pX1_sub(a, b, bw);
// }

posit_2_t positSub(posit_2_t a, posit_2_t b, int bw)
{
	return pX2_sub(a, b, bw);
}

// posit_1_t positMul(posit_1_t a, posit_1_t b, int bw)
// {
// 	return pX1_mul(a, b, bw);
// }

posit_2_t positMul(posit_2_t a, posit_2_t b, int bw)
{
	return pX2_mul(a, b, bw);
}

// posit_1_t positDiv(posit_1_t a, posit_1_t b, int bw)
// {
// 	return pX1_div(a, b, bw);
// }

posit_2_t positDiv(posit_2_t a, posit_2_t b, int bw)
{
	return pX2_div(a, b, bw);
}

// posit_1_t positSqrt(posit_1_t a, int bw) {
// 	return pX1_sqrt(a, bw);
// }

posit_2_t positSqrt(posit_2_t a, int bw) {
	return pX2_sqrt(a, bw);
}

// posit_1_t convertQuireToPosit(quire_1_t q, int bw)
// {
// 	return qX1_to_pX1(q, bw);
// }

posit_2_t convertQuireToPosit(quire_2_t q, int bw)
{
	return qX2_to_pX2(q, bw);
}

// quire_1_t positFMA(quire_1_t q, posit_1_t a, posit_1_t b, int bw)
// {
// 	return qX1_fdp_add(q, a, b);
// }

quire_2_t positFMA(quire_2_t q, posit_2_t a, posit_2_t b, int bw)
{
	return qX2_fdp_add(q, a, b);
}

// bool positLT(posit_1_t a, posit_1_t b, int bw)
// {
// 	return pX1_lt(a, b);
// }

bool positLT(posit_2_t a, posit_2_t b, int bw)
{
	return pX2_lt(a, b);
}

// bool positEQ(posit_1_t a, posit_1_t b, int bw)
// {
// 	return pX1_eq(a, b);
// }

bool positEQ(posit_2_t a, posit_2_t b, int bw)
{
	return pX2_eq(a, b);
}

// quire_1_t clearQuire(quire_1_t q, int bw)
// {
// 	return qX1_clr(q);
// }

quire_2_t clearQuire(quire_2_t q, int bw)
{
	return qX2_clr(q);
}

void debugPrint(posit8_t* A, int I, int J, std::string varName)
{
	#ifdef DEBUG
	std::ofstream f("debugLog", std::ios::app);
	
	f << varName<<std::endl;
	for(int i=0;i<I;i++)
	{
		for(int j=0;j<J;j++)
		{
			float a = convertP8ToDouble(A[i*J + j]);
			f<< a << " ";
		}
	}
	f<<std::endl<<std::endl;
	f.close();
	#endif
}

void debugPrint(posit16_t* A, int I, int J, std::string varName)
{
	#ifdef DEBUG
	std::ofstream f("debugLog", std::ios::app);
	
	f << varName<<std::endl;
	for(int i=0;i<I;i++)
	{
		for(int j=0;j<J;j++)
		{
			float a = convertP16ToDouble(A[i*J + j]);
			f<< a << " ";
		}
	}
	f<<std::endl<<std::endl;
	f.close();
	#endif
}

void debugPrint(posit32_t* A, int I, int J, std::string varName)
{
	#ifdef DEBUG
	std::ofstream f("debugLog", std::ios::app);
	
	f << varName<<std::endl;
	for(int i=0;i<I;i++)
	{
		for(int j=0;j<J;j++)
		{
			float a = convertP32ToDouble(A[i*J + j]);
			f<< a << " ";
		}
	}
	f<<std::endl<<std::endl;
	f.close();
	#endif
}

void debugPrint(posit_2_t* A, int I, int J, std::string varName, int bw)
{
	#ifdef DEBUG
	std::ofstream f("debugLog", std::ios::app);
	
	f << varName<<std::endl;
	for(int i=0;i<I;i++)
	{
		for(int j=0;j<J;j++)
		{
			float a = convertPositToDouble(A[i*J + j], bw);
			f<< a << " ";
		}
	}
	f<<std::endl<<std::endl;
	f.close();
	#endif
}

void debugPrint(posit8_t* A, int I, int J, int K, int L, std::string varName)
{
	#ifdef DEBUG
	std::ofstream f("debugLog", std::ios::app);
	
	f << varName<<std::endl;
	for(int i=0;i<I*J*K*L;i++)
	{
		float a = convertP8ToDouble(A[i]);
		f<< a << " ";
	}
	f<<std::endl<<std::endl;
	f.close();
	#endif
}

void debugPrint(posit16_t* A, int I, int J, int K, int L, std::string varName)
{
	#ifdef DEBUG
	std::ofstream f("debugLog", std::ios::app);
	
	f << varName<<std::endl;
	for(int i=0;i<I*J*K*L;i++)
	{
		float a = convertP16ToDouble(A[i]);
		f<< a << " ";
	}
	f<<std::endl<<std::endl;
	f.close();
	#endif
}

void debugPrint(posit32_t* A, int I, int J, int K, int L, std::string varName)
{
	#ifdef DEBUG
	std::ofstream f("debugLog", std::ios::app);
	
	f << varName<<std::endl;
	for(int i=0;i<I*J*K*L;i++)
	{
		float a = convertP32ToDouble(A[i]);
		f<< a << " ";
	}
	f<<std::endl<<std::endl;
	f.close();
	#endif
}

void debugPrint(posit_2_t* A, int I, int J, int K, int L, std::string varName, int bw)
{
	#ifdef DEBUG
	std::ofstream f("debugLog", std::ios::app);
	
	f << varName<<std::endl;
	for(int i=0;i<I*J*K*L;i++)
	{
		float a = convertPositToDouble(A[i], bw);
		f<< a << " ";
	}
	f<<std::endl<<std::endl;
	f.close();
	#endif
}

void debugPrint(std::string str)
{
	#ifdef DEBUG
	std::ofstream f("debugLog", std::ios::app);
	f << str;
	f<<std::endl<<std::endl;
	f.close();
	#endif	
}