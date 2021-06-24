// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include <iostream>
#include <cmath>

#include "datatypes.h"
#include "library_float.h"
#include "profile.h"

// This file contains floating point implementations of operations supported by SeeDot.

// C = A + B
void MatAddNN(FP_TYPE* A, FP_TYPE* B, FP_TYPE* C, MYINT I, MYINT J, MYINT shrA, MYINT shrB, MYINT shrC) {
	for (MYITE i = 0; i < I; i++) {
		for (MYITE j = 0; j < J; j++) {
			FP_TYPE a = A[i * J + j];
			FP_TYPE b = B[i * J + j];

			FP_TYPE c = a + b;

			C[i * J + j] = c;
		}
	}
	return;
}

// C = A + B
void MatAddCN(const FP_TYPE* A, FP_TYPE* B, FP_TYPE* C, MYINT I, MYINT J, MYINT shrA, MYINT shrB, MYINT shrC) {
	for (MYITE i = 0; i < I; i++) {
		for (MYITE j = 0; j < J; j++) {
			FP_TYPE a = A[i * J + j];
			FP_TYPE b = B[i * J + j];

			FP_TYPE c = a + b;

			C[i * J + j] = c;
		}
	}
	return;
}

// C = A + B
void MatAddNC(FP_TYPE* A, const FP_TYPE* B, FP_TYPE* C, MYINT I, MYINT J, MYINT shrA, MYINT shrB, MYINT shrC) {
	for (MYITE i = 0; i < I; i++) {
		for (MYITE j = 0; j < J; j++) {
			FP_TYPE a = A[i * J + j];
			FP_TYPE b = B[i * J + j];

			FP_TYPE c = a + b;

			C[i * J + j] = c;
		}
	}
	return;
}

// C = A + B
void MatAddCC(const FP_TYPE* A, const FP_TYPE* B, FP_TYPE* C, MYINT I, MYINT J, MYINT shrA, MYINT shrB, MYINT shrC) {
	for (MYITE i = 0; i < I; i++) {
		for (MYITE j = 0; j < J; j++) {
			FP_TYPE a = A[i * J + j];
			FP_TYPE b = B[i * J + j];

			FP_TYPE c = a + b;

			C[i * J + j] = c;
		}
	}
	return;
}

// C = a + B
void MatAddBroadCastA(FP_TYPE* A, FP_TYPE* B, FP_TYPE* C, MYINT I, MYINT J, MYINT shrA, MYINT shrB, MYINT shrC) {
	for (MYITE i = 0; i < I; i++) {
		for (MYITE j = 0; j < J; j++) {
			FP_TYPE a = *A;
			FP_TYPE b = B[i * J + j];

			FP_TYPE c = a + b;

			C[i * J + j] = c;
		}
	}
	return;
}

// C = A + b
void MatAddBroadCastB(FP_TYPE* A, FP_TYPE* B, FP_TYPE* C, MYINT I, MYINT J, MYINT shrA, MYINT shrB, MYINT shrC) {
	for (MYITE i = 0; i < I; i++) {
		for (MYITE j = 0; j < J; j++) {
			FP_TYPE a = A[i * J + j];
			FP_TYPE b = *B;

			FP_TYPE c = a + b;

			C[i * J + j] = c;
		}
	}
	return;
}

// C = A + B
void MatAdd4(FP_TYPE* A, FP_TYPE* B, FP_TYPE* X, MYINT N, MYINT H, MYINT W, MYINT C, MYINT shrA, MYINT shrB, MYINT shrC) {
	for (MYITE n = 0; n < N; n++) {
		for (MYITE h = 0; h < H; h++) {
			for (MYITE w = 0; w < W; w++) {
				for (MYITE c = 0; c < C; c++) {
					FP_TYPE a = A[n * H * W * C + h * W * C + w * C + c];
					FP_TYPE b = B[n * H * W * C + h * W * C + w * C + c];

					FP_TYPE x = a + b;

					X[n * H * W * C + h * W * C + w * C + c] = x;
				}
			}
		}
	}
	return;
}

// C = A - B
void MatSub(FP_TYPE* A, const FP_TYPE* B, FP_TYPE* C, MYINT I, MYINT J, MYINT shrA, MYINT shrB, MYINT shrC) {
	for (MYITE i = 0; i < I; i++) {
		for (MYITE j = 0; j < J; j++) {
			FP_TYPE a = A[i * J + j];
			FP_TYPE b = B[i * J + j];

			FP_TYPE c = a - b;

			C[i * J + j] = c;
		}
	}
	return;
}

// C = a - B
void MatSubBroadCastA(FP_TYPE* A, FP_TYPE* B, FP_TYPE* C, MYINT I, MYINT J, MYINT shrA, MYINT shrB, MYINT shrC) {
	for (MYITE i = 0; i < I; i++) {
		for (MYITE j = 0; j < J; j++) {
			FP_TYPE a = *A;
			FP_TYPE b = B[i * J + j];

			FP_TYPE c = a - b;

			C[i * J + j] = c;
		}
	}
	return;
}

// C = A - b
void MatSubBroadCastB(FP_TYPE* A, FP_TYPE* B, FP_TYPE* C, MYINT I, MYINT J, MYINT shrA, MYINT shrB, MYINT shrC) {
	for (MYITE i = 0; i < I; i++) {
		for (MYITE j = 0; j < J; j++) {
			FP_TYPE a = A[i * J + j];
			FP_TYPE b = *B;

			FP_TYPE c = a - b;

			C[i * J + j] = c;
		}
	}
	return;
}

// C = A * B
void MatMulNN(FP_TYPE* A, FP_TYPE* B, FP_TYPE* C, FP_TYPE* tmp, MYINT I, MYINT K, MYINT J, MYINT shrA, MYINT shrB, MYINT H1, MYINT H2) {
	for (MYITE i = 0; i < I; i++) {
		for (MYITE j = 0; j < J; j++) {
			for (MYITE k = 0; k < K; k++) {
				FP_TYPE a = A[i * K + k];
				FP_TYPE b = B[k * J + j];

				tmp[k] = a * b;
			}

			MYITE count = K, depth = 0;
			bool shr = true;

			while (depth < (H1 + H2)) {
				if (depth >= H1) {
					shr = false;
				}

				for (MYITE p = 0; p < (K / 2 + 1); p++) {
					FP_TYPE sum;
					if (p < (count >> 1)) {
						sum = tmp[2 * p] + tmp[(2 * p) + 1];
					} else if ((p == (count >> 1)) && ((count & 1) == 1)) {
						sum = tmp[2 * p];
					} else {
						sum = 0;
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

			C[i * J + j] = tmp[0];
		}
	}
	return;
}

// C = A * B
void MatMulCN(const FP_TYPE* A, FP_TYPE* B, FP_TYPE* C, FP_TYPE* tmp, MYINT I, MYINT K, MYINT J, MYINT shrA, MYINT shrB, MYINT H1, MYINT H2) {
	for (MYITE i = 0; i < I; i++) {
		for (MYITE j = 0; j < J; j++) {
			for (MYITE k = 0; k < K; k++) {
				FP_TYPE a = A[i * K + k];
				FP_TYPE b = B[k * J + j];

				tmp[k] = a * b;
			}

			MYITE count = K, depth = 0;
			bool shr = true;

			while (depth < (H1 + H2)) {
				if (depth >= H1) {
					shr = false;
				}

				for (MYITE p = 0; p < (K / 2 + 1); p++) {
					FP_TYPE sum;
					if (p < (count >> 1)) {
						sum = tmp[2 * p] + tmp[(2 * p) + 1];
					} else if ((p == (count >> 1)) && ((count & 1) == 1)) {
						sum = tmp[2 * p];
					} else {
						sum = 0;
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

			C[i * J + j] = tmp[0];
		}
	}
	return;
}

// C = A * B
void MatMulNC(FP_TYPE* A, const FP_TYPE* B, FP_TYPE* C, FP_TYPE* tmp, MYINT I, MYINT K, MYINT J, MYINT shrA, MYINT shrB, MYINT H1, MYINT H2) {
	for (MYITE i = 0; i < I; i++) {
		for (MYITE j = 0; j < J; j++) {
			for (MYITE k = 0; k < K; k++) {
				FP_TYPE a = A[i * K + k];
				FP_TYPE b = B[k * J + j];

				tmp[k] = a * b;
			}

			MYITE count = K, depth = 0;
			bool shr = true;

			while (depth < (H1 + H2)) {
				if (depth >= H1) {
					shr = false;
				}

				for (MYITE p = 0; p < (K / 2 + 1); p++) {
					FP_TYPE sum;
					if (p < (count >> 1)) {
						sum = tmp[2 * p] + tmp[(2 * p) + 1];
					} else if ((p == (count >> 1)) && ((count & 1) == 1)) {
						sum = tmp[2 * p];
					} else {
						sum = 0;
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

			C[i * J + j] = tmp[0];
		}
	}
	return;
}

// C = A * B
void MatMulCC(const FP_TYPE* A, const FP_TYPE* B, FP_TYPE* C, FP_TYPE* tmp, MYINT I, MYINT K, MYINT J, MYINT shrA, MYINT shrB, MYINT H1, MYINT H2) {
	for (MYITE i = 0; i < I; i++) {
		for (MYITE j = 0; j < J; j++) {
			for (MYITE k = 0; k < K; k++) {
				FP_TYPE a = A[i * K + k];
				FP_TYPE b = B[k * J + j];

				tmp[k] = a * b;
			}

			MYITE count = K, depth = 0;
			bool shr = true;

			while (depth < (H1 + H2)) {
				if (depth >= H1) {
					shr = false;
				}

				for (MYITE p = 0; p < (K / 2 + 1); p++) {
					FP_TYPE sum;
					if (p < (count >> 1)) {
						sum = tmp[2 * p] + tmp[(2 * p) + 1];
					} else if ((p == (count >> 1)) && ((count & 1) == 1)) {
						sum = tmp[2 * p];
					} else {
						sum = 0;
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

			C[i * J + j] = tmp[0];
		}
	}
	return;
}

// C = A |*| B
void SparseMatMulX(const MYINT* Aidx, const FP_TYPE* Aval, FP_TYPE** B, FP_TYPE* C, int16_t K, MYINT shrA, MYINT shrB, MYINT shrC) {
	MYITE ite_idx = 0, ite_val = 0;
	for (MYITE k = 0; k < K; k++) {
		FP_TYPE b = B[k * 1][0];

		MYINT idx = Aidx[ite_idx];
		while (idx != 0) {
			FP_TYPE a = Aval[ite_val];

			FP_TYPE c = a * b;

			C[idx - 1] += c;

			ite_idx++;
			ite_val++;

			idx = Aidx[ite_idx];
		}
		ite_idx++;
	}
	return;
}

// C = A |*| B
void SparseMatMul(const MYINT* Aidx, const FP_TYPE* Aval, FP_TYPE* B, FP_TYPE* C, int16_t K, MYINT shrA, MYINT shrB, MYINT shrC) {
	MYITE ite_idx = 0, ite_val = 0;
	for (MYITE k = 0; k < K; k++) {
		FP_TYPE b = B[k];

		MYINT idx = Aidx[ite_idx];
		while (idx != 0) {
			FP_TYPE a = Aval[ite_val];

			FP_TYPE c = a * b;

			C[idx - 1] += c;

			ite_idx++;
			ite_val++;

			idx = Aidx[ite_idx];
		}
		ite_idx++;
	}

	return;
}

// C = A <*> B
void MulCir(FP_TYPE* A, FP_TYPE* B, FP_TYPE* C, MYINT I, MYINT J, MYINT shrA, MYINT shrB) {
	for (MYITE i = 0; i < I; i++) {
		for (MYITE j = 0; j < J; j++) {
			FP_TYPE a = A[i * J + j];
			FP_TYPE b = B[i * J + j];

			C[i * J + j] = a * b;
		}
	}
	return;
}

// A = tanh(A)
void TanH(FP_TYPE* A, MYINT I, MYINT J, float scale_in, float scale_out, FP_TYPE* B) {
	for (MYITE i = 0; i < I; i++) {
		for (MYITE j = 0; j < J; j++) {
			FP_TYPE x = A[i * J + j], y;

			#ifdef FLOATEXP
				y = tanh(x);
			#else
				y = x > -1 ? x : -1;
				y = y < 1 ? y : 1;
			#endif

			B[i * J + j] = y;
		}
	}
	return;
}

// B = reverse(A, axis)
void Reverse2(FP_TYPE* A, MYINT axis, MYINT I, MYINT J, FP_TYPE* B) {
	for (MYITE i = 0; i < I; i++) {
		for (MYITE j = 0; j < J; j++) {
			MYINT i_prime = (axis == 0 ? (I - 1 - i) : i);
			MYINT j_prime = (axis == 1 ? (J - 1 - j) : j);

			B[i * J + j] = A[i_prime*J + j_prime];
		}
	}
	return;
}

// index = argmax(A)
void ArgMax(FP_TYPE* A, MYINT I, MYINT J, int* index) {
	FP_TYPE max = A[0];
	MYITE maxIndex = 0, counter = 0;
	for (MYITE i = 0; i < I; i++) {
		for (MYITE j = 0; j < J; j++) {
			FP_TYPE x = A[i * J + j];

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

// A = A^T
void Transpose(FP_TYPE* A, FP_TYPE* B, MYINT I, MYINT J) {
	for (MYITE i = 0; i < I; i++) {
		for (MYITE j = 0; j < J; j++) {
			B[i * J + j] = A[j * I + i];
		}
	}
	return;
}

// C = a * B
void ScalarMul(FP_TYPE* A, FP_TYPE* B, FP_TYPE* C, MYINT I, MYINT J, MYINT shrA, MYINT shrB) {
	FP_TYPE a = *A;
	for (MYITE i = 0; i < I; i++) {
		for (MYITE j = 0; j < J; j++) {
			FP_TYPE b = B[i * J + j];
			C[i * J + j] = a * b;
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
void MBConv(FP_TYPE* A, const FP_TYPE* F1, const FP_TYPE* BN1W, const FP_TYPE* BN1B, const FP_TYPE* F2, const FP_TYPE* BN2W, const FP_TYPE* BN2B, const FP_TYPE* F3, const FP_TYPE* BN3W, const FP_TYPE* BN3B, FP_TYPE* C, FP_TYPE* X, FP_TYPE* T, FP_TYPE* U, MYITE N, MYITE H, MYITE W, MYITE Cin, MYITE Ct, MYITE HF, MYITE WF, MYITE Cout, MYITE Hout, MYITE Wout, MYITE HPADL, MYITE HPADR, MYITE WPADL, MYITE WPADR, MYITE HSTR, MYITE WSTR, MYITE D1, MYITE D2, MYITE D3, MYINT SIX_1, MYINT SIX_2, MYINT shr1, MYINT shr2, MYINT shr3, MYINT shr4, MYINT shr5, MYINT shr6, MYINT shr7, MYINT shr8, MYINT shr9, MYINT shl1, MYINT shl2, MYINT shl3, MYINT shl4, MYINT shl5, MYINT shl6, MYINT shl7, MYINT shl8, MYINT shl9, std::string name) {
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
						U[l] = A[n * H * W * Cin + i * W * Cin + j * Cin + l] * F1[l * Ct + k];
					}
					MYITE totalEle = Cin;
					MYITE count = Cin;
					MYITE depth = 0;

					while (depth < D1) {
						for (MYITE p = 0; p < (totalEle / 2 + 1); p++) {
							if (p < count / 2) {
								U[p] = U[2 * p] + U[(2 * p) + 1];
							} else if ((p == (count / 2)) && ((count % 2) == 1)) {
								U[p] = U[2 * p];
							} else {
								U[p] = 0;
							}
						}

						count = (count + 1) / 2;
						depth++;
					}
	
					FP_TYPE ar = U[0] + BN1B[k];
					X[i * W * Ct + j * Ct + k] = (ar) * BN1W[k];
					Profile2(&ar, 1, 1, name + "t1");
					X[i * W * Ct + j * Ct + k] = X[i * W * Ct + j * Ct + k] < FP_TYPE(0.0) ? FP_TYPE(0.0) : X[i * W * Ct + j * Ct + k];
					X[i * W * Ct + j * Ct + k] = X[i * W * Ct + j * Ct + k] > 6.0 ? 6.0 : X[i * W * Ct + j * Ct + k];
				}
			}
		}

		for (MYITE h = HOffsetL, hout = 0; h < H - HOffsetR; hout++, h += HSTR) {

			for (MYITE i = 0; i < HSTR; i++) {
				for (MYITE j = 0; j < W; j++) {
					for (MYITE k = 0; k < Ct; k++) {
						MYITE iRed = (i + margin + hout * HSTR) % HF, iFull = i + margin + hout * HSTR;
						X[iRed * W * Ct + j * Ct + k] = FP_TYPE(0.0);
						for (MYITE l = 0; l < Cin; l++) {
							FP_TYPE a = iFull < H ? A[n * H * W * Cin + iFull * W * Cin + j * Cin + l] : FP_TYPE(0.0);
							U[l] = a * F1[l * Ct + k];
						}
						MYITE totalEle = Cin;
						MYITE count = Cin;
						MYITE depth = 0;

						while (depth < D1) {
							for (MYITE p = 0; p <(totalEle / 2 + 1); p++) {
								if (p < count / 2) {
									U[p] = U[2 * p] + U[(2 * p) + 1];
								} else if ((p == (count / 2)) && ((count % 2) == 1)) {
									U[p] = U[2 * p];
								} else {
									U[p] = 0;
								}
							}

							count = (count + 1) / 2;
							depth++;
						}

						FP_TYPE ar = U[0] + BN1B[k];
						X[iRed * W * Ct + j * Ct + k] = (ar) * BN1W[k];
						Profile2(&ar, 1, 1, name + "t1");
						X[iRed * W * Ct + j * Ct + k] = X[iRed * W * Ct + j * Ct + k] < FP_TYPE(0.0) ? FP_TYPE(0.0) : X[iRed * W * Ct + j * Ct + k];
						X[iRed * W * Ct + j * Ct + k] = X[iRed * W * Ct + j * Ct + k] > 6.0 ? 6.0 : X[iRed * W * Ct + j * Ct + k];
					}
				}
			}

			for (MYITE w = WOffsetL, wout = 0; w < W - WOffsetR; w += WSTR, wout++) {
				for (MYITE g = 0; g < Ct; g++) {
					MYITE counter = 0;
					for (MYITE hf = -(HF / 2); hf <= (HF / 2); hf++) {
						for (MYITE wf = -(WF / 2); wf <= (WF / 2); wf++) {
							FP_TYPE x = (((h + hf) < 0) || ((h + hf) >= H) || ((w + wf) < 0) || ((w + wf) >= W)) ? FP_TYPE(0.0) : X[((h + hf) % HF) * W * Ct + (w + wf) * Ct + g];
							FP_TYPE b = F2[g * HF * WF + (hf + HF / 2) * WF + (wf + WF / 2)];
							U[counter] = x * b;
							counter++;
						}
					}
					MYITE totalEle = HF * WF;
					MYITE count = HF * WF;
					MYITE depth = 0;

					while (depth < D2) {
						for (MYITE p = 0; p < (totalEle / 2 + 1); p++) {
							if (p < count / 2) {
								U[p] = U[2 * p] + U[(2 * p) + 1];
							} else if ((p == (count / 2)) && ((count % 2) == 1)) {
								U[p] = U[2 * p];
							} else {
								U[p] = 0;
							}
						}

						count = (count + 1) / 2;
						depth++;
					}

					FP_TYPE ar = U[0] + BN2B[g];
					T[g] = (ar) * BN2W[g];
					Profile2(&ar, 1, 1, name + "t3");
					T[g] = T[g] < FP_TYPE(0.0) ? FP_TYPE(0.0) : T[g];
					T[g] = T[g] > 6.0 ? 6.0 : T[g];
				}

				for (MYITE i = 0; i < Cout; i++) {
					for (MYITE g = 0; g < Ct; g++) {
						U[g] = T[g] * F3[g * Cout + i];
					}
					MYITE totalEle = Ct;
					MYITE count = Ct;
					MYITE depth = 0;

					while (depth < D3) {
						for (MYITE p = 0; p < (totalEle / 2 + 1); p++) {
							if (p < count / 2) {
								U[p] = U[2 * p] + U[(2 * p) + 1];
							} else if ((p == (count / 2)) && ((count % 2) == 1)) {
								U[p] = U[2 * p];
							} else {
								U[p] = 0;
							}
						}

						count = (count + 1) / 2;
						depth++;
					}

					FP_TYPE ar = U[0] + BN3B[i];
					C[n * Hout * Wout * Cout + hout * Wout * Cout + wout * Cout + i] = (ar) * BN3W[i];
					Profile2(&ar, 1, 1, name + "t5");
				}
			}
		}
	}
}

// C = conv(A, B, <params>)
// A[N][H][W][CIN], B[G][HF][WF][CINF][COUTF], C[N][HOUT][WOUT][COUTF*G]
void Convolution(FP_TYPE* A, const FP_TYPE* B, FP_TYPE* C, FP_TYPE* tmp, MYINT N, MYINT H, MYINT W, MYINT CIN, MYINT HF, MYINT WF, MYINT CINF, MYINT COUTF, MYINT HOUT, MYINT WOUT, MYINT HPADL, MYINT HPADR, MYINT WPADL, MYINT WPADR, MYINT HSTR, MYINT WSTR, MYINT HDL, MYINT WDL, MYINT G, MYINT shrA, MYINT shrB, MYINT H1, MYINT H2) {
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
									FP_TYPE a = (((h + HDL * hf) < 0) || ((h + HDL * hf) >= H) || ((w + WDL * wf) < 0) || ((w + WDL * wf) >= W)) ? FP_TYPE(0) : A[n * H * W * CIN + (h + HDL * hf) * W * CIN + (w + WDL * wf) * CIN + (ci + g * CINF)];
									FP_TYPE b = B[g * HF * WF * CINF * COUTF + (hf + HF / 2) * WF * CINF * COUTF + (wf + WF / 2) * CINF * COUTF + ci * COUTF + co];

									tmp[counter] = a * b;
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
								FP_TYPE sum;
								if (p < (count >> 1)) {
									sum = tmp[2 * p] + tmp[(2 * p) + 1];
								} else if ((p == (count >> 1)) && ((count & 1) == 1)) {
									sum = tmp[2 * p];
								} else {
									sum = 0;
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
void Conv(FP_TYPE* A, const FP_TYPE* B, FP_TYPE* C, FP_TYPE* tmp, MYINT N, MYINT H, MYINT W, MYINT CI, MYINT HF, MYINT WF, MYINT CO, MYINT shrA, MYINT shrB, MYINT H1, MYINT H2) {
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
								FP_TYPE a = (((((h + hf) < padH) || ((h + hf) >= (H + padH))) || (((w + wf) < padW) || ((w + wf) >= (W + padW)))) ? FP_TYPE(0) : A[n * H * W * CI + ((h + hf) - padH) * W * CI + ((w + wf) - padW) * CI + ci]);
								FP_TYPE b = B[hf * WF * CI * CO + wf * CI * CO + ci * CO + co];

								tmp[counter] = a * b;
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
							FP_TYPE sum;
							if (p < (count >> 1)) {
								sum = tmp[2 * p] + tmp[(2 * p) + 1];
							} else if ((p == (count >> 1)) && ((count & 1) == 1)) {
								sum = tmp[2 * p];
							} else {
								sum = 0;
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
void AddOrSubCir4D(FP_TYPE* A, const FP_TYPE* B, FP_TYPE* X, MYINT N, MYINT H, MYINT W, MYINT C, MYINT shrA, MYINT shrB, MYINT shrC, bool add) {
	for (MYITE n = 0; n < N; n++) {
		for (MYITE h = 0; h < H; h++) {
			for (MYITE w = 0; w < W; w++) {
				for (MYITE c = 0; c < C; c++) {
					FP_TYPE a = A[n * H * W * C + h * W * C + w * C + c];
					FP_TYPE b = B[c];

					FP_TYPE res;
					if (add) {
						res = a + b;
					} else {
						res = a - b;
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
void AddOrSubCir2D(FP_TYPE* A, const FP_TYPE* B, FP_TYPE* X, MYINT H, MYINT W, MYINT shrA, MYINT shrB, MYINT shrC, bool add) {
	for (MYITE h = 0; h < H; h++) {
		for (MYITE w = 0; w < W; w++) {
			FP_TYPE a = A[h * W + w];
			FP_TYPE b = B[w];

			FP_TYPE res;
			if (add) {
				res = a + b;
			} else {
				res = a - b;
			}

			X[h * W + w] = res;
		}
	}
	return;
}

// A = relu(A)
// A[N][H][W][C]
void Relu4D(FP_TYPE* A, MYINT N, MYINT H, MYINT W, MYINT C) {
	for (MYITE n = 0; n < N; n++) {
		for (MYITE h = 0; h < H; h++) {
			for (MYITE w = 0; w < W; w++) {
				for (MYITE c = 0; c < C; c++) {
					FP_TYPE a = A[n * H * W * C + h * W * C + w * C + c];
					if (a < 0) {
						a = 0;
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
void Relu6(FP_TYPE* A, FP_TYPE* B, MYINT N, MYINT H, MYINT W, MYINT C, MYINT six, MYINT div) {
	for (MYITE n = 0; n < N; n++) {
		for (MYITE h = 0; h < H; h++) {
			for (MYITE w = 0; w < W; w++) {
				for (MYITE c = 0; c < C; c++) {
					FP_TYPE a = A[n * H * W * C + h * W * C + w * C + c];
					if (a < 0) {
						a = 0;
					}
					if (a > 6) {
						a = 6;
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
void Relu2D(FP_TYPE* A, MYINT H, MYINT W) {
	for (MYITE h = 0; h < H; h++) {
		for (MYITE w = 0; w < W; w++) {
			FP_TYPE a = A[h * W + w];
			if (a < 0) {
				a = 0;
			}

			A[h * W + w] = a;
		}
	}
	return;
}

// B = maxpool(A)
// A[N][H][W][C], B[N][H][W][C]
void Maxpool(FP_TYPE* A, FP_TYPE* B, MYINT N, MYINT H, MYINT W, MYINT C, MYINT FH, MYINT FW, MYINT strideH, MYINT strideW, MYINT HPADL, MYINT HPADR, MYINT WPADL, MYINT WPADR) {
	MYITE HO = H / strideH;
	MYITE WO = W / strideW;

	for (MYITE n = 0; n < N; n++) {
		for (MYITE ho = 0; ho < HO; ho++) {
			for (MYITE wo = 0; wo < WO; wo++) {
				for (MYITE c = 0; c < C; c++) {

					FP_TYPE max = A[n * H * W * C + (strideH * ho) * W * C + (strideW * wo) * C + c];
					for (MYITE hs = 0; hs < FH; hs++) {
						for (MYITE ws = 0; ws < FW; ws++) {
							FP_TYPE a = A[n * H * W * C + ((strideH * ho) + hs) * W * C + ((strideW * wo) + ws) * C + c];
							if (a > max) {
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

void NormaliseL2(FP_TYPE* A, FP_TYPE* B, MYINT N, MYINT H, MYINT W, MYINT C, MYINT scaleA, MYINT shrA) {
	for (MYITE n = 0; n < N; n++) {
		for (MYITE h = 0; h < H; h++) {
			for (MYITE w = 0; w < W; w++) {

				// Calculate the sum square.
				FP_TYPE sumSquare(0);
				for (MYITE c = 0; c < C; c++) {
					FP_TYPE tmp = A[n * H * W * C + h * W * C + w * C + c];
					sumSquare += tmp * tmp;
				}

				// Calculate the inverse square root of sumSquare.
				if (sumSquare == 0) {
					sumSquare = 1e-5;
				}

				FP_TYPE inverseNorm(1 / sqrt(sumSquare));

				// Multiply all elements by the 1 / sqrt(sumSquare).
				for (MYITE c = 0; c < C; c++) {
					B[n * H * W * C + h * W * C + w * C + c]  = A[n * H * W * C + h * W * C + w * C + c]  * inverseNorm;
				}
			}
		}
	}
	return;
}

// B = exp(A)
void Exp(FP_TYPE* A, MYINT I, MYINT J, MYINT shrA, MYINT shrB, FP_TYPE* B) {
	for (MYITE i = 0; i < I; i++) {
		for (MYITE j = 0; j < J; j++) {
			FP_TYPE x = A[i * J + j];

			updateRangeOfExp(-x);

			B[i * J + j] = exp(x);
		}
	}
	return;
}

// A = sigmoid(A)
void Sigmoid(FP_TYPE* A, MYINT I, MYINT J, float div, float add, float sigmoid_limit, MYINT scale_in, MYINT scale_out, FP_TYPE* B) {
	for (MYITE i = 0; i < I; i++) {
		for (MYITE j = 0; j < J; j++) {
			FP_TYPE x = A[i * J + j], y;

#ifdef FLOATEXP
			y = 1 / (1 + exp(-x));
#else
			y = (x + 1) / 2;
			y = y > 0 ? y : 0;
			y = y < 1 ? y : 1;
#endif
			B[i * J + j] = y;
		}
	}
	return;
}

// A = AdjustScaleShr(A)
void AdjustScaleShr(FP_TYPE* A, MYINT I, MYINT J, MYINT scale) {
	return;
}

// A = AdjustScaleShl(A)
void AdjustScaleShl(FP_TYPE* A, MYINT I, MYINT J, MYINT scale) {
	return;
}
