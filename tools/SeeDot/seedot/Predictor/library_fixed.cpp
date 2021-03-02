// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include <iostream>
#include <cmath>

#include "datatypes.h"
#include "library_fixed.h"

// This file contains implementations of the linear algebra operators supported by SeeDot.
// Each function takes the scaling factors as arguments along with the pointers to the operands.

// C = A + B
void MatAddNN(MYINT* A, MYINT* B, MYINT* C, MYINT I, MYINT J, MYINT shrA, MYINT shrB, MYINT shrC) {
	#ifdef SHIFT
		MYINT lshrA = MultiplyDeBruijnBitPosition2[(uint32_t)(shrA* 0x077CB531U) >> 27];;
		MYINT lshrB = MultiplyDeBruijnBitPosition2[(uint32_t)(shrB* 0x077CB531U) >> 27];;
		MYINT lshrC = MultiplyDeBruijnBitPosition2[(uint32_t)(shrC* 0x077CB531U) >> 27];;
	#endif

	for (MYITE i = 0; i < I; i++) {
		for (MYITE j = 0; j < J; j++) {
			MYINT a = A[i * J + j];
			MYINT b = B[i * J + j];

			#ifndef SHIFT
				a = a / shrA;
				b = b / shrB;
			#endif

			#ifdef SHIFT
			MYINT c = Saturate<MYINT>((a >> (lshrA + lshrC)) + (b >> (lshrB + lshrC)));
			#else
			MYINT c = Saturate<MYINT>(a / shrC + b / shrC);
			#endif

			C[i * J + j] = c;
		}
	}
	return;
}

// C = A + B
void MatAddCN(const MYINT* A, MYINT* B, MYINT* C, MYINT I, MYINT J, MYINT shrA, MYINT shrB, MYINT shrC) {
	#ifdef SHIFT
		MYINT lshrA = MultiplyDeBruijnBitPosition2[(uint32_t)(shrA* 0x077CB531U) >> 27];;
		MYINT lshrB = MultiplyDeBruijnBitPosition2[(uint32_t)(shrB* 0x077CB531U) >> 27];;
		MYINT lshrC = MultiplyDeBruijnBitPosition2[(uint32_t)(shrC* 0x077CB531U) >> 27];;
	#endif

	for (MYITE i = 0; i < I; i++) {
		for (MYITE j = 0; j < J; j++) {
			MYINT a = A[i * J + j];
			MYINT b = B[i * J + j];
			
			#ifndef SHIFT
				a = a / shrA;
				b = b / shrB;
			#endif

			#ifdef SHIFT
			MYINT c = Saturate<MYINT>((a >> (lshrA + lshrC)) + (b >> (lshrB + lshrC)));
			#else
			MYINT c = Saturate<MYINT>(a / shrC + b / shrC);
			#endif
			C[i * J + j] = c;
		}
	}
	return;
}

// C = A + B
void MatAddNC(MYINT* A, const MYINT* B, MYINT* C, MYINT I, MYINT J, MYINT shrA, MYINT shrB, MYINT shrC) {
	
	#ifdef SHIFT
		MYINT lshrA = MultiplyDeBruijnBitPosition2[(uint32_t)(shrA* 0x077CB531U) >> 27];;
		MYINT lshrB = MultiplyDeBruijnBitPosition2[(uint32_t)(shrB* 0x077CB531U) >> 27];;
		MYINT lshrC = MultiplyDeBruijnBitPosition2[(uint32_t)(shrC* 0x077CB531U) >> 27];;
	#endif


	for (MYITE i = 0; i < I; i++) {
		for (MYITE j = 0; j < J; j++) {
			MYINT a = A[i * J + j];
			MYINT b = B[i * J + j];

			#ifndef SHIFT
				a = a / shrA;
				b = b / shrB;
			#endif

			#ifdef SHIFT
				MYINT c = Saturate<MYINT>((a >> (lshrA+lshrC)) + (b>> (lshrB+lshrC)));
			#else
				MYINT c = Saturate<MYINT>(a / shrC + b / shrC);
			#endif

			C[i * J + j] = c;
		}
	}
	return;
}

// C = A + B
void MatAddCC(const MYINT* A, const MYINT* B, MYINT* C, MYINT I, MYINT J, MYINT shrA, MYINT shrB, MYINT shrC) {
	
	#ifdef SHIFT
		MYINT lshrA = MultiplyDeBruijnBitPosition2[(uint32_t)(shrA* 0x077CB531U) >> 27];;
		MYINT lshrB = MultiplyDeBruijnBitPosition2[(uint32_t)(shrB* 0x077CB531U) >> 27];;
		MYINT lshrC = MultiplyDeBruijnBitPosition2[(uint32_t)(shrC* 0x077CB531U) >> 27];;
	#endif

	for (MYITE i = 0; i < I; i++) {
		for (MYITE j = 0; j < J; j++) {
			MYINT a = A[i * J + j];
			MYINT b = B[i * J + j];

			#ifndef SHIFT
				a = a / shrA;
				b = b / shrB;
			#endif

			#ifdef SHIFT
				MYINT c = Saturate<MYINT>((a >> (lshrA+lshrC)) + (b>> (lshrB+lshrC)));
			#else
				MYINT c = Saturate<MYINT>(a / shrC + b / shrC);
			#endif

			C[i * J + j] = c;
		}
	}
	return;
}

// C = a + B
void MatAddBroadCastA(MYINT* A, MYINT* B, MYINT* C, MYINT I, MYINT J, MYINT shrA, MYINT shrB, MYINT shrC) {
	#ifdef SHIFT
		MYINT lshrA = MultiplyDeBruijnBitPosition2[(uint32_t)(shrA* 0x077CB531U) >> 27];;
		MYINT lshrB = MultiplyDeBruijnBitPosition2[(uint32_t)(shrB* 0x077CB531U) >> 27];;
		MYINT lshrC = MultiplyDeBruijnBitPosition2[(uint32_t)(shrC* 0x077CB531U) >> 27];;
	#endif

	for (MYITE i = 0; i < I; i++) {
		for (MYITE j = 0; j < J; j++) {
			MYINT a = *A;
			MYINT b = B[i * J + j];

			#ifndef SHIFT
				a = a / shrA;
				b = b / shrB;
			#endif

			#ifdef SHIFT
				MYINT c = Saturate<MYINT>((a >> (lshrA+lshrC)) + (b>> (lshrB+lshrC)));
			#else
				MYINT c = Saturate<MYINT>(a / shrC + b / shrC);
			#endif

			C[i * J + j] = c;
		}
	}
	return;
}

// C = A + b
void MatAddBroadCastB(MYINT* A, MYINT* B, MYINT* C, MYINT I, MYINT J, MYINT shrA, MYINT shrB, MYINT shrC) {
	#ifdef SHIFT
		MYINT lshrA = MultiplyDeBruijnBitPosition2[(uint32_t)(shrA* 0x077CB531U) >> 27];;
		MYINT lshrB = MultiplyDeBruijnBitPosition2[(uint32_t)(shrB* 0x077CB531U) >> 27];;
		MYINT lshrC = MultiplyDeBruijnBitPosition2[(uint32_t)(shrC* 0x077CB531U) >> 27];;
	#endif
	for (MYITE i = 0; i < I; i++) {
		for (MYITE j = 0; j < J; j++) {
			MYINT a = A[i * J + j];
			MYINT b = *B;

			#ifndef SHIFT
				a = a / shrA;
				b = b / shrB;
			#endif

			#ifdef SHIFT
				MYINT c = Saturate<MYINT>((a >> (lshrA+lshrC)) + (b>> (lshrB+lshrC)));
			#else
				MYINT c = Saturate<MYINT>(a / shrC + b / shrC);
			#endif

			C[i * J + j] = c;
		}
	}
	return;
}

// C = A - B
// TODO: shrB is int32_t because in 8-bit/16-bit code, shrB is usually very high and int8_t/int16_t will overflow.
void MatSub(MYINT* A, const MYINT* B, MYINT* C, MYINT I, MYINT J, MYINT shrA, int32_t shrB, MYINT shrC) {
	
	#ifdef SHIFT
		MYINT lshrA = MultiplyDeBruijnBitPosition2[(uint32_t)(shrA* 0x077CB531U) >> 27];;
		MYINT lshrB = MultiplyDeBruijnBitPosition2[(uint32_t)(shrB* 0x077CB531U) >> 27];;
		MYINT lshrC = MultiplyDeBruijnBitPosition2[(uint32_t)(shrC* 0x077CB531U) >> 27];;
	#endif
	
	for (MYITE i = 0; i < I; i++) {
		for (MYITE j = 0; j < J; j++) {
			MYINT a = A[i * J + j];
			MYINT b = B[i * J + j];

			#ifndef SHIFT
				a = a / shrA;
				b = b / shrB;
			#endif

			#ifdef SHIFT
				MYINT c = Saturate<MYINT>((a >> (lshrA+lshrC)) - (b>> (lshrB+lshrC)));
			#else
				MYINT c = Saturate<MYINT>(a / shrC - b / shrC);
			#endif

			C[i * J + j] = c;
		}
	}
	return;
}

// C = a - B
// TODO: shrB is int32_t because in 8-bit/16-bit code, shrB is usually very high and int8_t/int16_t will overflow.
void MatSubBroadCastA(MYINT* A, MYINT* B, MYINT* C, MYINT I, MYINT J, MYINT shrA, int32_t shrB, MYINT shrC) {
	
	#ifdef SHIFT
		MYINT lshrA = MultiplyDeBruijnBitPosition2[(uint32_t)(shrA* 0x077CB531U) >> 27];;
		MYINT lshrB = MultiplyDeBruijnBitPosition2[(uint32_t)(shrB* 0x077CB531U) >> 27];;
		MYINT lshrC = MultiplyDeBruijnBitPosition2[(uint32_t)(shrC* 0x077CB531U) >> 27];;
	#endif

	for (MYITE i = 0; i < I; i++) {
		for (MYITE j = 0; j < J; j++) {
			MYINT a = *A;
			MYINT b = B[i * J + j];

			#ifndef SHIFT
				a = a / shrA;
				b = b / shrB;
			#endif

			#ifdef SHIFT
				MYINT c = Saturate<MYINT>((a >> (lshrA+lshrC)) - (b>> (lshrB+lshrC)));
			#else
				MYINT c = Saturate<MYINT>(a / shrC - b / shrC);
			#endif

			C[i * J + j] = c;
		}
	}
	return;
}

// C = A - b
// TODO: shrB is int32_t because in 8-bit/16-bit code, shrB is usually very high and int8_t/int16_t will overflow.
void MatSubBroadCastB(MYINT* A, MYINT* B, MYINT* C, MYINT I, MYINT J, MYINT shrA, int32_t shrB, MYINT shrC) {
	#ifdef SHIFT
		MYINT lshrA = MultiplyDeBruijnBitPosition2[(uint32_t)(shrA* 0x077CB531U) >> 27];;
		MYINT lshrB = MultiplyDeBruijnBitPosition2[(uint32_t)(shrB* 0x077CB531U) >> 27];;
		MYINT lshrC = MultiplyDeBruijnBitPosition2[(uint32_t)(shrC* 0x077CB531U) >> 27];;
	#endif
	
	for (MYITE i = 0; i < I; i++) {
		for (MYITE j = 0; j < J; j++) {
			MYINT a = A[i * J + j];
			MYINT b = *B;

			#ifndef SHIFT
				a = a / shrA;
				b = b / shrB;
			#endif

			#ifdef SHIFT
				MYINT c = Saturate<MYINT>((a >> (lshrA+lshrC)) - (b>> (lshrB+lshrC)));
			#else
				MYINT c = Saturate<MYINT>(a / shrC - b / shrC);
			#endif

			C[i * J + j] = c;
		}
	}
	return;
}

// C = A * B
void MatMulNN(MYINT* A, MYINT* B, MYINT* C, MYINT* tmp, MYINT I, MYINT K, MYINT J, MYINT shrA, MYINT shrB, MYINT H1, MYINT H2) {
	#ifdef SHIFT
		MYINT lshrA = MultiplyDeBruijnBitPosition2[(uint32_t)(shrA* 0x077CB531U) >> 27];;
		MYINT lshrB = MultiplyDeBruijnBitPosition2[(uint32_t)(shrB* 0x077CB531U) >> 27];;
	#endif
	
	for (MYITE i = 0; i < I; i++) {
		for (MYITE j = 0; j < J; j++) {
			for (MYITE k = 0; k < K; k++) {
				MYINT a = A[i * K + k];
				MYINT b = B[k * J + j];

				#ifdef FASTAPPROX
					#ifndef SHIFT
						a = a / shrA;
						b = b / shrB;
					#else
						a = a >> lshrA;
						b = b >> lshrB;
					#endif

					tmp[k] = a * b;
				#else
					int64_t prod = ((int64_t)a * (int64_t)b);
					#ifdef SHIFT
						tmp[k] = Saturate<MYINT>((prod >> (lshrB + lshrA)));
					#else
						tmp[k] = Saturate<MYINT>((prod / ((int64_t)shrB * (int64_t)shrA)));
					#endif
				#endif
			}

			MYITE count = K, depth = 0;
			bool shr = true;

			while (depth < (H1 + H2)) {
				if (depth >= H1) {
					shr = false;
				}

				for (MYITE p = 0; p < (K / 2 + 1); p++) { // TODO: Check if this needs changing K/2 -> K>>1 
					MYINT sum;
					if (p < (count >> 1)) {
						if (shr) {
							#ifdef SHIFT
								sum = (tmp[2 * p] >> 1) + (tmp[(2 * p) + 1] >> 1);
							#else 
								sum = tmp[2 * p] / 2 + tmp[(2 * p) + 1] / 2;
							#endif
						} else {
							sum = tmp[2 * p] + tmp[(2 * p) + 1];
						}
					} else if ((p == (count >> 1)) && ((count & 1) == 1)) {
						if (shr) {
							#ifdef SHIFT
								sum = tmp[2 * p] >> 1;
							#else 
								sum = tmp[2 * p] / 2;
							#endif
						} else {
							sum = tmp[2 * p];
						}
					} else {
						sum = 0;
					}

					tmp[p] = sum;
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
void MatMulCN(const MYINT* A, MYINT* B, MYINT* C, MYINT* tmp, MYINT I, MYINT K, MYINT J, MYINT shrA, MYINT shrB, MYINT H1, MYINT H2) {
	#ifdef SHIFT
		MYINT lshrA = MultiplyDeBruijnBitPosition2[(uint32_t)(shrA* 0x077CB531U) >> 27];;
		MYINT lshrB = MultiplyDeBruijnBitPosition2[(uint32_t)(shrB* 0x077CB531U) >> 27];;
	#endif

	for (MYITE i = 0; i < I; i++) {
		for (MYITE j = 0; j < J; j++) {
			for (MYITE k = 0; k < K; k++) {
				MYINT a = A[i * K + k];
				MYINT b = B[k * J + j];

				#ifdef FASTAPPROX
					#ifndef SHIFT
						a = a / shrA;
						b = b / shrB;
					#else
						a = a >> lshrA;
						b = b >> lshrB;
					#endif

					tmp[k] = a * b;
				#else
					int64_t prod = ((int64_t)a * (int64_t)b);
					#ifdef SHIFT
						tmp[k] = Saturate<MYINT>((prod >> (lshrB + lshrA)));
					#else
						tmp[k] = Saturate<MYINT>((prod / ((int64_t)shrB * (int64_t)shrA)));
					#endif
				#endif
			}

			MYITE count = K, depth = 0;
			bool shr = true;

			while (depth < (H1 + H2)) {
				if (depth >= H1) {
					shr = false;
				}

				for (MYITE p = 0; p < (K / 2 + 1); p++) {
					MYINT sum;
					if (p < (count >> 1)) {
						if (shr) {
							#ifdef SHIFT
								sum = (tmp[2 * p] >> 1) + (tmp[(2 * p) + 1] >> 1);
							#else
								sum = tmp[2 * p] / 2 + tmp[(2 * p) + 1] / 2;
							#endif
						} else {
							sum = tmp[2 * p] + tmp[(2 * p) + 1];
						}
					} else if ((p == (count >> 1)) && ((count & 1) == 1)) {
						if (shr) {
							#ifdef SHIFT
								sum = tmp[2 * p] >> 1;
							#else
								sum = tmp[2 * p] / 2;
							#endif
						} else {
							sum = tmp[2 * p];
						}
					} else {
						sum = 0;
					}

					tmp[p] = sum;
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
void MatMulNC(MYINT* A, const MYINT* B, MYINT* C, MYINT* tmp, MYINT I, MYINT K, MYINT J, MYINT shrA, MYINT shrB, MYINT H1, MYINT H2) {
	#ifdef SHIFT
		MYINT lshrA = MultiplyDeBruijnBitPosition2[(uint32_t)(shrA* 0x077CB531U) >> 27];;
		MYINT lshrB = MultiplyDeBruijnBitPosition2[(uint32_t)(shrB* 0x077CB531U) >> 27];;
	#endif

	for (MYITE i = 0; i < I; i++) {
		for (MYITE j = 0; j < J; j++) {
			for (MYITE k = 0; k < K; k++) {
				MYINT a = A[i * K + k];
				MYINT b = B[k * J + j];

				#ifdef FASTAPPROX
					#ifndef SHIFT
						a = a / shrA;
						b = b / shrB;
					#else
						a = a >> lshrA;
						b = b >> lshrB;
					#endif

					tmp[k] = a * b;
				#else
					int64_t prod = ((int64_t)a * (int64_t)b);
					#ifdef SHIFT
						tmp[k] = Saturate<MYINT>((prod >> (lshrB + lshrA)));
					#else
						tmp[k] = Saturate<MYINT>((prod / ((int64_t)shrB * (int64_t)shrA)));
					#endif
				#endif
			}

			MYITE count = K, depth = 0;
			bool shr = true;

			while (depth < (H1 + H2)) {
				if (depth >= H1) {
					shr = false;
				}

				for (MYITE p = 0; p < (K / 2 + 1); p++) {
					MYINT sum;
					if (p < (count >> 1)) {
						if (shr) {
							#ifdef SHIFT
								sum = (tmp[2 * p] >> 1) + (tmp[(2 * p) + 1] >> 1);
							#else
								sum = tmp[2 * p] / 2 + tmp[(2 * p) + 1] / 2;
							#endif
						} else {
							sum = tmp[2 * p] + tmp[(2 * p) + 1];
						}
					} else if ((p == (count >> 1)) && ((count & 1) == 1)) {
						if (shr) {
							#ifdef SHIFT
								sum = tmp[2 * p] >> 1;
							#else
								sum = tmp[2 * p] / 2;
							#endif
						} else {
							sum = tmp[2 * p];
						}
					} else {
						sum = 0;
					}

					tmp[p] = sum;
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
void MatMulCC(const MYINT* A, const MYINT* B, MYINT* C, MYINT* tmp, MYINT I, MYINT K, MYINT J, MYINT shrA, MYINT shrB, MYINT H1, MYINT H2) {
	
	#ifdef SHIFT
		MYINT lshrA = MultiplyDeBruijnBitPosition2[(uint32_t)(shrA* 0x077CB531U) >> 27];;
		MYINT lshrB = MultiplyDeBruijnBitPosition2[(uint32_t)(shrB* 0x077CB531U) >> 27];;
	#endif

	for (MYITE i = 0; i < I; i++) {
		for (MYITE j = 0; j < J; j++) {
			for (MYITE k = 0; k < K; k++) {
				MYINT a = A[i * K + k];
				MYINT b = B[k * J + j];

				#ifdef FASTAPPROX
					#ifndef SHIFT
						a = a / shrA;
						b = b / shrB;
					#else
						a = a >> lshrA;
						b = b >> lshrB;
					#endif

					tmp[k] = a * b;
				#else
					int64_t prod = ((int64_t)a * (int64_t)b);
					#ifdef SHIFT
						tmp[k] = Saturate<MYINT>((prod >> (lshrB + lshrA)));
					#else
						tmp[k] = Saturate<MYINT>((prod / ((int64_t)shrB * (int64_t)shrA)));
					#endif
				#endif
			}

			MYITE count = K, depth = 0;
			bool shr = true;

			while (depth < (H1 + H2)) {
				if (depth >= H1) {
					shr = false;
				}

				for (MYITE p = 0; p < (K / 2 + 1); p++) {
					MYINT sum;
					if (p < (count >> 1)) {
						if (shr) {
							#ifdef SHIFT
								sum = (tmp[2 * p] >> 1) + (tmp[(2 * p) + 1] >> 1);
							#else
								sum = tmp[2 * p] / 2 + tmp[(2 * p) + 1] / 2;
							#endif
						} else {
							sum = tmp[2 * p] + tmp[(2 * p) + 1];
						}
					} else if ((p == (count >> 1)) && ((count & 1) == 1)) {
						if (shr) {
							#ifdef SHIFT
								sum = tmp[2 * p] >> 1;
							#else
								sum = tmp[2 * p] / 2;
							#endif
						} else {
							sum = tmp[2 * p];
						}
					} else {
						sum = 0;
					}

					tmp[p] = sum;
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
// TODO: K is int16_t because K is usually very high and int8_t will overflow in 8-bit code.
void SparseMatMulX(const MYINT* Aidx, const MYINT* Aval, MYINT** B, MYINT* C, int16_t K, MYINT shrA, MYINT shrB, MYINT shrC) {
		
	#ifdef SHIFT
		MYINT lshrA = MultiplyDeBruijnBitPosition2[(uint32_t)(shrA* 0x077CB531U) >> 27];;
		MYINT lshrB = MultiplyDeBruijnBitPosition2[(uint32_t)(shrB* 0x077CB531U) >> 27];;
		MYINT lshrC = MultiplyDeBruijnBitPosition2[(uint32_t)(shrC* 0x077CB531U) >> 27];;
	#endif

	MYITE ite_idx = 0, ite_val = 0;
	for (MYITE k = 0; k < K; k++) {
		MYINT b = B[k * 1][0];

		#ifdef FASTAPPROX
			b = b / shrB;
		#endif
		MYITE idx = Aidx[ite_idx];
		while (idx != 0) {
			MYINT a = Aval[ite_val];

			#ifdef FASTAPPROX
				#ifdef SHIFT
					a = a >> lshrA;
				#else
					a = a / shrA;
				#endif

				MYINT c = a * b;
				#ifdef SHIFT
					c = c >> lshrC;
				#else
					c = c / shrC;
				#endif
			#else
				#ifdef SHIFT
					MYINT c = Saturate<MYINT>(((int64_t)a * (int64_t)b) >> (lshrC + lshrA + lshrB));
				#else
					MYINT c = Saturate<MYINT>(((int64_t)a * (int64_t)b) / ((int64_t)shrC * (int64_t)shrA * (int64_t)shrB));
				#endif
			#endif
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
// TODO: K is int16_t because K is usually very high and int8_t will overflow in 8-bit code.
void SparseMatMul(const MYINT* Aidx, const MYINT* Aval, MYINT* B, MYINT* C, int16_t K, MYINT shrA, MYINT shrB, MYINT shrC) {

	#ifdef SHIFT
		MYINT lshrA = MultiplyDeBruijnBitPosition2[(uint32_t)(shrA* 0x077CB531U) >> 27];;
		MYINT lshrB = MultiplyDeBruijnBitPosition2[(uint32_t)(shrB* 0x077CB531U) >> 27];;
		MYINT lshrC = MultiplyDeBruijnBitPosition2[(uint32_t)(shrC* 0x077CB531U) >> 27];;
	#endif

	MYITE ite_idx = 0, ite_val = 0;
	for (MYITE k = 0; k < K; k++) {
		MYINT b = B[k];

		#ifdef FASTAPPROX
			b = b / shrB;
		#endif
		MYITE idx = Aidx[ite_idx];
		while (idx != 0) {
			MYINT a = Aval[ite_val];

			#ifdef FASTAPPROX
				#ifdef SHIFT
					a = a >> lshrA;
				#else
					a = a / shrA;
				#endif
				

				MYINT c = a * b;
				#ifdef SHIFT
					c = c >> lshrC;
				#else
					c = c / shrC;
				#endif
				
			#else
				#ifdef SHIFT
					MYINT c = Saturate<MYINT>(((int64_t)a * (int64_t)b) >> (lshrC + lshrA + lshrB));
				#else
					MYINT c = Saturate<MYINT>(((int64_t)a * (int64_t)b) / ((int64_t)shrC * (int64_t)shrA * (int64_t)shrB));
				#endif
			#endif
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
void MulCir(MYINT* A, MYINT* B, MYINT* C, MYINT I, MYINT J, MYINT shrA, MYINT shrB) {
	
	#ifdef SHIFT
		MYINT lshrA = MultiplyDeBruijnBitPosition2[(uint32_t)(shrA* 0x077CB531U) >> 27];;
		MYINT lshrB = MultiplyDeBruijnBitPosition2[(uint32_t)(shrB* 0x077CB531U) >> 27];;
	#endif

	for (MYITE i = 0; i < I; i++) {
		for (MYITE j = 0; j < J; j++) {
			MYINT a = A[i * J + j];
			MYINT b = B[i * J + j];

			#ifdef FASTAPPROX
				#ifdef SHIFT
					a = a >> lshrA;
					b = b >> lshrB;
				#else
					a = a / shrA;
					b = b / shrB;
				#endif

				C[i * J + j] = a * b;
			#else
				int64_t prod = ((int64_t)a * (int64_t)b);
				#ifdef SHIFT
					C[i * J + j] = Saturate<MYINT>(prod >> (lshrB + lshrA));
				#else
					C[i * J + j] = Saturate<MYINT>(prod / ((int64_t)shrB * (int64_t)shrA));
				#endif
			#endif
		}
	}
	return;
}

// A = tanh(A)
void TanH(MYINT* A, MYINT I, MYINT J, MYINT scale_in, MYINT scale_out, MYINT* B) {
	#ifdef SHIFT
		MYINT lscale_in = log2(scale_in);
		MYINT lscale_out = log2(scale_out);
	#endif

	for (MYITE i = 0; i < I; i++) {
		for (MYITE j = 0; j < J; j++) {
			#ifdef FLOATEXP
				float x = float(A[i * J + j]) / scale_in;
				float y = tanh(x);
				MYINT z = MYINT(y * scale_out);

				B[i * J + j] = z;
			#else
				MYINT x = A[i * J + j], y;

				if (x >= scale_in) {
					y = scale_in;
				} else if (x <= -scale_in) {
					y = -scale_in;
				} else {
					y = x;
				}

				#ifdef SHIFT
					MYINT scale_diff = scale_out >> lscale_in;
				#else
					MYINT scale_diff = scale_out / scale_in;
				#endif

				y *= scale_diff;

				B[i * J + j] = y;
			#endif
		}
	}
	return;
}

// index = argmax(A)
void ArgMax(MYINT* A, MYINT I, MYINT J, int* index) {
	MYINT max = A[0];
	MYITE maxIndex = 0, counter = 0;
	for (MYITE i = 0; i < I; i++) {
		for (MYITE j = 0; j < J; j++) {
			MYINT x = A[i * J + j];

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

// B = reverse(A, axis)
void Reverse2(MYINT* A, MYINT axis, MYINT I, MYINT J, MYINT* B) {
	for (MYITE i = 0; i < I; i++) {
		for (MYITE j = 0; j < J; j++) {	
			MYINT i_prime = (axis == 0 ? (I - 1 - i) : i);
			MYINT j_prime = (axis == 1 ? (J - 1 - j) : j);
			B[i * J + j] = A[i_prime * J + j_prime];
		}
	}
	return;
}

// A = A^T
void Transpose(MYINT* A, MYINT* B, MYINT I, MYINT J) {
	for (MYITE i = 0; i < I; i++) {
		for (MYITE j = 0; j < J; j++) {
			B[i * J + j] = A[j * I + i];
		}
	}
	return;
}

// C = a * B
void ScalarMul(MYINT* A, MYINT* B, MYINT* C, MYINT I, MYINT J, MYINT shrA, MYINT shrB) {
	#ifdef SHIFT
		MYINT lshrA = MultiplyDeBruijnBitPosition2[(uint32_t)(shrA* 0x077CB531U) >> 27];;
		MYINT lshrB = MultiplyDeBruijnBitPosition2[(uint32_t)(shrB* 0x077CB531U) >> 27];;
	#endif

	MYINT a = *A;

	#ifdef FASTAPPROX
		#ifdef SHIFT
			a = a >> lshrA;
		#else
			a = a / shrA;
		#endif
	#endif
	for (MYITE i = 0; i < I; i++) {
		for (MYITE j = 0; j < J; j++) {
			MYINT b = B[i * J + j];

			#ifdef FASTAPPROX
				#ifdef SHIFT
					b = b >> lshrB;
				#else
					b = b / shrB;
				#endif
				C[i * J + j] = a * b;
			#else
				int64_t prod = ((int64_t)a * (int64_t)b);
				#ifdef SHIFT
					C[i * J + j] = Saturate<MYINT>(prod >> (lshrA + lshrB));
				#else
					C[i * J + j] = Saturate<MYINT>(prod / ((int64_t)shrA * (int64_t)shrB));
				#endif
			#endif
		}
	}
	return;
}

// C = A # B
// A[N][H][W][CI], B[HF][WF][CI][CO], C[N][H][W][CO]
void Conv(MYINT* A, const MYINT* B, MYINT* C, MYINT* tmp, MYINT N, MYINT H, MYINT W, MYINT CI, MYINT HF, MYINT WF, MYINT CO, MYINT shrA, MYINT shrB, MYINT H1, MYINT H2) {
	#ifdef SHIFT
		MYINT lshrA = MultiplyDeBruijnBitPosition2[(uint32_t)(shrA* 0x077CB531U) >> 27];;
		MYINT lshrB = MultiplyDeBruijnBitPosition2[(uint32_t)(shrB* 0x077CB531U) >> 27];;
	#endif
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
								MYINT a = (((((h + hf) < padH) || ((h + hf) >= (H + padH))) || (((w + wf) < padW) || ((w + wf) >= (W + padW)))) ? 0 : A[n * H * W * CI + ((h + hf) - padH) * W * CI + ((w + wf) - padW) * CI + ci]);
								MYINT b = B[hf * WF * CI * CO + wf * CI * CO + ci * CO + co];

								#ifdef FASTAPPROX
									#ifdef SHIFT
										a = a >> lshrA;
										b = b >> lshrB;
									#else
										a = a / shrA;
										b = b / shrB;
									#endif

									tmp[counter] = a * b;
								#else
									#ifdef SHIFT
										int64_t temp = (((int64_t) a) * ((int64_t)b)) >> (lshrA + lshrB);
									#else
										int64_t temp = (((int64_t) a) * ((int64_t)b)) / (((int64_t)shrA) * ((int64_t)shrB));
									#endif
									tmp[counter] = Saturate<MYINT>(temp);
								#endif
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
							MYINT sum;
							if (p < (count >> 1)) {
								if (shr) {
									#ifdef SHIFT
										sum = (tmp[2 * p] >> 1) + (tmp[(2 * p) + 1] >> 1);
									#else
										sum = tmp[2 * p] / 2 + tmp[(2 * p) + 1] / 2;
									#endif
								} else {
									sum = tmp[2 * p] + tmp[(2 * p) + 1];
								}
							} else if ((p == (count >> 1)) && ((count & 1) == 1)) {
								if (shr) {
									#ifdef SHIFT
										sum = tmp[2 * p] >> 1;
									#else
										sum = tmp[2 * p] / 2;
									#endif
								} else {
									sum = tmp[2 * p];
								}
							} else {
								sum = 0;
							}

							tmp[p] = sum;
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

// C = conv(A, B, <params>)
// A[N][H][W][CIN], B[G][HF][WF][CINF][COUTF], C[N][HOUT][WOUT][COUTF*G]
void Convolution(MYINT* A, const MYINT* B, MYINT* C, MYINT* tmp, MYINT N, MYINT H, MYINT W, MYINT CIN, MYINT HF, MYINT WF, MYINT CINF, MYINT COUTF, MYINT HOUT, MYINT WOUT, MYINT HPADL, MYINT HPADR, MYINT WPADL, MYINT WPADR, MYINT HSTR, MYINT WSTR, MYINT HDL, MYINT WDL, MYINT G, MYINT shrA, MYINT shrB, MYINT H1, MYINT H2) {
	#ifdef SHIFT
		MYINT lshrA = MultiplyDeBruijnBitPosition2[(uint32_t)(shrA* 0x077CB531U) >> 27];;
		MYINT lshrB = MultiplyDeBruijnBitPosition2[(uint32_t)(shrB* 0x077CB531U) >> 27];;
	#endif

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

									MYINT a = (((h + HDL * hf) < 0) || ((h + HDL * hf) >= H) || ((w + WDL * wf) < 0) || ((w + WDL * wf) >= W)) ? 0 : A[n * H * W * CIN + (h + HDL * hf) * W * CIN + (w + WDL * wf) * CIN + (ci + g * CINF)];
									MYINT b = B[g * HF * WF * CINF * COUTF + (hf + HF/2) * WF * CINF * COUTF + (wf + WF/2) * CINF * COUTF + ci * COUTF + co];

									#ifdef FASTAPPROX
										#ifdef SHIFT
											a = a >> lshrA;
											b = b >> lshrB;
										#else
											a = a / shrA;
											b = b / shrB;
										#endif
										tmp[counter] = a * b;
									#else
										#ifdef SHIFT
											int64_t temp = (((int64_t) a) * ((int64_t)b)) >> (lshrA + lshrB);
										#else
											int64_t temp = (((int64_t) a) * ((int64_t)b)) / (((int64_t)shrA) * ((int64_t)shrB));
										#endif
										tmp[counter] = Saturate<MYINT>(temp);
									#endif
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
								MYINT sum;
								if (p < (count >> 1)) {
									if (shr) {
										#ifdef SHIFT
											sum = (tmp[2 * p] >> 1) + (tmp[(2 * p) + 1] >> 1);
										#else
											sum = tmp[2 * p] / 2 + tmp[(2 * p) + 1] / 2;
										#endif
									} else {
										sum = tmp[2 * p] + tmp[(2 * p) + 1];
									}
								} else if ((p == (count >> 1)) && ((count & 1) == 1)) {
									if (shr) {
										#ifdef SHIFT
											sum = tmp[2 * p] >> 1;
										#else
											sum = tmp[2 * p] / 2;
										#endif
									} else {
										sum = tmp[2 * p];
									}
								} else {
									sum = 0;
								}

								tmp[p] = sum;
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

// A = A <+> B
// A[N][H][W][C], B[C]
void AddOrSubCir4D(MYINT* A, const MYINT* B, MYINT* X, MYINT N, MYINT H, MYINT W, MYINT C, MYINT shrA, MYINT shrB, MYINT shrC, bool add) {
	#ifdef SHIFT
		MYINT lshrA = MultiplyDeBruijnBitPosition2[(uint32_t)(shrA* 0x077CB531U) >> 27];;
		MYINT lshrB = MultiplyDeBruijnBitPosition2[(uint32_t)(shrB* 0x077CB531U) >> 27];;
		MYINT lshrC = MultiplyDeBruijnBitPosition2[(uint32_t)(shrC* 0x077CB531U) >> 27];;
	#endif

	for (MYITE n = 0; n < N; n++) {
		for (MYITE h = 0; h < H; h++) {
			for (MYITE w = 0; w < W; w++) {
				for (MYITE c = 0; c < C; c++) {
					MYINT a = A[n * H * W * C + h * W * C + w * C + c];
					#ifndef SHIFT
						a = a / shrA;
					#endif

					MYINT b = B[c];
					#ifndef SHIFT
						b = b / shrB;
					#endif

					MYINT res;
					if (add) {
						#ifdef SHIFT
							res = Saturate<MYINT>((a >> (lshrA + lshrC)) + (b >> (lshrB + lshrC)));
						#else
							res = Saturate<MYINT>(a / shrC + b / shrC);
						#endif
					} else {
						#ifdef SHIFT
							res = Saturate<MYINT>((a >> (lshrA + lshrC)) - (b >> (lshrB + lshrC)));
						#else
							res = Saturate<MYINT>(a / shrC - b / shrC);
						#endif
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
void AddOrSubCir2D(MYINT* A, const MYINT* B, MYINT* X, MYINT H, MYINT W, MYINT shrA, MYINT shrB, MYINT shrC, bool add) {
	#ifdef SHIFT
		MYINT lshrA = MultiplyDeBruijnBitPosition2[(uint32_t)(shrA* 0x077CB531U) >> 27];;
		MYINT lshrB = MultiplyDeBruijnBitPosition2[(uint32_t)(shrB* 0x077CB531U) >> 27];;
		MYINT lshrC = MultiplyDeBruijnBitPosition2[(uint32_t)(shrC* 0x077CB531U) >> 27];;
	#endif
	
	for (MYITE h = 0; h < H; h++) {
		for (MYITE w = 0; w < W; w++) {
			MYINT a = A[h * W + w];

			#ifndef SHIFT
			a = a / shrA;
			#endif

			MYINT b = B[w];
			#ifndef SHIFT
			b = b / shrB;
			#endif

			MYINT res;
			if (add) {
				#ifdef SHIFT
				res = Saturate<MYINT>((a >> (lshrA + lshrC)) + (b >> (lshrB + lshrC)));
				#else
				res = Saturate<MYINT>(a / shrC + b / shrC);
				#endif
			} else {
				#ifdef SHIFT
				res = Saturate<MYINT>((a >> (lshrA + lshrC)) - (b >> (lshrB + lshrC)));
				#else
				res = Saturate<MYINT>(a / shrC - b / shrC);
				#endif
			}

			X[h * W + w] = res;
		}
	}
	return;
}

// A = relu(A)
// A[N][H][W][C]
void Relu4D(MYINT* A, MYINT N, MYINT H, MYINT W, MYINT C) {
	for (MYITE n = 0; n < N; n++) {
		for (MYITE h = 0; h < H; h++) {
			for (MYITE w = 0; w < W; w++) {
				for (MYITE c = 0; c < C; c++) {
					MYINT a = A[n * H * W * C + h * W * C + w * C + c];
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
void Relu6(MYINT* A, MYINT* B, MYINT N, MYINT H, MYINT W, MYINT C,MYINT six, MYINT div) {
	for (MYITE n = 0; n < N; n++) {
		for (MYITE h = 0; h < H; h++) {
			for (MYITE w = 0; w < W; w++) {
				for (MYITE c = 0; c < C; c++) {
					MYINT a = A[n * H * W * C + h * W * C + w * C + c];
					if (a < 0) {
						a = 0;
					}
					if (a > six) {
						a = six;
					}

					B[n * H * W * C + h * W * C + w * C + c] = a / div;
				}
			}
		}
	}
	return;
}

// A = relu(A)
// A[N][H][W][C]
void Relu2D(MYINT* A, MYINT H, MYINT W) {
	for (MYITE h = 0; h < H; h++) {
		for (MYITE w = 0; w < W; w++) {
			MYINT a = A[h * W + w];
			if (a < 0){
				a = 0;
			}

			A[h * W + w] = a;
		}
	}
	return;
}

// B = maxpool(A)
// A[N][H][W][C], B[N][H][W][C]
void Maxpool(MYINT* A, MYINT* B, MYINT N, MYINT H, MYINT W, MYINT C, MYINT FH, MYINT FW, MYINT strideH, MYINT strideW, MYINT HPADL, MYINT HPADR, MYINT WPADL, MYINT WPADR) {
	MYITE HO = H / strideH;
	MYITE WO = W / strideW;

	for (MYITE n = 0; n < N; n++) {
		for (MYITE ho = 0; ho < HO; ho++) {
			for (MYITE wo = 0; wo < WO; wo++) {
				for (MYITE c = 0; c < C; c++) {

					MYINT max = A[n * H * W * C + (strideH * ho) * W * C + (strideW * wo) * C + c];
					for (MYITE hs = 0; hs < FH; hs++) {
						for (MYITE ws = 0; ws < FW; ws++) {
							MYINT a = A[n * H * W * C + ((strideH * ho) + hs) * W * C + ((strideW * wo) + ws) * C + c];
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

// A = Normalise(A)
void NormaliseL2(MYINT* A, MYINT* B, MYINT N, MYINT H, MYINT W, MYINT C, MYINT scaleA, MYINT shrA) {
	for (MYITE n = 0; n < N; n++) {
		for (MYITE h = 0; h < H; h++) {
			for (MYITE w = 0; w < W; w++) {

				// Calculate the sum square.
				int32_t sumSquare = 0;
				MYINT shrAdiv = (1 << shrA);

				for (MYITE c = 0; c < C; c++) {
					#ifdef FASTAPPROX
						#ifdef SHIFT
							MYINT tmp = (A[n * H * W * C + h * W * C + w * C + c] >> shrA);
						#else
							MYINT tmp = (A[n * H * W * C + h * W * C + w * C + c] / shrAdiv);
						#endif
						sumSquare += tmp*tmp;
					#else
						int32_t tmp = A[n * H * W * C + h * W * C + w * C + c];
						#ifdef SHIFT
							sumSquare += (((tmp * tmp) >> shrA) >> shrA);
						#else
							sumSquare += (((tmp * tmp) / shrAdiv) / shrAdiv);
						#endif
					#endif
				}

				// Calculate the inverse square root of sumSquare.
				MYINT yLow = 1;

				// yHigh: A number of length shrA with all 1s in binary representation e.g. for shrA=8 --> y_high = 0b11111111
				MYINT yHigh = (1 << shrA - 1);

				// one: value of 1 with same scale as y * y * sumSquare.
				// scale of sumSquare = 2*scale_in + 2*shrA
				// since we assume scale of y = 1 - shrA
				// scale of y*y*sumSquare =  2*scale_in + 2*shrA + 2(1-shrA) = 2*scale_in + 2
				int32_t one = (1 << (-(2 * scaleA + 2)));

				// Binary search for the inverse square root.
				while (yLow + 1 < yHigh) {

					// Using int32_t sotherwise (y*y*sumSquare) will overflow.
					MYINT yMid = ((yHigh + yLow) >> 1);

					int64_t cmpValue = (int64_t)sumSquare * yMid * yMid;

					if (cmpValue > one) {
						yHigh = yMid;	
					} else {
						yLow = yMid;
					}
				}
				MYINT inverseNorm = yLow;

				// Multiply all elements by the 1/sqrt(sumSquare).
				for (MYITE c = 0; c < C; c++) {
					#ifdef SHIFT
						B[n * H * W * C + h * W * C + w * C + c]  = (A[n * H * W * C + h * W * C + w * C + c]  >> shrA) * inverseNorm;
					#else
						B[n * H * W * C + h * W * C + w * C + c]  = (A[n * H * W * C + h * W * C + w * C + c]  / shrAdiv) * inverseNorm;
					#endif
				}
			}
		}
	}
	return;
}

// B = exp(A)
void Exp(MYINT* A, MYINT I, MYINT J, MYINT shrA, MYINT shrB, MYINT* B) {
	for (MYITE i = 0; i < I; i++) {
		for (MYITE j = 0; j < J; j++) {
			B[i * J + j] = ((MYINT)(exp(((float) A[i * J + j]) / shrA) * shrB));
		}
	}
	return;
}

// Check Shift Protocol
// B = Sigmoid(A)
void Sigmoid(MYINT* A, MYINT I, MYINT J, MYINT div, MYINT add, MYINT sigmoid_limit, MYINT scale_in, MYINT scale_out, MYINT* B) {
	MYINT scale_diff = scale_out / scale_in;
	for (MYITE i = 0; i < I; i++) {
		for (MYITE j = 0; j < J; j++) {
			#ifdef FLOATEXP
				float x = float(A[i * J + j]) / scale_in;

				float y = 1 / (1 + exp(-x));

				MYINT z = MYINT(y * scale_out);

				B[i * J + j] = z;
			#else
				MYINT x = A[i * J + j];

				x = (x / div) + add;

				MYINT y;
				if (x >= sigmoid_limit) {
					y = sigmoid_limit;
				} else if (x <= 0) {
					y = 0;
				} else {
					y = x;
				}

				y = y * scale_diff;

				B[i * J + j] = y;
			#endif
		}
	}
	return;
}

// A = AdjustScaleShr(A)
void AdjustScaleShr(MYINT* A, MYINT I, MYINT J, MYINT K, MYINT L, MYINT scale) {
	#ifdef SHIFT
		MYINT lscale = log2(scale);
	#endif
	for (MYITE i = 0; i < I; i++) {
		for (MYITE j = 0; j < J; j++) {
			for (MYITE k = 0; k < K; k++) {
				for (MYITE l = 0; l < L; l++) {
					MYINT a = A[i * J * K * L + j * K * L + k * L + l];
					#ifdef SHIFT
						A[i * J * K * L + j * K * L + k * L + l] = a >> lscale;
					#else
						A[i * J * K * L + j * K * L + k * L + l] = a / scale;
					#endif
				}
			}
		}
	}
	return;
}

// A = AdjustScaleShr(A)
void AdjustScaleShr(MYINT* A, MYINT I, MYINT J, MYINT scale) {
	for (MYITE i = 0; i < I; i++) {
		for (MYITE j = 0; j < J; j++) {
			MYINT a = A[i * J + j];
			A[i * J + j] = a / scale;
		}
	}
	return;
}

// A = AdjustScaleShl(A)
void AdjustScaleShl(MYINT* A, MYINT I, MYINT J, MYINT scale) {
	for (MYITE i = 0; i < I; i++) {
		for (MYITE j = 0; j < J; j++) {
			MYINT a = A[i * J + j];
			A[i * J + j] = a * scale;
		}
	}
	return;
}

// A = AdjustScaleShl(A)
void AdjustScaleShl(MYINT* A, MYINT I, MYINT J, MYINT K, MYINT L, MYINT scale) {
	for (MYITE i = 0; i < I; i++) {
		for (MYITE j = 0; j < J; j++) {
			for (MYITE k = 0; k < K; k++) {
				for (MYITE l = 0; l < L; l++) {
					MYINT a = A[i * J * K * L + j * K * L + k * L + l];
					A[i * J * K * L + j * K * L + k * L + l] = a * scale;
				}
			}
		}
	}
	return;
}

MYINT treeSum(MYINT* arr, MYINT count, MYINT height_shr, MYINT height_noshr) {
	if (count == 1) {
		return arr[0];
	}

	bool shr = true;

	for (MYITE depth = 0; depth < (height_shr + height_noshr); depth++) {
		if (depth >= height_shr) {
			shr = false;
		}

		for (MYITE index = 0; index < (count / 2); index++) {
			MYINT sum = arr[2 * index] + arr[(2 * index) + 1];

			if (shr) {
				arr[index] = sum / 2;
			} else {
				arr[index] = sum;
			}
		}

		if (count % 2 == 1) {
			MYITE index = (count / 2) + 1;
			if (shr) {
				arr[index - 1] = arr[count - 1] / 2;
			} else {
				arr[index - 1] = arr[count - 1];
			}
		}

		// Debugging
		if (count % 2 == 1) {
			arr[count / 2 + 1] = 0;
		} else {
			arr[count / 2] = 0;
		}

		count = (count + 1) >> 1;
	}

	return arr[0];
}
