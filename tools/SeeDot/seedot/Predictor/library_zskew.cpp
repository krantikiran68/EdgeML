// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include <iostream>
#include <cmath>

#include "datatypes.h"
#include "library_fixed.h"

// This file contains implementations of the linear algebra operators supported by SeeDot.
// Each function takes the scaling factors as arguments along with the pointers to the operands.

// C = A + B
void MatAdd(MYINT* A, MYINT* B, MYINT* C, MYITE I, MYITE J, MYITE left_shift, ACINT zeroA, ACINT shrA, MYITE nA, ACINT zeroB, ACINT shrB, MYITE nB, ACINT zeroC, ACINT shrC, MYITE nC) {
	for (MYITE i = 0; i < I; i++) {
		for (MYITE j = 0; j < J; j++) {
			ACINT a = A[i * J + j];
			ACINT b = B[i * J + j];

			a += zeroA;
			b += zeroB;
			a *= (1 << left_shift);
			b *= (1 << left_shift);

			a = MulQuantMultiplierLTO<ACINT>(a, shrA, nA);
			b = MulQuantMultiplierLTO<ACINT>(b, shrB, nB);

			ACINT c = MulQuantMultiplierLTO<ACINT>(a + b, shrC, nC);
			C[i * J + j] = Saturate<ACINT, MYINT>(zeroC + c);
		}
	}
	return;
}

// C = a + B
void MatAddBroadCastA(MYINT* A, MYINT* B, MYINT* C, MYITE I, MYITE J, MYITE left_shift, ACINT zeroA, ACINT shrA, MYITE nA, ACINT zeroB, ACINT shrB, MYITE nB, ACINT zeroC, ACINT shrC, MYITE nC) {
	ACINT a = (ACINT) *A;
	a += zeroA;
	a *= (1 << left_shift);
	a = MulQuantMultiplierLTO<ACINT>(a, shrA, nA);

	for (MYITE i = 0; i < I; i++) {
		for (MYITE j = 0; j < J; j++) {
			ACINT b = B[i * J + j];
			b += zeroB;
			b *= (1 << left_shift);
			b = MulQuantMultiplierLTO<ACINT>(b, shrB, nB);

			ACINT c = MulQuantMultiplierLTO<ACINT>(a + b, shrC, nC);
			C[i * J + j] = Saturate<ACINT, MYINT>(zeroC + c);
		}
	}
	return;
}

// C = a - B
void MatSubBroadCastA(MYINT* A, MYINT* B, MYINT* C, MYITE I, MYITE J, MYITE left_shift, ACINT zeroA, ACINT shrA, MYITE nA, ACINT zeroB, ACINT shrB, MYITE nB, ACINT zeroC, ACINT shrC, MYITE nC) {
	ACINT a = (ACINT) *A;
	a += zeroA;
	a *= (1 << left_shift);
	a = MulQuantMultiplierLTO<ACINT>(a, shrA, nA);

	for (MYITE i = 0; i < I; i++) {
		for (MYITE j = 0; j < J; j++) {
			ACINT b = B[i * J + j];
			b += zeroB;
			b *= (1 << left_shift);
			b = MulQuantMultiplierLTO<ACINT>(b, shrB, nB);

			ACINT c = MulQuantMultiplierLTO<ACINT>(a - b, shrC, nC);
			C[i * J + j] = Saturate<ACINT, MYINT>(zeroC + c);
		}
	}
	return;
}

// C = A * B
void MatMul(MYINT* A, MYINT* B, MYINT* C, MYITE I, MYITE K, MYITE J, ACINT zeroA, ACINT zeroB, ACINT zeroC, ACINT M0, MYITE N) {
	for (MYITE i = 0; i < I; i++) {
		for (MYITE j = 0; j < J; j++) {
			ACINT sum = 0;

			for (MYITE k = 0; k < K; k++) {
				ACINT a = A[i * K + k];
				ACINT b = B[k * J + j];

				sum += (a + zeroA) * (b + zeroB);
			}

			sum = MulQuantMultiplierLTO<ACINT>(sum, M0, N);
			C[i * J + j] = Saturate<ACINT, MYINT>(zeroC + sum);
		}
	}
	return;
}

// C = A <*> B
void Hadamard(MYINT* A, MYINT* B, MYINT* C, MYITE I, MYITE J, ACINT zeroA, ACINT zeroB, ACINT zeroC, ACINT M0, MYITE N) {
	for (MYITE i = 0; i < I; i++) {
		for (MYITE j = 0; j < J; j++) {
			ACINT a = A[i * K + k];
			ACINT b = B[k * J + j];

			ACINT prod = (a + zeroA) * (b + zeroB);

			prod = MulQuantMultiplierLTO<ACINT>(prod, M0, N);
			C[i * J + j] = Saturate<ACINT, MYINT>(zeroC + prod);
		}
	}
	return;
}

// C = a * B
void MatMulBroadcastA(MYINT* A, MYINT* B, MYINT* C, MYITE I, MYITE J, ACINT zeroA, ACINT zeroB, ACINT zeroC, ACINT M0, MYITE N) {
	ACINT a = (ACINT) *A;
	a += zeroA;

	for (MYITE i = 0; i < I; i++) {
		for (MYITE j = 0; j < J; j++) {
			ACINT b = B[k * J + j];
			ACINT prod = a * (b + zeroB);

			prod = MulQuantMultiplierLTO<ACINT>(prod, M0, N);
			C[i * J + j] = Saturate<ACINT, MYINT>(zeroC + prod);
		}
	}
	return;
}

// A = tanh(A)
void TanH(MYINT* A, MYINT I, MYINT J, MYINT scale_in, MYINT scale_out, MYINT* B) {
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

				MYINT scale_diff = scale_out / scale_in;

				y *= scale_diff;

				B[i * J + j] = y;
			#endif
		}
	}
	return;
}

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

