// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include <iostream>
#include <cmath>

#include "datatypes.h"
#include "library_zskew.h"

// This file contains implementations of the linear algebra operators supported by SeeDot.
// Each function takes the scaling factors as arguments along with the pointers to the operands.

// C = A + B
void MatAdd(MYINT* A, MYINT* B, MYINT* C, MYITE I, MYITE J, MYITE left_shift, ACINT zeroA, ACINT shrA, MYITE nA, ACINT zeroB, ACINT shrB, MYITE nB, ACINT zeroC, ACINT shrC, MYITE nC, ACINT clamp_min, ACINT clamp_max) {
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
			C[i * J + j] = Saturate<ACINT, MYINT>(zeroC + c, clamp_min, clamp_max);
		}
	}
	return;
}

// C = a + B
void MatAddBroadCastA(MYINT* A, MYINT* B, MYINT* C, MYITE I, MYITE J, MYITE left_shift, ACINT zeroA, ACINT shrA, MYITE nA, ACINT zeroB, ACINT shrB, MYITE nB, ACINT zeroC, ACINT shrC, MYITE nC, ACINT clamp_min, ACINT clamp_max) {
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
			C[i * J + j] = Saturate<ACINT, MYINT>(zeroC + c, clamp_min, clamp_max);
		}
	}
	return;
}

// C = a - B
void MatSubBroadCastA(MYINT* A, MYINT* B, MYINT* C, MYITE I, MYITE J, MYITE left_shift, ACINT zeroA, ACINT shrA, MYITE nA, ACINT zeroB, ACINT shrB, MYITE nB, ACINT zeroC, ACINT shrC, MYITE nC, ACINT clamp_min, ACINT clamp_max) {
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
			C[i * J + j] = Saturate<ACINT, MYINT>(zeroC + c, clamp_min, clamp_max);
		}
	}
	return;
}

// C = A * B
void MatMul(MYINT* A, MYINT* B, MYINT* C, MYITE I, MYITE K, MYITE J, ACINT zeroA, ACINT zeroB, ACINT zeroC, ACINT M0, MYITE N, ACINT clamp_min, ACINT clamp_max) {
	for (MYITE i = 0; i < I; i++) {
		for (MYITE j = 0; j < J; j++) {
			ACINT sum = 0;

			for (MYITE k = 0; k < K; k++) {
				ACINT a = A[i * K + k];
				ACINT b = B[k * J + j];

				sum += (a + zeroA) * (b + zeroB);
			}

			sum = MulQuantMultiplierLTO<ACINT>(sum, M0, N);
			C[i * J + j] = Saturate<ACINT, MYINT>(zeroC + sum, clamp_min, clamp_max);
		}
	}
	return;
}

// C = A <*> B
void Hadamard(MYINT* A, MYINT* B, MYINT* C, MYITE I, MYITE J, ACINT zeroA, ACINT zeroB, ACINT zeroC, ACINT M0, MYITE N, ACINT clamp_min, ACINT clamp_max) {
	for (MYITE i = 0; i < I; i++) {
		for (MYITE j = 0; j < J; j++) {
			ACINT a = A[i * K + k];
			ACINT b = B[k * J + j];

			ACINT prod = (a + zeroA) * (b + zeroB);

			prod = MulQuantMultiplierLTO<ACINT>(prod, M0, N);
			C[i * J + j] = Saturate<ACINT, MYINT>(zeroC + prod, clamp_min, clamp_max);
		}
	}
	return;
}

// C = a * B
void MatMulBroadcastA(MYINT* A, MYINT* B, MYINT* C, MYITE I, MYITE J, ACINT zeroA, ACINT zeroB, ACINT zeroC, ACINT M0, MYITE N, ACINT clamp_min, ACINT clamp_max) {
	ACINT a = (ACINT) *A;
	a += zeroA;

	for (MYITE i = 0; i < I; i++) {
		for (MYITE j = 0; j < J; j++) {
			ACINT b = B[k * J + j];
			ACINT prod = a * (b + zeroB);

			prod = MulQuantMultiplierLTO<ACINT>(prod, M0, N);
			C[i * J + j] = Saturate<ACINT, MYINT>(zeroC + prod, clamp_min, clamp_max);
		}
	}
	return;
}

// A = tanh(A)
void TanH(MYINT* A, MYINT* B, MYITE I, MYITE J, ACINT zeroA, ACINT M0, MYITE N, ACINT clamp_radius) {
	for (MYITE i = 0; i < I; i++) {
		for (MYITE j = 0; j < J; j++) {
			ACINT x = A[i * J + j];
			x += zeroA;

			MYINT y;
			if (x < -clamp_radius) {
				y = std::numeric_limits<MYINT>::min();
			} else if (x > clamp_radius) {
				y = std::numeric_limits<MYINT>::max();
			} else {
				const ACINT x_rescaled = MulQuantMultiplier<ACINT>(x, M0, N);
				using FixedPoint4 = gemmlowp::FixedPoint<TypeAc, 4>;
				using FixedPoint0 = gemmlowp::FixedPoint<TypeAc, 0>;
				const FixedPoint4 x_f4 = FixedPoint4::FromRaw(x_rescaled);
				const FixedPoint0 y_f0 = gemmlowp::tanh(x_f4);

				ACINT y_s32 = RoundingDivideByPOT(y_f0.raw(), 24);
				if (y_s32 == 256) {
					y_s32 = std::numeric_limits<MYINT>::max();
				}
			}

			B[i * J + j] = Saturate<ACINT, MYINT>(y, std::numeric_limits<MYINT>::min(), std::numeric_limits<MYINT>::max());
		}
	}
	return;
}

// B = Sigmoid(A)
void Sigmoid(MYINT* A, MYINT* B, MYITE I, MYITE J, ACINT zeroA, ACINT M0, MYITE N, ACINT clamp_radius) {
	for (MYITE i = 0; i < I; i++) {
		for (MYITE j = 0; j < J; j++) {
			ACINT x = A[i * J + j];
			x += zeroA;

			MYINT y;
			if (x < -clamp_radius) {
				y = std::numeric_limits<MYINT>::min();
			} else if (x > clamp_radius) {
				y = std::numeric_limits<MYINT>::max();
			} else {
				const ACINT x_rescaled = MulQuantMultiplier<ACINT>(x, M0, N);
				using FixedPoint4 = gemmlowp::FixedPoint<TypeAc, 4>;
				using FixedPoint0 = gemmlowp::FixedPoint<TypeAc, 0>;
				const FixedPoint4 x_f4 = FixedPoint4::FromRaw(x_rescaled);
				const FixedPoint0 y_f0 = gemmlowp::logistic(x_f4);

				ACINT y_s32 = RoundingDivideByPOT(y_f0.raw(), 23);
				if (y_s32 == 256) {
					y_s32 = std::numeric_limits<MYINT>::max();
				}
			}

			B[i * J + j] = Saturate<ACINT, MYINT>(y, std::numeric_limits<MYINT>::min(), std::numeric_limits<MYINT>::max());
		}
	}
	return;
}
