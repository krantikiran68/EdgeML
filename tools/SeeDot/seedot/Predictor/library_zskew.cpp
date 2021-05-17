// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include <iostream>
#include <cmath>

#include "datatypes.h"
#include "library_zskew.h"

int64_t SaturatingRoundingDoublingHighMul(int64_t a, int64_t b) {
  bool overflow = a == b && a == std::numeric_limits<int64_t>::min();
  __int128 a_128(a);
  __int128 b_128(b);
  __int128 ab_128 = a_128 * b_128;
  int64_t nudge = ab_128 >= 0 ? (1 << 62) : (1 - (1 << 62));
  int64_t ab_x2_high64 = static_cast<int64_t>((ab_128 + nudge) / (1ll << 63));
  return overflow ? std::numeric_limits<int64_t>::max() : ab_x2_high64;
}

int32_t SaturatingRoundingDoublingHighMul(int32_t a, int32_t b) {
  bool overflow = a == b && a == std::numeric_limits<int32_t>::min();
  int64_t a_64(a);
  int64_t b_64(b);
  int64_t ab_64 = a_64 * b_64;
  int32_t nudge = ab_64 >= 0 ? (1 << 30) : (1 - (1 << 30));
  int32_t ab_x2_high32 = static_cast<int32_t>((ab_64 + nudge) / (1ll << 31));
  return overflow ? std::numeric_limits<int32_t>::max() : ab_x2_high32;
}

int16_t SaturatingRoundingDoublingHighMul(int16_t a, int16_t b) {
  bool overflow = a == b && a == std::numeric_limits<int16_t>::min();
  int32_t a_32(a);
  int32_t b_32(b);
  int32_t ab_32 = a_32 * b_32;
  int16_t nudge = ab_32 >= 0 ? (1 << 14) : (1 - (1 << 14));
  int16_t ab_x2_high16 = static_cast<int16_t>((ab_32 + nudge) / (1 << 15));
  return overflow ? std::numeric_limits<int16_t>::max() : ab_x2_high16;
}

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
			ACINT a = A[i * J + j];
			ACINT b = B[i * J + j];

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
			ACINT b = B[i * J + j];
			ACINT prod = a * (b + zeroB);

			prod = MulQuantMultiplierLTO<ACINT>(prod, M0, N);
			C[i * J + j] = Saturate<ACINT, MYINT>(zeroC + prod);
		}
	}
	return;
}

// A = tanh(A)
void TanHZSkew(MYINT* A, MYINT I, MYINT J, MYINT scale_in, MYINT scale_out, MYINT* B) {
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
void SigmoidZSkew(MYINT* A, MYINT I, MYINT J, MYINT div, MYINT add, MYINT sigmoid_limit, MYINT scale_in, MYINT scale_out, MYINT* B) {
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

