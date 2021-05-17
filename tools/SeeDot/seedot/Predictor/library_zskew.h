// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#pragma once

#include "datatypes.h"

/**
 * Notation used:
 * 		By default, 'matrix' is to be interpreted as a matrix in fixed point representation.
 * 		dim(X) = dimension of matrix X.
 * 		bw(X) = number of bits each value of X uses.
 * 		sc(X) = scale of matrix X.
 * 		scale of a fixed point matrix X is an integer S such that
 * 			Xq (floating point matrix) = (2 ^ -S) * X where
 * 				a ^ b (a and b are integers) is a raised to the power b, and
 * 				a * b (a is integer and b is a matrix) is a multiplied to each element of b.
 **/

/**
 * Dimensions: 	A, B, C are matrices, dim(A) = dim(B) = dim(C) = [I][J]; I, J, shrA, shrB, shrC are integers
 *
 * Matrix Addition
 * Compute A + B and store it in C.
 * shrA, shrB, shrC are scaling constants which are computed in irBuilder.py::getScaleForAddAndSub(sc(A), sc(B), sc(C)).
 * 		shrA, shrB are used to bring matrices A and B to the same scale for addition.
 * 		shrC adjusts the output matrix if required to prevent overflows.
 * The last two letters, which can be either C or N, denote the following:
 * 		If the last letter is N, it means the matrix B is an intermediate variable in RAM.
 * 		If the last letter is C, it means the matrix B is a read only parameter which must be extracted from flash.
 * 		Similarly, the second last letter controls the input of matrix A.
 * 		On Arduino-like devices with Harvard architecture, the reading of RAM and flash variables is different, hence the different functions.
 **/
void MatAdd(MYINT* A, MYINT* B, MYINT* C, MYITE I, MYITE J, MYITE left_shift, ACINT zeroA, ACINT shrA, MYITE nA, ACINT zeroB, ACINT shrB, MYITE nB, ACINT zeroC, ACINT shrC, MYITE nC, ACINT clamp_min, ACINT clamp_max);

/**
 * Dimensions: 	I, J, shrA, shrB, shrC are integers
 * 				C is a matrix, dim(C) = [I][J]
 * 				For MatAddBroadCastA, B is a matrix, dim(B) = [I][J], A represents a scalar
 * 				For MatAddBroadCastB, A is a matrix, dim(A) = [I][J], B represents a scalar
 *
 * Broadcasted Matrix Addition
 * 		For MatAddBroadCastA, add scalar A to all elements of B and store result in C.
 * 		For MatAddBroadCastB, add scalar B to all elements of A and store result in C.
 * shrA, shrB, shrC are scaling constants which are computed in irBuilder.py::getScaleForAddAndSub(sc(A), sc(B), sc(C)).
 * 		shrA, shrB are used to bring matrices A and B to the same scale for addition.
 * 		shrC adjusts the output matrix if required to prevent overflows.
 **/
void MatAddBroadCastA(MYINT* A, MYINT* B, MYINT* C, MYITE I, MYITE J, MYITE left_shift, ACINT zeroA, ACINT shrA, MYITE nA, ACINT zeroB, ACINT shrB, MYITE nB, ACINT zeroC, ACINT shrC, MYITE nC, ACINT clamp_min, ACINT clamp_max);

/**
 * Dimensions: 	I, J, shrA, shrB, shrC are integers
 * 				C is a matrix, dim(C) = [I][J]
 * 				For MatSubBroadCastA, B is a matrix, dim(B) = [I][J], A represents a scalar
 * 				For MatSubBroadCastB, A is a matrix, dim(A) = [I][J], B represents a scalar
 *
 * Broadcasted Matrix Subtraction
 * 		For MatSubBroadCastA, add scalar A to all elements of B and store result in C.
 * 		For MatSubBroadCastB, add scalar B to all elements of A and store result in C.
 * shrA, shrB, shrC are scaling constants which are computed in irBuilder.py::getScaleForAddAndSub(sc(A), sc(B), sc(C)).
 * 		shrA, shrB are used to bring matrices A and B to the same scale for addition.
 * 		shrC adjusts the output matrix if required to prevent overflows.
 **/
void MatSubBroadCastA(MYINT* A, MYINT* B, MYINT* C, MYITE I, MYITE J, MYITE left_shift, ACINT zeroA, ACINT shrA, MYITE nA, ACINT zeroB, ACINT shrB, MYITE nB, ACINT zeroC, ACINT shrC, MYITE nC, ACINT clamp_min, ACINT clamp_max);

/**
 * Dimensions: 	A, B, C are matrices, dim(A) = [I][J], dim(B) = [J][K], dim(C) = [I][K]; tmp is a vector, dim(tmp) = [J] I, K, J, shrA, shrB, H1, H2 are integers.
 *
 * Matrix Multiplication
 * Compute A * B and store it in C, using tmp as a buffer.
 * 		To compute C[i][k], we have to compute summation_j[0:J](A[i][j]*B[j][k]). We store the J values in the vector tmp,
 * 		and carry out Tree Sum (described below) on the vector to ensure minimum loss of bits
 * shrA, shrB, H1, H2 are scaling constants which are computed in irBuilder.py::getShrTreeSumAndDemoteParamsForMul(bw(A), sc(A), bw(B), sc(B), bw(tmp), sc(tmp), bw(C), sc(C), J).
 * 		shrA, shrB are used to alter the scales of matrices A and B so that the multiplication avoids overflows but maintains as many bits as possible.
 * 		H1, H2 are used for Tree Sum. Usage is described below
 * The last two letters, which can be either C or N, denote the following:
 * 		If the last letter is N, it means the matrix B is an intermediate variable in RAM.
 * 		If the last letter is C, it means the matrix B is a read only parameter which must be extracted from flash.
 * 		Similarly, the second last letter controls the input of matrix A.
 * 		On Arduino-like devices with Harvard architecture, the reading of RAM and flash variables is different, hence the different functions.
 *
 * Tree Sum
 * This is a technique used to sum up a long vector. To sum up a vector [a0, a1, a2, a3, a4, a5, a6...],
 * in the first stage we first store a0 + a1 at index 0, a2 + a3 at index 2, a4 + a5 at index 4 and so on.
 * Next stage we store index 0 + index 2 at index 0, index 4 + index 6 at index 4, and so on.
 * We continue this till all elements are summed up at index 0.
 * For fixed point arithmetic, in the first H1 (parameter) stages, we divide the addition result by 2 to avoid overflows,
 * and in the next H2 (parameter) stages (assuming no overflows), we do not do the division to conserve precision.
 **/
void MatMul(MYINT* A, MYINT* B, MYINT* C, MYITE I, MYITE K, MYITE J, ACINT zeroA, ACINT zeroB, ACINT zeroC, ACINT M0, MYITE N, ACINT clamp_min, ACINT clamp_max);

/**
 * Dimensions: 	A, B, C are matrices, dim(A) = dim(B) = dim(C) = [I][J]; I, J, shrA, shrB, shrC are integers.
 *
 * Hadamard Matrix Product
 * Compute A * B element-wise and store it in C.
 * shrA, shrB are scaling constants which are computed in irBuilder.py::getShrTreeSumAndDemoteParamsForMul(bw(A), sc(A), bw(B), sc(B), bw(C), sc(C), bw(C), sc(C), 1).
 * 		shrA, shrB are used to alter the scales of matrices A and B so that the multiplication avoids overflows but maintains as many bits as possible.
 **/
void Hadamard(MYINT* A, MYINT* B, MYINT* C, MYITE I, MYITE J, ACINT zeroA, ACINT zeroB, ACINT zeroC, ACINT M0, MYITE N, ACINT clamp_min, ACINT clamp_max);

/**
 * Dimensions: 	I, J, shrA, shrB are integers
 * 				B, C is are matrices, dim(B) = dim(C) = [I][J]
 * 				A represents a scalar
 *
 * Scalar Matrix Addition
 * 		Multiply scalar A to all elements of B and store result in C.
 * shrA, shrB are scaling constants which are computed in irBuilder.py::getShrTreeSumAndDemoteParamsForMul(bw(A), sc(A), bw(B), sc(B), bw(C), sc(C), bw(C), sc(C), 1).
 * 		shrA, shrB are used to alter the scales of matrices A and B so that the multiplication avoids overflows but maintains as many bits as possible.
 */
void MatMulBroadcastA(MYINT* A, MYINT* B, MYINT* C, MYITE I, MYITE J, ACINT zeroA, ACINT zeroB, ACINT zeroC, ACINT M0, MYITE N, ACINT clamp_min, ACINT clamp_max);

/**
 * Dimensions:	A, B are matrices, dim(A) = dim(B) = [I][J]. I, J, scale_in, scale_out are integers.
 *
 * TanH
 * Computes tanH(A) element-wise and stores the result in B.
 * scale_in is the scale of the input matrix A, and scale_out is the scale of the output matrix B.
 */
void TanH(MYINT* A, MYINT* B, MYITE I, MYITE J, ACINT zeroA, ACINT M0, MYITE N, ACINT clamp_radius);

/**
 * Dimensions:	A, B are matrices, dim(A) = dim(B) = [I][J]; div, add, sigmoid_limit, scale_in, scale_out are integers.
 *
 * Sigmoid activation
 * Computes the sigmoid activation for all elements of A and stores the result in B.
 * scale_in, scale_out are integers which satisfy the following:
 * 		Dividing (float division) each element of matrix A by scale_in gives the floating point matrix of A.
 * 		Dividing (float division) each element of matrix B by scale_out gives the floating point matrix of B.
 *
 * In some cases, a piecewise linear approximation is used for sigmoid: min(max((X+2.0)/4.0, 0.0), 1.0) in floating point version.
 * In this case,
 * 		div represents the fixed point version of 4.0 in the expression.
 * 		add represents the fixed point version of 2.0 in the expression.
 * 		sigmoid_limit represents the fixed point version of 1.0 in the expression.
 * If flag FLOATEXP is disabled, and if new table exponentiation (Util.py::class Config) is not used, this piecewise approximation is used. Else, the above 3 parameters are not used.
 */
void Sigmoid(MYINT* A, MYINT* B, MYITE I, MYITE J, ACINT zeroA, ACINT M0, MYITE N, ACINT clamp_radius);

//Templated Operations: For cases when Variable BitWidth is enabled.

template<class InputType, class OutputType>
inline OutputType Saturate(InputType inp, InputType min_value, InputType max_value) {
  inp = inp < min_value ? min_value : inp;
  return (OutputType)(inp > max_value ? max_value : inp);
}

template <typename IntegerType>
IntegerType SaturatingRoundingDoublingHighMul(IntegerType a, IntegerType b) {
  static_assert(std::is_same<IntegerType, void>::value, "Unimplemented");
  return a;
}

template <>
int64_t SaturatingRoundingDoublingHighMul(int64_t a, int64_t b) {
  bool overflow = a == b && a == std::numeric_limits<int64_t>::min();
  int128_t a_128(a);
  int128_t b_128(b);
  int128_t ab_128 = a_128 * b_128;
  int64_t nudge = ab_128 >= 0 ? (1 << 62) : (1 - (1 << 62));
  int64_t ab_x2_high64 = static_cast<int64_t>((ab_128 + nudge) / (1ll << 63));
  return overflow ? std::numeric_limits<int64_t>::max() : ab_x2_high64;
}

template <>
int32_t SaturatingRoundingDoublingHighMul(int32_t a, int32_t b) {
  bool overflow = a == b && a == std::numeric_limits<int32_t>::min();
  int64_t a_64(a);
  int64_t b_64(b);
  int64_t ab_64 = a_64 * b_64;
  int32_t nudge = ab_64 >= 0 ? (1 << 30) : (1 - (1 << 30));
  int32_t ab_x2_high32 = static_cast<int32_t>((ab_64 + nudge) / (1ll << 31));
  return overflow ? std::numeric_limits<int32_t>::max() : ab_x2_high32;
}

template <>
int16_t SaturatingRoundingDoublingHighMul(int16_t a, int16_t b) {
  bool overflow = a == b && a == std::numeric_limits<int16_t>::min();
  int32_t a_32(a);
  int32_t b_32(b);
  int32_t ab_32 = a_32 * b_32;
  int16_t nudge = ab_32 >= 0 ? (1 << 14) : (1 - (1 << 14));
  int16_t ab_x2_high16 = static_cast<int16_t>((ab_32 + nudge) / (1 << 15));
  return overflow ? std::numeric_limits<int16_t>::max() : ab_x2_high16;
}

// Correctly-rounded-to-nearest division by a power-of-two.
// Also known as a rounding arithmetic right shift.
template <typename IntegerType, typename ExponentType>
IntegerType RoundingDivideByPOT(IntegerType x, ExponentType exponent) {
  assert(exponent >= 0);
  assert(exponent <= 31);
  const IntegerType mask = (1ll << exponent) - 1;
  const IntegerType remainder = x & mask;
  IntegerType threshold = (mask >> 1);
  if (x < 0) {
	threshold += 1;
  }

  if (remainder > threshold) {
	return (x >> exponent) + 1;
  }

  return (x >> exponent);
}

template<class InputType>
inline InputType MulQuantMultiplierLTO(InputType x, InputType multiplier, MYITE right_shift) {
	return RoundingDivideByPOT(SaturatingRoundingDoublingHighMul(x, multiplier), right_shift);
}

template<class InputType>
inline InputType MulQuantMultiplierGTO(InputType x, InputType multiplier, MYITE left_shift) {
	return SaturatingRoundingDoublingHighMul(x * (1 << left_shift), multiplier);
}

template<class TypeA, class TypeB, class TypeAc, class TypeC>
void MatAdd(TypeA* A, TypeB* B, TypeC* C, MYITE I, MYITE J, MYITE left_shift, TypeAc zeroA, TypeAc shrA, MYITE nA, TypeAc zeroB, TypeAc shrB, MYITE nB, TypeAc zeroC, TypeAc shrC, MYITE nC, TypeAc clamp_min, TypeAc clamp_max) {	
	for (MYITE i = 0; i < I; i++) {
		for (MYITE j = 0; j < J; j++) {
			TypeAc a = A[i * J + j];
			TypeAc b = B[i * J + j];

			a += zeroA;
			b += zeroB;
			a *= (1 << left_shift);
			b *= (1 << left_shift);

			a = MulQuantMultiplierLTO<TypeAc>(a, shrA, nA);
			b = MulQuantMultiplierLTO<TypeAc>(b, shrB, nB);

			TypeAc c = MulQuantMultiplierLTO<TypeAc>(a + b, shrC, nC);
			C[i * J + j] = Saturate<TypeAc, TypeC>(zeroC + c, clamp_min, clamp_max);
		}
	}
	return;
}

template<class TypeA, class TypeB, class TypeAc, class TypeC>
void MatAddBroadCastA(TypeA* A, TypeB* B, TypeC* C, MYITE I, MYITE J, MYITE left_shift, TypeAc zeroA, TypeAc shrA, MYITE nA, TypeAc zeroB, TypeAc shrB, MYITE nB, TypeAc zeroC, TypeAc shrC, MYITE nC, TypeAc clamp_min, TypeAc clamp_max) {
	TypeAc a = (TypeAc) *A;
	a += zeroA;
	a *= (1 << left_shift);
	a = MulQuantMultiplierLTO<TypeAc>(a, shrA, nA);

	for (MYITE i = 0; i < I; i++) {
		for (MYITE j = 0; j < J; j++) {
			TypeAc b = B[i * J + j];
			b += zeroB;
			b *= (1 << left_shift);
			b = MulQuantMultiplierLTO<TypeAc>(b, shrB, nB);

			TypeAc c = MulQuantMultiplierLTO<TypeAc>(a + b, shrC, nC);
			C[i * J + j] = Saturate<TypeAc, TypeC>(zeroC + c, clamp_min, clamp_max);
		}
	}
	return;
}

template<class TypeA, class TypeB, class TypeAc, class TypeC>
void MatSubBroadCastA(TypeA* A, TypeB* B, TypeC* C, MYITE I, MYITE J, MYITE left_shift, TypeAc zeroA, TypeAc shrA, MYITE nA, TypeAc zeroB, TypeAc shrB, MYITE nB, TypeAc zeroC, TypeAc shrC, MYITE nC, TypeAc clamp_min, TypeAc clamp_max) {
	TypeAc a = (TypeAc) *A;
	a += zeroA;
	a *= (1 << left_shift);
	a = MulQuantMultiplierLTO<TypeAc>(a, shrA, nA);

	for (MYITE i = 0; i < I; i++) {
		for (MYITE j = 0; j < J; j++) {
			TypeAc b = B[i * J + j];
			b += zeroB;
			b *= (1 << left_shift);
			b = MulQuantMultiplierLTO<TypeAc>(b, shrB, nB);

			TypeAc c = MulQuantMultiplierLTO<TypeAc>(a - b, shrC, nC);
			C[i * J + j] = Saturate<TypeAc, TypeC>(zeroC + c, clamp_min, clamp_max);
		}
	}
	return;
}

template<class TypeA, class TypeB, class TypeAc, class TypeC>
void MatMul(TypeA* A, TypeB* B, TypeC* C, MYITE I, MYITE K, MYITE J, TypeAc zeroA, TypeAc zeroB, TypeAc zeroC, TypeAc M0, MYITE N, TypeAc clamp_min, TypeAc clamp_max) {
	for (MYITE i = 0; i < I; i++) {
		for (MYITE j = 0; j < J; j++) {
			TypeAc sum = 0;

			for (MYITE k = 0; k < K; k++) {
				TypeAc a = A[i * K + k];
				TypeAc b = B[k * J + j];

				sum += (a + zeroA) * (b + zeroB);
			}

			sum = MulQuantMultiplierLTO<TypeAc>(sum, M0, N);
			C[i * J + j] = Saturate<TypeAc, TypeC>(zeroC + sum, clamp_min, clamp_max);
		}
	}
	return;
}

template<class TypeA, class TypeB, class TypeAc, class TypeC>
void Hadamard(TypeA* A, TypeB* B, TypeC* C, MYITE I, MYITE J, TypeAc zeroA, TypeAc zeroB, TypeAc zeroC, TypeAc M0, MYITE N, , TypeAc clamp_min, TypeAc clamp_max) {
	for (MYITE i = 0; i < I; i++) {
		for (MYITE j = 0; j < J; j++) {
			TypeAc a = A[i * K + k];
			TypeAc b = B[k * J + j];

			TypeAc prod = (a + zeroA) * (b + zeroB);

			prod = MulQuantMultiplierLTO<TypeAc>(prod, M0, N);
			C[i * J + j] = Saturate<TypeAc, TypeC>(zeroC + prod, clamp_min, clamp_max);
		}
	}
	return;
}

template<class TypeA, class TypeB, class TypeAc, class TypeC>
void MatMulBroadcastA(TypeA* A, TypeB* B, TypeC* C, MYITE I, MYITE J, TypeAc zeroA, TypeAc zeroB, TypeAc zeroC, TypeAc M0, MYITE N, TypeAc clamp_min, TypeAc clamp_max) {
	TypeAc a = (TypeAc) *A;
	a += zeroA;

	for (MYITE i = 0; i < I; i++) {
		for (MYITE j = 0; j < J; j++) {
			TypeAc b = B[k * J + j];
			TypeAc prod = a * (b + zeroB);

			prod = MulQuantMultiplierLTO<TypeAc>(prod, M0, N);
			C[i * J + j] = Saturate<TypeAc, TypeC>(zeroC + prod, clamp_min, clamp_max);
		}
	}
	return;
}

template<class TypeA>
void Sigmoid(TypeA* A, TypeA* B, MYITE I, MYITE J, TypeAc zeroA, TypeAc M0, MYITE N, TypeAc clamp_radius) {
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
				const ACINT x_rescaled = MultiplyByQuantizedMultiplierGreaterThanOne(x, M0, N);
				using FixedPoint4 = gemmlowp::FixedPoint<ACINT, 4>;
				using FixedPoint0 = gemmlowp::FixedPoint<ACINT, 0>;
				const FixedPoint4 x_f4 = FixedPoint4::FromRaw(x_rescaled);
				const FixedPoint0 y_f0 = gemmlowp::logistic(x_f4);

				ACINT y_s32 = RoundingDivideByPOT(y_f0.raw(), 23);
				if (y_s32 == 256) {
					y_s32 = std::numeric_limits<MYINT>::max();
				}
			}

			B[i * J + j] = Saturate<ACINT, MYINT>(y, 0, 255);
		}
	}
	return;
}

template<class TypeA>
void TanH(TypeA* A, TypeA* B, MYITE I, MYITE J, ) {
	for (MYITE i = 0; i < I; i++) {
		for (MYITE j = 0; j < J; j++) {
		}
	}
	return;
}
