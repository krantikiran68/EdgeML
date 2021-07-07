// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#pragma once

#include <limits>

#include "datatypes.h"
#include <cassert>
#include <limits>
#include "gemmlowp/fixedpoint/fixedpoint.h"

/**
 * Note: This implementation only works for 8-bit quantization. TFLite follows a different
 * scheme for 16-bit quantization.
 */

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
void MatAddBroadCastB(MYINT* A, MYINT* B, MYINT* C, MYITE I, MYITE J, MYITE left_shift, ACINT zeroA, ACINT shrA, MYITE nA, ACINT zeroB, ACINT shrB, MYITE nB, ACINT zeroC, ACINT shrC, MYITE nC, ACINT clamp_min, ACINT clamp_max);


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
 * 		
 * shrC adjusts the output matrix if required to prevent overflows.
 **/
void MatSubBroadCastA(MYINT* A, MYINT* B, MYINT* C, MYITE I, MYITE J, MYITE left_shift, ACINT zeroA, ACINT shrA, MYITE nA, ACINT zeroB, ACINT shrB, MYITE nB, ACINT zeroC, ACINT shrC, MYITE nC, ACINT clamp_min, ACINT clamp_max);
void MatSubBroadCastB(MYINT* A, MYINT* B, MYINT* C, MYITE I, MYITE J, MYITE left_shift, ACINT zeroA, ACINT shrA, MYITE nA, ACINT zeroB, ACINT shrB, MYITE nB, ACINT zeroC, ACINT shrC, MYITE nC, ACINT clamp_min, ACINT clamp_max);

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

void Sigmoid(MYINT* A, MYINT* B, MYITE I, MYITE J, ACINT zeroA, ACINT shrA, MYITE nA, ACINT zeroB, ACINT shrB, MYITE nB, ACINT clamp_radius);
void Sigmoid(MYINT* A, MYINT* B, MYITE I, MYITE J, float scale_in, ACINT zeroA, ACINT shrA, MYITE nA, float scale_out, ACINT zeroB, ACINT shrB, MYITE nB, ACINT clamp_radius);
/**
 * Dimensions:	A, B are matrices, dim(A) = dim(B) = [I][J]. I, J, scale_in, scale_out are integers.
 *
 * TanH
 * Computes tanH(A) element-wise and stores the result in B.
 * scale_in is the scale of the input matrix A, and scale_out is the scale of the output matrix B.
 */
void TanH(MYINT* A, MYINT* B, MYITE I, MYITE J, ACINT zeroA, ACINT shrA, MYITE nA, ACINT zeroB, ACINT shrB, MYITE nB, ACINT clamp_radius);
void TanH(MYINT* A, MYINT* B, MYITE I, MYITE J, float scale_in, ACINT zeroA, ACINT shrA, MYITE nA, float scale_out, ACINT zeroB, ACINT shrB, MYITE nB, ACINT clamp_radius);
void ArgMax(MYINT* A, MYINT I, MYINT J, float scale_in, MYINT zero_in, int* index);


//Templated Operations: For cases when Variable BitWidth is enabled.

template<class InputType, class OutputType>
inline OutputType Saturate(InputType inp, InputType min_value, InputType max_value) {
	inp = inp < min_value ? min_value : inp;
	return (OutputType)(inp > max_value ? max_value : inp);
}

void debugPrint(MYINT* A, int I, int J, float scale, int zero, std::string varName);
void debugPrint(MYINT* A, int N, int H, int W, int C, float scale, int zero, std::string varName);

// template <typename IntegerType>
// IntegerType SaturatingRoundingDoublingHighMul(IntegerType a, IntegerType b) {
// 	static_assert(std::is_same<IntegerType, void>::value, "Unimplemented");
// 	return a;
// }

// int64_t SaturatingRoundingDoublingHighMul(int64_t a, int64_t b);

// int32_t SaturatingRoundingDoublingHighMul(int32_t a, int32_t b);

// int16_t SaturatingRoundingDoublingHighMul(int16_t a, int16_t b);

// Correctly-rounded-to-nearest division by a power-of-two.
// Also known as a rounding arithmetic right shift.
// template <typename IntegerType, typename ExponentType>
// IntegerType RoundingDivideByPOT(IntegerType x, ExponentType exponent) {
// 	assert(exponent >= 0);
// 	assert(exponent <= 31);
// 	const IntegerType mask = (1ll << exponent) - 1;
// 	const IntegerType remainder = x & mask;
// 	IntegerType threshold = (mask >> 1);
// 	if (x < 0) {
// 		threshold += 1;
// 	}

// 	if (remainder > threshold) {
// 		return (x >> exponent) + 1;
// 	}

// 	return (x >> exponent);
// }

template<class InputType>
inline InputType MulQuantMultiplier(InputType x, InputType quantized_multiplier, MYITE shift) {
  int left_shift = shift > 0 ? shift : 0;
  int right_shift = shift > 0 ? 0 : -shift;
  return gemmlowp::RoundingDivideByPOT(gemmlowp::SaturatingRoundingDoublingHighMul(x * InputType(1LL << left_shift), quantized_multiplier), right_shift);
}

template<class InputType>
inline InputType MulQuantMultiplierLTO(InputType x, InputType multiplier, MYITE left_shift) {
	return gemmlowp::RoundingDivideByPOT(gemmlowp::SaturatingRoundingDoublingHighMul(x, multiplier), -left_shift);
}

template<class InputType>
inline InputType MulQuantMultiplierGTO(InputType x, InputType multiplier, MYITE left_shift) {
	return gemmlowp::SaturatingRoundingDoublingHighMul(x * InputType(1LL << left_shift), multiplier);
}

template <typename T>
int CountLeadingZeros(T integer_input) {
	return integer_input ? __builtin_clz(integer_input) : std::numeric_limits<T>::digits;
}

template<typename InputType>
inline void InvSqrtQuantizedMultiplier(InputType input, InputType* output_inv_sqrt, MYITE* shift) {
	if (input <= 1) {
		*output_inv_sqrt = std::numeric_limits<InputType>::max();
		*shift = 0;
		return;
	}

	*shift = 11;
	while (input >= (1 << 29)) {
		input /= 4;
		++*shift;
	}
	unsigned max_left_shift_bits;
	if (std::is_same<InputType, std::int32_t>::value) {
		max_left_shift_bits = CountLeadingZeros(static_cast<uint32_t>(input)) - 1;
	} else if (std::is_same<InputType, std::int64_t>::value) {
		max_left_shift_bits = CountLeadingZeros(static_cast<uint64_t>(input)) - 1;
	}
	const unsigned max_left_shift_bit_pairs = max_left_shift_bits / 2;
	const unsigned left_shift_bit_pairs = max_left_shift_bit_pairs - 1;
	*shift -= left_shift_bit_pairs;
	input <<= 2 * left_shift_bit_pairs;
	using gemmlowp::FixedPoint;
	using gemmlowp::Rescale;
	using gemmlowp::SaturatingRoundingMultiplyByPOT;
	// Using 3 integer bits gives us enough room for the internal arithmetic in
	// this Newton-Raphson iteration.
	using F3 = FixedPoint<int32_t, 3>;
	using F0 = FixedPoint<int32_t, 0>;
	const F3 fixedpoint_input = F3::FromRaw(input >> 1);
	const F3 fixedpoint_half_input = SaturatingRoundingMultiplyByPOT<-1>(fixedpoint_input);
	const F3 fixedpoint_half_three = GEMMLOWP_CHECKED_FIXEDPOINT_CONSTANT(F3, (1 << 28) + (1 << 27), 1.5);
	// Newton-Raphson iteration
	// Naive unoptimized starting guess: x = 1
	F3 x = F3::One();
	// Naive unoptimized number of iterations: 5
	for (int i = 0; i < 5; i++) {
		const F3 x3 = Rescale<3>(x * x * x);
		x = Rescale<3>(fixedpoint_half_three * x - fixedpoint_half_input * x3);
	}
	const F0 fixedpoint_half_sqrt_2 = GEMMLOWP_CHECKED_FIXEDPOINT_CONSTANT(F0, 1518500250, std::sqrt(2.) / 2.);
	x = x * fixedpoint_half_sqrt_2;
	*output_inv_sqrt = x.raw();
	if (*shift < 0) {
		*output_inv_sqrt <<= -*shift;
		*shift = 0;
	}
	// Convert right shift (right is positive) to left shift.
	*shift *= -1;
}

template<class TypeA, class TypeB, class TypeAc, class TypeC>
void MatAdd(TypeA* A, TypeB* B, TypeC* C, MYITE I, MYITE J, float scaleA, float scaleB, float scaleC, MYITE left_shift, TypeAc zeroA, TypeAc shrA, MYITE nA, TypeAc zeroB, TypeAc shrB, MYITE nB, TypeAc zeroC, TypeAc shrC, MYITE nC, TypeAc clamp_min, TypeAc clamp_max) {
	for (MYITE i = 0; i < I; i++) {
		for (MYITE j = 0; j < J; j++) {
			TypeAc a = A[i * J + j];
			TypeAc b = B[i * J + j];

			a += zeroA;
			b += zeroB;
			a *= TypeAc(1LL << left_shift);
			b *= TypeAc(1LL << left_shift);

			a = MulQuantMultiplier<TypeAc>(a, shrA, nA);
			b = MulQuantMultiplier<TypeAc>(b, shrB, nB);

			TypeAc c = MulQuantMultiplier<TypeAc>(a + b, shrC, nC);
			C[i * J + j] = Saturate<TypeAc, TypeC>(zeroC + c, clamp_min, clamp_max);
		}
	}
	return;
}

template<class TypeA, class TypeB, class TypeAc, class TypeC>
void MatAddBroadCastA(TypeA* A, TypeB* B, TypeC* C, MYITE I, MYITE J, float scaleA, float scaleB, float scaleC, MYITE left_shift, TypeAc zeroA, TypeAc shrA, MYITE nA, TypeAc zeroB, TypeAc shrB, MYITE nB, TypeAc zeroC, TypeAc shrC, MYITE nC, TypeAc clamp_min, TypeAc clamp_max) {
	TypeAc a = (TypeAc) *A;
	a += zeroA;
	a *= TypeAc(1LL << left_shift);
	a = MulQuantMultiplier<TypeAc>(a, shrA, nA);

	for (MYITE i = 0; i < I; i++) {
		for (MYITE j = 0; j < J; j++) {
			TypeAc b = B[i * J + j];
			b += zeroB;
			b *= TypeAc(1LL << left_shift);
			b = MulQuantMultiplier<TypeAc>(b, shrB, nB);

			TypeAc c = MulQuantMultiplier<TypeAc>(a + b, shrC, nC);
			C[i * J + j] = Saturate<TypeAc, TypeC>(zeroC + c, clamp_min, clamp_max);
		}
	}
	return;
}

template<class TypeA, class TypeB, class TypeAc, class TypeC>
void MatAddBroadCastB(TypeA* A, TypeB* B, TypeC* C, MYITE I, MYITE J, float scaleA, float scaleB, float scaleC, MYITE left_shift, TypeAc zeroA, TypeAc shrA, MYITE nA, TypeAc zeroB, TypeAc shrB, MYITE nB, TypeAc zeroC, TypeAc shrC, MYITE nC, TypeAc clamp_min, TypeAc clamp_max) {
	TypeAc b = (TypeAc) *B;
	b += zeroB;
	b *= TypeAc(1LL << left_shift);
	b = MulQuantMultiplier<TypeAc>(b, shrB, nB);

	for (MYITE i = 0; i < I; i++) {
		for (MYITE j = 0; j < J; j++) {
			TypeAc a = A[i * J + j];
			a += zeroA;
			a *= TypeAc(1LL << left_shift);
			a = MulQuantMultiplier<TypeAc>(a, shrA, nA);

			TypeAc c = MulQuantMultiplier<TypeAc>(a + b, shrC, nC);
			C[i * J + j] = Saturate<TypeAc, TypeC>(zeroC + c, clamp_min, clamp_max);
		}
	}
	return;
}

template<class TypeA, class TypeB, class TypeAc, class TypeX>
void MatAdd4(TypeA* A, TypeB* B, TypeX* X, MYITE N, MYITE H, MYITE W, MYITE C, MYITE left_shift, TypeAc zeroA, TypeAc shrA, MYITE nA, TypeAc zeroB, TypeAc shrB, MYITE nB, TypeAc zeroX, TypeAc shrX, MYITE nX, TypeAc clamp_min, TypeAc clamp_max) {
	for (MYITE n = 0; n < N; n++) {
		for (MYITE h = 0; h < H; h++) {
			for (MYITE w = 0; w < W; w++) {
				for (MYITE c = 0; c < C; c++) {
					TypeAc a = A[n * H * W * C + h * W * C + w * C + c];
					TypeAc b = B[n * H * W * C + h * W * C + w * C + c];

					a += zeroA;
					b += zeroB;
					a *= TypeAc(1LL << left_shift);
					b *= TypeAc(1LL << left_shift);

					a = MulQuantMultiplier<TypeAc>(a, shrA, nA);
					b = MulQuantMultiplier<TypeAc>(b, shrB, nB);

					TypeAc x = MulQuantMultiplier<TypeAc>(a + b, shrX, nX);
					X[n * H * W * C + h * W * C + w * C + c] = Saturate<TypeAc, TypeX>(zeroX + x, clamp_min, clamp_max);
				}
			}
		}
	}
	return;
}

template<class TypeA, class TypeB, class TypeAc, class TypeC>
void MatSub(TypeA* A, TypeB* B, TypeC* C, MYITE I, MYITE J, float scaleA, float scaleB, float scaleC, MYITE left_shift, TypeAc zeroA, TypeAc shrA, MYITE nA, TypeAc zeroB, TypeAc shrB, MYITE nB, TypeAc zeroC, TypeAc shrC, MYITE nC, TypeAc clamp_min, TypeAc clamp_max) {
	// #define MATSUB_APPROX
	for (MYITE i = 0; i < I; i++) {
		for (MYITE j = 0; j < J; j++) {
			#ifdef MATSUB_APPROX
				float a = A[i * J + j];
				float b = B[i * J + j];

				a += zeroA;
				b += zeroB;
				a *= scaleA;
				b *= scaleB;

				float c = (a - b)/scaleC;
				C[i * J + j] = Saturate<TypeAc, TypeC>(TypeAc(zeroC + c), clamp_min, clamp_max);
			#else
				TypeAc a = A[i * J + j];
				TypeAc b = B[i * J + j];

				a += zeroA;
				b += zeroB;
				a *= TypeAc(1LL << left_shift);
				b *= TypeAc(1LL << left_shift);

				a = MulQuantMultiplier<TypeAc>(a, shrA, nA);
				b = MulQuantMultiplier<TypeAc>(b, shrB, nB);

				TypeAc c = MulQuantMultiplier<TypeAc>(a - b, shrC, nC);
				C[i * J + j] = Saturate<TypeAc, TypeC>(zeroC + c, clamp_min, clamp_max);
			#endif
		}
	}
	return;
}

template<class TypeA, class TypeB, class TypeAc, class TypeC>
void MatSubBroadCastA(TypeA* A, TypeB* B, TypeC* C, MYITE I, MYITE J, float scaleA, float scaleB, float scaleC, MYITE left_shift, TypeAc zeroA, TypeAc shrA, MYITE nA, TypeAc zeroB, TypeAc shrB, MYITE nB, TypeAc zeroC, TypeAc shrC, MYITE nC, TypeAc clamp_min, TypeAc clamp_max) {
	TypeAc a = (TypeAc) *A;
	a += zeroA;
	a *= TypeAc(1LL << left_shift);
	a = MulQuantMultiplier<TypeAc>(a, shrA, nA);

	for (MYITE i = 0; i < I; i++) {
		for (MYITE j = 0; j < J; j++) {
			TypeAc b = B[i * J + j];
			b += zeroB;
			b *= TypeAc(1LL << left_shift);
			b = MulQuantMultiplier<TypeAc>(b, shrB, nB);

			TypeAc c = MulQuantMultiplier<TypeAc>(a - b, shrC, nC);
			C[i * J + j] = Saturate<TypeAc, TypeC>(zeroC + c, clamp_min, clamp_max);
		}
	}
	return;
}

template<class TypeA, class TypeB, class TypeAc, class TypeC>
void MatSubBroadCastB(TypeA* A, TypeB* B, TypeC* C, MYITE I, MYITE J, float scaleA, float scaleB, float scaleC, MYITE left_shift, TypeAc zeroA, TypeAc shrA, MYITE nA, TypeAc zeroB, TypeAc shrB, MYITE nB, TypeAc zeroC, TypeAc shrC, MYITE nC, TypeAc clamp_min, TypeAc clamp_max) {
	TypeAc b = (TypeAc) *B;
	b += zeroB;
	b *= TypeAc(1LL << left_shift);
	b = MulQuantMultiplier<TypeAc>(b, shrB, nB);

	for (MYITE i = 0; i < I; i++) {
		for (MYITE j = 0; j < J; j++) {
			TypeAc a = A[i * J + j];
			a += zeroA;
			a *= TypeAc(1LL << left_shift);
			a = MulQuantMultiplier<TypeAc>(a, shrA, nA);

			TypeAc c = MulQuantMultiplier<TypeAc>(a - b, shrC, nC);
			C[i * J + j] = Saturate<TypeAc, TypeC>(zeroC + c, clamp_min, clamp_max);
		}
	}
	return;
}

template<class TypeA, class TypeB, class TypeAc, class TypeC>
void MatMul(TypeA* A, TypeB* B, TypeC* C, MYITE I, MYITE K, MYITE J, float scaleA, float scaleB, float scaleC, TypeAc zeroA, TypeAc zeroB, TypeAc zeroC, TypeAc M0, MYITE N, TypeAc clamp_min, TypeAc clamp_max) {
	// #define MATMUL_APPROX
	for (MYITE i = 0; i < I; i++) {
		for (MYITE j = 0; j < J; j++) {
			#ifdef MATMUL_APPROX
				float sum = 0;

				for (MYITE k = 0; k < K; k++) {
					float a = A[i * K + k];
					a += zeroA;
					a *= scaleA;

					float b = B[k * J + j];
					b += zeroB;
					b *= scaleB;

					sum += (a) * (b);
				}
				sum /= scaleC;
				C[i * J + j] = Saturate<TypeAc, TypeC>(TypeAc(zeroC + sum), clamp_min, clamp_max);
			#else
				TypeAc sum = 0;

				for (MYITE k = 0; k < K; k++) {
					TypeAc a = A[i * K + k];
					TypeAc b = B[k * J + j];

					sum += (a + zeroA) * (b + zeroB);
				}

				sum = MulQuantMultiplier<TypeAc>(sum, M0, N);
				C[i * J + j] = Saturate<TypeAc, TypeC>(zeroC + sum, clamp_min, clamp_max);
			#endif
		}
	}
	return;
}

template<class TypeA, class TypeB, class TypeAc, class TypeC>
void Hadamard(TypeA* A, TypeB* B, TypeC* C, MYITE I, MYITE J, TypeAc zeroA, TypeAc zeroB, TypeAc zeroC, TypeAc M0, MYITE N, TypeAc clamp_min, TypeAc clamp_max) {
	for (MYITE i = 0; i < I; i++) {
		for (MYITE j = 0; j < J; j++) {
			TypeAc a = A[i * J + j];
			TypeAc b = B[i * J + j];

			TypeAc prod = (a + zeroA) * (b + zeroB);

			prod = MulQuantMultiplier<TypeAc>(prod, M0, N);
			C[i * J + j] = Saturate<TypeAc, TypeC>(zeroC + prod, clamp_min, clamp_max);
		}
	}
	return;
}

template<class TypeA, class TypeB, class TypeAc, class TypeC>
void MatMulBroadcastA(TypeA* A, TypeB* B, TypeC* C, MYITE I, MYITE J, float scaleA, float scaleB, float scaleC, TypeAc zeroA, TypeAc zeroB, TypeAc zeroC, TypeAc M0, MYITE N, TypeAc clamp_min, TypeAc clamp_max) {
	// #define MATMULBROADCAST_A_APPROX
	
	#ifdef MATMULBROADCAST_A_APPROX
			float a = (float) *A;
			a += zeroA;
			a *= scaleA;

			for (MYITE i = 0; i < I; i++) {
				for (MYITE j = 0; j < J; j++) {
					float b = B[i * J + j];
					b += zeroB;
					b *= scaleB;
					float prod = (a * b)/scaleC;

					C[i * J + j] = Saturate<TypeAc, TypeC>(TypeAc(zeroC + prod), clamp_min, clamp_max);
				}
			}
	#else
		TypeAc a = (TypeAc) *A;
		a += zeroA;

		for (MYITE i = 0; i < I; i++) {
			for (MYITE j = 0; j < J; j++) {
				TypeAc b = B[i * J + j];
				TypeAc prod = a * (b + zeroB);

				prod = MulQuantMultiplier<TypeAc>(prod, M0, N);
				C[i * J + j] = Saturate<TypeAc, TypeC>(zeroC + prod, clamp_min, clamp_max);
			}
		}
	#endif
	return;
}

template<class TypeA, class TypeAc>
void Sigmoid(TypeA* A, TypeA* B, MYITE I, MYITE J, float scale_in, TypeAc zeroA, TypeAc shrA, MYITE nA, float scale_out, TypeAc zeroB, TypeAc shrB, MYITE nB, TypeAc clamp_radius) {
	for (MYITE i = 0; i < I; i++) {
		for (MYITE j = 0; j < J; j++) {
			TypeAc x = A[i * J + j];
			x += zeroA;

			TypeAc y;
			if (x < -clamp_radius) {
				B[i * J + j] = std::numeric_limits<TypeA>::min();
				continue;
			} else if (x > clamp_radius) {
				B[i * J + j] = std::numeric_limits<TypeA>::max();
				continue;
			} else {
				const int32_t x_rescaled = MulQuantMultiplier<TypeAc>(x, shrA, nA);
				using FixedPoint4 = gemmlowp::FixedPoint<int32_t, 13>;
				using FixedPoint0 = gemmlowp::FixedPoint<int32_t, 0>;
				const FixedPoint4 x_f4 = FixedPoint4::FromRaw(x_rescaled);
				const FixedPoint0 y_f0 = gemmlowp::logistic(x_f4);

				y = gemmlowp::RoundingDivideByPOT(y_f0.raw(), 23);
			}
			y = MulQuantMultiplier<TypeAc>(y, shrB, nB);

			B[i * J + j] = Saturate<TypeAc, TypeA>(y + zeroB, std::numeric_limits<TypeA>::min(), std::numeric_limits<TypeA>::max());
		}
	}
	return;
}

template<class TypeA, class TypeAc>
void TanH(TypeA* A, TypeA* B, MYITE I, MYITE J, float scale_in, TypeAc zeroA, TypeAc shrA, MYITE nA, float scale_out, TypeAc zeroB, TypeAc shrB, MYITE nB, TypeAc clamp_radius) {
	for (MYITE i = 0; i < I; i++) {
		for (MYITE j = 0; j < J; j++) {
			TypeAc x = A[i * J + j];
			x += zeroA;

			TypeAc y;
			if (x < -clamp_radius) {
				B[i * J + j] = std::numeric_limits<TypeA>::min();
				continue;
			} else if (x > clamp_radius) {
				B[i * J + j] = std::numeric_limits<TypeA>::max();
				continue;
			} else {
				const int32_t x_rescaled = MulQuantMultiplier<TypeAc>(x, shrA, nA);
				using FixedPoint4 = gemmlowp::FixedPoint<int32_t, 13>;
				using FixedPoint0 = gemmlowp::FixedPoint<int32_t, 0>;
				const FixedPoint4 x_f4 = FixedPoint4::FromRaw(x_rescaled);
				const FixedPoint0 y_f0 = gemmlowp::tanh(x_f4);

				y = gemmlowp::RoundingDivideByPOT(y_f0.raw(), 24);
			}
			y = MulQuantMultiplier<TypeAc>(y, shrB, nB);

			B[i * J + j] = Saturate<TypeAc, TypeA>(y + zeroB, std::numeric_limits<TypeA>::min(), std::numeric_limits<TypeA>::max());
		}
	}
	return;
}

template<class TypeA, class TypeAc>
void ArgMax(TypeA* A, MYITE I, MYITE J, float scale_in, TypeAc zero_in, int* index) {
	TypeAc a = A[0];
	a += zero_in;
	float max = a * scale_in;
	MYITE maxIndex = 0, counter = 0;
	for (MYITE i = 0; i < I; i++) {
		for (MYITE j = 0; j < J; j++) {
			float x = A[i * J + j];
			x += zero_in;
			x *= scale_in;

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

template<class TypeA, class TypeAidx, class TypeB, class TypeAc, class TypeC>
void SparseMatMulX(const TypeAidx* Aidx, const TypeA* Aval, TypeB** B, TypeC* C, TypeAc* tmp, MYITE P, MYITE K, float scaleA, float scaleB, float scale_out, MYITE left_shift, TypeAc zeroA, TypeAc zeroB, TypeAc zeroC, TypeAc M0, MYITE N, TypeAc clamp_min, TypeAc clamp_max) {
	MYITE ite_idx = 0, ite_val = 0;
	memset(tmp, 0, sizeof(TypeAc) * P);
	for (MYITE k = 0; k < K; k++) {
		// #define SPARSE_FLT_APPROX
		#ifdef SPARSE_FLT_APPROX
			float b = (float) B[k * 1][0];
			b += zeroB;

			b *= scaleB;

			MYITE idx = Aidx[ite_idx];
			while (idx != 0) {
				float a = (float) Aval[ite_val];
				a += zeroA;
				a *= scaleA;
				float c = (a * b);

				float c2 = C[idx - 1];
				c2 -= zeroC;
				c2 *= scale_out;

				c = c + c2;

				c /= scale_out;

				C[idx -1] = Saturate<TypeAc, TypeC>(TypeAc(c + zeroC), clamp_min, clamp_max);

				ite_idx++;
				ite_val++;

				idx = Aidx[ite_idx];
			}
		#else
			TypeAc b = (TypeAc) B[k * 1][0];
			b += zeroB;

			MYITE idx = Aidx[ite_idx];
			while (idx != 0) {
				TypeAc a = (TypeAc) Aval[ite_val];
				a += zeroA;
				tmp[idx - 1] += (a * b);

				ite_idx++;
				ite_val++;

				idx = Aidx[ite_idx];
			}
		#endif
		ite_idx++;
	}

	for (MYITE p = 0; p < P; p++) {
		tmp[p] = MulQuantMultiplier<TypeAc>(tmp[p], M0, N);
		C[p] = Saturate<TypeAc, TypeC>(tmp[p] + zeroC, clamp_min, clamp_max);
	}
	return;
}

template<class TypeA, class TypeAidx, class TypeB, class TypeAc, class TypeC>
void SparseMatMul(const TypeAidx* Aidx, const TypeA* Aval, TypeB* B, TypeC* C, TypeAc* tmp, MYITE P, MYITE K, float scaleA, float scaleB, float scale_out, MYITE left_shift, TypeAc zeroA, TypeAc zeroB, TypeAc zeroC, TypeAc M0, MYITE N, TypeAc clamp_min, TypeAc clamp_max) {
	MYITE ite_idx = 0, ite_val = 0;
	memset(tmp, 0, sizeof(TypeAc) * P);
	for (MYITE k = 0; k < K; k++) {
		TypeAc b = (TypeAc) B[k];
		b += zeroB;

		MYITE idx = Aidx[ite_idx];
		while (idx != 0) {
			TypeAc a = (TypeAc) Aval[ite_val];
			a += zeroA;
			tmp[idx - 1] += (a * b);

			ite_idx++;
			ite_val++;

			idx = Aidx[ite_idx];
		}
		ite_idx++;
	}

	for (MYITE p = 0; p < P; p++) {
		tmp[p] = MulQuantMultiplier<TypeAc>(tmp[p], M0, N);
		C[p] = Saturate<TypeAc, TypeC>(tmp[p] + zeroC, clamp_min, clamp_max);
	}
	return;
}

template<class TypeA, class TypeAc, class TypeB>
TypeB AdjustScaleZero(TypeA A, TypeAc zeroA, TypeAc zeroOut, TypeAc M, MYITE N, TypeAc clamp_min, TypeAc clamp_max) {
	TypeAc a = A;
	a -= zeroA;

	a = MulQuantMultiplier<TypeAc>(a, M, N);

	return Saturate<TypeAc, TypeB>(a + zeroOut, clamp_min, clamp_max);
}

template<class TypeA>
float ConvertZSkewToFloat(TypeA A, TypeA zeroA, float scaleA) {
	float a = A;
	a -= zeroA;

	return (a * scaleA);
}

template<class TypeA, class TypeAc, class TypeB>
void AddInPlace(TypeA* A, TypeB* B, MYITE I, MYITE J, float scaleA, float scaleB, TypeAc zero_in, TypeAc zero_out, MYITE left_shift, TypeAc shrA, MYITE nA, TypeAc shrB, MYITE nB, TypeAc shrC, MYITE nC, TypeAc clamp_min, TypeAc clamp_max) {
	// #define AddInPlace_APPROX

	for (MYITE i = 0; i < I; i++) {
		for (MYITE j = 0; j < J; j++) {
			#ifdef AddInPlace_APPROX
				float a = A[i * J + j];
				float b = B[i * J + j];

				a -= zero_in;
				b -= zero_out;
				a *= scaleA;
				b *= scaleB;

				float c = (a + b)/scaleB;
				B[i * J + j] = Saturate<TypeAc, TypeB>(TypeAc(zero_out + c), clamp_min, clamp_max);
			#else
				TypeAc a = A[i * J + j];
				TypeAc b = B[i * J + j];

				a -= zero_in;
				b -= zero_out;
				a *= TypeAc(1LL << left_shift);
				b *= TypeAc(1LL << left_shift);

				a = MulQuantMultiplier<TypeAc>(a, shrA, nA);
				b = MulQuantMultiplier<TypeAc>(b, shrB, nB);

				TypeAc c = MulQuantMultiplier<TypeAc>(a + b, shrC, nC);
				B[i * J + j] = Saturate<TypeAc, TypeB>(zero_out + c, clamp_min, clamp_max);
			#endif
		}
	}
	return;
}

template<class TypeA, class TypeAc, class TypeB>
void Exp(TypeA* A, TypeB* B, MYITE I, MYITE J, float scale_in, float scale_out, MYITE left_shift, TypeAc zeroA, TypeAc zeroB, TypeAc shrA, MYITE nA, TypeAc shrB1, MYITE nB1, TypeAc shrB2, MYITE nB2, TypeAc clamp_min, TypeAc clamp_max) {
	for (MYITE i = 0; i < I; i++) {
		for (MYITE j = 0; j < J; j++) {
			#ifdef FLOAT_APPROX
				float x = A[i * J + j];
				x += zeroA;

				float y = exp(x*scale_in);

				y /= scale_out;
				y += zeroB;

				B[i*J + j] = Saturate<TypeAc, TypeB>(TypeAc(y), clamp_min, clamp_max);
			#else
				TypeAc x = A[i * J + j];
				x += zeroA;

				TypeAc y;

				if (x < 0)
				{
					const int32_t x_rescaled = MulQuantMultiplier<TypeAc>(x, shrA, nA);
					using FixedPoint4 = gemmlowp::FixedPoint<int32_t, 13>;
					using FixedPoint0 = gemmlowp::FixedPoint<int32_t, 0>;
					const FixedPoint4 x_f4 = FixedPoint4::FromRaw(x_rescaled);
					const FixedPoint0 y_f0 = gemmlowp::exp_on_negative_values(x_f4);

					y = gemmlowp::RoundingDivideByPOT(y_f0.raw(), 24);

					y = MulQuantMultiplier<TypeAc>(y, shrB1, nB1);
				}
				else{
					x = -x;
					const int32_t x_rescaled = MulQuantMultiplier<TypeAc>(x, shrA, nA);
					using FixedPoint4 = gemmlowp::FixedPoint<int32_t, 13>;
					using FixedPoint0 = gemmlowp::FixedPoint<int32_t, 0>;
					const FixedPoint4 x_f4 = FixedPoint4::FromRaw(x_rescaled);
					const FixedPoint0 y_f0 = gemmlowp::exp_on_negative_values(x_f4);

					y = gemmlowp::RoundingDivideByPOT(y_f0.raw(), 24);
					// scale of y after this operation is -7
					y = TypeAc(1LL<< left_shift) / (y);
					// scale after this operation is -(left_shift -7)
					y = MulQuantMultiplier<TypeAc>(y, shrB2, nB2);
				}

				B[i * J + j] = Saturate<TypeAc, TypeA>(y + zeroB, std::numeric_limits<TypeA>::min(), std::numeric_limits<TypeA>::max());
			#endif
		}
	}
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

template<class TypeA>
void Reverse2(TypeA* A, MYINT axis, MYINT I, MYINT J, TypeA* B) {
	for (MYITE i = 0; i < I; i++) {
		for (MYITE j = 0; j < J; j++) {
			MYINT i_prime = (axis == 0 ? (I-1-i) : i);
			MYINT j_prime = (axis == 1 ? (J-1-j) : j);

			B[i * J + j] = A[i_prime*J + j_prime];
		}
	}
	return;
}

template<class TypeA, class TypeAc>
TypeA UnaryNegate(TypeA A, TypeAc zeroA, TypeAc zeroOut, TypeAc clamp_min, TypeAc clamp_max)
{
	TypeAc a = A;
	a += zeroA;
	a *= -1;
	return Saturate<TypeAc, TypeA> (a + zeroOut, clamp_min, clamp_max);
}

template<class TypeA, class TypeB, class TypeAc, class TypeC>
void AddOrSubCir4D(TypeA* A, TypeB* B, TypeC* X, MYITE N, MYITE H, MYITE W, MYITE C, float scaleA, float scaleB, float scaleC, MYITE left_shift, TypeAc zeroA, TypeAc shrA, MYITE nA, TypeAc zeroB, TypeAc shrB, MYITE nB, TypeAc zeroC, TypeAc shrC, MYITE nC, TypeAc clamp_min, TypeAc clamp_max, bool add) {
	for (MYITE n = 0; n < N; n++) {
		for (MYITE h = 0; h < H; h++) {
			for (MYITE w = 0; w < W; w++) {
				for (MYITE c = 0; c < C; c++) {
					TypeAc a = A[n * H * W * C + h * W * C + w * C + c];
					TypeAc b = B[c];

					a += zeroA;
					b += zeroB;
					a *= TypeAc(1LL << left_shift);
					b *= TypeAc(1LL << left_shift);

					a = MulQuantMultiplier<TypeAc>(a, shrA, nA);
					b = MulQuantMultiplier<TypeAc>(b, shrB, nB);

					TypeAc res;
					if (add) {
						res = MulQuantMultiplier<TypeAc>(a + b, shrC, nC);
					} else {
						res = MulQuantMultiplier<TypeAc>(a - b, shrC, nC);
					}

					X[n * H * W * C + h * W * C + w * C + c] = Saturate<TypeAc, TypeC>(zeroC + res, clamp_min, clamp_max);
				}
			}
		}
	}
	return;
}

template<class TypeA, class TypeB, class TypeAc, class TypeC>
void AddOrSubCir2D(TypeA* A, TypeB* B, TypeC* X, MYITE H, MYITE W, float scaleA, float scaleB, float scaleC, MYITE left_shift, TypeAc zeroA, TypeAc shrA, MYITE nA, TypeAc zeroB, TypeAc shrB, MYITE nB, TypeAc zeroC, TypeAc shrC, MYITE nC, TypeAc clamp_min, TypeAc clamp_max, bool add) {
	for (MYITE h = 0; h < H; h++) {
		for (MYITE w = 0; w < W; w++) {
			TypeAc a = A[h * W + w];
			TypeAc b = B[w];

			a += zeroA;
			b += zeroB;
			a *= TypeAc(1LL << left_shift);
			b *= TypeAc(1LL << left_shift);

			a = MulQuantMultiplier<TypeAc>(a, shrA, nA);
			b = MulQuantMultiplier<TypeAc>(b, shrB, nB);

			TypeAc res;
			if (add) {
				res = MulQuantMultiplier<TypeAc>(a + b, shrC, nC);
			} else {
				res = MulQuantMultiplier<TypeAc>(a - b, shrC, nC);
			}

			X[h * W  + w] = Saturate<TypeAc, TypeC>(zeroC + res, clamp_min, clamp_max);
		}
	}
	return;
}

template<class TypeA, class TypeAc>
void Relu4D(TypeA* A, MYITE N, MYITE H, MYITE W, MYITE C, float scale_in, float scale_out, TypeAc zeroA, TypeAc zeroOut, TypeAc M0, MYITE N0, TypeAc clamp_min, TypeAc clamp_max) {
	for (MYITE n = 0; n < N; n++) {
		for (MYITE h = 0; h < H; h++) {
			for (MYITE w = 0; w < W; w++) {
				for (MYITE c = 0; c < C; c++) {
					A[n * H * W * C + h * W * C + w * C + c] = AdjustScaleZero<TypeA, TypeAc, TypeA>(A[n * H * W * C + h * W * C + w * C + c], zeroA, zeroOut, M0, N0, clamp_min, clamp_max);
				}
			}
		}
	}
	return;
}

template<class TypeA, class TypeAc>
void Relu2D(TypeA* A, MYITE H, MYITE W, float scale_in, float scale_out, TypeAc zeroA, TypeAc zeroOut, TypeAc M0, MYITE N0, TypeAc clamp_min, TypeAc clamp_max) {
	for (MYITE h = 0; h < H; h++) {
		for (MYITE w = 0; w < W; w++) {
			A[h * W + w] = AdjustScaleZero<TypeA, TypeAc, TypeA>(A[h * W + w], zeroA, zeroOut, M0, N0, clamp_min, clamp_max);
		}
	}
	return;
}

template<class TypeA, class TypeAc, class TypeB>
void Relu6(TypeA* A, TypeB* B, MYITE N, MYITE H, MYITE W, MYITE C, float scale_in, float scale_out, TypeAc zeroA, TypeAc zeroOut, TypeAc M0, MYITE N0, TypeAc clamp_min, TypeAc clamp_max) {
	for (MYITE n = 0; n < N; n++) {
		for (MYITE h = 0; h < H; h++) {
			for (MYITE w = 0; w < W; w++) {
				for (MYITE c = 0; c < C; c++) {
					B[n * H * W * C + h * W * C + w * C + c] = AdjustScaleZero<TypeA, TypeAc, TypeB>(A[n * H * W * C + h * W * C + w * C + c], zeroA, zeroOut, M0, N0, clamp_min, clamp_max);
				}
			}
		}
	}
	return;
}

template<class TypeA, class TypeB, class TypeAc, class TypeC>
void Convolution(TypeA* A, const TypeB* B, TypeC* C, MYINT N, MYINT H, MYINT W, MYINT CIN, MYINT HF, MYINT WF, MYINT CINF, MYINT COUTF, MYINT HOUT, MYINT WOUT, MYINT HPADL, MYINT HPADR, MYINT WPADL, MYINT WPADR, MYINT HSTR, MYINT WSTR, MYINT HDL, MYINT WDL, MYINT G, float scaleA, float scaleB, float scaleC, TypeAc zeroA, TypeAc zeroB, TypeAc zeroC, TypeAc M0, MYITE N0, TypeAc clamp_min, TypeAc clamp_max) {
	MYITE HOffsetL = HDL*(HF/2) - HPADL;
	MYITE WOffsetL = WDL*(WF/2) - WPADL;
	MYITE HOffsetR = HDL*(HF/2) - HPADR;
	MYITE WOffsetR = WDL*(WF/2) - WPADR;

	for (MYITE n = 0; n < N; n++) {
		for (MYITE h = HOffsetL, hout = 0; h < H - HOffsetR; h += HSTR, hout++) {
			for (MYITE w = WOffsetL, wout = 0; w < W - WOffsetR; w += WSTR, wout++) {
				for (MYITE g = 0; g < G; g++) {
					for (MYITE co = 0; co < COUTF; co++) {

						TypeAc sum = 0;
						for (MYITE hf = -(HF / 2); hf <= HF / 2; hf++) {
							for (MYITE wf = -(WF / 2); wf <= WF / 2; wf++) {
								for (MYITE ci = 0; ci < CINF; ci++) {

									TypeAc a = (TypeAc) (((h + HDL * hf) < 0) || ((h + HDL * hf) >= H) || ((w + WDL * wf) < 0) || ((w + WDL * wf) >= W)) ? -zeroA : A[n * H * W * CIN + (h + HDL * hf) * W * CIN + (w + WDL * wf) * CIN + (ci + g * CINF)];

									TypeAc b = (TypeAc) B[g * HF * WF * CINF * COUTF + (hf + HF / 2) * WF * CINF * COUTF + (wf + WF / 2) * CINF * COUTF + ci * COUTF + co];

									sum += (a + zeroA) * (b + zeroB);
								}
							}
						}

						sum = MulQuantMultiplier<TypeAc>(sum, M0, N0);
						C[n * HOUT * WOUT * (COUTF * G) + hout * WOUT * (COUTF * G) + wout * (COUTF * G) + (co + g * COUTF)] = Saturate<TypeAc, TypeC>(zeroC + sum, clamp_min, clamp_max);
					}
				}
			}
		}
	}
}

template<class TypeA, class TypeF1, class TypeB1W, class TypeB1B, class TypeF2, class TypeB2W, class TypeB2B, class TypeF3, class TypeB3W, class TypeB3B, class TypeC, class TypeX, class TypeT, class TypeUB1W, class TypeUB2W, class TypeUB3W, class TypeAc>
void MBConv(TypeA* A, TypeF1* F1, TypeB1W* BN1W, TypeB1B* BN1B, TypeF2* F2, TypeB2W* BN2W, TypeB2B* BN2B, TypeF3* F3, TypeB3W* BN3W, TypeB3B* BN3B, TypeC* C, TypeX* X, TypeT* T, MYITE N, MYITE H, MYITE W, MYITE Cin, MYITE Ct, MYITE HF, MYITE WF, MYITE Cout, MYITE Hout, MYITE Wout, MYITE HPADL, MYITE HPADR, MYITE WPADL, MYITE WPADR, MYITE HSTR, MYITE WSTR, TypeAc zeroA, TypeAc zeroF1, TypeUB1W zeroBN1W, TypeUB1W zeroBN1B, TypeAc zeroF2, TypeUB2W zeroBN2W, TypeUB2W zeroBN2B, TypeAc zeroF3, TypeUB3W zeroBN3W, TypeUB3W zeroBN3B, TypeUB3W zeroC, TypeUB1W zeroX, TypeUB2W zeroT, MYITE left_shift1, MYITE left_shift2, MYITE left_shift3, TypeAc M11, MYITE N11, TypeUB1W M12, MYITE N12, TypeUB1W M13, MYITE N13, TypeUB1W M14, MYITE N14, TypeUB1W M15, MYITE N15, TypeAc M21, MYITE N21, TypeUB2W M22, MYITE N22, TypeUB2W M23, MYITE N23, TypeUB2W M24, MYITE N24, TypeUB2W M25, MYITE N25, TypeAc M31, MYITE N31, TypeUB3W M32, MYITE N32, TypeUB3W M33, MYITE N33, TypeUB3W M34, MYITE N34, TypeUB3W M35, MYITE N35, TypeUB1W clamp_min_X, TypeUB1W clamp_max_X, TypeUB2W clamp_min_T, TypeUB2W clamp_max_T, TypeUB3W clamp_min_C, TypeUB3W clamp_max_C) {
	MYITE HOffsetL = (HF / 2) - HPADL;
	MYITE WOffsetL = (WF / 2) - WPADL;
	MYITE HOffsetR = (HF / 2) - HPADR;
	MYITE WOffsetR = (WF / 2) - WPADR;

	for (MYITE n = 0; n < N; n++) {
		MYITE margin = HOffsetL + (HF / 2 + 1) - HSTR > 0 ? HOffsetL + (HF / 2 + 1) - HSTR : 0;
		MYITE nstart = HOffsetL - (HF / 2) < 0 ? 0 : HOffsetL - (HF / 2);
		for (MYITE i = nstart; i < margin; i++) {
			for (MYITE j = 0; j < W; j++) {
				for (MYITE k = 0; k < Ct; k++) {
					TypeAc sum = 0;
					for (MYITE l = 0; l < Cin; l++) {
						TypeAc a = A[n * H * W * Cin + i * W * Cin + j * Cin + l];
						TypeAc f = F1[l * Ct + k];
						sum += (a + zeroA) * (f + zeroF1);
					}

					sum = MulQuantMultiplier<TypeAc>(sum, M11, N11);
					sum *= TypeAc(1LL << left_shift1);
					sum = MulQuantMultiplier<TypeUB1W>(sum, M12, N12);

					TypeUB1W b = BN1B[k];
					b += zeroBN1B;
					b *= TypeUB1W(1LL << left_shift1);
					b = MulQuantMultiplier<TypeUB1W>(b, M13, N13);

					TypeUB1W c = MulQuantMultiplier<TypeUB1W>(sum + b, M14, N14);

					TypeUB1W w = BN1W[k];
					w += zeroBN1W;
					TypeUB1W x = MulQuantMultiplier<TypeUB1W>(c * w, M15, N15);

					X[i * W * Ct + j * Ct + k] = Saturate<TypeUB1W, TypeX>(x + zeroX, clamp_min_X, clamp_max_X);
				}
			}
		}

		for (MYITE h = HOffsetL, hout = 0; h < H - HOffsetR; hout++, h += HSTR) {

			for (MYITE i = 0; i < HSTR; i++) {
				for (MYITE j = 0; j < W; j++) {
					for (MYITE k = 0; k < Ct; k++) {
						MYITE iRed = (i + margin + hout * HSTR) % HF, iFull = i + margin + hout * HSTR;
						X[iRed * W * Ct + j * Ct + k] = 0;
						TypeAc sum = 0;
						for (MYITE l = 0; l < Cin; l++) {
							TypeAc a = iFull < H ? A[n * H * W * Cin + iFull * W * Cin + j * Cin + l] : -zeroA;
							TypeAc f = F1[l * Ct + k];

							sum += (a + zeroA) * (f + zeroF1);
						}

						sum = MulQuantMultiplier<TypeAc>(sum, M11, N11);
						sum *= TypeAc(1LL << left_shift1);
						sum = MulQuantMultiplier<TypeUB1W>(sum, M12, N12);

						TypeUB1W b = BN1B[k];
						b += zeroBN1B;
						b *= TypeUB1W(1LL << left_shift1);
						b = MulQuantMultiplier<TypeUB1W>(b, M13, N13);

						TypeUB1W c = MulQuantMultiplier<TypeUB1W>(sum + b, M14, N14);

						TypeUB1W w = BN1W[k];
						w += zeroBN1W;
						TypeUB1W x = MulQuantMultiplier<TypeUB1W>(c * w, M15, N15);

						X[iRed * W * Ct + j * Ct + k] = Saturate<TypeUB1W, TypeX>(x + zeroX, clamp_min_X, clamp_max_X);
					}
				}
			}

			for (MYITE w = WOffsetL, wout = 0; w < W - WOffsetR; w += WSTR, wout++) {
				for (MYITE g = 0; g < Ct; g++) {
					TypeAc sum = 0;
					for (MYITE hf = -(HF / 2); hf <= (HF / 2); hf++) {
						for (MYITE wf = -(WF / 2); wf <= (WF / 2); wf++) {
							TypeAc x = (((h + hf) < 0) || ((h + hf) >= H) || ((w + wf) < 0) || ((w + wf) >= W)) ? -zeroX : X[((h + hf) % HF) * W * Ct + (w + wf) * Ct + g];
							TypeAc f = F2[g * HF * WF + (hf + HF / 2) * WF + (wf + WF / 2)];
							sum += (x - zeroX) * (f + zeroF2);
						}
					}

					sum = MulQuantMultiplier<TypeAc>(sum, M21, N21);
					sum *= TypeAc(1LL << left_shift2);
					sum = MulQuantMultiplier<TypeUB2W>(sum, M22, N22);

					TypeUB2W b = BN2B[g];
					b += zeroBN2B;
					b *= TypeUB2W(1LL << left_shift2);
					b = MulQuantMultiplier<TypeUB2W>(b, M23, N23);

					TypeUB2W c = MulQuantMultiplier<TypeUB2W>(sum + b, M24, N24);

					TypeUB2W w = BN2W[g];
					w += zeroBN2W;
					TypeUB2W t = MulQuantMultiplier<TypeUB2W>(c * w, M25, N25);

					T[g] = Saturate<TypeUB2W, TypeT>(t + zeroT, clamp_min_T, clamp_max_T);
				}

				for (MYITE i = 0; i < Cout; i++) {
					TypeAc sum = 0;
					for (MYITE g = 0; g < Ct; g++) {
						TypeAc t = T[g];
						TypeAc f = F3[g * Cout + i];
						sum += (t - zeroT) * (f + zeroF3);
					}

					sum = MulQuantMultiplier<TypeAc>(sum, M31, N31);
					sum *= TypeAc(1LL << left_shift3);
					sum = MulQuantMultiplier<TypeUB3W>(sum, M32, N32);

					TypeUB3W b = BN3B[i];
					b += zeroBN3B;
					b *= TypeUB3W(1LL << left_shift3);
					b = MulQuantMultiplier<TypeUB3W>(b, M33, N33);

					TypeUB3W c = MulQuantMultiplier<TypeUB3W>(sum + b, M34, N34);

					TypeUB3W w = BN3W[i];
					w += zeroBN3W;
					c = MulQuantMultiplier<TypeUB3W>(c * w, M35, N35);

					C[n * Hout * Wout * Cout + hout * Wout * Cout + wout * Cout + i] = Saturate<TypeUB3W, TypeT>(c + zeroC, clamp_min_C, clamp_max_C);
				}
			}
		}
	}
}

template<class TypeA, class TypeAc>
void NormaliseL2(TypeA* A, TypeA* B, MYITE N, MYITE H, MYITE W, MYITE C, float scale_in, float scale_out, TypeAc zeroA, TypeAc zeroOut, TypeAc clamp_min, TypeAc clamp_max) {
	for (MYITE n = 0; n < N; n++) {
		for (MYITE h = 0; h < H; h++) {
			for (MYITE w = 0; w < W; w++) {
				TypeAc sum = 0;
				for (MYITE c = 0; c < C; c++) {
					TypeAc a = A[n * H * W * C + h * W * C + w * C + c];
					a += zeroA;
					sum += a * a;
				}

				TypeAc norm_multiplier;
				MYITE norm_shift;
				InvSqrtQuantizedMultiplier<TypeAc>(sum, &norm_multiplier, &norm_shift);

				for (MYITE c = 0; c < C; c++) {
					TypeAc a = A[n * H * W * C + h * W * C + w * C + c];
					a += zeroA;
					a <<= 7;

					TypeAc a_rescaled = MulQuantMultiplier<TypeAc>(a, norm_multiplier, norm_shift);
					B[n * H * W * C + h * W * C + w * C + c] = Saturate<TypeAc, TypeA>(a_rescaled + zeroOut, clamp_min, clamp_max);
				}
			}
		}
	}
	return;
}
