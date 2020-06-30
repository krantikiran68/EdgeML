// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include <iostream>
#include <cmath>

#include "datatypes.h"
#include "library_float.h"
#include "profile.h"

// C = A + B
void MatAddNN(float *A, float *B, float *C, MYINT I, MYINT J, MYINT shrA, MYINT shrB, MYINT shrC)
{
	for (MYITE i = 0; i < I; i++)
	{
		for (MYITE j = 0; j < J; j++)
		{
			float a = A[i * J + j];
			float b = B[i * J + j];

			float c = a + b;

			C[i * J + j] = c;
		}
	}
	return;
}

// C = A + B
void MatAddCN(const float *A, float *B, float *C, MYINT I, MYINT J, MYINT shrA, MYINT shrB, MYINT shrC)
{
	for (MYITE i = 0; i < I; i++)
	{
		for (MYITE j = 0; j < J; j++)
		{
			float a = A[i * J + j];
			float b = B[i * J + j];

			float c = a + b;

			C[i * J + j] = c;
		}
	}
	return;
}

// C = A + B
void MatAddNC(float *A, const float *B, float *C, MYINT I, MYINT J, MYINT shrA, MYINT shrB, MYINT shrC)
{
	for (MYITE i = 0; i < I; i++)
	{
		for (MYITE j = 0; j < J; j++)
		{
			float a = A[i * J + j];
			float b = B[i * J + j];

			float c = a + b;

			C[i * J + j] = c;
		}
	}
	return;
}

// C = A + B
void MatAddCC(const float *A, const float *B, float *C, MYINT I, MYINT J, MYINT shrA, MYINT shrB, MYINT shrC)
{
	for (MYITE i = 0; i < I; i++)
	{
		for (MYITE j = 0; j < J; j++)
		{
			float a = A[i * J + j];
			float b = B[i * J + j];

			float c = a + b;

			C[i * J + j] = c;
		}
	}
	return;
}

// C = a + B
void MatAddBroadCastA(float *A, float *B, float *C, MYINT I, MYINT J, MYINT shrA, MYINT shrB, MYINT shrC)
{
	for (MYITE i = 0; i < I; i++)
	{
		for (MYITE j = 0; j < J; j++)
		{
			float a = *A;
			float b = B[i * J + j];

			float c = a + b;

			C[i * J + j] = c;
		}
	}
	return;
}

// C = A + b
void MatAddBroadCastB(float *A, float *B, float *C, MYINT I, MYINT J, MYINT shrA, MYINT shrB, MYINT shrC)
{
	for (MYITE i = 0; i < I; i++)
	{
		for (MYITE j = 0; j < J; j++)
		{
			float a = A[i * J + j];
			float b = *B;

			float c = a + b;

			C[i * J + j] = c;
		}
	}
	return;
}

// C = A - B
void MatSub(float *A, const float *B, float *C, MYINT I, MYINT J, MYINT shrA, MYINT shrB, MYINT shrC)
{
	for (MYITE i = 0; i < I; i++)
	{
		for (MYITE j = 0; j < J; j++)
		{
			float a = A[i * J + j];
			float b = B[i * J + j];

			float c = a - b;

			C[i * J + j] = c;
		}
	}
	return;
}

// C = a - B
void MatSubBroadCastA(float *A, float *B, float *C, MYINT I, MYINT J, MYINT shrA, MYINT shrB, MYINT shrC)
{
	for (MYITE i = 0; i < I; i++)
	{
		for (MYITE j = 0; j < J; j++)
		{
			float a = *A;
			float b = B[i * J + j];

			float c = a - b;

			C[i * J + j] = c;
		}
	}
	return;
}

// C = A - b
void MatSubBroadCastB(float *A, float *B, float *C, MYINT I, MYINT J, MYINT shrA, MYINT shrB, MYINT shrC)
{
	for (MYITE i = 0; i < I; i++)
	{
		for (MYITE j = 0; j < J; j++)
		{
			float a = A[i * J + j];
			float b = *B;

			float c = a - b;

			C[i * J + j] = c;
		}
	}
	return;
}

// C = A * B
void MatMulNN(float *A, float *B, float *C, float *tmp, MYINT I, MYINT K, MYINT J, MYINT shrA, MYINT shrB, MYINT H1, MYINT H2)
{

	for (MYITE i = 0; i < I; i++)
	{
		for (MYITE j = 0; j < J; j++)
		{
			for (MYITE k = 0; k < K; k++)
			{
				float a = A[i * K + k];
				float b = B[k * J + j];

				tmp[k] = a * b;
			}

			MYITE count = K, depth = 0;
			bool shr = true;

			while (depth < (H1 + H2))
			{
				if (depth >= H1)
					shr = false;

				for (MYITE p = 0; p < (K / 2 + 1); p++)
				{
					float sum;
					if (p < (count >> 1))
						sum = tmp[2 * p] + tmp[(2 * p) + 1];
					else if ((p == (count >> 1)) && ((count & 1) == 1))
						sum = tmp[2 * p];
					else
						sum = 0;

					if (shr)
						tmp[p] = sum;
					else
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
void MatMulCN(const float *A, float *B, float *C, float *tmp, MYINT I, MYINT K, MYINT J, MYINT shrA, MYINT shrB, MYINT H1, MYINT H2)
{

	for (MYITE i = 0; i < I; i++)
	{
		for (MYITE j = 0; j < J; j++)
		{
			for (MYITE k = 0; k < K; k++)
			{
				float a = A[i * K + k];
				float b = B[k * J + j];

				tmp[k] = a * b;
			}

			MYITE count = K, depth = 0;
			bool shr = true;

			while (depth < (H1 + H2))
			{
				if (depth >= H1)
					shr = false;

				for (MYITE p = 0; p < (K / 2 + 1); p++)
				{
					float sum;
					if (p < (count >> 1))
						sum = tmp[2 * p] + tmp[(2 * p) + 1];
					else if ((p == (count >> 1)) && ((count & 1) == 1))
						sum = tmp[2 * p];
					else
						sum = 0;

					if (shr)
						tmp[p] = sum;
					else
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
void MatMulNC(float *A, const float *B, float *C, float *tmp, MYINT I, MYINT K, MYINT J, MYINT shrA, MYINT shrB, MYINT H1, MYINT H2)
{

	for (MYITE i = 0; i < I; i++)
	{
		for (MYITE j = 0; j < J; j++)
		{
			for (MYITE k = 0; k < K; k++)
			{
				float a = A[i * K + k];
				float b = B[k * J + j];

				tmp[k] = a * b;
			}

			MYITE count = K, depth = 0;
			bool shr = true;

			while (depth < (H1 + H2))
			{
				if (depth >= H1)
					shr = false;

				for (MYITE p = 0; p < (K / 2 + 1); p++)
				{
					float sum;
					if (p < (count >> 1))
						sum = tmp[2 * p] + tmp[(2 * p) + 1];
					else if ((p == (count >> 1)) && ((count & 1) == 1))
						sum = tmp[2 * p];
					else
						sum = 0;

					if (shr)
						tmp[p] = sum;
					else
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
void MatMulCC(const float *A, const float *B, float *C, float *tmp, MYINT I, MYINT K, MYINT J, MYINT shrA, MYINT shrB, MYINT H1, MYINT H2)
{

	for (MYITE i = 0; i < I; i++)
	{
		for (MYITE j = 0; j < J; j++)
		{
			for (MYITE k = 0; k < K; k++)
			{
				float a = A[i * K + k];
				float b = B[k * J + j];

				tmp[k] = a * b;
			}

			MYITE count = K, depth = 0;
			bool shr = true;

			while (depth < (H1 + H2))
			{
				if (depth >= H1)
					shr = false;

				for (MYITE p = 0; p < (K / 2 + 1); p++)
				{
					float sum;
					if (p < (count >> 1))
						sum = tmp[2 * p] + tmp[(2 * p) + 1];
					else if ((p == (count >> 1)) && ((count & 1) == 1))
						sum = tmp[2 * p];
					else
						sum = 0;

					if (shr)
						tmp[p] = sum;
					else
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
void SparseMatMul(const MYINT *Aidx, const float *Aval, float **B, float *C, int16_t K, MYINT shrA, MYINT shrB, MYINT shrC)
{

	MYITE ite_idx = 0, ite_val = 0;
	for (MYITE k = 0; k < K; k++)
	{
		// float b = getIntFeature(k);
		float b = B[k * 1][0];

		MYINT idx = Aidx[ite_idx];
		while (idx != 0)
		{
			float a = Aval[ite_val];

			float c = a * b;

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
void MulCir(float *A, float *B, float *C, MYINT I, MYINT J, MYINT shrA, MYINT shrB)
{
	for (MYITE i = 0; i < I; i++)
	{
		for (MYITE j = 0; j < J; j++)
		{
			float a = A[i * J + j];
			float b = B[i * J + j];

			C[i * J + j] = a * b;
		}
	}
	return;
}

// A = tanh(A)
void TanH(float *A, MYINT I, MYINT J, float scale_in, float scale_out, float *B)
{
	for (MYITE i = 0; i < I; i++)
	{
		for (MYITE j = 0; j < J; j++)
		{
			float x = A[i * J + j], y;

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
void Reverse2(float *A, MYINT axis, MYINT I, MYINT J, float *B)
{
	for (MYITE i = 0; i < I; i++)
	{
		for (MYITE j = 0; j < J; j++)
		{	
			MYINT i_prime = (axis == 0 ? (I-1-i) : i);
			MYINT j_prime = (axis == 1 ? (J-1-j) : j); 

			B[i * J + j] = A[i_prime*J + j_prime];
		}
	}
	return;
}

// index = argmax(A)
void ArgMax(float *A, MYINT I, MYINT J, MYINT *index)
{

	float max = A[0];
	MYITE maxIndex = 0, counter = 0;
	for (MYITE i = 0; i < I; i++)
	{
		for (MYITE j = 0; j < J; j++)
		{
			float x = A[i * J + j];

			if (max < x)
			{
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
void Transpose(float *A, float *B, MYINT I, MYINT J)
{
	for (MYITE i = 0; i < I; i++)
	{
		for (MYITE j = 0; j < J; j++)
		{
			B[i * J + j] = A[j * I + i];
		}
	}
	return;
}

// C = a * B
void ScalarMul(float *A, float *B, float *C, MYINT I, MYINT J, MYINT shrA, MYINT shrB)
{

	float a = *A;

	for (MYITE i = 0; i < I; i++)
	{
		for (MYITE j = 0; j < J; j++)
		{
			float b = B[i * J + j];

			C[i * J + j] = a * b;
		}
	}

	return;
}

// C = conv(A, B, <params>)
// A[N][H][W][CIN], B[G][HF][WF][CINF][COUTF], C[N][HOUT][WOUT][COUTF*G]
void Convolution(float *A, const float *B, float *C, float *tmp, MYINT N, MYINT H, MYINT W, MYINT CIN, MYINT HF, MYINT WF, MYINT CINF, MYINT COUTF, MYINT HOUT, MYINT WOUT, MYINT HPADL, MYINT HPADR, MYINT WPADL, MYINT WPADR, MYINT HSTR, MYINT WSTR, MYINT HDL, MYINT WDL, MYINT G, MYINT shrA, MYINT shrB, MYINT H1, MYINT H2) {
	MYITE HOffsetL = HDL*(HF/2) - HPADL;
	MYITE WOffsetL = WDL*(WF/2) - WPADL;
	MYITE HOffsetR = HDL*(HF/2) - HPADR;
	MYITE WOffsetR = WDL*(WF/2) - WPADR;

	for(MYITE n = 0; n < N; n++) {
		for(MYITE h = HOffsetL, hout = 0; h < H - HOffsetR; h += HSTR, hout++) {
			for(MYITE w = WOffsetL, wout = 0; w < W - WOffsetR; w += WSTR, wout++) {
				for(MYITE g = 0; g < G; g++) {
					for(MYITE co = 0; co < COUTF; co ++) {

						MYITE counter = 0;
						for(MYITE hf = -(HF/2); hf <= HF/2; hf++) {
							for(MYITE wf = -(WF/2); wf <= WF/2; wf++) {
								for(MYITE ci = 0; ci < CINF; ci++) {

									float a = (((h + HDL * hf) < 0) || ((h + HDL * hf) >= H) || ((w + WDL * wf) < 0) || ((w + WDL * wf) >= W)) ? 0 : A[n * H * W * CIN + (h + HDL * hf) * W * CIN + (w + WDL * wf) * CIN + (ci + g * CINF)];

									float b = B[(hf + HF/2) * WF * CINF * COUTF + (wf + WF/2) * CINF * COUTF + ci * COUTF + co];

									tmp[counter] = a * b;
									counter++;
								}
							}
						}

						MYITE totalEle = HF * WF * CINF;
						MYITE count = HF * WF * CINF, depth = 0;
						bool shr = true;

						while (depth < (H1 + H2)) {
							if (depth >= H1)
								shr = false;

							for (MYITE p = 0; p < (totalEle / 2 + 1); p++) {
								float sum;
								if (p < (count >> 1))
									sum = tmp[2 * p] + tmp[(2 * p) + 1];
								else if ((p == (count >> 1)) && ((count & 1) == 1))
									sum = tmp[2 * p];
								else
									sum = 0;

								if (shr)
									tmp[p] = sum;
								else
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


// C = A # B
// A[N][H][W][CI], B[HF][WF][CI][CO], C[N][H][W][CO]
void Conv(float *A, const float *B, float *C, float *tmp, MYINT N, MYINT H, MYINT W, MYINT CI, MYINT HF, MYINT WF, MYINT CO, MYINT shrA, MYINT shrB, MYINT H1, MYINT H2)
{
	MYITE padH = (HF - 1) / 2;
	MYITE padW = (WF - 1) / 2;

	for (MYITE n = 0; n < N; n++)
	{
		for (MYITE h = 0; h < H; h++)
		{
			for (MYITE w = 0; w < W; w++)
			{
				for (MYITE co = 0; co < CO; co++)
				{

					MYITE counter = 0;
					for (MYITE hf = 0; hf < HF; hf++)
					{
						for (MYITE wf = 0; wf < WF; wf++)
						{
							for (MYITE ci = 0; ci < CI; ci++)
							{
								float a = (((((h + hf) < padH) || ((h + hf) >= (H + padH))) || (((w + wf) < padW) || ((w + wf) >= (W + padW)))) ? 0 : A[n * H * W * CI + ((h + hf) - padH) * W * CI + ((w + wf) - padW) * CI + ci]);

								float b = B[hf * WF * CI * CO + wf * CI * CO + ci * CO + co];

								tmp[counter] = a * b;
								counter++;
							}
						}
					}

					MYITE totalEle = HF * WF * CI;
					MYITE count = HF * WF * CI, depth = 0;
					bool shr = true;

					while (depth < (H1 + H2))
					{
						if (depth >= H1)
							shr = false;

						for (MYITE p = 0; p < (totalEle / 2 + 1); p++)
						{
							float sum;
							if (p < (count >> 1))
								sum = tmp[2 * p] + tmp[(2 * p) + 1];
							else if ((p == (count >> 1)) && ((count & 1) == 1))
								sum = tmp[2 * p];
							else
								sum = 0;

							if (shr)
								tmp[p] = sum;
							else
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

// A = A <+> B
// A[N][H][W][C], B[C]
void AddOrSubCir4D(float *A, const float *B, float *X, MYINT N, MYINT H, MYINT W, MYINT C, MYINT shrA, MYINT shrB, MYINT shrC, bool add)
{

	for (MYITE n = 0; n < N; n++)
	{
		for (MYITE h = 0; h < H; h++)
		{
			for (MYITE w = 0; w < W; w++)
			{
				for (MYITE c = 0; c < C; c++)
				{
					float a = A[n * H * W * C + h * W * C + w * C + c];

					float b = B[c];

					float res;
					if (add)
						res = a + b;
					else
						res = a - b;

					X[n * H * W * C + h * W * C + w * C + c] = res;
				}
			}
		}
	}

	return;
}

// A = A <+> B
// A[N][H][W][C], B[C]
void AddOrSubCir2D(float *A, const float *B, float *X, MYINT H, MYINT W, MYINT shrA, MYINT shrB, MYINT shrC, bool add)
{

	for (MYITE h = 0; h < H; h++)
	{
		for (MYITE w = 0; w < W; w++)
		{
			float a = A[h * W + w];

			float b = B[w];

			float res;
			if (add)
				res = a + b;
			else
				res = a - b;

			X[h * W + w] = res;
		}
	}

	return;
}

// A = relu(A)
// A[N][H][W][C]
void Relu4D(float *A, MYINT N, MYINT H, MYINT W, MYINT C)
{

	for (MYITE n = 0; n < N; n++)
	{
		for (MYITE h = 0; h < H; h++)
		{
			for (MYITE w = 0; w < W; w++)
			{
				for (MYITE c = 0; c < C; c++)
				{
					float a = A[n * H * W * C + h * W * C + w * C + c];
					if (a < 0)
						a = 0;

					A[n * H * W * C + h * W * C + w * C + c] = a;
				}
			}
		}
	}

	return;
}

// A = relu(A)
// A[N][H][W][C]
void Relu2D(float *A, MYINT H, MYINT W)
{

	for (MYITE h = 0; h < H; h++)
	{
		for (MYITE w = 0; w < W; w++)
		{
			float a = A[h * W + w];
			if (a < 0)
				a = 0;

			A[h * W + w] = a;
		}
	}

	return;
}

// B = maxpool(A)
// A[N][H][W][C], B[N][H][W][C]
void Maxpool(float *A, float *B, MYINT N, MYINT H, MYINT W, MYINT C, MYINT FH, MYINT FW, MYINT strideH, MYINT strideW, MYINT HPADL, MYINT HPADR, MYINT WPADL, MYINT WPADR)
{
	MYITE HO = H / strideH;
	MYITE WO = W / strideW;

	for (MYITE n = 0; n < N; n++)
	{
		for (MYITE ho = 0; ho < HO; ho++)
		{
			for (MYITE wo = 0; wo < WO; wo++)
			{
				for (MYITE c = 0; c < C; c++)
				{

					float max = A[n * H * W * C + (strideH * ho) * W * C + (strideW * wo) * C + c];
					for (MYITE hs = 0; hs < FH; hs++)
					{
						for (MYITE ws = 0; ws < FW; ws++)
						{
							float a = A[n * H * W * C + ((strideH * ho) + hs) * W * C + ((strideW * wo) + ws) * C + c];
							if (a > max)
								max = a;
						}
					}

					B[n * HO * WO * C + ho * WO * C + wo * C + c] = max;
				}
			}
		}
	}

	return;
}

void NormaliseL2(float* A, MYINT N, MYINT H, MYINT W, MYINT C, MYINT scaleA, MYINT shrA) {
	for (MYITE n = 0; n < N; n++) {
		for (MYITE h = 0; h < H; h++) {
			for (MYITE w = 0; w < W; w++) {

				// calculate the sum square
				float sumSquare = 0;
				for (MYITE c = 0; c < C; c++) {
						float tmp = A[n * H * W * C + h * W * C + w * C + c];
						sumSquare += tmp*tmp;
				}

				// calculate the inverse square root of sumSquare

				if(sumSquare == 0){
					sumSquare = 1e-5;
				}

				float inverseNorm = 1/sqrt(sumSquare);

				// multiply all elements by the 1/sqrt(sumSquare)
				for (MYITE c = 0; c < C; c++) {
						A[n * H * W * C + h * W * C + w * C + c]  = A[n * H * W * C + h * W * C + w * C + c]  *inverseNorm;  
				}								
			}
		}
	}
	return;
}

// B = exp(A)
void Exp(float *A, MYINT I, MYINT J, MYINT shrA, MYINT shrB, float *B)
{

	for (MYITE i = 0; i < I; i++)
	{
		for (MYITE j = 0; j < J; j++)
		{
			float x = A[i * J + j];

			updateRangeOfExp(-x);

			B[i * J + j] = exp(x);
		}
	}

	return;
}

// A = sigmoid(A)
void Sigmoid(float *A, MYINT I, MYINT J, float div, float add, float sigmoid_limit, MYINT scale_in, MYINT scale_out, float *B)
{
	for (MYITE i = 0; i < I; i++)
	{
		for (MYITE j = 0; j < J; j++)
		{
			float x = A[i * J + j], y;
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
void AdjustScaleShr(float *A, MYINT I, MYINT J, MYINT scale)
{
	return;
}

// A = AdjustScaleShl(A)
void AdjustScaleShl(float *A, MYINT I, MYINT J, MYINT scale)
{
	return;
}
