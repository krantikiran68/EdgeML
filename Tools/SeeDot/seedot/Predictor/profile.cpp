// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include <iostream>
#include <fstream>
#include <limits>
#include <cmath>
#include <unordered_map>

#include "datatypes.h"
#include "profile.h"

using namespace std;

float m_all, M_all;
float m_exp, M_exp;

void initializeProfiling()
{
	m_all = numeric_limits<float>::max();
	M_all = -numeric_limits<float>::max();

	m_exp = numeric_limits<float>::max();
	M_exp = -numeric_limits<float>::max();

	return;
}

void updateRange(float x)
{
	if (x < m_all)
		m_all = x;
	if (x > M_all)
		M_all = x;
	return;
}

void updateRangeOfExp(float x)
{
	if (x < m_exp)
		m_exp = x;
	if (x > M_exp)
		M_exp = x;
	return;
}

void dumpRange(string outputFile)
{
	ofstream fout(outputFile);

	fout.precision(6);
	fout << fixed;
	fout << m_all << ", " << M_all << endl;
	fout << m_exp << ", " << M_exp << endl;

	return;
}

unordered_map<string, float> min_all;
unordered_map<string, float> max_all;

unordered_map<string, float> min_temp;
unordered_map<string, float> max_temp;

bool range_exceeded = false;

void dumpProfile() {
	if (min_all.size() == 0)
		return;
	ofstream outfile("dump.profile");
	auto min_i = min_all.begin();
	while (min_i != min_all.end()) {
		string key = min_i->first;
		outfile << key << "," << min_all[key] << "," << max_all[key] << endl;
		min_i++;
	}
	outfile.close();
}

void flushProfile() {
	if (range_exceeded == false) {
		for(auto it = min_temp.begin(); it != min_temp.end(); it ++) {
			string name = it->first;
			if(min_all.find(name) == min_all.end()) {
				min_all[name] = min_temp[name];
				max_all[name] = max_temp[name];
			} else {
				min_all[name] = min_all[name] < min_temp[name] ? min_all[name] : min_temp[name];
				max_all[name] = max_all[name] > max_temp[name] ? max_all[name] : max_temp[name];
			}
			min_temp[name] = FLT_MAX;
			max_temp[name] = -FLT_MAX;
		}
	}
	range_exceeded = false;
}

void checkRange2(float* A, int I, int J) {
	for (int i = 0; i < I; i++) {
		for (int j = 0; j < J; j++) {
			if (fabs(A[i * J + j]) >= 32) {
				range_exceeded = true;
			}
		}
	}
}

void Profile4(float* A, int I, int J, int K, int L, string name) {
	return;
}

void Profile2(float* A, int I, int J, string name) {
	if (min_temp.find(name) == min_temp.end()) {
		min_temp[name] = FLT_MAX;
		max_temp[name] = -FLT_MAX;
	}
	for (int i = 0; i < I; i++) {
		for (int j = 0; j < J; j++) {
			min_temp[name] = min_temp[name] < A[i * J + j] ? min_temp[name] : A[i * J + j];
			max_temp[name] = max_temp[name] > A[i * J + j] ? max_temp[name] : A[i * J + j];
		}
	}
}

void diff(float *A, MYINT *B, MYINT scale, MYINT I, MYINT J)
{

	float min = numeric_limits<float>::max(), max = 0, sum = 0;
	float min_relative = numeric_limits<float>::max(), max_relative = 0, sum_relative = 0;
	int count = 0;

	for (MYINT i = 0; i < I; i++)
	{
		for (MYINT j = 0; j < J; j++)
		{
			float a = A[i * J + j];

			MYINT b = B[i * J + j];
			//float b_float = float(b) / scale;
			float b_float = float(ldexp(double(b), scale));

			float diff = abs(a - b_float);
			float diff_relative = diff / abs(a);

			if (diff < min)
				min = diff;
			if (diff > max)
				max = diff;

			if (diff_relative < min_relative)
				min_relative = diff_relative;
			if (diff_relative > max_relative)
				max_relative = diff_relative;

			sum += diff;
			sum_relative += diff_relative;

			count++;
		}
	}

	float avg = sum / count;
	float avg_relative = sum_relative / count;

	cout << max << "\t" << avg << "\t" << min << "\t" << max_relative << "\t" << avg_relative << "\t" << min_relative << endl;

	return;
}

void diff(float *A, MYINT *B, MYINT scale, MYINT I, MYINT J, MYINT K)
{

	float min = numeric_limits<float>::max(), max = 0, sum = 0;
	float min_relative = numeric_limits<float>::max(), max_relative = 0, sum_relative = 0;
	int count = 0;

	for (MYINT i = 0; i < I; i++)
	{
		for (MYINT j = 0; j < J; j++)
		{
			for (MYINT k = 0; k < K; k++)
			{
				float a = A[i * J * K + j * K + k];

				MYINT b = B[i * J * K + j * K + k];
				//float b_float = float(b) / scale;
				float b_float = float(ldexp(double(b), scale));

				float diff = abs(a - b_float);
				float diff_relative = diff / abs(a);

				if (diff < min)
					min = diff;
				if (diff > max)
					max = diff;

				if (diff_relative < min_relative)
					min_relative = diff_relative;
				if (diff_relative > max_relative)
					max_relative = diff_relative;

				sum += diff;
				sum_relative += diff_relative;

				count++;
			}
		}
	}

	float avg = sum / count;
	float avg_relative = sum_relative / count;

	cout << max << "\t" << avg << "\t" << min << "\t" << max_relative << "\t" << avg_relative << "\t" << min_relative << endl;

	return;
}
