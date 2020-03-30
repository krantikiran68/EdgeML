// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <cstring>
#include <cmath>
#include <cstdlib>
#include <thread>

#include "datatypes.h"
#include "predictors.h"
#include "profile.h"

using namespace std;

enum Version
{
	Fixed,
	Float
};
enum DatasetType
{
	Training,
	Testing
};

bool profilingEnabled = false;

// Split the CSV row into multiple values
vector<string> readCSVLine(string line)
{
	vector<string> tokens;

	stringstream stream(line);
	string str;

	while (getline(stream, str, ','))
		tokens.push_back(str);

	return tokens;
}

vector<string> getFeatures(string line)
{
	static int featuresLength = -1;

	vector<string> features = readCSVLine(line);

	if (featuresLength == -1)
		featuresLength = (int)features.size();

	if ((int)features.size() != featuresLength)
		throw "Number of row entries in X is inconsistent";

	return features;
}

int getLabel(string line)
{
	static int labelLength = -1;

	vector<string> labels = readCSVLine(line);

	if (labelLength == -1)
		labelLength = (int)labels.size();

	if ((int)labels.size() != labelLength || labels.size() != 1)
		throw "Number of row entries in Y is inconsistent";

	return (int)atoi(labels.front().c_str());
}

void populateFixedVector(MYINT **features_int, vector<string> features, int scale)
{
	int features_size = (int)features.size();

	for (int i = 0; i < features_size; i++)
	{
		double f = (double)(atof(features.at(i).c_str()));
		double f_int = ldexp(f, -scale);
		features_int[i][0] = (MYINT)(f_int);
	}

	return;
}

void populateFloatVector(float **features_float, vector<string> features)
{
	int features_size = (int)features.size();
	for (int i = 0; i < features_size; i++)
		features_float[i][0] = (float)(atof(features.at(i).c_str()));
	return;
}

void launchThread(int features_size, MYINT **features_int, MYINT *** features_intV, float **features_float, int counter, int* float_res, int* res, int* resV) {
	*res = seedotFixed(features_int);
	*float_res = seedotFloat(features_float);

	for (int i = 0; i < switches; i++) {
		seedotFixedSwitch(i, features_intV[i], resV[i]);
	}

	for(int i = 0; i < features_size; i++) {
		delete features_int[i];
		delete features_float[i];
		for (int j = 0; j < switches; j++) {
			delete features_intV[j][i];
		}
	}	
	delete[] features_int;
	delete[] features_float;
	for (int j = 0; j < switches; j++) {
		delete[] features_intV[j];
	}
	delete[] features_intV;
}

int main(int argc, char *argv[])
{
	if (argc == 1)
	{
		cout << "No arguments supplied" << endl;
		return 1;
	}

	Version version;
	if (strcmp(argv[1], "fixed") == 0)
		version = Fixed;
	else if (strcmp(argv[1], "float") == 0)
		version = Float;
	else
	{
		cout << "Argument mismatch for version\n";
		return 1;
	}
	string versionStr = argv[1];

	DatasetType datasetType;
	if (strcmp(argv[2], "training") == 0)
		datasetType = Training;
	else if (strcmp(argv[2], "testing") == 0)
		datasetType = Testing;
	else
	{
		cout << "Argument mismatch for dataset type\n";
		return 1;
	}
	string datasetTypeStr = argv[2];

	// Reading the dataset
	string inputDir = "input/";

	ifstream featuresFile(inputDir + "X.csv");
	ifstream lablesFile(inputDir + "Y.csv");

	if (featuresFile.good() == false || lablesFile.good() == false)
		throw "Input files doesn't exist";

	// Create output directory and files
	string outputDir = "output/" + versionStr;

	string outputFile = outputDir + "/prediction-info-" + datasetTypeStr + ".txt";
	string statsFile = outputDir + "/stats-" + datasetTypeStr + ".txt";

	ofstream output(outputFile);
	ofstream stats(statsFile);

	bool alloc = false;
	int features_size = -1;
	MYINT **features_int = NULL;
	vector<MYINT **> features_intV(switches, NULL);
	float **features_float = NULL;

	// Initialize variables used for profiling
	initializeProfiling();

	vector<int*> vector_float_res;
	vector<int*> vector_int_res;
	vector<int> labels;
	vector<int*> vector_int_resV;
	vector<thread> threads;

	MYINT*** features_intV_copy;

	string line1, line2;
	int counter = 0;

	if(version == Float)
		profilingEnabled = true;

	while (getline(featuresFile, line1) && getline(lablesFile, line2))
	{
		// Read the feature vector and class ID
		vector<string> features = getFeatures(line1);
		int label = getLabel(line2);

		// Allocate memory to store the feature vector as arrays
		if (alloc == false)
		{
			features_size = (int)features.size();

			features_int = new MYINT *[features_size];
			for (int i = 0; i < features_size; i++)
				features_int[i] = new MYINT[1];

			for (int i = 0; i < switches; i++) {
				features_intV[i] = new MYINT *[features_size];
				for (int j = 0; j < features_size; j++)
					features_intV[i][j] = new MYINT[1];
			}

			features_float = new float *[features_size];
			for (int i = 0; i < features_size; i++)
				features_float[i] = new float[1];

			alloc = true;
		}

		// Populate the array using the feature vector
		if (debugMode || version == Fixed)
		{
			populateFixedVector(features_int, features, scaleForX);
			for (int i = 0; i < switches; i++) {
				populateFixedVector(features_intV[i], features, scalesForX[i]);
			}
			populateFloatVector(features_float, features);
		}
		else
			populateFloatVector(features_float, features);

		// Invoke the predictor function
		int res = -1, float_res = -1;
		vector <int> resV(switches, -1);

		if (debugMode)
		{
			int* res_float = new int(seedotFloat(features_float));
			int* res_fixed = new int(seedotFixed(features_int));
			//debug();
			res = *res_fixed;
			vector_float_res.push_back(res_float);
			vector_int_res.push_back(res_fixed);
			labels.push_back(label);
			vector_int_resV.push_back(NULL);
		}
		else
		{
			if (version == Fixed) {
				vector_float_res.push_back(new int(-1));
				vector_int_res.push_back(new int(-1));
				labels.push_back(label);
				int* switchRes = new int[switches];
				for(int i = 0; i < switches; i++) {
					switchRes[i] = -1;
				}
				vector_int_resV.push_back(switchRes);
				MYINT** features_int_copy = new MYINT*[features_size];
				for(int i = 0; i < features_size; i++) {
					features_int_copy[i] = new MYINT[1];
					features_int_copy[i][0] = features_int[i][0];
				}
				float** features_float_copy = new float*[features_size];
				for(int i = 0; i < features_size; i++) {
					features_float_copy[i] = new float[1];
					features_float_copy[i][0] = features_float[i][0];
				}
				features_intV_copy = new MYINT**[switches];
				for(int j = 0; j < switches; j++) {
					features_intV_copy[j] = new MYINT*[features_size];
					for(int i = 0; i < features_size; i++) {
						features_intV_copy[j][i] = new MYINT[1];
						features_intV_copy[j][i][0] = features_intV[j][i][0];
					}
				}
				threads.push_back(thread(launchThread, features_size, features_int_copy, features_intV_copy, features_float_copy, counter, vector_float_res.back(), vector_int_res.back(), vector_int_resV.back()));
			}
			else if (version == Float) {
				res = seedotFloat(features_float);
				vector_float_res.push_back(new int(res));
				vector_int_res.push_back(new int(-1));
				labels.push_back(label);
				vector_int_resV.push_back(NULL);
			}
		}

		if(!logProgramOutput) {
			output << "Inputs handled = " << counter + 1 << endl;
		}

		flushProfile();
		counter ++;
	}

	for(int i = 0; i < threads.size(); i++) {
		threads[i].join();
	}


	int disagreements = 0, reduced_disagreements = 0;

	vector<int> correctV(switches, 0), totalV(switches, 0);
	vector<int> disagreementsV(switches, 0), reduced_disagreementsV(switches, 0);

	int correct = 0, total = 0;
	for(int i = 0; i < counter; i++) {
		int res = *vector_int_res[i];
		int float_res = *vector_float_res[i];
		int *resV = vector_int_resV[i];
		int label = labels[i];

		if(version == Float)
			res = float_res;

		if (res != float_res) {
			if (float_res == label) {
				reduced_disagreements++;
			}
			disagreements++;
		}

		if (res == label)
		{
			correct++;
		}
		else
		{
			if(logProgramOutput)
				output << "Incorrect prediction for input " << total + 1 << ". Predicted " << res << " Expected " << label << endl;
		}
		total++;

		for (int i = 0; i < switches; i++) {

			if (resV[i] != float_res) {
				if (float_res == label) {
					reduced_disagreementsV[i]++;
				}
				disagreementsV[i]++;
			}

			if (resV[i] == label)
			{
				correctV[i]++;
			}
			else
			{
				if(logProgramOutput)
					output << "Incorrect prediction for input " << totalV[i] + 1 << ". Predicted " << resV[i] << " Expected " << label << endl;
			}
			totalV[i]++;
		}

		delete vector_int_res[i];
		delete vector_float_res[i];
		delete[] vector_int_resV[i];
	}

	// Deallocate memory
	for (int i = 0; i < features_size; i++)
		delete features_int[i];
	delete[] features_int;

	for (int i = 0; i < features_size; i++)
		delete features_float[i];
	delete[] features_float;

	for (int i = 0; i < switches; i++) {
		for (int j = 0; j < features_size; j++)
			delete features_intV[i][j];
		delete[] features_intV[i];
	}

	float accuracy = (float)correct / total * 100.0f;

	cout.precision(3);
	cout << fixed;
	cout << "\n\n#test points = " << total << endl;
	cout << "Correct predictions = " << correct << endl;
	cout << "Accuracy = " << accuracy << "\n\n";

	output.precision(3);
	output << fixed;
	output << "\n\n#test points = " << total << endl;
	output << "Correct predictions = " << correct << endl;
	output << "Accuracy = " << accuracy << "\n\n";
	output.close();

	stats.precision(3);
	stats << fixed;
	stats << "default" << "\n";
	stats << accuracy << "\n";
	stats << disagreements << "\n";
	stats << reduced_disagreements << "\n";

	if (version == Fixed) 
	{
		for (int i = 0; i < switches; i++) 
		{
			stats << i+1 << "\n";
			stats << (float)correctV[i] / totalV[i] * 100.0f << "\n";
			stats << disagreementsV[i] << "\n";
			stats << reduced_disagreementsV[i] << "\n";
		}
	}

	stats.close();

	if (version == Float)
		dumpProfile();

	if (datasetType == Training)
		dumpRange(outputDir + "/profile.txt");

	return 0;
}
