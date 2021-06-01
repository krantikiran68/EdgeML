// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#pragma once

#define INT16
typedef int64_t ACINT;
typedef int16_t MYINT;
typedef int16_t MYITE;
typedef uint16_t MYUINT;

const int scaleForX = -12;

const bool debugMode = false;

const bool logProgramOutput = false;

const int scalesForX[16] = {-12, -12, -12, -12, -12, -12, -12, -12, -12, -12, -12, -12};

const int scaleForY = 0;

const int scalesForY[16] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

const float scaleXZeroSkew = 0.0002;

const int zeroPointXZeroSkew = 72;

const float scalesXZeroSkew[1] = {0.0002};

const int zeroPointsXZeroSkew[1] = {72}; 

const float scaleYZeroSkew = 0.0002;

const int zeroPointYZeroSkew = 72;


const float scalesYZeroSkew[1] = {1.0};

const int zeroPointsYZeroSkew[1] = {0}; 

//#define SATURATE
//#define FASTAPPROX
//#define FLOATEXP
