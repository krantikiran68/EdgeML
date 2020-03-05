// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#pragma once

// Sketch operates in two modes.
// 1. Accuracy mode: Each data point is sent to the device using serial communication for prediction.
//    The predicted class label is sent back to the master
// 2. Prediction time mode: A single data point is used to compute the prediction time per data point.
//    This data point is placed in the flash memory of the device.
//
// Uncomment the below #define to choose the mode. Default is the prediction time mode.


//#define ACCURACY
#define PREDICTIONTIME


// The datatype of the fixed-point representation is specified below.
// The selection below should be equal to the Common.wordLength variable in Common.py
// Uncomment the below #define to choose the datatype. Default is INT16.

//define INT8
#define INT16
//#define INT32

#define XFLOAT
//#define XINT8
//#define XINT16

#ifdef INT8
typedef int8_t MYINT;
typedef uint8_t MYUINT;
#endif

#ifdef INT16
typedef int16_t MYINT;
typedef uint16_t MYUINT;
#endif

#ifdef INT32
typedef int32_t MYINT;
typedef uint32_t MYUINT;
#endif

typedef int16_t MYITE;
