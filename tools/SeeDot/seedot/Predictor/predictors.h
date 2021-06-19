// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#pragma once

void seedotFixed(MYINT** X, int32_t* res);
void seedotPosit(float** X, float* res);
void seedotFloat(float** X, float* res);
void seedotFixedSwitch(int i, MYINT** X, int32_t* res);
void seedotPositSwitch(int i, float** X, float* res);

extern const int switches;
extern const int positSwitches;
extern int switchCount;
