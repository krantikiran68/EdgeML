// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#pragma once

int seedotFixed(MYINT **X);
int seedotFloat(float **X);
void seedotFixedSwitch(int i, MYINT** X, int& res);

extern const int switches;