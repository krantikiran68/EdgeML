// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#pragma once

void seedotFixed(MYINT** X, int32_t* res);
void seedotFloat(FP_TYPE** X, FP_TYPE* res);
void seedotFixedSwitch(int i, MYINT** X, int32_t* res);

extern const int switches;
