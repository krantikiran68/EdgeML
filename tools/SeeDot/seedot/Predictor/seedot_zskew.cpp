#include <iostream>
#include <cstring>
#include <cmath>

#include "datatypes.h"
#include "predictors.h"
#include "profile.h"
#include "library_zskew.h"
#include "model_zskew.h"
#include "vars_zskew.h"

using namespace std;
using namespace seedot_zskew;
void seedotZeroSkew(MYINT **X, int32_t* res) {
	*res = 0;
	return;
}

const int zSwitches = 0;

void seedotZeroSkewSwitch(int i, MYINT** X_temp, int32_t* res) {
  switch(i) {
    default: res[0] = -1;
             return;
  }
}

