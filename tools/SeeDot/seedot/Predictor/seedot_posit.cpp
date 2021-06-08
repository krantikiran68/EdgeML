#include <iostream>
#include <cstring>
#include <cmath>

#include "datatypes.h"
#include "predictors.h"
#include "profile.h"
#include "library_posit.h"
#include "model_posit.h"
#include "vars_posit.h"

using namespace std;
using namespace seedot_posit;
void seedotPosit(float **X_temp, int32_t* res) {
}

const int positSwitches = 0;

void seedotPositSwitch(int i, float** X_temp, int32_t* res) {
  switch(i) {
    default: res[0] = -1;
             return;
  }
}
