// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include <iostream>
#include <fstream>
#include "library_zskew.h"
#include "datatypes.h"
#include <bits/stdc++.h>
#include <cassert>

using namespace std;
int clam_limit = 126;
int right_shift = 31;
int bitwidth = 32;
float getScale(float *a, int I)
{
    float min = a[0];
    float max = a[0];
    for(int i=1;i<I;i++)
    {
        min = min > a[i]?a[i]:min;
        max = max < a[i]?a[i]:max;
    }
    cout<<"max, min "<< min<< ", "<<max<<endl;
    float scale = (max - min)/(clam_limit*2);
    return scale;
}

int getZero(float *A, int I, float scale)
{
    float min = A[0];
    float max = A[0];
    for(int i=1;i<I;i++)
    {
        min = min > A[i]?A[i]:min;
        max = max < A[i]?A[i]:max;
    }
    float zero = -1*(min + max) / 2;
    zero = zero/scale;
    return int(zero);
}

void getZeroSkewArray(float *A, int I, MYINT* B, float scale, int zero)
{
    for(int i=0;i<I;i++)
    {
        B[i] = int(A[i]/scale) + zero;
    }
}

void getQuantizedMultiplierLTO(float m, int bitwidth, ACINT& M, MYITE& N)
{
    cout<<"m "<<m<<endl;
    float m_ = log2(m);
    cout<<"m_ "<<m_<<endl;

    
    int m_int = int(m_);
    if (fabs(m_ - round(m_))  < 0.00000000000001 )
    {
        m_int = round(m_) + 1;
    }
    else
    {
        m_int = ceil(m_);
    }
    std::cout<<m_int << ", " << int(m_)<<endl;

    
    std::cout<<m_int<<endl;

    int scale = -1*(bitwidth - 1 - m_int);
    assert(scale <= 0);
    float exp;
    if(scale > 64)
    {
        exp =  m * float(1LL << (-(scale/2)));
        exp *= float(1LL << (-(scale - (scale/2))));
    }
    else
    {
        exp = m * float(1LL << (-(scale)));
    }

    cout<<exp << ", " << float(1LL << (-scale)) <<endl;

    M = ACINT(exp);
    cout<<M<<endl;
    cout<<m<<endl;
    cout<<scale<<endl;
    N = -scale;
}

void getScaleAndZeroForAddorSub(float scaleA, ACINT zeroA, float  scaleB, ACINT zeroB, float  scaleC, ACINT zeroC, ACINT& m1, MYITE& n1, ACINT& m2, MYITE& n2, ACINT& m3, MYITE& n3)
{
    float s3 = 1/scaleC;
    getQuantizedMultiplierLTO(scaleA, bitwidth, m1, n1);
    getQuantizedMultiplierLTO(scaleB, bitwidth, m2, n2);
    getQuantizedMultiplierLTO(s3, bitwidth, m3, n3);

    n1 -= right_shift;
    n2 -= right_shift;
    n3 -= right_shift;
}
void test_MatAdd()
{
    ACINT left_shit = 0;

    int I = 10;
    int J = 5;

    float A_[50] = {-6.697373993405011, -4.007664222081393, -0.31776172891011, -2.8401345538930194, -0.7118823524317559, -4.988148691534757, 8.383677342098341, 2.0274583853521797, 7.795551165584875, -1.8101323150070492, -7.389897886085828, -9.49440018385403, -6.409200938898129, 5.0913711783916344, -6.418038608572328, 0.9756384844831167, 6.6971631695377205, -5.127905500900541, -8.355524302927614, -1.7042096216437326, 3.8870330553238546, 0.4422296735976978, -4.953386772623974, -3.8662991359007766, 0.560960830648586, -0.6893724269609294, 1.2986523627051803, 4.690635943590307, 7.148931054894703, -1.8778305503692732, 4.250002849983739, 6.012257243013252, -3.707826369795024, 1.47019580413534, 0.6742013407090912, -3.3810424334646916, -5.042774582299961, 2.711417287173223, -4.43605713643739, 2.61543457104972, 1.8448067786309963, 8.485714082662362, -6.652463705159968, 0.5102530093489843, 5.342176313927446, 4.232384114071422, 7.520847467781117, 3.771220873891398, -2.815750902308407, -5.850184028104975};
    float B_[50] = {7.756451167414895, 7.333002590956523, -0.048726320020087144, 0.7714985421337728, -3.3854080890311744, 10.848370884467876, 1.7142799120375702, 6.453061018933472, -1.9626191979244823, 10.231102876277953, 5.212509368380733, 11.735465509846733, -2.0834535277420163, -2.6150803359123787, 1.7886297079225848, 3.34798758447816, 11.165349725652714, -4.41185131784583, 4.687017496229204, 11.767612334756222, 9.06322668739003, 5.460507064107764, -3.729517843303145, 13.62854404341845, -2.161953889740711, 6.247579952566117, 8.506664185985125, -4.80717292179005, -3.749053202177546, -1.3464950675311305, 9.851083528552403, 9.877619569914458, -1.25733870092682, 12.927703725344148, -2.1460421046296796, 10.69441778045486, 8.378915169143815, -1.7920656104098809, 6.710800961543814, -2.2194166118158742, 1.8195425938133747, 1.2618378053202184, 8.738742757913396, 5.753949892234642, 7.268285334412539, 1.859581490639366, 12.90859161216667, 1.1720542739996587, 0.8717967286137789, -3.7671524526208877};
    float C_exptd[50] = {1.0590771740098832, 3.32533836887513, -0.36648804893019715, -2.0686360117592466, -4.09729044146293, 5.860222192933119, 10.097957254135911, 8.480519404285651, 5.832931967660393, 8.420970561270904, -2.1773885177050953, 2.241065325992704, -8.492654466640145, 2.476290842479256, -4.629408900649743, 4.323626068961277, 17.862512895190434, -9.53975681874637, -3.66850680669841, 10.06340271311249, 12.950259742713884, 5.902736737705462, -8.682904615927118, 9.762244907517674, -1.6009930590921249, 5.558207525605187, 9.805316548690305, -0.1165369781997434, 3.3998778527171574, -3.2243256179004036, 14.101086378536142, 15.88987681292771, -4.965165070721843, 14.397899529479488, -1.4718407639205884, 7.313375346990168, 3.336140586843854, 0.9193516767633421, 2.274743825106424, 0.39601795923384575, 3.664349372444371, 9.747551887982581, 2.0862790527534285, 6.2642029015836265, 12.610461648339985, 6.091965604710788, 20.429439079947787, 4.943275147891057, -1.9439541736946282, -9.617336480725863};

    MYINT* A = new MYINT[50];
    MYINT* B = new MYINT[50];
    MYINT* C = new MYINT[50];

    
    float scale1 = getScale(A_, I*J);
    int zero1 = getZero(A_, I*J, scale1);
    
    float scale2 = getScale(B_, I*J);
    int zero2 = getZero(B_, I*J, scale2);

    float scale3 = getScale(C_exptd, I*J);
    int zero3 = getZero(C_exptd, I*J, scale3);
    
    
    getZeroSkewArray(A_, I*J, A, scale1, zero1);
    getZeroSkewArray(B_, I*J, B, scale2, zero2);

    ACINT m1, m2, m3;
    MYITE n1, n2, n3;
    getScaleAndZeroForAddorSub(scale1, zero1, scale2, zero2, scale3, zero3, m1, n1, m2, n2, m3, n3);
    MatAdd(A, B, C, I, J, 0, -zero1, m1, -n1, -zero2, m2, -n2, zero3, m3, -n3, -clam_limit, clam_limit);
    std::cout<<m1 << ", " << n1 << ", " << m2 << ", "<< n2 << ", " << m3 << ", " << n3 << std::endl;
    int max_err = numeric_limits<MYINT>::min();
    float max_err_flt = numeric_limits<float>::min();
    cout<<max_err_flt<<endl;
    float max_err_relative_flt = float(max_err_flt);
    for(int i=0;i<I*J;i++)
    {
        float flt_err = fabs(C_exptd[i] - (scale3*(float(C[i])-float(zero3))));
        int err = abs(C[i] - MYINT(int(C_exptd[i]/scale3) + zero3));
        max_err_flt = max_err_flt < flt_err? flt_err: max_err_flt;
        max_err = max_err < err ? err: max_err;
        float relative_error = (flt_err)/fabs(C_exptd[i]);
        max_err_relative_flt = max_err_relative_flt < relative_error?relative_error:max_err_relative_flt;
        std::cout<< int(C[i]) << ", "<< int(MYINT(int(C_exptd[i]/scale3) + zero3)) << ", " << (scale3*float(float(C[i])-float(zero3))) <<", " << C_exptd[i] << ", " << abs(C[i] - MYINT(int(C_exptd[i]/scale3) + zero3))<< ", " <<fabs(C_exptd[i] - (scale3*(C[i]-zero3)))<<", " << relative_error<<std::endl;
    }
    std::cout<<max_err << ", "<<max_err_flt<< ", " << max_err_relative_flt<<endl;
    std::cout<<scale1 << ", "<<zero1<<endl;
    std::cout<<scale2 << ", "<<zero2<<endl;
    std::cout<<scale3 << ", "<<zero3<<endl;

}

void test_Sigmoid()
{
    float A_[50] = {-6.697373993405011, -4.007664222081393, -0.31776172891011, -2.8401345538930194, -0.7118823524317559, -4.988148691534757, 8.383677342098341, 2.0274583853521797, 7.795551165584875, -1.8101323150070492, -7.389897886085828, -9.49440018385403, -6.409200938898129, 5.0913711783916344, -6.418038608572328, 0.9756384844831167, 6.6971631695377205, -5.127905500900541, -8.355524302927614, -1.7042096216437326, 3.8870330553238546, 0.4422296735976978, -4.953386772623974, -3.8662991359007766, 0.560960830648586, -0.6893724269609294, 1.2986523627051803, 4.690635943590307, 7.148931054894703, -1.8778305503692732, 4.250002849983739, 6.012257243013252, -3.707826369795024, 1.47019580413534, 0.6742013407090912, -3.3810424334646916, -5.042774582299961, 2.711417287173223, -4.43605713643739, 2.61543457104972, 1.8448067786309963, 8.485714082662362, -6.652463705159968, 0.5102530093489843, 5.342176313927446, 4.232384114071422, 7.520847467781117, 3.771220873891398, -2.815750902308407, -5.850184028104975};
    
    MYINT* A = new MYINT[50];
    MYINT* C = new MYINT[50];
    float C_exptd[50];
    int I=10;
    int J=5;
    
    for(int i=0;i<50;i++)
    {
        C_exptd[i] = 1 / (1 + exp(-1*A_[i]));
    }

    
    float scale1 = getScale(A_, I*J);
    int zero1 = getZero(A_, I*J, scale1);
    
    getZeroSkewArray(A_, I*J, A, scale1, zero1);
    float scale2 = getScale(C_exptd, I*J);
    int zero2 = getZero(C_exptd, I*J, scale2);

    ACINT m1, m2;
    MYITE n1, n2;
    cout<<scale1 << ", "<<zero1<<endl;
    getQuantizedMultiplierLTO(scale1, 32, m1, n1);
    n1 -= (31 + 27);

    getQuantizedMultiplierLTO(1/scale2, 32, m2, n2);
    n2 -= 23;
    cout<< "vals " <<m1 << ", "<<n1<<endl;
    Sigmoid(A, C, I, J, -zero1, m1, -n1, zero2, m2, -n2, clam_limit);

    int max_err = numeric_limits<MYINT>::min();
    float max_err_flt = numeric_limits<float>::min();
    cout<<max_err_flt<<endl;
    float max_err_relative_flt = float(max_err_flt);
    for(int i=0;i<I*J;i++)
    {
        float flt_err = fabs(C_exptd[i] - (scale2*(float(C[i])-float(zero2))));
        int err = abs(C[i] - MYINT(int(C_exptd[i]/scale2) + zero2));
        max_err_flt = max_err_flt < flt_err? flt_err: max_err_flt;
        max_err = max_err < err ? err: max_err;
        float relative_error = (flt_err)/fabs(C_exptd[i]);
        max_err_relative_flt = max_err_relative_flt < relative_error?relative_error:max_err_relative_flt;
        // std::cout<< int(C[i]) << ", "<< int(MYINT(int(C_exptd[i]/scale2) + zero2)) << ", " << (scale2*float(float(C[i])-float(zero2))) <<", " << C_exptd[i] << ", " << abs(C[i] - MYINT(int(C_exptd[i]/scale2) + zero2))<< ", " <<fabs(C_exptd[i] - (scale2*(C[i]-zero2)))<<std::endl;
    }
    std::cout<<max_err << ", "<<max_err_flt<< ", " << max_err_relative_flt<<endl;
    std::cout<<scale2 << ", "<<zero2<<endl;



}

void test_TanH()
{
    float A_[50] = {7.795551165584875, -6.697373993405011, -4.007664222081393, -0.31776172891011, -2.8401345538930194, -0.7118823524317559, -4.988148691534757, 8.383677342098341, 2.0274583853521797, -1.8101323150070492, -7.389897886085828, -9.49440018385403, -6.409200938898129, 5.0913711783916344, -6.418038608572328, 0.9756384844831167, 6.6971631695377205, -5.127905500900541, -8.355524302927614, -1.7042096216437326, 3.8870330553238546, 0.4422296735976978, -4.953386772623974, -3.8662991359007766, 0.560960830648586, -0.6893724269609294, 1.2986523627051803, 4.690635943590307, 7.148931054894703, -1.8778305503692732, 4.250002849983739, 6.012257243013252, -3.707826369795024, 1.47019580413534, 0.6742013407090912, -3.3810424334646916, -5.042774582299961, 2.711417287173223, -4.43605713643739, 2.61543457104972, 1.8448067786309963, 8.485714082662362, -6.652463705159968, 0.5102530093489843, 5.342176313927446, 4.232384114071422, 7.520847467781117, 3.771220873891398, -2.815750902308407, -5.850184028104975};
    
    MYINT* A = new MYINT[50];
    MYINT* C = new MYINT[50];
    float C_exptd[50];
    int I=10;
    int J=5;
    
    for(int i=0;i<50;i++)
    {
        C_exptd[i] = tanh(A_[i]);
    }

    
    float scale1 = getScale(A_, I*J);
    int zero1 = getZero(A_, I*J, scale1);
    
    getZeroSkewArray(A_, I*J, A, scale1, zero1);
    float scale2 = getScale(C_exptd, I*J);
    int zero2 = getZero(C_exptd, I*J, scale2);

    ACINT m1, m2;
    MYITE n1, n2;
    cout<<scale1 << ", "<<zero1<<endl;
    getQuantizedMultiplierLTO(scale1, 32, m1, n1);
    n1 -= (31 + 27);

    getQuantizedMultiplierLTO(1/scale2, 32, m2, n2);
    cout<<n2 <<endl;
    n2 -= 24;
    cout<< "vals " <<m1 << ", "<<n1<<endl;
    cout<< "vals " <<m2 << ", "<<n2<<endl;

    TanH(A, C, I, J, -zero1, m1, -n1, zero2, m2, -n2, clam_limit);

    int max_err = numeric_limits<MYINT>::min();
    float max_err_flt = numeric_limits<float>::min();
    cout<<max_err_flt<<endl;
    float max_err_relative_flt = float(max_err_flt);
    for(int i=0;i<I*J;i++)
    {
        float flt_err = fabs(C_exptd[i] - (scale2*(float(C[i])-float(zero2))));
        int err = abs(C[i] - MYINT(int(C_exptd[i]/scale2) + zero2));
        max_err_flt = max_err_flt < flt_err? flt_err: max_err_flt;
        max_err = max_err < err ? err: max_err;
        float relative_error = (flt_err)/fabs(C_exptd[i]);
        max_err_relative_flt = max_err_relative_flt < relative_error?relative_error:max_err_relative_flt;
        std::cout<< int(C[i]) << ", "<< int(MYINT(int(C_exptd[i]/scale2) + zero2)) << ", " << (scale2*float(float(C[i])-float(zero2))) <<", " << C_exptd[i] << ", " << abs(C[i] - MYINT(int(C_exptd[i]/scale2) + zero2))<< ", " <<fabs(C_exptd[i] - (scale2*(C[i]-zero2)))<<std::endl;
    }
    std::cout<<max_err << ", "<<max_err_flt<< ", " << max_err_relative_flt<<endl;
    std::cout<<scale2 << ", "<<zero2<<endl;

}

void test_Hadamard()
{
    ACINT left_shit = 0;

    int I = 10;
    int J = 5;

    float A_[50] = {-6.697373993405011, -4.007664222081393, -0.31776172891011, -2.8401345538930194, -0.7118823524317559, -4.988148691534757, 8.383677342098341, 2.0274583853521797, 7.795551165584875, -1.8101323150070492, -7.389897886085828, -9.49440018385403, -6.409200938898129, 5.0913711783916344, -6.418038608572328, 0.9756384844831167, 6.6971631695377205, -5.127905500900541, -8.355524302927614, -1.7042096216437326, 3.8870330553238546, 0.4422296735976978, -4.953386772623974, -3.8662991359007766, 0.560960830648586, -0.6893724269609294, 1.2986523627051803, 4.690635943590307, 7.148931054894703, -1.8778305503692732, 4.250002849983739, 6.012257243013252, -3.707826369795024, 1.47019580413534, 0.6742013407090912, -3.3810424334646916, -5.042774582299961, 2.711417287173223, -4.43605713643739, 2.61543457104972, 1.8448067786309963, 8.485714082662362, -6.652463705159968, 0.5102530093489843, 5.342176313927446, 4.232384114071422, 7.520847467781117, 3.771220873891398, -2.815750902308407, -5.850184028104975};
    float B_[50] = {7.756451167414895, 7.333002590956523, -0.048726320020087144, 0.7714985421337728, -3.3854080890311744, 10.848370884467876, 1.7142799120375702, 6.453061018933472, -1.9626191979244823, 10.231102876277953, 5.212509368380733, 11.735465509846733, -2.0834535277420163, -2.6150803359123787, 1.7886297079225848, 3.34798758447816, 11.165349725652714, -4.41185131784583, 4.687017496229204, 11.767612334756222, 9.06322668739003, 5.460507064107764, -3.729517843303145, 13.62854404341845, -2.161953889740711, 6.247579952566117, 8.506664185985125, -4.80717292179005, -3.749053202177546, -1.3464950675311305, 9.851083528552403, 9.877619569914458, -1.25733870092682, 12.927703725344148, -2.1460421046296796, 10.69441778045486, 8.378915169143815, -1.7920656104098809, 6.710800961543814, -2.2194166118158742, 1.8195425938133747, 1.2618378053202184, 8.738742757913396, 5.753949892234642, 7.268285334412539, 1.859581490639366, 12.90859161216667, 1.1720542739996587, 0.8717967286137789, -3.7671524526208877};

    float C_exptd[50];

    for(int i=0;i<I*J;i++)
    {
        C_exptd[i] = A_[i] * B_[i];
    }
    
    MYINT* A = new MYINT[50];
    MYINT* B = new MYINT[50];
    MYINT* C = new MYINT[50];

    
    float scale1 = getScale(A_, I*J);
    int zero1 = getZero(A_, I*J, scale1);
    
    float scale2 = getScale(B_, I*J);
    int zero2 = getZero(B_, I*J, scale2);

    float scale3 = getScale(C_exptd, I*J);
    int zero3 = getZero(C_exptd, I*J, scale3);
    
    
    getZeroSkewArray(A_, I*J, A, scale1, zero1);
    getZeroSkewArray(B_, I*J, B, scale2, zero2);

    ACINT m1, m2, m3;
    MYITE n1, n2, n3;
    getQuantizedMultiplierLTO((scale1 * scale2)/scale3, bitwidth, m1, n1);

    n1 -= right_shift;
    Hadamard(A, B, C, I, J, -zero1, -zero2, zero3, m1, -n1, -clam_limit, clam_limit);

    int max_err = numeric_limits<MYINT>::min();
    float max_err_flt = numeric_limits<float>::min();
    cout<<max_err_flt<<endl;
    float max_err_relative_flt = float(max_err_flt);
    for(int i=0;i<I*J;i++)
    {
        float flt_err = fabs(C_exptd[i] - (scale3*(float(C[i])-float(zero3))));
        int err = abs(C[i] - MYINT(int(C_exptd[i]/scale3) + zero3));
        max_err_flt = max_err_flt < flt_err? flt_err: max_err_flt;
        max_err = max_err < err ? err: max_err;
        float relative_error = (flt_err)/fabs(C_exptd[i]);
        max_err_relative_flt = max_err_relative_flt < relative_error?relative_error:max_err_relative_flt;
        // std::cout<< int(C[i]) << ", "<< int(MYINT(int(C_exptd[i]/scale3) + zero3)) << ", " << (scale3*float(float(C[i])-float(zero3))) <<", " << C_exptd[i] << ", " << abs(C[i] - MYINT(int(C_exptd[i]/scale3) + zero3))<< ", " <<fabs(C_exptd[i] - (scale3*(C[i]-zero3)))<<std::endl;
    }
    std::cout<<max_err << ", "<<max_err_flt<< ", " << max_err_relative_flt<<endl;
    std::cout<<scale3 << ", "<<zero3<<endl;
    
}

void test_ScalarMul()
{
    ACINT left_shit = 0;

    int I = 10;
    int J = 5;

    float A_[50] = {-6.697373993405011, -4.007664222081393, -0.31776172891011, -2.8401345538930194, -0.7118823524317559, -4.988148691534757, 8.383677342098341, 2.0274583853521797, 7.795551165584875, -1.8101323150070492, -7.389897886085828, -9.49440018385403, -6.409200938898129, 5.0913711783916344, -6.418038608572328, 0.9756384844831167, 6.6971631695377205, -5.127905500900541, -8.355524302927614, -1.7042096216437326, 3.8870330553238546, 0.4422296735976978, -4.953386772623974, -3.8662991359007766, 0.560960830648586, -0.6893724269609294, 1.2986523627051803, 4.690635943590307, 7.148931054894703, -1.8778305503692732, 4.250002849983739, 6.012257243013252, -3.707826369795024, 1.47019580413534, 0.6742013407090912, -3.3810424334646916, -5.042774582299961, 2.711417287173223, -4.43605713643739, 2.61543457104972, 1.8448067786309963, 8.485714082662362, -6.652463705159968, 0.5102530093489843, 5.342176313927446, 4.232384114071422, 7.520847467781117, 3.771220873891398, -2.815750902308407, -5.850184028104975};
    float B_[50] = {7.756451167414895, 7.333002590956523, -0.048726320020087144, 0.7714985421337728, -3.3854080890311744, 10.848370884467876, 1.7142799120375702, 6.453061018933472, -1.9626191979244823, 10.231102876277953, 5.212509368380733, 11.735465509846733, -2.0834535277420163, -2.6150803359123787, 1.7886297079225848, 3.34798758447816, 11.165349725652714, -4.41185131784583, 4.687017496229204, 11.767612334756222, 9.06322668739003, 5.460507064107764, -3.729517843303145, 13.62854404341845, -2.161953889740711, 6.247579952566117, 8.506664185985125, -4.80717292179005, -3.749053202177546, -1.3464950675311305, 9.851083528552403, 9.877619569914458, -1.25733870092682, 12.927703725344148, -2.1460421046296796, 10.69441778045486, 8.378915169143815, -1.7920656104098809, 6.710800961543814, -2.2194166118158742, 1.8195425938133747, 1.2618378053202184, 8.738742757913396, 5.753949892234642, 7.268285334412539, 1.859581490639366, 12.90859161216667, 1.1720542739996587, 0.8717967286137789, -3.7671524526208877};

    float C_exptd[50];

    for(int i=0;i<I*J;i++)
    {
        C_exptd[i] = A_[0] * B_[i];
    }
    
    MYINT* A = new MYINT[1];
    MYINT* B = new MYINT[50];
    MYINT* C = new MYINT[50];

    
    float m_ = log2(A_[0]);
    
    int m_int = int(m_);
    if (fabs(m_ - round(m_))  < 0.00000000000001 )
    {
        m_int = round(m_) + 1;
    }
    else
    {
        m_int = ceil(m_);
    }

    int scale = -1*(31 - m_int);

    float scale1 = (A_[0] > 1.0)? fabs(float(1) / float(A_[0])): fabs(float(A_[0]));
    int zero1 = 0;

    float scale2 = getScale(B_, I*J);
    int zero2 = getZero(B_, I*J, scale2);

    float scale3 = getScale(C_exptd, I*J);
    int zero3 = getZero(C_exptd, I*J, scale3);
    
    
    getZeroSkewArray(A_, 1, A, scale1, zero1);
    getZeroSkewArray(B_, I*J, B, scale2, zero2);

    ACINT m1, m2, m3;
    MYITE n1, n2, n3;

    cout<<scale1 << ", "<<scale2 << ", "<< scale3<<endl;

    getQuantizedMultiplierLTO((scale1 * scale2)/scale3, bitwidth, m1, n1);
    n1 -= right_shift;

    MatMulBroadcastA(A, B, C, I, J, -zero1, -zero2, zero3, m1, -n1, -clam_limit, clam_limit);

    int max_err = numeric_limits<MYINT>::min();
    float max_err_flt = numeric_limits<float>::min();
    cout<<max_err_flt<<endl;
    float max_err_relative_flt = float(max_err_flt);
    for(int i=0;i<I*J;i++)
    {
        float flt_err = fabs(C_exptd[i] - (scale3*(float(C[i])-float(zero3))));
        int err = abs(C[i] - MYINT(int(C_exptd[i]/scale3) + zero3));
        max_err_flt = max_err_flt < flt_err? flt_err: max_err_flt;
        max_err = max_err < err ? err: max_err;
        float relative_error = (flt_err)/fabs(C_exptd[i]);
        max_err_relative_flt = max_err_relative_flt < relative_error?relative_error:max_err_relative_flt;
        // std::cout<< int(C[i]) << ", "<< int(MYINT(int(C_exptd[i]/scale3) + zero3)) << ", " << (scale3*float(float(C[i])-float(zero3))) <<", " << C_exptd[i] << ", " << abs(C[i] - MYINT(int(C_exptd[i]/scale3) + zero3))<< ", " <<fabs(C_exptd[i] - (scale3*(C[i]-zero3)))<<std::endl;
    }
    std::cout<<max_err << ", "<<max_err_flt<< ", " << max_err_relative_flt<<endl;
    std::cout<<scale3 << ", "<<zero3<<endl;    
}

void test_MatSub()
{
    ACINT left_shit = 0;

    int I = 10;
    int J = 5;

    float A_[50] = {-6.697373993405011, -4.007664222081393, -0.31776172891011, -2.8401345538930194, -0.7118823524317559, -4.988148691534757, 8.383677342098341, 2.0274583853521797, 7.795551165584875, -1.8101323150070492, -7.389897886085828, -9.49440018385403, -6.409200938898129, 5.0913711783916344, -6.418038608572328, 0.9756384844831167, 6.6971631695377205, -5.127905500900541, -8.355524302927614, -1.7042096216437326, 3.8870330553238546, 0.4422296735976978, -4.953386772623974, -3.8662991359007766, 0.560960830648586, -0.6893724269609294, 1.2986523627051803, 4.690635943590307, 7.148931054894703, -1.8778305503692732, 4.250002849983739, 6.012257243013252, -3.707826369795024, 1.47019580413534, 0.6742013407090912, -3.3810424334646916, -5.042774582299961, 2.711417287173223, -4.43605713643739, 2.61543457104972, 1.8448067786309963, 8.485714082662362, -6.652463705159968, 0.5102530093489843, 5.342176313927446, 4.232384114071422, 7.520847467781117, 3.771220873891398, -2.815750902308407, -5.850184028104975};
    float B_[50] = {7.756451167414895, 7.333002590956523, -0.048726320020087144, 0.7714985421337728, -3.3854080890311744, 10.848370884467876, 1.7142799120375702, 6.453061018933472, -1.9626191979244823, 10.231102876277953, 5.212509368380733, 11.735465509846733, -2.0834535277420163, -2.6150803359123787, 1.7886297079225848, 3.34798758447816, 11.165349725652714, -4.41185131784583, 4.687017496229204, 11.767612334756222, 9.06322668739003, 5.460507064107764, -3.729517843303145, 13.62854404341845, -2.161953889740711, 6.247579952566117, 8.506664185985125, -4.80717292179005, -3.749053202177546, -1.3464950675311305, 9.851083528552403, 9.877619569914458, -1.25733870092682, 12.927703725344148, -2.1460421046296796, 10.69441778045486, 8.378915169143815, -1.7920656104098809, 6.710800961543814, -2.2194166118158742, 1.8195425938133747, 1.2618378053202184, 8.738742757913396, 5.753949892234642, 7.268285334412539, 1.859581490639366, 12.90859161216667, 1.1720542739996587, 0.8717967286137789, -3.7671524526208877};
    float C_exptd[50];

    MYINT* A = new MYINT[50];
    MYINT* B = new MYINT[50];
    MYINT* C = new MYINT[50];

    for(int i=0;i<I*J;i++)
    {
        C_exptd[i] = A_[0] - B_[i];
    }

    
    float scale1 = (A_[0] > 1.0)? fabs(float(1) / float(A_[0])): fabs(float(A_[0]));
    int zero1 = 0;
    
    float scale2 = getScale(B_, I*J);
    int zero2 = getZero(B_, I*J, scale2);

    float scale3 = getScale(C_exptd, I*J);
    int zero3 = getZero(C_exptd, I*J, scale3);
    
    
    getZeroSkewArray(A_, 1, A, scale1, zero1);
    getZeroSkewArray(B_, I*J, B, scale2, zero2);

    ACINT m1, m2, m3;
    MYITE n1, n2, n3;
    getScaleAndZeroForAddorSub(scale1, zero1, scale2, zero2, scale3, zero3, m1, n1, m2, n2, m3, n3);
    MatSubBroadCastA(A, B, C, I, J, 0, -zero1, m1, -n1, -zero2, m2, -n2, zero3, m3, -n3, -clam_limit, clam_limit);
    // std::cout<<m1 << ", " << n1 << ", " << m2 << ", "<< n2 << ", " << m3 << ", " << n3 << std::endl;
    int max_err = numeric_limits<MYINT>::min();
    float max_err_flt = numeric_limits<float>::min();
    cout<<max_err_flt<<endl;
    float max_err_relative_flt = float(max_err_flt);
    for(int i=0;i<I*J;i++)
    {
        float flt_err = fabs(C_exptd[i] - (scale3*(float(C[i])-float(zero3))));
        int err = abs(C[i] - MYINT(int(C_exptd[i]/scale3) + zero3));
        max_err_flt = max_err_flt < flt_err? flt_err: max_err_flt;
        max_err = max_err < err ? err: max_err;
        float relative_error = (flt_err)/fabs(C_exptd[i]);
        max_err_relative_flt = max_err_relative_flt < relative_error?relative_error:max_err_relative_flt;
        // std::cout<< int(C[i]) << ", "<< int(MYINT(int(C_exptd[i]/scale3) + zero3)) << ", " << (scale3*float(float(C[i])-float(zero3))) <<", " << C_exptd[i] << ", " << abs(C[i] - MYINT(int(C_exptd[i]/scale3) + zero3))<< ", " <<fabs(C_exptd[i] - (scale3*(C[i]-zero3)))<<std::endl;
    }
    std::cout<<max_err << ", "<<max_err_flt<< ", " << max_err_relative_flt<<endl;
    std::cout<<scale3 << ", "<<zero3<<endl;

}

void test_MatMul()
{
    int I = 10;
    int K = 5;
    int J = 10;

    float A_[50] = {-6.697373993405011, -4.007664222081393, -0.31776172891011, -2.8401345538930194, -0.7118823524317559, -4.988148691534757, 8.383677342098341, 2.0274583853521797, 7.795551165584875, -1.8101323150070492, -7.389897886085828, -9.49440018385403, -6.409200938898129, 5.0913711783916344, -6.418038608572328, 0.9756384844831167, 6.6971631695377205, -5.127905500900541, -8.355524302927614, -1.7042096216437326, 3.8870330553238546, 0.4422296735976978, -4.953386772623974, -3.8662991359007766, 0.560960830648586, -0.6893724269609294, 1.2986523627051803, 4.690635943590307, 7.148931054894703, -1.8778305503692732, 4.250002849983739, 6.012257243013252, -3.707826369795024, 1.47019580413534, 0.6742013407090912, -3.3810424334646916, -5.042774582299961, 2.711417287173223, -4.43605713643739, 2.61543457104972, 1.8448067786309963, 8.485714082662362, -6.652463705159968, 0.5102530093489843, 5.342176313927446, 4.232384114071422, 7.520847467781117, 3.771220873891398, -2.815750902308407, -5.850184028104975};
    float B_[50] = {7.756451167414895, 7.333002590956523, -0.048726320020087144, 0.7714985421337728, -3.3854080890311744, 10.848370884467876, 1.7142799120375702, 6.453061018933472, -1.9626191979244823, 10.231102876277953, 5.212509368380733, 11.735465509846733, -2.0834535277420163, -2.6150803359123787, 1.7886297079225848, 3.34798758447816, 11.165349725652714, -4.41185131784583, 4.687017496229204, 11.767612334756222, 9.06322668739003, 5.460507064107764, -3.729517843303145, 13.62854404341845, -2.161953889740711, 6.247579952566117, 8.506664185985125, -4.80717292179005, -3.749053202177546, -1.3464950675311305, 9.851083528552403, 9.877619569914458, -1.25733870092682, 12.927703725344148, -2.1460421046296796, 10.69441778045486, 8.378915169143815, -1.7920656104098809, 6.710800961543814, -2.2194166118158742, 1.8195425938133747, 1.2618378053202184, 8.738742757913396, 5.753949892234642, 7.268285334412539, 1.859581490639366, 12.90859161216667, 1.1720542739996587, 0.8717967286137789, -3.7671524526208877};
    
    float C_exptd[100];

    for(int i=0;i<I;i++)
    {
        for(int j=0;j<J;j++)
        {
            double sum = 0;
            for(int k=0;k<K;k++)
            {
                sum += A_[i*K + k] * B_[k*J + j];
            }
            cout<<sum<<endl;
            C_exptd[i*J + j] = sum;
        }
    }
    MYINT* A = new MYINT[50];
    MYINT* B = new MYINT[50];
    MYINT* C = new MYINT[100];

    
    float scale1 = getScale(A_, I*K);
    int zero1 = getZero(A_, I*K, scale1);
    
    float scale2 = getScale(B_, K*J);
    int zero2 = getZero(B_, K*J, scale2);

    float scale3 = getScale(C_exptd, I*J);
    int zero3 = getZero(C_exptd, I*J, scale3);
    
    
    getZeroSkewArray(A_, I*K, A, scale1, zero1);
    getZeroSkewArray(B_, K*J, B, scale2, zero2);

    ACINT m1, m2, m3;
    MYITE n1, n2, n3;
    getQuantizedMultiplierLTO(((scale1 * scale2) / scale3), bitwidth, m1, n1);
    n1 -= right_shift;
    MatMul(A, B, C, I, K, J, -zero1, -zero2, zero3, m1, -n1, -clam_limit, clam_limit);

    int max_err = numeric_limits<MYINT>::min();
    float max_err_flt = numeric_limits<float>::min();
    cout<<max_err_flt<<endl;
    float max_err_relative_flt = float(max_err_flt);
    for(int i=0;i<I*J;i++)
    {
        float flt_err = fabs(C_exptd[i] - (scale3*(float(C[i])-float(zero3))));
        int err = abs(C[i] - MYINT(int(C_exptd[i]/scale3) + zero3));
        max_err_flt = max_err_flt < flt_err? flt_err: max_err_flt;
        max_err = max_err < err ? err: max_err;
        float relative_error = (flt_err)/fabs(C_exptd[i]);
        max_err_relative_flt = max_err_relative_flt < relative_error?relative_error:max_err_relative_flt;
        // std::cout<< int(C[i]) << ", "<< int(MYINT(int(C_exptd[i]/scale3) + zero3)) << ", " << (scale3*float(float(C[i])-float(zero3))) <<", " << C_exptd[i] << ", " << abs(C[i] - MYINT(int(C_exptd[i]/scale3) + zero3))<< ", " <<fabs(C_exptd[i] - (scale3*(C[i]-zero3)))<<std::endl;
    }
    std::cout<<max_err << ", "<<max_err_flt<< ", " << max_err_relative_flt<<endl;
    std::cout<<scale3 << ", "<<zero3<<endl;

}
int main()
{   
    test_MatAdd();
    // test_Sigmoid();
    // test_TanH();
    // test_Hadamard();
    // test_ScalarMul();
    // test_MatSub();
    // test_MatMul();
}

