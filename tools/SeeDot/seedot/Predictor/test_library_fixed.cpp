// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include <iostream>
#include <fstream>
#include "library_fixed.h"
#include "datatypes.h"
#include <bits/stdc++.h>
#include <cassert>
#include <cmath>

using namespace std;
int clam_limit = 126;
int right_shift = 31;
int bitwidth = 8;

int getScale(float *A, int I)
{
    float min = A[0];
    float max_ = A[0];
    for(int i=1;i<I;i++)
    {
        min = min > A[i]?A[i]:min;
        max_ = max_ < A[i]?A[i]:max_;
    }
    float m_ = log2(max(fabs(min), fabs(max_)));
    int m_int = int(m_);
    if (fabs(m_ - round(m_))  < 0.00000000000001 )
    {
        m_int = round(m_) + 1;
    }
    else
    {
        m_int = ceil(m_);
    }
    // std::cout<<m_int << ", " << int(m_)<<endl;

    
    // std::cout<<m_int<<endl;

    int scale = -1*(bitwidth - 1 - m_int);
    return scale;
}

// int getZero(float *A, int I, float scale)
// {
    // float min = A[0];
    // float max = A[0];
    // for(int i=1;i<I;i++)
    // {
    //     min = min > A[i]?A[i]:min;
    //     max = max < A[i]?A[i]:max;
    // }
//     float zero = -1*(min + max) / 2;
//     zero = zero/scale;
//     return int(zero);
// }

void getFixedArray(float *A, int I, MYINT* B, int scale)
{
    for(int i=0;i<I;i++)
    {
        B[i] = A[i] * (1LL << (-(scale)));
    }
}

// void getQuantizedMultiplierLTO(float m, int bitwidth, ACINT& M, MYITE& N)
// {
//     cout<<"m "<<m<<endl;
//     float m_ = log2(m);
//     cout<<"m_ "<<m_<<endl;

    
//     int m_int = int(m_);
//     if (fabs(m_ - round(m_))  < 0.00000000000001 )
//     {
//         m_int = round(m_) + 1;
//     }
//     else
//     {
//         m_int = ceil(m_);
//     }
//     std::cout<<m_int << ", " << int(m_)<<endl;

    
//     std::cout<<m_int<<endl;

//     int scale = -1*(bitwidth - 1 - m_int);
//     assert(scale <= 0);
//     float exp =  m * float(1LL << (-(scale/2)));
//     exp *= float(1LL << (-(scale - (scale/2))));

//     cout<<exp << ", " << float(1LL << (-scale)) <<endl;

//     M = ACINT(exp);
//     cout<<M<<endl;
//     cout<<m<<endl;
//     cout<<scale<<endl;
//     N = -scale;
// }

// void getScaleAndZeroForAddorSub(float scaleA, ACINT zeroA, float  scaleB, ACINT zeroB, float  scaleC, ACINT zeroC, ACINT& m1, MYITE& n1, ACINT& m2, MYITE& n2, ACINT& m3, MYITE& n3)
// {
//     float s3 = 1/scaleC;
//     getQuantizedMultiplierLTO(scaleA, bitwidth, m1, n1);
//     getQuantizedMultiplierLTO(scaleB, bitwidth, m2, n2);
//     getQuantizedMultiplierLTO(s3, bitwidth, m3, n3);

//     n1 -= right_shift;
//     n2 -= right_shift;
//     n3 -= right_shift;
// }

void AdjustScale(MYINT* a, int old_scale, int new_scale)
{
    if(new_scale != old_scale)
    {
        if(new_scale > old_scale)
        {
            *a = *a / ( 1 << (new_scale -old_scale));
        }
        else{
            *a = *a * ( 1 << (-new_scale + old_scale));
        }
    }
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

    
    int scale1 = getScale(A_, I*J);
    
    int scale2 = getScale(B_, I*J);

    int scale3 = getScale(C_exptd, I*J);

    int scale_comm = scale1 > scale2?scale1: scale2;
    int d1 =scale1 > scale2?0:scale2 - scale1; 
    int d2 =scale1 > scale2?scale1 - scale2:0; 
    int diff = scale3 - max(scale1, scale2);   
    int scale_out_new = diff >=0?scale3:scale_comm;
    diff = diff > 0? diff: 0;
    int demoteLog = diff >= 8 ? diff - 8:0;
    diff = min(diff, 8);


    getFixedArray(A_, I*J, A, scale1);
    getFixedArray(B_, I*J, B, scale2);
    cout<<(1 << d1) << ", "<< (1<< d2) << ", " << (1 << diff) << ", " << ( 1 << demoteLog) <<endl;
    cout<<scale1 << ", "<< scale2 << ", " << scale3 << "," << scale_out_new<< endl;
    MatAddNN<MYINT, MYINT, MYINT, MYINT>(A, B, C, I, J, 1 << d1, 1 << d2, 1 << diff, 1 << demoteLog);

    int max_err = numeric_limits<MYINT>::min();
    float max_err_flt = numeric_limits<float>::min();
    cout<<max_err_flt<<endl;
    float max_err_relative_flt = float(max_err_flt);
    for(int i=0;i<I*J;i++)
    {
        AdjustScale(&C[i], scale_out_new, scale3);
        
        float flt_err = fabs(C_exptd[i] - (float(C[i] / (1 << (-scale3)))));
        int err = abs(C[i] - (MYINT(C_exptd[i] * (1 << -scale3))));
        max_err_flt = max_err_flt < flt_err? flt_err: max_err_flt;
        max_err = max_err < err ? err: max_err;
        float relative_error = (flt_err)/fabs(C_exptd[i]);
        max_err_relative_flt = max_err_relative_flt < relative_error?relative_error:max_err_relative_flt;
        std::cout<< int(A[i]) << ", "<<int(B[i]) << ", "<<int(C[i]) << ", "<< int(C_exptd[i] * (1 << -scale3)) << ", " << (float(float(C[i]) / (1 << (-scale3)))) <<", " << C_exptd[i] << ", " <<  err << ", " <<flt_err<< ", " << relative_error<<std::endl;
    }
    std::cout<<max_err << ", "<<max_err_flt<< ", " << max_err_relative_flt<<endl;
    std::cout<<scale3 <<endl;

}

void test_Sigmoid()
{
    float A_[50] = {-2.2452245708608247, 3.009875338014462, -1.7002412111109821, -1.9049264433636206, -2.0182750807793712, -0.5432766723508782, -1.7432700285712415, 0.30189251298798414, 2.5838316613677446, -0.3143724663629377, -1.8505789548909433, 0.05082527135955672, 2.6554193128240735, 3.9778999851631527, -1.5081166909725068, -0.35576575053478976, 3.5395032860593236, -2.7228117698286494, 3.1376137878675134, -2.4624751124417115, 0.1958889247055975, -1.9613318006582434, -2.5840202433380095, 1.510661269277815, 3.596748092641704, -0.7635623106205034, 2.87103826511569, 2.651333365901154, 3.074900175663762, -2.189857861117537, -1.1747240542994373, -2.136933872646459, 0.0141745465708909, 1.2023797439864916, 0.18558777990798614, 2.4994670186671577, 0.6809709546575329, 2.25371928340655, 2.909832736228255, 2.3704825815807196, 1.7662742825338107, 0.5322344767157219, 2.869563030200495, -2.2815409226073164, 2.9823727627373025, -0.8581122748414729, 2.987002276679913, -1.0532018811175756, 2.1268778201173193, 3.5355006374176794};
    
    MYINT* A = new MYINT[50];
    MYINT* C = new MYINT[50];
    float C_exptd[50];
    int I=10;
    int J=5;
    
    for(int i=0;i<50;i++)
    {
        C_exptd[i] = 1 / (1 + exp(-1*A_[i]));
    }

    
    int scale1 = getScale(A_, I*J);
    
    int scale2 = getScale(C_exptd, I*J);
    
    getFixedArray(A_, I*J, A, scale1);

    cout<<scale1 << ", "<< scale2<<endl;


    Sigmoid<MYINT>(A, I, J, 2, 0.5 * (1 << -scale1), (1 << -scale1), (1 << -(scale1+1)), (1L << -(scale2+1)), C);

    int max_err = numeric_limits<MYINT>::min();
    float max_err_flt = numeric_limits<float>::min();
    cout<<max_err_flt<<endl;
    float max_err_relative_flt = float(max_err_flt);
    for(int i=0;i<I*J;i++)
    {
        
        float flt_err = fabs(C_exptd[i] - (float(C[i] / (1 << (-scale2)))));
        int err = abs(C[i] - (MYINT(C_exptd[i] * (1 << -scale2))));
        max_err_flt = max_err_flt < flt_err? flt_err: max_err_flt;
        max_err = max_err < err ? err: max_err;
        float relative_error = (flt_err)/fabs(C_exptd[i]);
        max_err_relative_flt = max_err_relative_flt < relative_error?relative_error:max_err_relative_flt;
        std::cout<< int(A[i]) << ", "<<int(C[i]) << ", "<< int(C_exptd[i] * (1 << -scale2)) << ", " << (float(float(C[i]) / (1 << (-scale2)))) <<", " << C_exptd[i] << ", " <<  err << ", " <<flt_err<< ", " << relative_error<<std::endl;
    }
    std::cout<<max_err << ", "<<max_err_flt<< ", " << max_err_relative_flt<<endl;
    std::cout<<scale2 <<endl;

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
    int scale1 = getScale(A_, I*J);
    
    int scale2 = getScale(C_exptd, I*J);
    
    getFixedArray(A_, I*J, A, scale1);

    cout<<scale1 << ", "<< scale2<<endl;


    TanH<MYINT>(A,  I, J, (1 << -scale1), (1 << -scale2), C);

    
    int max_err = numeric_limits<MYINT>::min();
    float max_err_flt = numeric_limits<float>::min();
    cout<<max_err_flt<<endl;
    float max_err_relative_flt = float(max_err_flt);
    for(int i=0;i<I*J;i++)
    {
        
        float flt_err = fabs(C_exptd[i] - (float(C[i] / (1 << (-scale2)))));
        int err = abs(C[i] - (MYINT(C_exptd[i] * (1 << -scale2))));
        max_err_flt = max_err_flt < flt_err? flt_err: max_err_flt;
        max_err = max_err < err ? err: max_err;
        float relative_error = (flt_err)/fabs(C_exptd[i]);
        max_err_relative_flt = max_err_relative_flt < relative_error?relative_error:max_err_relative_flt;
        std::cout<< int(A[i]) << ", "<<int(C[i]) << ", "<< int(C_exptd[i] * (1 << -scale2)) << ", " << (float(float(C[i]) / (1 << (-scale2)))) <<", " << C_exptd[i] << ", " <<  err << ", " <<flt_err<< ", " << relative_error<<std::endl;
    }
    std::cout<<max_err << ", "<<max_err_flt<< ", " << max_err_relative_flt<<endl;
    std::cout<<scale2 <<endl;

}

void getTreeSumShrAndDemoteForMul(int scaleA, int scaleB, int scale_temp, int scale_out, int hidden_dim, int& shrA, int& shrB, int& H1, int& H2, int& demote, int& scale_out_new)
{
    int bitwidth_temp = 2*8*sizeof(MYINT);

    int totalShr = 0;

    int scaleAfterMulOp = scaleA + scaleB;
    int bitsAfterMulOp = 15;
    int bitsAfterMulStore = bitwidth_temp;
    int scaleAfterMulStore = min(scaleAfterMulOp + max(bitsAfterMulOp - bitsAfterMulStore, 0), 1000 - max(bitsAfterMulStore - 8, 0));
    totalShr += scaleAfterMulStore - scaleAfterMulOp;

    if (scaleA > scaleB)
    {
        shrA = totalShr / 2;
        shrB = totalShr - totalShr / 2;    
    }
    else
    {
        shrB = totalShr / 2;
        shrA = totalShr - totalShr / 2;    
    }

    int scaleAfterAddOp = max(scale_temp, scaleAfterMulStore);

    totalShr += (scaleAfterAddOp - scaleAfterMulStore);

    H1 = (scaleAfterAddOp - scaleAfterMulStore);

    int scaleAfterAddStore = scale_out;

    totalShr += (scaleAfterAddStore - scaleAfterAddOp);

    demote = totalShr - shrA - shrB - H1;

    int height = int(ceil(log2(hidden_dim)));

    if (height < H1)
    {
        int diff = H1 - height;
        H1 = height;

        if (shrA >= shrB)
        {
            shrA += diff / 2;
            shrB += diff - diff/2;
        }
        else{
            shrB += diff / 2;
            shrA += diff - diff/2;
        }
    }

    if(demote < 0)
    {
        if (demote + H1 >= 0)
        {
            H1 += demote;
            demote = totalShr - shrA - shrB - H1;
        }
        else
        {
            H1 = 0;
            demote = totalShr - shrA - shrB - H1;

            if ((demote + shrA + shrB) >= 0)
            {
                int toAdd = demote;
                shrA += toAdd /2;
                shrB += toAdd - toAdd / 2;
                demote = totalShr - shrA -shrB - H1;
            }
            else
            {
                shrA = 0;
                shrB = 0;
                demote = 0;
                scale_out_new = scaleA + scaleB;
            }
        }
    }

    demote = 1 << demote;
    H2 = height - H1;

}
void test_Hadamard()
{

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

    
    int scale1 = getScale(A_, I*J);
    
    int scale2 = getScale(B_, I*J);

    int scale3 = getScale(C_exptd, I*J);
    
    
    getFixedArray(A_, I*J, A, scale1);
    getFixedArray(B_, I*J, B, scale2);

    int shrA, shrB, H1, H2, demote, scale_out_new = scale3;

    getTreeSumShrAndDemoteForMul(scale1, scale2, scale3, scale3, 1, shrA, shrB, H1, H2, demote, scale_out_new);
    cout<<shrA<<","<< shrB<<","<< H1<<","<< H2<<","<< demote<<","<< scale_out_new<<endl;
    cout <<scale1 <<","<<scale2 <<","<<scale3 <<endl;
    
    MulCir<MYINT, MYINT, int16_t, MYINT>(A, B, C, I, J, 1<<shrA, 1 << shrB, demote);

    int max_err = numeric_limits<MYINT>::min();
    float max_err_flt = numeric_limits<float>::min();

    float max_err_relative_flt = float(max_err_flt);
    for(int i=0;i<I*J;i++)
    {
        AdjustScale(&C[i], scale_out_new, scale3);
        
        float flt_err = fabs(C_exptd[i] - (float(C[i] / (1 << (-scale3)))));
        int err = abs(C[i] - (MYINT(C_exptd[i] * (1 << -scale3))));
        max_err_flt = max_err_flt < flt_err? flt_err: max_err_flt;
        max_err = max_err < err ? err: max_err;
        float relative_error = (flt_err)/fabs(C_exptd[i]);
        max_err_relative_flt = max_err_relative_flt < relative_error?relative_error:max_err_relative_flt;
        std::cout<< int(A[i]) << ", "<<int(B[i]) << ", "<<int(C[i]) << ", "<< int(C_exptd[i] * (1 << -scale3)) << ", " << (float(float(C[i]) / (1 << (-scale3)))) <<", " << C_exptd[i] << ", " <<  err << ", " <<flt_err<< ", " << relative_error<<std::endl;
    }
    std::cout<<max_err << ", "<<max_err_flt<< ", " << max_err_relative_flt<<endl;
    std::cout<<scale3 <<endl;
    
}

void test_ScalarMul()
{

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

    int scale1 = getScale(A_, 1);
    
    int scale2 = getScale(B_, I*J);

    int scale3 = getScale(C_exptd, I*J);
    
    
    getFixedArray(A_, 1, A, scale1);
    getFixedArray(B_, I*J, B, scale2);

    int shrA, shrB, H1, H2, demote, scale_out_new = scale3;

    getTreeSumShrAndDemoteForMul(scale1, scale2, scale3, scale3, 1, shrA, shrB, H1, H2, demote, scale_out_new);
    cout<<shrA<<","<< shrB<<","<< H1<<","<< H2<<","<< demote<<","<< scale_out_new<<endl;
    cout <<scale1 <<","<<scale2 <<","<<scale3 <<endl;
    
    ScalarMul<MYINT, MYINT, int16_t, MYINT>(A, B, C, I, J, 1<<shrA, 1 << shrB, demote);
int max_err = numeric_limits<MYINT>::min();
    float max_err_flt = numeric_limits<float>::min();

    float max_err_relative_flt = float(max_err_flt);
    for(int i=0;i<I*J;i++)
    {
        AdjustScale(&C[i], scale_out_new, scale3);
        
        float flt_err = fabs(C_exptd[i] - (float(C[i] / (1 << (-scale3)))));
        int err = abs(C[i] - (MYINT(C_exptd[i] * (1 << -scale3))));
        max_err_flt = max_err_flt < flt_err? flt_err: max_err_flt;
        max_err = max_err < err ? err: max_err;
        float relative_error = (flt_err)/fabs(C_exptd[i]);
        max_err_relative_flt = max_err_relative_flt < relative_error?relative_error:max_err_relative_flt;
        std::cout<< int(A[0]) << ", "<<int(B[i]) << ", "<<int(C[i]) << ", "<< int(C_exptd[i] * (1 << -scale3)) << ", " << (float(float(C[i]) / (1 << (-scale3)))) <<", " << C_exptd[i] << ", " <<  err << ", " <<flt_err<< ", " << relative_error<<std::endl;
    }
    std::cout<<max_err << ", "<<max_err_flt<< ", " << max_err_relative_flt<<endl;
    std::cout<<scale3 <<endl;
    
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
        C_exptd[i] = A_[i] - B_[i];
    }
    
    int scale1 = getScale(A_, I*J);
    
    int scale2 = getScale(B_, I*J);

    int scale3 = getScale(C_exptd, I*J);

    int scale_comm = scale1 > scale2?scale1: scale2;
    int d1 =scale1 > scale2?0:scale2 - scale1; 
    int d2 =scale1 > scale2?scale1 - scale2:0; 
    int diff = scale3 - max(scale1, scale2);   
    int scale_out_new = diff >=0?scale3:scale_comm;
    diff = diff > 0? diff: 0;
    int demoteLog = diff >= 8 ? diff - 8:0;
    diff = min(diff, 8);


    getFixedArray(A_, I*J, A, scale1);
    getFixedArray(B_, I*J, B, scale2);
    cout<<(1 << d1) << ", "<< (1<< d2) << ", " << (1 << diff) << ", " << ( 1 << demoteLog) <<endl;
    cout<<scale1 << ", "<< scale2 << ", " << scale3 << "," << scale_out_new<< endl;
    
    MatSub<MYINT, MYINT, MYINT, MYINT>(A, B, C, I, J, 1 << d1, 1 << d2, 1 << diff, 1 << demoteLog);

    int max_err = numeric_limits<MYINT>::min();
    float max_err_flt = numeric_limits<float>::min();
    cout<<max_err_flt<<endl;
    float max_err_relative_flt = float(max_err_flt);
    for(int i=0;i<I*J;i++)
    {
        AdjustScale(&C[i], scale_out_new, scale3);
        
        float flt_err = fabs(C_exptd[i] - (float(C[i] / (1 << (-scale3)))));
        int err = abs(C[i] - (MYINT(C_exptd[i] * (1 << -scale3))));
        max_err_flt = max_err_flt < flt_err? flt_err: max_err_flt;
        max_err = max_err < err ? err: max_err;
        float relative_error = (flt_err)/fabs(C_exptd[i]);
        max_err_relative_flt = max_err_relative_flt < relative_error?relative_error:max_err_relative_flt;
        std::cout<< int(A[i]) << ", "<<int(B[i]) << ", "<<int(C[i]) << ", "<< int(C_exptd[i] * (1 << -scale3)) << ", " << (float(float(C[i]) / (1 << (-scale3)))) <<", " << C_exptd[i] << ", " <<  err << ", " <<flt_err<< ", " << relative_error<<std::endl;
    }
    std::cout<<max_err << ", "<<max_err_flt<< ", " << max_err_relative_flt<<endl;
    std::cout<<scale3 <<endl;
}

void test_MatMul()
{
    int I = 10;
    int K = 5;
    int J = 10;

    float A_[50] = {-2.2452245708608247, 3.009875338014462, -1.7002412111109821, -1.9049264433636206, -2.0182750807793712, -0.5432766723508782, -1.7432700285712415, 0.30189251298798414, 2.5838316613677446, -0.3143724663629377, -1.8505789548909433, 0.05082527135955672, 2.6554193128240735, 3.9778999851631527, -1.5081166909725068, -0.35576575053478976, 3.5395032860593236, -2.7228117698286494, 3.1376137878675134, -2.4624751124417115, 0.1958889247055975, -1.9613318006582434, -2.5840202433380095, 1.510661269277815, 3.596748092641704, -0.7635623106205034, 2.87103826511569, 2.651333365901154, 3.074900175663762, -2.189857861117537, -1.1747240542994373, -2.136933872646459, 0.0141745465708909, 1.2023797439864916, 0.18558777990798614, 2.4994670186671577, 0.6809709546575329, 2.25371928340655, 2.909832736228255, 2.3704825815807196, 1.7662742825338107, 0.5322344767157219, 2.869563030200495, -2.2815409226073164, 2.9823727627373025, -0.8581122748414729, 2.987002276679913, -1.0532018811175756, 2.1268778201173193, 3.5355006374176794};
    float B_[50] = {-0.9721936964304785, -3.062643085506733, -1.448129883728786, 0.4855276507598596, -0.030127186366902592, 0.6338516216963654, -2.0636227164697862, 0.9118327939224438, -0.04727907671301068, -2.263524100910411, 0.36390725935374313, -2.5664325851473997, 0.12077767124192551, -3.2174269379389666, -3.5561563576585007, -2.2851978513851474, -3.721980298015348, -3.137903361186443, -0.3812338965600297, -1.8301802563374192, -2.612776678267217, 0.28309669484675837, 0.9737223890192617, -3.365990244563438, -4.737755263837463, -3.797768981725157, -2.758091077500203, -1.5415114559401197, 0.5606473716827729, -3.8901759783074032, -4.462544059872947, -2.9975945989535235, -1.8482802777037879, -3.6350307683051177, 0.9922215187825749, -1.0897637049797382, -4.396286432954785, -1.9312483795880353, -1.43168457777392, -1.3103665976563303, -0.4061508514028258, -2.2044440346764014, -1.0576693684876712, -0.978664325435731, -2.8626912091862557, -2.4752199387408256, -1.9016323185735842, -1.633689858925294, -4.351377621341459, -3.1048891161510275};
    
    float C_exptd[100];

    float sum_max = numeric_limits<float>::min();

    for(int i=0;i<I;i++)
    {
        for(int j=0;j<J;j++)
        {
            double sum = 0;
            for(int k=0;k<K;k++)
            {
                sum += A_[i*K + k] * B_[k*J + j];
            }
            sum_max = sum_max < sum?sum:sum_max;
            C_exptd[i*J + j] = sum;
        }
    }
    cout<<sum_max<<endl;
    MYINT* A = new MYINT[50];
    MYINT* B = new MYINT[50];
    MYINT* C = new MYINT[100];

    int scale1 = getScale(A_, I*K);
    
    int scale2 = getScale(B_, K*J);

    int scale3 = getScale(C_exptd, I*J);
    
    
    getFixedArray(A_, I*K, A, scale1);
    getFixedArray(B_, K*J, B, scale2);

    int shrA, shrB, H1, H2, demote, scale_out_new = scale3;

    getTreeSumShrAndDemoteForMul(scale1, scale2, scale3, scale3, K, shrA, shrB, H1, H2, demote, scale_out_new);
    cout<<shrA<<","<< shrB<<","<< H1<<","<< H2<<","<< demote<<","<< scale_out_new<<endl;
    cout <<scale1 <<","<<scale2 <<","<<scale3 <<endl;
    int16_t *temp_arr = new int16_t[K];
    MatMulNN<MYINT, MYINT, int16_t, MYINT>(A, B, C, temp_arr, I, K, J, 1<<shrA, 1 << shrB, H1, H2, demote);
    delete[] temp_arr;

    int max_err = numeric_limits<MYINT>::min();
    float max_err_flt = numeric_limits<float>::min();

    float max_err_relative_flt = float(max_err_flt);
    for(int i=0;i<I*J;i++)
    {
        AdjustScale(&C[i], scale_out_new, scale3);
        
        float flt_err = fabs(C_exptd[i] - (float(C[i] / (1 << (-scale3)))));
        int err = abs(C[i] - (MYINT(C_exptd[i] * (1 << -scale3))));
        max_err_flt = max_err_flt < flt_err? flt_err: max_err_flt;
        max_err = max_err < err ? err: max_err;
        float relative_error = (flt_err)/fabs(C_exptd[i]);
        max_err_relative_flt = max_err_relative_flt < relative_error?relative_error:max_err_relative_flt;
        std::cout<< int(A[i]) << ", "<<int(B[i]) << ", "<<int(C[i]) << ", "<< int(MYINT(C_exptd[i] * (1 << -scale3))) << ", " << (float(float(C[i]) / (1 << (-scale3)))) <<", " << C_exptd[i] << ", " <<  err << ", " <<flt_err<< ", " << relative_error<<std::endl;
    }
    std::cout<<max_err << ", "<<max_err_flt<< ", " << max_err_relative_flt<<endl;
    std::cout<<scale3 <<endl;
    
}
int main()
{   
    // test_MatAdd();
    // test_Sigmoid();
    // test_TanH();
    // test_Hadamard();
    // test_ScalarMul();
    // test_MatSub();
    // test_MatMul();
}

