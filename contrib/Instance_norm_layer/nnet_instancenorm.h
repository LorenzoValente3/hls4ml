//
//    rfnoc-hls-neuralnet: Vivado HLS code for neural-net building blocks
//
//    Copyright (C) 2017 EJ Kreinar
//
//    This program is free software: you can redistribute it and/or modify
//    it under the terms of the GNU General Public License as published by
//    the Free Software Foundation, either version 3 of the License, or
//    (at your option) any later version.
//
//    This program is distributed in the hope that it will be useful,
//    but WITHOUT ANY WARRANTY; without even the implied warranty of
//    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//    GNU General Public License for more details.
//
//    You should have received a copy of the GNU General Public License
//    along with this program.  If not, see <http://www.gnu.org/licenses/>.
//

#ifndef NNET_INSTANCENORM_H_
#define NNET_INSTANCENORM_H_

#include "nnet_common.h"
#include "nnet_dense.h"
#include "hls_stream.h"
#include "/home/marco/tools/Vivado/2021.1/include/hls_math.h"
#include "ap_fixed.h"
//#include <math.h>

namespace nnet {

template<class data_T, class res_T>
void fixedsqrt(
    data_T    data,
    res_T&     res
)
{
    #pragma HLS PIPELINE
    if (data == 0){
        res = 0;
        return;
    }
 
    int msb = 0;
    int n = data / 2;
    while (n != 0) {
        n = n / 2;
        msb++;
    }
    unsigned int a = 1 << msb;
    res = 0;
 
    //data_T result = 0;
    while (a != 0) {
        // Check whether the current value
        // of 'res' can be added or not
        if ((res + a) * (res + a) <= data) {
            res += a;
        }
 
        // (a = a/2)
        a >>= 1;
    }
}



struct instancenorm_config
{
    // Internal data type definitions
    typedef float gamma_t;
    typedef float beta_t;
    typedef float eps_t;

    // Layer Sizes
    static const unsigned n_in = 10;
    static const unsigned n_filt = -1;
    static const unsigned n_gamma_beta = 10;


    
    // Resource reuse info
    static const unsigned io_type = io_parallel;
    static const unsigned reuse_factor = 1;
    static const bool store_weights_in_bram = false;
    static const unsigned n_zeros = 0;

    // partitioning arrays cyclically to go with roll factors?
    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};

template<class data_T, class res_T, typename CONFIG_T>
void instancenorm(
    data_T    data[CONFIG_T::n_in],
    res_T     res[CONFIG_T::n_in],
    typename CONFIG_T::gamma_t  gamma[CONFIG_T::n_gamma_beta],
    typename CONFIG_T::beta_t   beta[CONFIG_T::n_gamma_beta],
    typename CONFIG_T::eps_t  eps
)
{
    // data_T cache;
   
    // Use a function_instantiate in case it helps to explicitly optimize unchanging weights/biases
    #pragma HLS function_instantiate variable=eps

    // For parallel inputs:
    //   - completely partition arrays -- target fabric
    //   - if we have an unroll factor, limit number of multipliers
    //#pragma HLS PIPELINE II=CONFIG_T::reuse_factor

    // #pragma HLS ARRAY_PARTITION variable=weights complete // remove this line for now, it breaks compression sometimes
    // #pragma HLS ARRAY_PARTITION variable=scale complete
    // #pragma HLS ARRAY_PARTITION variable=bias complete

    // for loop mean & var
    typename CONFIG_T::ops_t var[CONFIG_T::n_filt];
    typename CONFIG_T::ops_t mean[CONFIG_T::n_filt];
    typename CONFIG_T::ops_t scale[CONFIG_T::n_filt];
    typename CONFIG_T::ops_t bias[CONFIG_T::n_filt];
    typename CONFIG_T::ops_t sqrsize = CONFIG_T::n_in/CONFIG_T::n_filt;
    typename CONFIG_T::ops_t sizehw;
    typename CONFIG_T::ops_t fixedeps = eps;
    //fixedsqrt(sqrsize,sizehw);
    sizehw = hls::sqrt(sqrsize);
    typename CONFIG_T::sum_t sum = 0;
    for (int i = 0; i < CONFIG_T::n_filt; i++){
        sum = 0;
        var[i] = 0;
        mean[i] = 0;
        for (int j = sqrsize*i; j < sqrsize*(i+1); j++){
                sum += data[j];
            
        }
            mean[i] = sum / sqrsize;

        for (int j = sqrsize*i; j < sqrsize*(i+1); j++){
                typename CONFIG_T::ops_t tmp = (data[j] - mean[i]);
                var[i] += tmp*tmp;
        }
           var[i] /= (sqrsize - 1);
           //fixedsqrt(var[i], var[i]);
           //var[i] = hls::sqrt(var[i]);
           //var[i] = sqrt(var[i]);
           typename CONFIG_T::ops_t denominator;
           //fixedsqrt(var[i] + fixedeps,denominator);
           denominator = hls::sqrt(var[i] + fixedeps);
           scale[i] = gamma[i] / denominator;
           bias[i] = beta[i] - gamma[i] * mean[i] / denominator;
    }

    int multiplier_limit  = ceil(float(CONFIG_T::n_in) / float(CONFIG_T::reuse_factor));
    CONFIG_T::template product<data_T, typename CONFIG_T::ops_t>::limit(multiplier_limit);

    // Calcuate result
    Result: for (int ires = 0; ires < CONFIG_T::n_in; ires++) {
        if (CONFIG_T::n_filt==-1) {
            res[ires] = CONFIG_T::template product<data_T, typename CONFIG_T::ops_t>::product(data[ires], scale[ires]) + bias[ires];
	    } else {
            int norm_index = ires%CONFIG_T::n_filt;
            res[ires] = CONFIG_T::template product<data_T, typename CONFIG_T::ops_t>::product(data[ires], scale[norm_index]) + bias[norm_index];
        }
	}
}


}






#endif
