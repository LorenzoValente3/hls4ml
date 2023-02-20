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
#include <math.h>

namespace nnet {

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
    static const float eps = 1e-3;

    
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
    data_T cache;
   
    // Use a function_instantiate in case it helps to explicitly optimize unchanging weights/biases
    #pragma HLS function_instantiate variable=eps

    // For parallel inputs:
    //   - completely partition arrays -- target fabric
    //   - if we have an unroll factor, limit number of multipliers
    #pragma HLS PIPELINE II=CONFIG_T::reuse_factor

    // #pragma HLS ARRAY_PARTITION variable=weights complete // remove this line for now, it breaks compression sometimes
    // #pragma HLS ARRAY_PARTITION variable=scale complete
    // #pragma HLS ARRAY_PARTITION variable=bias complete

    // for loop mean & var
    float var[n_filt];
    float mean[n_filt];
    float scale[n_filt];
    float bias[n_filt];


    for (int i = 0; i < CONFIG_T::n_filt, i++){
        float sum = 0;

        for (int j = 0; j < data.size, j++){
            for(int k = 0; k < data.size, k++){
                sum += data[j][k];
            }
        }
            mean[i] = sum / data.size**2;

        for (int j = 0; j < data.size, j++){
            for(int k = 0; k < data.size, k++){
                var[i] += (data[j][k] - mean)**2;
            }
        }
           var[i] /= (data.size**2 - 1);
           var[i] = sqrt(var);

           scale[i] = gamma[i] / sqrt(var[i] + eps);
           bias[i] = beta[i] - gamma[i] * mean[i] / (var[i] + eps);
    }

    int multiplier_limit  = ceil(float(CONFIG_T::n_in) / float(CONFIG_T::reuse_factor));
    CONFIG_T::template product<data_T, typename CONFIG_T::scale_t>::limit(multiplier_limit);

    // Calcuate result
    Result: for (int ires = 0; ires < CONFIG_T::n_in; ires++) {
        if (CONFIG_T::n_filt==-1) {
            res[ires] = CONFIG_T::template product<data_T, typename CONFIG_T::scale_t>::product(data[ires], scale[ires]) + bias[ires];
	    } else {
            int norm_index = ires%CONFIG_T::n_filt;
            res[ires] = CONFIG_T::template product<data_T, typename CONFIG_T::scale_t>::product(data[ires], scale[norm_index]) + bias[norm_index];
        }
	}
}


}


#endif
