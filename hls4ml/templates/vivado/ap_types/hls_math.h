#ifndef X_HLS_MATH_H
#define X_HLS_MATH_H

#include <cmath>
#include "ap_fixed.h"

namespace hls {

template<class T>
static T exp(const T x) {
  return (T) std::exp(x.to_double());
}
// sqrt
  double sqrt(double);
  float  sqrt(float);
  float  sqrtf(float);
  int8_t sqrt(int8_t);
  uint8_t sqrt(uint8_t);
  int16_t sqrt(int16_t);
  uint16_t sqrt(uint16_t);
  int32_t sqrt(int32_t);
  uint32_t sqrt(uint32_t);

  // template<int W, int I>
  // ap_fixed<W,I> sqrt(ap_fixed<W,I> x){
  // 	return sqrt_fixed(x);
  // };
  // template<int W, int I>
  // ap_ufixed<W,I> sqrt(ap_ufixed<W,I> x){
  //   return sqrt_fixed(x);
  // }
  // template<int I>
  // ap_int<I> sqrt(ap_int<I> x){
  //   return sqrt_fixed(x);
  // }
  // template<int I>
  // ap_uint<I> sqrt(ap_uint<I> x){
  //   return sqrt_fixed(x);
  // }
}
#include "hls_sqrt_apfixed.h"
namespace hls {

// sqrt(ap_fixed)
  template<int W, int I>
  ap_fixed<W,I> sqrt(ap_fixed<W,I> x){
    return sqrt_fixed(x);
  }
  template<int W, int I>
  ap_ufixed<W,I> sqrt(ap_ufixed<W,I> x){
    return sqrt_fixed(x);
  }
  template<int I>
  ap_int<I> sqrt(ap_int<I> x){
    return sqrt_fixed(x);
  }
  template<int I>
  ap_uint<I> sqrt(ap_uint<I> x){
    return sqrt_fixed(x);
  }
  
}
#endif
