#include "structset.hpp"

#if STRUCT_EXPERIMENT

// This is the poset leq comparator: define as needed
// row1 <= row2
bool leq(SetFreq freq, SetX x, int32_t idx) {
  int32_t featbits = x.featbits;
  uint64_t* row2 = x.data + featbits * idx;
  if (featbits == 1) {
    return (row2[0] & freq.data_l) == freq.data_l;
  } else if (featbits == 2) {
    return (row2[0] & freq.data_l) == freq.data_l && (row2[1] & freq.data_h) == freq.data_h;
  } else {
    uint64_t* row1 = freq.data;
    for (int32_t j = 0; j < featbits; ++j) {
      if ((row2[j] & row1[j]) != row1[j]) {
        return false;
      }
    }
    return true;
  }
}

bool leq(SetX x, int32_t idx1, int32_t idx2) {
  int32_t featbits = x.featbits;
  uint64_t* row1 = x.data + featbits * idx1;
  uint64_t* row2 = x.data + featbits * idx2;
  if (featbits == 1) {
    return (row2[0] & row1[0]) == row1[0];
  } else if (featbits == 2) {
    return (row2[0] & row1[0]) == row1[0] && (row2[1] & row1[1]) == row1[1];
  } else {
    for (int32_t j = 0; j < featbits; ++j) {
      if ((row2[j] & row1[j]) != row1[j]) {
        return false;
      }
    }
    return true;
  }
}

void load_poset(SetX& x, uint64_t*& order, int32_t samples) {
  int32_t featbits = x.featbits;
  int32_t features = x.features;
  uint64_t* data = x.data;
  int32_t sambits = ((samples + 63) >> 6);
  order = new uint64_t[samples * sambits];
  for (int32_t i = 0; i < samples; ++i) {
    for (int32_t j = 0; j < sambits; ++j) {
      int32_t oidx = i * sambits + j;
      uint64_t obit = 0;
      for (int32_t jj = 0; jj < 64; ++jj) {
        int32_t jjdx = j * 64 + jj;
        if (jjdx >= features) {
          break;
        }
        bool comp = leq(x, i, jjdx);
        obit |= ((uint64_t)comp << jj);
      }
      order[oidx] = obit;
    }
  }
}
  
#endif
