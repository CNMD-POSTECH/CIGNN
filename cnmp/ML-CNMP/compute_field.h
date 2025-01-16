#ifdef COMPUTE_CLASS
// clang-format off
ComputeStyle(field, ComputeField);
// clang-format on
#else

#ifndef LMP_COMPUTE_FIELD_H
#define LMP_COMPUTE_FIELD_H

#include <iostream>

#include "error.h"
#include "force.h"
#include "modify.h"
#include "update.h"
#include "compute.h"
#include "pair_CNMP.h"

namespace LAMMPS_NS {

class ComputeField : public Compute {
 public:
  ComputeField(class LAMMPS *, int, char **);
  ~ComputeField() override = default;
  void init() override {}
  double compute_scalar() override;

 private:
  class PairCNMP *pair_CNMP;  // Declare a pointer to PairCNMP
};

} // namespace LAMMPS_NS

#endif
#endif