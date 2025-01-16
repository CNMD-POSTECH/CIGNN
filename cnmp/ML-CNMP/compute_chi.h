#ifdef COMPUTE_CLASS
// clang-format off
ComputeStyle(chi,ComputeChi);
// clang-format on
#else

#ifndef LMP_COMPUTE_CHI_H
#define LMP_COMPUTE_CHI_H

#include "compute.h"

namespace LAMMPS_NS {

class ComputeChi : public Compute {
public:
    ComputeChi(class LAMMPS *, int, char **);
    ~ComputeChi();
    void init() override;
    void compute_peratom() override;
    double memory_usage() override;

private:
    int nmax;
    double *chi_array;
};

}

#endif
#endif
