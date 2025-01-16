#ifdef PAIR_CLASS

PairStyle(CNMP/gpu, PairCNMPGPU)

#else

#ifndef LMP_PAIR_CNMP_GPU_H_
#define LMP_PAIR_CNMP_GPU_H_

#include "pair_CNMP.h"

namespace LAMMPS_NS
{

class PairCNMPGPU: public PairCNMP
{
public:
    PairCNMPGPU(class LAMMPS*);

    virtual ~PairCNMPGPU() override;

protected:
    int withGPU() override;
};

}  // namespace LAMMPS_NS

#endif /* LMP_PAIR_CNMP_GPU_H_ */
#endif