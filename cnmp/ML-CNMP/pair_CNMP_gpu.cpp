#include "pair_CNMP_gpu.h"

using namespace LAMMPS_NS;

PairCNMPGPU::PairCNMPGPU(LAMMPS *lmp) : PairCNMP(lmp)
{
    if (copymode)
    {
        return;
    }

    // NOP
}

PairCNMPGPU::~PairCNMPGPU()
{
    // NOP
}

int PairCNMPGPU::withGPU()
{
    return 1;
}