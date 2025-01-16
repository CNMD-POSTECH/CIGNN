#include "compute_chi.h"

#include "pair_CNMP.h"
#include "atom.h"
#include "error.h"
#include "update.h"
#include "force.h"
#include "domain.h"
#include "memory.h"

using namespace LAMMPS_NS;

ComputeChi::ComputeChi(LAMMPS *lmp, int narg, char **arg) : Compute(lmp, narg, arg), chi_array(nullptr) {
    peratom_flag = 1;
    size_peratom_cols = 0;
    nmax = 0;
}

ComputeChi::~ComputeChi() {
    memory->destroy(chi_array);
}

void ComputeChi::init() {
    // 초기화 작업
}

void ComputeChi::compute_peratom() {
    invoked_peratom = update->ntimestep;

    // 필요한 경우 chi 배열을 확장
    if (atom->nmax > nmax) {
        memory->destroy(chi_array);
        nmax = atom->nmax;
        memory->create(chi_array, nmax, "chi/atom:chi_array");
        vector_atom = chi_array;
    }

    int nlocal = atom->nlocal;

    // chi 배열 초기화
    for (int i = 0; i < nlocal; i++) chi_array[i] = 0.0;

    // PairCNMP 클래스에서 chi 값을 가져옴
    PairCNMP *pair_CNMP = (PairCNMP *) force->pair_match("CNMP", 0);
    if (!pair_CNMP) {
        error->all(FLERR, "Pair style CNMP not found");
    }

    // 각 원자에 대해 chi 값 계산
    for (int i = 0; i < nlocal; i++) {
        chi_array[i] = pair_CNMP->compute_chi(i);
    }
}

double ComputeChi::memory_usage()
{
  double bytes = (double) nmax * sizeof(double);
  return bytes;
}