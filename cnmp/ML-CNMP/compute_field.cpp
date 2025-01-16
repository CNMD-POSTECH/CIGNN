#include "compute_field.h"

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

ComputeField::ComputeField(LAMMPS *lmp, int narg, char **arg)
    : Compute(lmp, narg, arg), pair_CNMP(nullptr) {
    scalar_flag = 1;
    extscalar = 1;
}

/* ---------------------------------------------------------------------- */

double ComputeField::compute_scalar() {
    invoked_scalar = update->ntimestep;

    // Find the CNMP pair style in the force object
    pair_CNMP = (PairCNMP *) force->pair_match("CNMP", 0);
    if (!pair_CNMP) {
        error->all(FLERR, "Pair style CNMP not found");
    }

    // Compute the field energy using the PairCNMP method
    double field = pair_CNMP->compute_field();

    scalar = field;  // Add the computed field energy to the scalar
    return scalar;  // Return the computed field energy
}