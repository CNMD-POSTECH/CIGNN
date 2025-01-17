#include "pair_CNMP.h"
#include <iostream>

using namespace LAMMPS_NS;

// 사용 변수 초기화
PairCNMP::PairCNMP(LAMMPS *lmp) : Pair(lmp) 
{
    // pair_style이 단일 원자 간의 에너지 및 힘 계산을 지원하는지 여부를 나타냄. 값이 0 이면 비 활성화
    single_enable               = 0; 
    // 재시작 파일에 pair_style 관련 정보를 저장할지 여부를 결정함.
    restartinfo                 = 0;
    // 한 번의 pair_coeff 호출로 모든 원자 타입 쌍에 대한 계수를 설정할 수 있는지 여부를 나타냄.
    one_coeff                   = 1;
    // 해당 pair_style이 다체(many-body) 상호작용을 포함하는지 여부를 나타냄.
    manybody_flag               = 1;
    // pair_style이 virial 계산을 위해 fdotr 배열을 사용하지 않음을 나타냄.
    no_virial_fdotr_compute     = 1;
    // 원심(centroid) 스트레스 계산을 지원하는지 여부를 나타냄.
    centroidstressflag          = CENTROID_NOTAVAIL;
    // e_field_list 초기값 설정
    e_field_list = std::make_tuple(0.0f, 0.0f, 0);
    // device의 기본값을 cpu로 설정
    gpuName                     = "cpu";

    this->atomNumMap            = nullptr;
    this->maxinum               = 10;
    this->initializedPython     = 0;
    this->virialWarning         = 0;
    this->cutoff                = 0.0;
    this->npythonPath           = 0;
    this->pythonPath           = nullptr;
    this->pModule              = nullptr;
    this->pFunc                = nullptr;
}

// 할당된 메모리와 초기화된 리소스를 해제
PairCNMP::~PairCNMP()
{
    if (copymode)
    {
        return;
    }

    if (this->atomNumMap != nullptr)
    {
        delete[] this->atomNumMap;
    }

    if (allocated)
    {
        memory->destroy(cutsq);
        memory->destroy(setflag);
        memory->destroy(this->cell);
        memory->destroy(this->chi);
        memory->destroy(this->charge);
        memory->destroy(this->forces);
        memory->destroy(this->atomNums);
        memory->destroy(this->positions);
    }

    if (this->pythonPath != nullptr)
    {
        for (int i = 0; i < this->npythonPath; i++)
        {
            delete[] this->pythonPath[i];
        }
        delete[] this->pythonPath;
    }

    if (this->initializedPython)
    {
        this->finalizePython();
    }
}

// 메모리 할당 및 초기화
void PairCNMP::allocate()
{
    allocated = 1;

    const int ntypes = atom->ntypes;

    memory->create(cutsq,   ntypes+1, ntypes+1, "pair:cutsq");
    memory->create(setflag, ntypes+1, ntypes+1, "pair:setflag");

    memory->create(this->cell,     3, 3,             "pair:cell");
    memory->create(this->chi,      this->maxinum,    "pair:chi");
    memory->create(this->charge,   this->maxinum,    "pair:charge");
    memory->create(this->forces,   this->maxinum, 3, "pair:forces");
    memory->create(this->atomNums, this->maxinum,    "pair:atomNums");
    memory->create(this->positions,this->maxinum, 3, "pair:positions");

}

// 모델을 사용하여 Energy 및 Force 계산
void PairCNMP::compute(int eflag, int vflag)
{
    ev_init(eflag, vflag);

    if (eflag_atom)
    {
        error->all(FLERR, "Pair style CNMP does not support atomic energy");
    }

    if (vflag_atom)
    {
        error->all(FLERR, "Pair style CNMP does not support atomic virial pressure");
    }

    if (vflag)
    {
        if (this->virialWarning == 0)
        {
            this->virialWarning = 1;
            error->warning(FLERR, "Pair style CNMP does currently not support virial pressure");
        }
    }

    this->prepareGNN();
    this->performGNN();
    
}

// 모델 준비
void PairCNMP::prepareGNN()
{
    int i;
    int iatom;

    int* type       = atom->type;
    double** x      = atom->x;

    int inum        = list->inum;
    int* ilist      = list->ilist;

    double* boxlo   = domain->boxlo;

    if (inum > this->maxinum)
    {
        this->maxinum = inum + this->maxinum / 2;

        memory->grow(this->forces, this->maxinum, 3,    "pair:forces");
        memory->grow(this->charge, this->maxinum,       "pair:charge");
        memory->grow(this->chi,    this->maxinum,       "pair:chi");
        memory->grow(this->atomNums, this->maxinum,     "pair:atomNums");
        memory->grow(this->positions, this->maxinum, 3, "pair:positions");
    }

    // set cell (domain->h[] 는 cell 길이와 각도를 포함)
    this->cell[0][0] = domain->h[0]; // xx (길이)
    this->cell[1][1] = domain->h[1]; // yy (길이)
    this->cell[2][2] = domain->h[2]; // zz (길이)
    this->cell[2][1] = domain->h[3]; // yz (yz 평면에서의 기울기)
    this->cell[2][0] = domain->h[4]; // xz (xz 평면에서의 기울기)
    this->cell[1][0] = domain->h[5]; // xy (xy 평면에서의 기울기)
    this->cell[0][1] = 0.0;
    this->cell[0][2] = 0.0;
    this->cell[1][2] = 0.0;

    // set atomNums and positions
    // MPI 환경에서 반복문이 병렬로 실행되도록 설정하는 명령어
    #pragma omp parallel for private(iatom, i)
    for (iatom = 0; iatom < inum; ++iatom)
    {
        i = ilist[iatom];

        this->atomNums[iatom] = this->atomNumMap[type[i]];
        
        this->positions[iatom][0] = x[i][0] - boxlo[0];
        this->positions[iatom][1] = x[i][1] - boxlo[1];
        this->positions[iatom][2] = x[i][2] - boxlo[2];
    }
}

// 모델 실행
void PairCNMP::performGNN()
{
    int i;
    int iatom;

    double** f = atom->f;
    double* q = atom->q;
    double e_field = 0.0;

    int inum = list->inum;
    int* ilist = list->ilist;

    double evdwl = 0.0;

    evdwl = this->calculatePython();

    // set total energy
    if (eflag_global)
    {
        eng_vdwl += evdwl;
    }

    // set atomic forces and charges (forces와 charge 값은 calculatePython에서 업데이트가 이루어짐.)
    for (iatom = 0; iatom < inum; ++iatom)
    {
        i = ilist[iatom];

        f[i][0] += this->forces[iatom][0];
        f[i][1] += this->forces[iatom][1];
        f[i][2] += this->forces[iatom][2];

        q[i] = this->charge[iatom];
    }

}

// set pair_style (pair_style CNMP cuda:1 ../../potentials/prediction)
void PairCNMP::settings(int narg, char **arg)
{
    //if (comm->nprocs > 1)
    //{
    //    error->all(FLERR, 'Pair style CNMP does not support MPI parallelization')
    //}

    if (narg < 2)
    {
        return;
    }

    this->gpuName = arg[0];

    this->npythonPath = narg -1;
    this->npythonPath = narg -4;
    this->pythonPath = new char*[this->npythonPath];

    for (int i = 0; i < this->npythonPath; ++i)
    {
        this->pythonPath[i] = new char[512];
        //strcpy는 문자열 복사 첫번째 인자에 두번째 인자 주소를 복사
        strcpy(this->pythonPath[i], arg[i+1]);
    }

    float e_start = atof(arg[narg-3]);
    float e_end = atof(arg[narg-2]);
    int e_axis = atoi(arg[narg-1]);

    std::cout << "E-field list: " << e_start << ", " << e_end << ", " << e_axis << std::endl;

    e_field_list = std::make_tuple(e_start, e_end, e_axis);
}

// pair_coeff * * CNMP Hf O
void PairCNMP::coeff(int narg, char **arg)
{
    int i, j;
    int count;

    int ntypes = atom->ntypes;
    int ntypesEff;

    int gpu = withGPU();

    if (narg != (3+ntypes))
    {
        error->all(FLERR, "Incorrect number of arguments for pair_coeff of pair_style CNMP");
    }

    if (strcmp(arg[0], "*") != 0 || strcmp(arg[1], "*") != 0)                                                  
    {
        error->all(FLERR, "Only wildcard asterisk is allowed in place of atom types for pair_coeff.");
    }

    if (this->atomNumMap != nullptr)
    {
        delete this->atomNumMap;
    }

    this->atomNumMap = new int[ntypes + 1];

    // ntypesEff 는 실제로 처리된 원소 타입의 수를 의미함.
    ntypesEff = 0;
    for (i = 0; i < ntypes; ++i)
    {
        if (strcmp(arg[i + 3], "NULL") == 0)
        {
            this->atomNumMap[i + 1] = 0;
        }
        // NULL이 아닌 원소 타입에 대해서만 atomNumMap에 원자 번호를 저장함.
        else
        {
            this->atomNumMap[i + 1] = this->elementToAtomNum(arg[i + 3]);
            ntypesEff++;
        }
    }
      

    if (ntypesEff < 1)
    {
        error->all(FLERR, "There are no elements for pair_coeff of CNMP.");
    }

    if (!allocated)
    {
        allocate();
    }

    if (this->initializedPython)
    {
        this->finalizePython();
    }

    // 모델 이름과 GPU 사용 여부를 인자로 전달해서 cutoff를 return으로 받음.
    this->cutoff = this->initializePython(gpu); 

    if (this->cutoff <= 0.0)
    {
        error->all(FLERR, "Cutoff is not positive for pair_coeff of CNMP.");
    }

    count = 0;

    // 각 원소 타입 쌍에 대해 설정이 적용될 수 있는지 확인하고, setflag 배열을 업데이트 함.
    // 실제로 적용 가능한 설정의 수 count를 계산함.
    for (i = 1; i <= ntypes; ++i)
    {
        for (j = i; j <= ntypes; ++j)
        {
            if (this->atomNumMap[i] > 0 && this->atomNumMap[j] > 0)
            {
                setflag[i][j] = 1;
                count++;
            }
            else
            {
                setflag[i][j] = 0;
            }
        }
    }  
      
    if (count == 0)
    {
        error->all(FLERR, "Incorrect args for pair coefficients");
    }
}

// 원자 Pair 간의 거리의 제곱을 사용하여 cutsq에 저장함.
// LAMMPS에서는 효율성을 높이기 위해 거리의 제곱을 사용함 (거리를 사용하면 Root를 수행해줘야 하는 번거로움 발생).
double PairCNMP::init_one(int i, int j)
{
    if (setflag[i][j] == 0)
    {
        error->all(FLERR, "All pair coeffs are not set");
    }

    double r, rr;

    r = this->cutoff;
    rr = r * r;

    cutsq[i][j] = rr;
    cutsq[j][i] = rr;

    return r;
}

void PairCNMP::init_style()
{
    if (strcmp(update->unit_style, "metal") != 0)
    {
        error->all(FLERR, "Pair style CNMP requires 'units metal'");
    }

    int* periodicity = domain->periodicity;

    //if (!(periodicity[0] && periodicity[1] && periodicity[2]))
    //{
    //    error->all(FLERR, "Pair style CNMP requires periodic boundary condition");
    //}

    // 시스템 초기화 단계에서 완전한 이웃 리스트를 필요로 한다는 것을 의미함.
    neighbor->add_request(this, NeighConst::REQ_FULL); 
}

// 현재 cpp 파일은 gpu version이 아님.
int PairCNMP::withGPU()
{
    return 0;
}

// e_field_list를 설정하는 메서드
void PairCNMP::setList(const std::tuple<float, float, int>& list)
{
    this->e_field_list = list;
}

// e_field_list를 반환하는 메서드
std::tuple<float, float, int> PairCNMP::getList() const
{
    return this->e_field_list;
}

void PairCNMP::finalizePython()
{
    if (this->initializedPython == 0)
    {
        return;
    }

    Py_XDECREF(this->pFunc);
    Py_XDECREF(this->pModule);

    Py_Finalize();
}

double PairCNMP::initializePython(int gpu)
{
    if (this->initializedPython != 0)
    {
        return this->cutoff;
    }

    double cutoff = -1.0;

    PyObject* pySys    = nullptr;
    PyObject* pyPath   = nullptr;
    PyObject* pyName   = nullptr;
    PyObject* pModule  = nullptr;
    PyObject* pFunc    = nullptr;
    PyObject* pyArgs   = nullptr;
    PyObject* pyArg1   = nullptr;
    PyObject* pyArg2   = nullptr;
    PyObject* pyArg3   = nullptr;
    PyObject* pyValue  = nullptr;

    Py_Initialize();

    pySys  = PyImport_ImportModule("sys");
    pyPath = PyObject_GetAttrString(pySys, "path");

    pyName = PyUnicode_DecodeFSDefault(".");
    if (pyName != nullptr)
    {
        PyList_Append(pyPath, pyName);
        Py_DECREF(pyName);
    }

    if (this->pythonPath != nullptr)
    {
        for (int i = 0; i < this->npythonPath; ++i)
        {
            pyName = PyUnicode_DecodeFSDefault(this->pythonPath[i]);
            if (pyName != nullptr)
            {
                PyList_Append(pyPath, pyName);
                Py_DECREF(pyName);
            }
        }
    }

    pyName = PyUnicode_DecodeFSDefault("prediction");
    if (pyName != nullptr)
    {
        pModule = PyImport_Import(pyName);
        Py_DECREF(pyName);
    }

    if (pModule != nullptr)
    {
        pFunc = PyObject_GetAttrString(pModule, "CNMP_initialize");

        if (pFunc != nullptr && PyCallable_Check(pFunc))
        {
            pyArg1 = PyBool_FromLong(gpu);
            pyArg2 = PyUnicode_FromString(gpuName.c_str());  // GPU 이름을 Python 문자열로 변환

            pyArgs = PyTuple_New(2);
            //e_field_list 값을 파싱하여 Python 리스트로 반환
            pyArg3 = PyList_New(3);
            PyList_SetItem(pyArg3, 0, PyFloat_FromDouble(std::get<0>(this->e_field_list)));
            PyList_SetItem(pyArg3, 1, PyFloat_FromDouble(std::get<1>(this->e_field_list)));
            PyList_SetItem(pyArg3, 2, PyLong_FromLong(std::get<2>(this->e_field_list)));
            
            pyArgs = PyTuple_New(3);
            PyTuple_SetItem(pyArgs, 0, pyArg1);
            PyTuple_SetItem(pyArgs, 1, pyArg2);
            PyTuple_SetItem(pyArgs, 2, pyArg3);

            pyValue = PyObject_CallObject(pFunc, pyArgs);

            Py_DECREF(pyArgs);

            if (pyValue != nullptr && PyFloat_Check(pyValue))
            {
                this->initializedPython = 1;
                cutoff = PyFloat_AsDouble(pyValue);
            }
            else
            {
                if (PyErr_Occurred()) PyErr_Print();
            }

            Py_XDECREF(pyValue);
        }

        else
        {
            if (PyErr_Occurred()) PyErr_Print();
        }

        Py_XDECREF(pFunc);

        pFunc = PyObject_GetAttrString(pModule, "CNMP_get_energy_forces_and_charge");

        if (pFunc != nullptr && PyCallable_Check(pFunc))
        {
            // NOP
        }
        else
        {
            this->initializedPython = 0;
            if (PyErr_Occurred()) PyErr_Print();
        }

        //Py_XDECREF(pFunc);
        //Py_DECREF(pModule);
    }

    else
    {
        if (PyErr_Occurred()) PyErr_Print();
    }

    if (this->initializedPython == 0)
    {
        Py_XDECREF(pFunc);
        Py_XDECREF(pModule);

        Py_Finalize();

        error->all(FLERR, "Cannot initialize python for pair_coeff of CNMP.");
    }

    this->pModule = pModule;
    this->pFunc   = pFunc;

    return cutoff;
}

double PairCNMP::calculatePython()
{
    int i;
    int iatom;
    int natom = list->inum;

    double energy      = 0.0; // 계산된 에너지를 저장할 변수
    double fieldEnergy = 0.0; // 에너지를 계산할 때 필요한 필드 에너지를 저장할 변수
    int hasFieldEnergy = 0;   // 에너지 계산 여부를 저장할 변수
    int hasEnergy      = 0;   // 에너지 계산 여부를 저장할 변수
    int hasForces      = 0;   // 힘 계산 여부를 저장할 변수
    int hasCharge      = 0;   // 전하 계산 여부를 저장할 변수
    int hasChi         = 0;   // Chi 계산 여부를 저장할 변수

    PyObject* pFunc  = this->pFunc; // pFunc은 Python 함수를 가리키는 포인터 (CNMP_initializer 메서드에서 설정)
    PyObject* pyArgs  = nullptr;
    PyObject* pyArg1  = nullptr;
    PyObject* pyArg2  = nullptr;
    PyObject* pyArg3  = nullptr;
    PyObject* pyAsub  = nullptr;
    PyObject* pyValue = nullptr;
    PyObject* pyVal1  = nullptr;
    PyObject* pyVal2  = nullptr;
    PyObject* pyVal3  = nullptr;
    PyObject* pyVal4  = nullptr;
    PyObject* pyVal5  = nullptr;
    PyObject* pyVsub  = nullptr;
    PyObject* pyVobj  = nullptr;

    // set cell -> pyArgs1 시뮬레이션 Cell 정보를 Python List 로 변환
    pyArg1 = PyList_New(3);

    for (i = 0; i < 3; ++i)
    {
        pyAsub = PyList_New(3);
        PyList_SetItem(pyAsub, 0, PyFloat_FromDouble(this->cell[i][0]));
        PyList_SetItem(pyAsub, 1, PyFloat_FromDouble(this->cell[i][1]));
        PyList_SetItem(pyAsub, 2, PyFloat_FromDouble(this->cell[i][2]));
        PyList_SetItem(pyArg1, i, pyAsub);
    }

    // set atomNums -> pyArgs2 각 원자의 번호를 나타내는 리스트를 생성함.
    pyArg2 = PyList_New(natom);

    for (iatom = 0; iatom < natom; ++iatom)
    {
        PyList_SetItem(pyArg2, iatom, PyLong_FromLong(this->atomNums[iatom]));
    }

    // set positions -> pyArgs3 각 원자의 위치를 나타내는 리스트를 생성함.
    pyArg3 = PyList_New(natom);

    for (iatom = 0; iatom < natom; ++iatom)
    {
        pyAsub = PyList_New(3);
        PyList_SetItem(pyAsub, 0, PyFloat_FromDouble(this->positions[iatom][0]));
        PyList_SetItem(pyAsub, 1, PyFloat_FromDouble(this->positions[iatom][1]));
        PyList_SetItem(pyAsub, 2, PyFloat_FromDouble(this->positions[iatom][2]));
        PyList_SetItem(pyArg3, iatom, pyAsub);
    }

    // call function Cell 정보, 원자 번호, 원자 위치 정보를 인자로 하는 튜플을 생성하고, 이를 사용하여 Python 함수를 호출함.
    pyArgs = PyTuple_New(3);
    PyTuple_SetItem(pyArgs, 0, pyArg1);
    PyTuple_SetItem(pyArgs, 1, pyArg2);
    PyTuple_SetItem(pyArgs, 2, pyArg3);

    pyValue = PyObject_CallObject(pFunc, pyArgs); // Python 함수를 호출해서 Energy, Force 및 Charge 계산
/*
    // Code Check
    PyObject* pyArgsStr = PyObject_Str(pyArgs); // PyObject_Repr도 사용 가능
    if (pyArgsStr != nullptr) {
        const char* argsStr = PyUnicode_AsUTF8(pyArgsStr);
        std::cout << "pyArgs: " << argsStr << std::endl;
        Py_DECREF(pyArgsStr);
    } else {
        std::cout << "Failed to convert pyArgs to string" << std::endl;
    }

    // 함수 호출 결과인 pyValue의 내용을 출력
    if (pyValue != nullptr) {
        PyObject* pyValueStr = PyObject_Str(pyValue); // PyObject_Repr도 사용 가능
        if (pyValueStr != nullptr) {
            const char* valueStr = PyUnicode_AsUTF8(pyValueStr);
            std::cout << "pyValue: " << valueStr << std::endl;
            Py_DECREF(pyValueStr);
        } else {
            std::cout << "Failed to convert pyValue to string" << std::endl;
        }
    } else {
        std::cout << "pyValue is nullptr" << std::endl;
    }
*/
    Py_DECREF(pyArgs);

    if (pyValue != nullptr && PyTuple_Check(pyValue) && PyTuple_Size(pyValue) >= 5)
    {
        // get energy <- pyValue
        pyVal1 = PyTuple_GetItem(pyValue, 0); // 에너지는 튜플의 첫 번째 요소로
        /*
        // energy type check
        if (pyVal1 != nullptr) {
            PyObject* pyType = PyObject_Type(pyVal1); // pyVal1의 타입 객체를 얻습니다.
            PyObject* pyTypeName = PyObject_Str(pyType); // 타입 객체의 문자열 표현을 얻습니다.
            const char* typeName = PyUnicode_AsUTF8(pyTypeName); // 문자열 표현을 C 스타일 문자열로 변환합니다.
            
            if (PyFloat_Check(pyVal1)) {
                double energyValue = PyFloat_AsDouble(pyVal1);
                std::cout << "The energy value (pyVal1) is a float and its value is: " << energyValue << std::endl;
            } else {
                std::cout << "The extracted pyVal1 is not a float. It is: " << typeName << std::endl;
            }
            
            // 사용이 끝난 후 PyObject의 참조 카운트를 감소시켜 메모리 누수를 방지합니다.
            Py_DECREF(pyType);
            Py_DECREF(pyTypeName);
        } else {
            std::cout << "Failed to extract pyVal1 from pyValue." << std::endl;
        }
        */
        if (pyVal1 != nullptr && PyFloat_Check(pyVal1))
        {
            hasEnergy = 1;
            energy = PyFloat_AsDouble(pyVal1);
            //std::cout << "Energy from pyValue: " << energy << std::endl;
        }
        else
        {
            std::cout << "Failed to get energy from pyValue." << std::endl;
            if (PyErr_Occurred()) PyErr_Print();
        }

        pyVal2 = PyTuple_GetItem(pyValue, 1); // Energy E-field는 튜플의 두 번째 요소로
        if (pyVal2 != nullptr && PyFloat_Check(pyVal2))
        {
            hasFieldEnergy = 1;
            fieldEnergy = PyFloat_AsDouble(pyVal2);
            this->energy_e_field = fieldEnergy;
            //std::cout << "Field Energy from pyValue: " << fieldEnergy << std::endl;
        }
        else
        {
            std::cout << "Failed to get field energy from pyValue." << std::endl;
            if (PyErr_Occurred()) PyErr_Print();
        }

        // get forces <- pyValue
        pyVal3 = PyTuple_GetItem(pyValue, 2); // Forces는 튜플의 세 번째 요소로
        /*
        if (pyVal2 != nullptr) {
            PyObject* pyType = PyObject_Type(pyVal2); // pyVal1의 타입 객체를 얻습니다.
            PyObject* pyTypeName = PyObject_Str(pyType); // 타입 객체의 문자열 표현을 얻습니다.
            const char* typeName = PyUnicode_AsUTF8(pyTypeName); // 문자열 표현을 C 스타일 문자열로 변환합니다.
            
            if (PyList_Check(pyVal2)) {
                std::cout << "The float value (pyVal2) is a list" << std::endl;
            } else {
                std::cout << "The extracted pyVal2 is not a list. It is: " << typeName << std::endl;
            }
            
            // 사용이 끝난 후 PyObject의 참조 카운트를 감소시켜 메모리 누수를 방지합니다.
            Py_DECREF(pyType);
            Py_DECREF(pyTypeName);
        } else {
            std::cout << "Failed to extract pyVal2 from pyValue." << std::endl;
        }
        */
        if (pyVal3 != nullptr && PyList_Check(pyVal3) && PyList_Size(pyVal3) >= natom)
        {
            hasForces = 1;
            //std::cout << "Forces array size: " << PyList_Size(pyVal2) << std::endl;
            
            for (iatom = 0; iatom < natom; ++iatom)
            {
                pyVsub = PyList_GetItem(pyVal3, iatom); // Atom 별로 Force를 받음.
                if (pyVsub != nullptr && PyList_Check(pyVsub) && PyList_Size(pyVsub) >= 3)
                {
                    for (i = 0; i < 3; ++i)
                    {
                        pyVobj = PyList_GetItem(pyVsub, i);
                        if (pyVobj != nullptr && PyFloat_Check(pyVobj))
                        {
                            this->forces[iatom][i] = PyFloat_AsDouble(pyVobj);
                            //std::cout << this->forces[iatom][i] << " ";
                        }
                        else
                        {
                            if (PyErr_Occurred()) PyErr_Print();
                            hasForces = 0;
                            break;
                        }
                    }
                }
                else
                {
                    if (PyErr_Occurred()) PyErr_Print();
                    hasForces = 0;
                    break;
                }

                if (hasForces == 0)
                {
                    break;
                }
            }
        }
        else
        {
            if (PyErr_Occurred()) PyErr_Print();
        }

        // get charge <- pyValue
        pyVal4 = PyTuple_GetItem(pyValue, 3); // Charge는 튜플의 네 번째 요소로
        /*
        if (pyVal3 != nullptr) {
            PyObject* pyType = PyObject_Type(pyVal3); // pyVal1의 타입 객체를 얻습니다.
            PyObject* pyTypeName = PyObject_Str(pyType); // 타입 객체의 문자열 표현을 얻습니다.
            const char* typeName = PyUnicode_AsUTF8(pyTypeName); // 문자열 표현을 C 스타일 문자열로 변환합니다.
            
            if (PyList_Check(pyVal3)) {
                std::cout << "The float value (pyVal3) is a list" << std::endl;
            } else {
                std::cout << "The extracted pyVal2 is not a list. It is: " << typeName << std::endl;
            }
            
            // 사용이 끝난 후 PyObject의 참조 카운트를 감소시켜 메모리 누수를 방지합니다.
            Py_DECREF(pyType);
            Py_DECREF(pyTypeName);
        } else {
            std::cout << "Failed to extract pyVal3 from pyValue." << std::endl;
        }
        */
        if (pyVal4 != nullptr && PyList_Check(pyVal4) && PyList_Size(pyVal4) >= natom)
        {
            hasCharge = 1;
            //std::cout << "Charge array size: " << PyList_Size(pyVal3) << std::endl;
            
            for (iatom = 0; iatom < natom; ++iatom)
            {
                pyVsub = PyList_GetItem(pyVal4, iatom); // Atom 별로 Charge를 받음.
                if (pyVsub != nullptr && PyFloat_Check(pyVsub))
                {
                    this->charge[iatom] = PyFloat_AsDouble(pyVsub);
                }
                else
                {
                    if (PyErr_Occurred()) PyErr_Print();
                    hasCharge = 0;
                    break;
                }
                if (hasCharge == 0)
                {
                    break;
                }
            }
        }
        else
        {
            if (PyErr_Occurred()) PyErr_Print();
        }

        pyVal5 = PyTuple_GetItem(pyValue, 4); // Chi는 튜플의 다섯 번째 요소로
        if (pyVal5 != nullptr && PyList_Check(pyVal5) && PyList_Size(pyVal5) >= natom)
        {
            hasChi = 1;
            //std::cout << "Chi array size: " << PyList_Size(pyVal5) << std::endl;
            
            for (iatom = 0; iatom < natom; ++iatom)
            {
                pyVsub = PyList_GetItem(pyVal5, iatom); // Atom 별로 Chi를 받음.
                if (pyVsub != nullptr && PyFloat_Check(pyVsub))
                {
                    this->chi[iatom] = PyFloat_AsDouble(pyVsub);
                }
                else
                {
                    if (PyErr_Occurred()) PyErr_Print();
                    hasChi = 0;
                    break;
                }
                if (hasChi == 0)
                {
                    break;
                }
            }
        }
        else
        {
            if (PyErr_Occurred()) PyErr_Print();
        }

    }

    else
    {
        if (PyErr_Occurred()) PyErr_Print();
    }

    Py_XDECREF(pyValue); // 함수 호출 결과에 대한 참조를 해제

    if (hasEnergy == 0 || hasFieldEnergy == 0 || hasForces == 0 || hasCharge == 0 || hasChi == 0)
    {   
        //std::cout << "Energy Check: " << hasEnergy << std::endl;
        //std::cout << "Forces Check: " << hasForces << std::endl;
        error->all(FLERR, "Cannot calculate energy, forces and charge by python of CNMP.");
    }

    return energy; // 계산된 에너지를 반환 f는 이미 사용 변수 초기화시 정의된 변수 사용.
}

double PairCNMP::compute_chi(int atomID)
{
    return this->chi[atomID];
}

double PairCNMP::compute_field()
{
    return this->energy_e_field;
}

static const int NUM_ELEMENTS = 118;

static const char* ALL_ELEMENTS[] = {
    "H",  "He", "Li", "Be", "B",  "C",  "N",  "O",  "F",  "Ne", "Na", "Mg", "Al", "Si", "P",  "S",
    "Cl", "Ar", "K",  "Ca", "Sc", "Ti", "V",  "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge",
    "As", "Se", "Br", "Kr", "Rb", "Sr", "Y",  "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd",
    "In", "Sn", "Sb", "Te", "I",  "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd",
    "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "Hf", "Ta", "W",  "Re", "Os", "Ir", "Pt", "Au", "Hg",
    "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th", "Pa", "U",  "Np", "Pu", "Am", "Cm",
    "Bk", "Cf", "Es", "Fm", "Md", "No", "Lr", "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds", "Rg", "Cn",
    "Nh", "Fl", "Mc", "Lv", "Ts", "Og"
};

int PairCNMP::elementToAtomNum(const char *elem)
{
    char elem1[16];

    strcpy(elem1, elem);

    this->toRealElement(elem1);

    if (strlen(elem1) > 0)
    {
        for (int i = 0; i < NUM_ELEMENTS; ++i)
        {
            if (strcasecmp(elem1, ALL_ELEMENTS[i]) == 0)
            {
                return (i + 1);
            }
        }
    }

    char estr[256];
    sprintf(estr, "Incorrect name of element: %s", elem);
    error->all(FLERR, estr);

    return 0;
}

void PairCNMP::toRealElement(char *elem)
{
    int n = strlen(elem);
    n = n > 2 ? 2 : n;

    int m = n;

    for (int i = 0; i < n; ++i)
    {
        char c = elem[i];
        if (c == '0' || c == '1' || c == '2' || c == '3' || c == '4' ||
            c == '5' || c == '6' || c == '7' || c == '8' || c == '9' || c == ' ' ||
            c == '_' || c == '-' || c == '+' || c == '*' || c == '~' || c == ':' || c == '#')
        {
            m = i;
            break;
        }

        elem[i] = c;
    }

    elem[m] = '\0';
}