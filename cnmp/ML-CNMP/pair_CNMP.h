#ifdef PAIR_CLASS

PairStyle(CNMP, PairCNMP)

# else

// header 파일이 중복으로 포함되지 않도록 보호함.
#ifndef LMP_PAIR_CNMP_H_
#define LMP_PAIR_CNMP_H_

#include <Python.h>                 // Python.h 파일을 include함.
#include <string>                   // string 헤더 파일을 include함.
#include "atom.h"                   // LAMMPS내 원자들에 대한 정보를 관리하는 클래스의 정의
#include "comm.h"                   // LAMMPS의 병렬 처리를 위한 프로세서 간 통신 기능
#include "pair.h"                   // LAMMPS의 Pair 클래스를 상속하기 위해 필요한 기본 Pair 클래스의 정의
#include "error.h"                  // LAMMPS의 에러 처리 및 경고 메시지를 위한 클래스의 정의
#include "input.h"                  // LAMMPS의 입력 파일을 읽고 파싱하는 클래스의 정의
#include "force.h"                  // LAMMPS내 Force Field를 계산하는데 사용되는 다양한 Force Field 관련 클래스
#include "update.h"                 // LAMMPS내 업데이트 관련 클래스의 정의
#include "memory.h"                 // LAMMPS내 동적 메모리 할당과 관련된 기능 (1D, 2D 배열 등)
#include "domain.h"                 // LAMMPS내 구조 박스의 정보를 관리하는 클래스의 정의 
#include "neighbor.h"               // LAMMPS내 이웃 목록 생성 및 관리를 담당하는 클래스의 정의
#include "neigh_list.h"             // LAMMPS내 이웃 목록과 관련된 데이터 구조를 정의
#include "neigh_request.h"          // LAMMPS내 이웃 목록 생성 요청을 관리하는 클래스의 정의

// LAMMPS 네임스페이스에 PairCNMP 클래스를 정의
namespace LAMMPS_NS
{
    class PairCNMP: public Pair {
        // Public은 클래스 외부에서 접근 가능 > LAMMPS 시 특정 작업을 수행하여야 할때 호출되어야 하는 함수들은 Public으로 정의 
        public:
        // PairCNMP 클래스의 생성자와 소멸자를 정의
            PairCNMP(class LAMMPS*);
            
            virtual ~PairCNMP() override;

            void compute(int, int) override;        // Model을 통해 Energy 및 Force를 계산하는 함수

            void settings(int, char **) override;   // Pair Style의 설정을 조절함 (pair_style CNMP ../../potentials/CNMP_driver)

            void coeff(int, char **) override;      // Pair Coeff의 설정을 조절함 (pair_coeff * * CNMP Hf O)

            double init_one(int, int) override;     // 모든 Pair의 초기화 작업을 수행함

            void init_style() override;             // Pair Style의 초기화 작업을 수행함

            void setList(const std::tuple<float, float, int>& list); // e_filed list를 설정하는 메서드
            std::tuple<float, float, int> getList() const;           // e_filed list를 반환하는 메서드

            double compute_chi(int atomID);         // chi 값을 계산하는 함수
            double compute_field();                 // e_field 값을 계산하는 함수
        
        protected:
            virtual int withGPU();                       // GPU를 사용하여 계산을 수행할지 여부를 결정하는 함수
            std::string gpuName;                         // GPU 이름
            std::tuple<float, float, int> e_field_list;  // e_field_list

        private:
            int*     atomNumMap;                     // 원자 번호 매핑
            int*     atomNums;                       // 원자 번호
            double*  chi;                            // chi 정보
            double*  charge;                         // 전하 정보
            double** cell;                           // 셀 정보
            double** positions;                      // 위치 정보
            double** forces;                         // 힘 정보
            double   energy_e_field;                 // 에너지 정보

            int      maxinum;                        // 최대값
            int      initializedPython;              // Python 초기화 여부
            int      virialWarning;                  // Virial Warning
            double   cutoff;                         // Cut-off 거리

            int      npythonPath;                    // Python 경로
            char**   pythonPath;                     // Python 경로

            PyObject* pModule;                       // Python 모듈
            PyObject* pFunc;                         // Python 함수

            void allocate();               

            void prepareGNN();               

            void performGNN();                

            void finalizePython();            

            double initializePython(int gpu);

            double calculatePython();            

            int elementToAtomNum(const char *elem);

            void toRealElement(char *elem);

    };
}

#endif
#endif