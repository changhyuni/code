# GPU란 무엇일까
<aside>

## A. 왜, 만들어 졌을까?

### 1.1 그래픽 처리용

![Image](https://github.com/user-attachments/assets/35171368-22e2-4b9a-882d-2cf2642a85bf)
![Image](https://github.com/user-attachments/assets/326d797e-676b-4475-a9a1-1b63a44972b8)

- 처음 GPU가 만들어졌을 때의 주된 목적은 **3D 그래픽 렌더링**의 **실시간 처리**를 가능케 하려는 것
- 고해상도·고프레임으로 게임이나 3D 애플리케이션을 구동하려면, 초당 수천만~수억 픽셀 연산(색상, 텍스처, 조명 계산)이 필요함
- **CPU**는 소수의 강력한 코어로 순차 처리하기에, 픽셀 단위의 방대한 동일 연산을 빠르게 감당하기 어려움

### 1.2 그래픽 파이프라인 간단히 살펴보기
<img width="764" alt="Image" src="https://github.com/user-attachments/assets/5ed24cec-9790-484c-83f4-50f60457c2cb" />

<aside>
💡

일반적인 그래픽 처리는 수많은 “점”을 찍어서 삼각형을 만들고 그 삼각형을 모여서 3D를 만들어낸다

</aside>

1. **Vertex Processing(정점 처리)**
    - 3D 모델의 각 정점(Vertex) 좌표, 조명(노말 벡터) 계산
2. **Rasterization(래스터화)**
    - 3D 공간에 있는 폴리곤(삼각형)을 2D 픽셀로 쪼개는 과정
3. **Pixel/Fragment Shader**
    - 각 픽셀에 텍스처, 조명 효과, 그림자 등을 연산해 색상 결정
4. **Blending/Output**
    - 최종 픽셀 값 합성, 프레임 버퍼에 저장 후 화면에 출력

이 전체 단계를 실시간(초당 수십 프레임 이상)으로 수행해야 하므로, 결국 **GPU 병렬 연산**이 필수적

### 1.3 결론, 대규모 픽셀 연산 = 병렬성 극대화
![Image](https://github.com/user-attachments/assets/4be825c3-57e0-4b24-9e89-52f798bad79e)

- 픽셀/프래그먼트 연산은 사실상 “동일 수학적 계산”을 엄청나게 많이 반복해야 하는 구조 → 자연스럽게 **병렬화**에 잘 맞음
- **GPU**는 수백수천 개의 간단한 코어를 통해 **동시에** 대량의 연산을 수행 → 실시간 그래픽(초당 3060프레임 이상의 3D 장면) 구현 가능
- 예: 1920×1080 해상도라면 약 207만 픽셀, 초당 60프레임을 그리려면 1초에 1억 2천만 개 이상의 픽셀 연산
- 각 픽셀별 색상, 조명, 텍스처, 그림자 계산 등은 수학적으로 동일한 계산을 반복하는 구조

---

### GPGPU

GPU를 그래픽 렌더링에 말고 다른산업(AI, 암호화페 채굴)에 사용하면서 GPGPU라는 용어가 등장한다

**GPGPU(General-Purpose Computing on GPUs)**:

- 그래픽 작업 이외에도 GPU의 **‘병렬 연산 능력’을 일반(범용) 계산**에 활용하는 접근.
- 예: 과학 시뮬레이션(유체역학, 분자 시뮬), 금융 모델링, 이미지·영상 처리, 암호화폐 채굴, AI/딥러닝 등.
- NVIDIA의 **CUDA**, Khronos의 **OpenCL** 등이 GPGPU 프로그래밍을 지원하는 대표 플랫폼.

---

## B. GPU와 CPU

### CPU 동작방식

- **Central Processing Unit**, 컴퓨터 시스템의 “두뇌” 역할
- 운영체제(OS)와 애플리케이션에서 내려오는 **명령어**를 해석(Decode)하고, **연산(Execute)**을 수행하는 핵심 장치

<img width="473" alt="Image" src="https://github.com/user-attachments/assets/d7a25cc9-9a26-486d-84b5-ced0a348879c" />

**특징**

- 프로그램(명령어)과 데이터가 **단일 메모리**에 저장
- CPU는 이 메모리에서 **명령어를 순차적으로 Fetch**하고 **해석(Decode)** 하여 **실행(Execute)**
    
    ![Image](https://github.com/user-attachments/assets/1f1f0d7a-6fe6-4196-b877-6005a9bdcaf9)
    
- 즉, **프로그램 자체를 메모리에 저장**해 소프트웨어적으로 바꿔주기만 하면, 하드웨어 배선 변경 없이 새로운 연산을 수행 가능

---

**그렇다면 Register, ALU, CU는 어떤 역할은?** 

1.1 레지스터(Register)

- CPU 내부에 존재하는 **초고속 메모리**.
- 명령어 실행 과정에서 **즉시 사용**할 데이터를 저장·전달하기 위해 사용함.
- 접근 속도가 L1 캐시보다도 더 빠르며, 용량은 매우 작음(수십~수백 개).

1.2 ALU(Arithmetic Logic Unit)

- CPU 내부에서 **산술 연산(덧셈, 뺄셈 등)** 및 **논리 연산(AND, OR, XOR, NOT 등)**을 수행
- Instruction Cycle의 **Execute** 단계에서 가장 핵심적인 계산 기능 담당

1.3 CU(Control Unit, 제어 장치)

1. **역할**
    - CPU 내부의 “지휘자 역할”
    - Instruction Register(IR)에 있는 명령어를 해석(**Decode**)하고, **ALU**나 **레지스터**, **메모리**가 어떻게 동작해야 할지를 지시

**1.4 코어(Core)**

1. **정의**
    - 현대 CPU에서 “코어”는 **독립된 실행 유닛**을 의미
    - 각 코어가 **자체적인 레지스터, ALU, CU** 등을 갖추고 있어, 한 코어가 하나의 프로그램(스레드)을 실행 가능
    - 사실상, 가장 중요한 부분
2. **멀티코어(Multi-Core)**
    - 하나의 CPU 패키지(실리콘 칩)에 **2개 이상**의 코어를 탑재
    - 예: 듀얼코어(2코어), 쿼드코어(4코어), 8코어, 16코어 등
    - 각 코어는 **명령어 파이프라인**과 **ALU**, **CU**, **레지스터 세트**를 독립적으로 가짐
    - 메모리, 캐시, 시스템 버스는 공유하거나 일부 분할하여 사용(L1은 코어별, L2/L3는 공유 등)
3. **SMT(동시 멀티스레딩, Hyper-Threading)**
    - 물리 코어 하나가 논리적 코어(스레드) 두 개 이상을 동시에 스케줄링
    - 레지스터 세트나 파이프라인 일부를 분산해 동시 처리 가능
        
        ![Image](https://github.com/user-attachments/assets/a9b35ba1-c3da-4e28-ac20-3b227b256264)
        

<aside>
📌

요약하자면

- **메모리에 프로그램 + 데이터**가 저장
- **CPU(코어) 내부**
    - **CU**가 명령어를 Fetch → Decode
    - Decode 결과를 따라 **ALU**가 연산, **레지스터**에 입출력
    - **PC**와 **IR**는 명령어 사이클의 순서와 내용 관리
</aside>

### GPU 동작방식

**가장 큰 특징은 동일 연산**(예: 픽셀 셰이더, 행렬 연산)을 수천 개 코어가 한꺼번에 수행 → **높은 처리량(Throughput) 이다**
아키텍처를 보면 수많은 단순 코어(Shader, CUDA core)를 갖춘 스트리밍 멀티프로세서(SM) 단위로 구성되어 있다

![Image](https://github.com/user-attachments/assets/d46b5ba2-ba31-44ab-9fcb-8110a14027c4)

- **SM(Streaming Multiprocessor)**
    - GPU 내부에서 실제 연산(Arithmetic, Load/Store 등)을 수행하는 핵심 단위
    - SM 내부에 여러 실행 파이프라인(코어, Tensor Core, 스케줄러 등) + 레지스터 파일 + 공유 메모리(Shared Memory)
- **L2 캐시 + DRAM(HBM/GDDR)**
    - 모든 SM이 공유하는 **L2 캐시**
    - 고대역폭 메모리(HBM, GDDR)를 통해 외부로부터 데이터를 가져옴
    - 예: NVIDIA A100은 40MB L2와 최대 2TB/s 대역폭의 HBM 메모리를 가짐

### 그럼 왜, GPU가 AI에 유리한가?

**Amdahl의 법칙(Amdahl’s Law)**

1. **정의**
    - 시스템(프로그램) 전체가 일정 부분은 병렬화할 수 있고, 나머지는 순차적(병렬화가 불가능)인 경우, 
    병렬화 가능한 부분을 빠르게(병렬 프로세서 증가) 만들어도, **전체 성능 향상의 한계**가 존재한다는 법칙.
    - 하지만 **병렬화 비율이 클수록**(= AI/딥러닝처럼 대규모 행렬 연산), 코어 수를 늘렸을 때 **성능 향상**(Speedup)이 매우 커짐
2. **공식**

![Image](https://github.com/user-attachments/assets/aa39f2b5-8367-4be2-bb24-f1ad4f68c617)

- p: 병렬화 가능한 비율(0~1)
- NNN: 병렬 프로세서(코어) 수
- **결과**: 병렬화 비율 p가 높을수록, 코어 수 N 증가에 따른 Speedup이 커짐

AI/딥러닝의 대규모 행렬·벡터 계산(Convolution, GEMM 등)은 병렬화 비율이 매우 높음(p≈1에 가깝다).
코어(프로세서) 수를 늘리면 성능 향상이 크며, **GPU**가 **수천 개 코어**로 이러한 패턴에 최적화.

<img width="772" alt="Image" src="https://github.com/user-attachments/assets/49c55d58-087b-4bf4-a3d9-d8752e26ed1d" />

1. **딥러닝은 대규모 병렬 연산 덩어리**
    - 인공신경망(Neural Network) 훈련(Training)은 주로 **행렬 곱셈**과 **가중치 업데이트** 과정으로 구성
    - 수백만~수억 개의 파라미터(Weight)를 가진 모델에서, 반복적으로 **행렬 연산**을 수행해야 함
    - CPU의 소수 코어로는 연산량이 방대해 **훈련 시간이 매우 길어짐**
2. **GPU의 구조와 딥러닝의 궁합**
    - **GPU**: 대량의 코어(Streaming Multiprocessor) + 고대역폭 메모리(GDDR, HBM) → 행렬 곱셈을 병렬로 동시에 처리
    - **반복적이고 단순한 연산**(예: 벡터·행렬 연산)을 **수천 개 코어**로 나누어 처리하므로 CPU 대비 속도 향상이 큼
    - 대규모 데이터셋(이미지, 텍스트)을 GPU에 태워, **한 번에 많은 샘플**(mini-batch)을 병렬 학습 가능
3. **학습(Training)과 추론(Inference)** 에 모두 사용
    - **Training**: 모델 파라미터를 최적화하기 위해, 엄청난 양의 연산(Forward+Backward Propagation)을 매번 반복 → GPU로 몇 배~수십 배 이상 빠르게 가능
    - **Inference**: 실제 배포 환경에서 이미지 분류, 음성 인식 등 실시간 추론 시에도 GPU가 처리량을 극대화
4. **Tensor Core 등 전용 AI 하드웨어**
    - 최신 GPU(NVIDIA Volta/Turing/Ampere 등)는 **Tensor Core**를 추가해 FP16/FP32 혼합 정밀도 연산에 특화 → 딥러닝 훈련 속도를 더욱 가속
    - CPU에는 없는 AI 특화 유닛이라, **딥러닝 모델 트레이닝 시 GPU 우위**가 더 커짐

<aside>
📌

요약하자면 

**CPU는 “낮은 지연시간(Low Latency)” 최적화**

- **CPU 코어**는 상대적으로 수가 적고(수 개~수십 개), 각 코어가 복잡한 제어 로직(Out-of-Order, 분기 예측 등)을 갖춤
- **순차적·복잡 로직**에 강점 → OS 운영, 다양한 애플리케이션
- 병렬화가 잘 되는 대규모 연산에서는 코어 수가 제한적이므로, Amdahl의 법칙 상 **코어 증가 효과가 한계**가 있음

**GPU는 “높은 처리량(High Throughput)” 최적화**

- **GPU**는 **수백~수천 개**의 단순 코어(Shader/CUDA Core) → 동일 연산을 대규모 병렬 실행
- 병렬화 가능 부분(p)이 크다면, Amdahl의 법칙에 따라 **성능이 급격히 향상**
    
    pp
    
- AI/딥러닝(행렬 곱, 수백만 파라미터 연산 등)은 거의 전부 병렬화 가능 → GPU가 탁월한 처리속도를 낼 수 있음
</aside>

### GPU와 CPU는 같이 쓰인다

![Image](https://github.com/user-attachments/assets/74628c94-b663-4fb7-abc0-c9b2ce8d2f99)
<aside>
💡

GPU는 OS나 시스템 전체 로직을 수행하지 않음. CPU가 여전히 시스템 제어, 프로그램 흐름, OS 운영 담당

</aside>

1. **호스트 CPU 메모리 → GPU 메모리 복사**
    - CPU가 메인 메모리(RAM)에서 데이터를 준비 → CUDA API(`cudaMemcpy`) 등으로 GPU 메모리(Global Memory)로 전송
    - 예: 이미지, 행렬 등
2. **GPU 내부에서 커널(Kernel) 실행**
    - CPU 코드(`cudaLaunchKernel`) 또는 CUDA API를 통해 “GPU에서 실행할 함수(커널)” 호출
    - 대규모 병렬 스레드(블록, 그리드)로 연산 수행
    - Shared Memory, 레지스터 등을 활용해 고속 연산 진행
3. **결과물은 GPU 메모리에 저장**
    - 커널이 끝나면, GPU 메모리(Global Memory)에 최종 결과가 남음
    - 중간 과정은 GPU 내부 공유 메모리 / 레지스터에서 처리하지만, 최종은 Global Memory로 귀결
4. **GPU 메모리 → 호스트 CPU 메모리로 결과 복사**
    - CPU가 다시 `cudaMemcpy`(반대 방향)로 결과를 가져옴
    - 애플리케이션 로직(호스트 측)이 이 결과를 이용해 후속 처리(파일 저장, 시각화, 모델 저장 등)

</aside>

## C. 결론

<aside>
💡

컴퓨터가 GPU를 사용한다

</aside>

<aside>

**CPU 제어(Host)**
- CUDA 프로그래밍에서 **CPU 코드**가 “어떤 커널을 GPU에서 실행할지, 어떤 데이터 범위를 처리할지, 블록/그리드 크기 등”을 결정
- 즉, **GPU는 직접 OS나 메모리 관리**를 하지 않으며, CPU가 모든 지시를 담당

**GPU 병렬 실행(Device)**
- GPU 내부에는 **수백~수천 개**의 코어(Shader/CUDA Core), SM(Streaming Multiprocessor) 단위로 병렬 연산
- CUDA **스레드 블록**과 **그리드** 개념으로 대량 스레드를 효율적으로 스케줄링
- AI/딥러닝, 영상 처리, 과학 계산처럼 **반복적/동일 연산**이 많은 병렬 workload에 최적화

**메모리 이동이 필수**
- GPU와 CPU는 서로 독립적인 메모리(Host RAM vs GPU VRAM)을 사용
- 연산 전에 CPU→GPU로 데이터를 복사, 연산 완료 후 결과를 GPU→CPU로 가져와야 함
- 이 전송 비용이 상당할 수도 있으므로, **연산량 대비 전송 오버헤드**도 고려 필요
</aside>