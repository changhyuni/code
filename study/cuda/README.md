# 2주차 CUDA
## 1. CUDA(Compute Unified Device Architecture)
### 1.1 용어정리
- 호스트(Host): CPU와 시스템 메모리(RAM)를 의미
- 디바이스(Device): GPU와 전용 메모리(DRAM)를 의미
-  SIMT(Single Instruction Multiple Thread) : NVIDIA에서는 이를 SIMT라고 부르며, “한 가지 명령을 여러 스레드가 동시에 수행” / SIMD(Single Instruction Multiple Data)와 유사함

### 1.2 왜, 만들어 졌을까?
![Image](https://github.com/user-attachments/assets/74628c94-b663-4fb7-abc0-c9b2ce8d2f99)
- CUDA는 C 언어 문법에 몇 가지 확장을 추가해 GPU 함수를 작성할 수 있도록 함
- 일반 C/C++ 프로그램처럼 CPU에서 코드를 실행하다가, 대규모 병렬 처리가 필요한 부분만 GPU로 오프로드(Offload)
GPU 연산이 끝나면 결과를 다시 CPU로 받아와 후속 연산 or 결과 처리

### 1.3 GPU 함수?
- 일반 C/C++ 함수와 구분하기 위한 용어이다
- CUDA 프로그래밍에서는 CPU에서 실행되는 함수와 GPU에서 실행되는 함수 구분해야함
- 예를 들어 CUDA에서는 __global__ 키워드를 사용해 “이 함수는 GPU(디바이스)에서 실행될 함수다” 라고 선언
- 이렇게 선언된 함수를 흔히 “GPU 함수”, “커널(kernel)”이라고 부름

#### 1.3.1 컴파일 후 ‘GPU용 바이너리’로 만들어짐
- GPU용 함수는 CPU용 코드와는 별도로 NVCC(NVIDIA CUDA 컴파일러) 등을 통해 GPU에서 실행 가능한 형태(PTX, SASS 등)로 컴파일됨 (row언어)
![Image](https://github.com/user-attachments/assets/b6ea7aec-9c21-48ec-83ea-47115863886d)
- 실행 시점에 CPU와 GPU가 통신하면서, 해당 GPU 함수(커널)에 대한 호출(런치, launch)이 이뤄지면 GPU가 그 함수를 실행하는 구조입

#### 1.3.2 GPU에 함수? 헷갈리지 말자
- 물리적으로 GPU 내부에 '함수'가 상주해 있는 것이 아니라,
- GPU용으로 컴파일된 코드(커널)가 메모리에 올라가고, GPU는 이 코드를 자신이 가진 수많은 코어에서 실행된다는 뜻
---
## 2. 커널(Kernel)과 쓰레드(Threads)
<img width="328" alt="Image" src="https://github.com/user-attachments/assets/ac2d2fc3-3a53-4a6e-ba6b-f0d7b43b1c97" />    


- GPU에서 실행되는 함수를 “커널(Kernel)”이라고 지칭  
- 커널을 호출 할 때, 동시에 수많은 스레드(Thread)가 생성  
- 각 스레드는 하나의 CUDA 코어 위에서 병렬로 동작

### 2.1 쓰레드(Threads)
<img width="861" alt="Image" src="https://github.com/user-attachments/assets/9de61b44-ef45-4927-b8fc-5bf66b6a6660" />

- 스레드(Thread): 커널 코드의 실행 단위(각 스레드는 한 점/한 픽셀/한 데이터 요소를 처리)  
- 블록(Block): 여러 스레드를 그룹화한 단위  
- 그리드(Grid): 여러 블록을 모은 최상위 계층 구조  

### 2.1.1 계층 구조 예시
<img width="609" alt="Image" src="https://github.com/user-attachments/assets/9a80fb38-f015-47fe-be9c-17184ab534e8" />

- 그리드(Grid) 안에 여러 블록(Block) 존재
- 각 블록(Block) 안에 여러 스레드(Thread) 존재
- 한 번의 커널 런치 → 하나의 그리드가 GPU 전체 자원에 매핑 → 그 아래 블록과 스레드가 자동으로 할당

### 2.1.2 예시 (1D, 2D, 3D) 만든다면?
- 블록과 그리드는 각각 1차원, 2차원, 3차원 형태로 구성 가능
- 예) dim3 grid(3, 2); → X방향 블록 3개, Y방향 블록 2개 = 총 6개 블록
- 예) dim3 block(4, 3); → X방향 스레드 4개, - Y방향 스레드 3개 = 블록당 12개 스레드
최종적으로 그리드 × 블록 × 스레드를 통해 GPU에 동시에 할당될 스레드 수가 결정

Grid: 3 × 2 → 6개의 블록  
Block: 4 × 3 → 12개의 스레드  
총 스레드 수 = 6(블록 수) × 12(블록당 스레드) = 72개의 스레드
---
# 3. Cuda Programming Model
### 3.1 CUDA 프로그램의 기본 흐름
#### 3.1.1 CPU(호스트) 제어
<img width="876" alt="Image" src="https://github.com/user-attachments/assets/40445b80-307a-46ee-9cf5-0eb5e392a826" />

- 전통적인 C/C++ 프로그램처럼 main() 함수부터 시작
- 순차적으로 코드를 실행하다가, 병렬 처리가 필요한 코드 블록에 도달하면 해당 부분만 GPU로 오프로드
- GPU(디바이스)에서 특정 함수를 실행(커널 런치)한 직후, 다시 호스트로 제어가 돌아옴

#### 3.1.2 GPU에서는 커널(Kernel)이 대량의 스레드로 병렬 실행
- 실행이 끝나더라도, 호스트가 명시적으로 대기(Synchronize) 명령을 하지 않으면 CPU는 계속 다음 코드를 실행
- 즉, CPU와 GPU가 비동기적으로 동작 가능

(GPU에서 계산이 끝난 결과를 호스트가 필요로 할 때, `cudaDeviceSynchronize()` 같은 함수를 통해 대기 결과적으로 커널 종료 시점을 정확히 맞춰주어야 데이터 무결성을 보장)

#### 3.1.3 메모리 관리는?
![Image](https://github.com/user-attachments/assets/d46b5ba2-ba31-44ab-9fcb-8110a14027c4)

호스트 메모리 vs. 디바이스 메모리
- CPU는 시스템 RAM(호스트 메모리)을 사용
- GPU는 전용 VRAM(디바이스 메모리)을 사용
- 서로 다른 메모리 공간이므로, 커널에서 사용할 데이터는 디바이스 메모리에 직접 할당 후 복사해야 함
- 디바이스 메모리 할당
  - C언어의 `malloc/free`처럼, CUDA는 `cudaMalloc/cudaFree` 사용
- 데이터 전송(cudaMemcpy)
  - 호스트 ↔ 디바이스 간 메모리 복사 함수
  - 방향에 따라 `cudaMemcpyHostToDevice`, `cudaMemcpyDeviceToHost` 등을 지정
  - PCI Express 버스를 통해 이동하므로 전송 비용이 큼 → 최소화가 중요

정리
1. 호스트 메모리에 있는 데이터 준비
2. 디바이스 메모리 할당 (cudaMalloc)
3. 호스트 → 디바이스로 데이터 복사 (cudaMemcpyHostToDevice)
4. 커널 런치 (kernel<<<grid, block>>>(...))
5. 필요한 경우 디바이스 → 호스트로 결과 복사 (cudaMemcpyDeviceToHost)
6. 디바이스 메모리 해제 (cudaFree)

