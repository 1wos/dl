# 심층 학습 핵심 개념 기출 문제 답안 정리

---

## I. 기본 수식 및 정의

### 1️⃣ Binary Cross Entropy Loss
**문제**: Write mathematical definitions for binary cross entropy loss  
**한글**: 이진 교차 엔트로피 손실 함수의 수학적 정의를 작성하시오

$$\mathcal{L}(\hat{y}, y) = -y \log(\hat{y}) - (1-y) \log(1-\hat{y})$$

**설명**: 이진 분류 문제의 손실 함수

---

### 2️⃣ Sigmoid
**문제**: Write mathematical definitions for sigmoid  
**한글**: 시그모이드 함수의 수학적 정의를 작성하시오

$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

**특징**: 출력 범위 $[0, 1]$, 확률값 표현에 사용

---

### 3️⃣ tanh (Hyperbolic Tangent)
**문제**: Write mathematical definitions for tanh  
**한글**: 쌍곡탄젠트 함수의 수학적 정의를 작성하시오

$$\tanh(z) = \frac{e^{z} - e^{-z}}{e^{z} + e^{-z}}$$

**특징**: 출력 범위 $[-1, 1]$, 제로 중심(zero-centered)

---

### 4️⃣ ReLU (Rectified Linear Unit)
**문제**: Write mathematical definitions for ReLU  
**한글**: ReLU 함수의 수학적 정의를 작성하시오

$$\text{ReLU}(z) = \max(0, z)$$

**특징**: 계산 간단, 기울기 소실 문제 완화

---

### 5️⃣ Leaky ReLU
**문제**: Write mathematical definitions for Leaky ReLU  
**한글**: Leaky ReLU 함수의 수학적 정의를 작성하시오

$$\text{Leaky ReLU}(z) = \max(0.01z, z)$$

**특징**: 음수 영역에서도 작은 기울기($0.01$) 유지

---

### 6️⃣ ELU (Exponential Linear Unit)
**문제**: Write mathematical definitions for ELU  
**한글**: ELU 함수의 수학적 정의를 작성하시오

$$f(x) = \begin{cases} 
x & \text{if } x > 0 \\
\alpha(e^{x} - 1) & \text{otherwise}
\end{cases}$$

**특징**: 음수 영역에서 지수 함수 사용

---

## II. 정규화 기법

### 7️⃣ Dropout
**문제**: Explain how dropout layers work. Assume $p$ is the probability that a node is kept.  
**한글**: 드롭아웃 레이어의 작동 방식을 설명하시오. $p$는 노드가 유지될 확률이다.

**작동 방식**:
- **훈련 시**: 각 노드를 확률 $p$ (keep_prob)로 유지, $(1-p)$로 제거
- **Inverted Dropout 스케일링**: 
$$a^{[l]} = \frac{a^{[l]}}{\text{keep\_prob}}$$

훈련 시 활성화를 keep_prob로 나누어 스케일링
- **테스트 시**: 드롭아웃 적용 안 함

**효과**:
- 과적합 방지 (Prevent overfitting)
- 분산 감소 (Reduce variance)
- 특정 특징 의존도 감소

---

### 8️⃣ Batch Normalization
**문제**: Write the final formula for batch normalisation. Assume $m$ is the training set size.  
**한글**: 배치 정규화의 최종 공식을 작성하시오. $m$은 훈련 세트 크기이다.

**4단계 수식**:

1. **평균 계산**:
$$\mu = \frac{1}{m} \sum_{i=1}^{m} z^{(i)}$$

2. **분산 계산**:
$$\sigma^2 = \frac{1}{m} \sum_{i=1}^{m} (z^{(i)} - \mu)^2$$

3. **정규화**:
$$z_{\text{norm}}^{(i)} = \frac{z^{(i)} - \mu}{\sqrt{\sigma^2 + \epsilon}}$$

4. **스케일 및 이동** (최종 공식):
$$\tilde{z}^{(i)} = \gamma \cdot z_{\text{norm}}^{(i)} + \beta$$

여기서 $\gamma$ (감마), $\beta$ (베타)는 학습 가능한 파라미터

---

### 9️⃣ Batch Normalization이 왜 작동하는가?
**문제**: Why does Batch Normalization work?  
**한글**: 배치 정규화가 왜 효과적인가?

**효과 3가지**:

1. **하이퍼파라미터 탐색 용이**
   - 하이퍼파라미터 선택 범위가 넓어짐
   - 네트워크가 더 강건(robust)해짐

2. **학습 속도 향상**
   - 은닉 유닛의 값 $z$를 정규화
   - 학습 속도 증가

3. **Covariate Shift 완화**
   - 이전 계층의 파라미터 변화로 인한 입력 분포 변화 안정화
   - 각 계층이 독립적으로 학습 가능

---

### 🔟 정규화 기법들의 Bias/Variance 효과
**문제**: When applying Batch Normalization, Dropout, or adding training data, indicate whether bias and variance increase or decrease.  
**한글**: 배치 정규화, 드롭아웃, 훈련 데이터 추가 시 편향과 분산이 증가/감소하는지 표시하시오.

| 기법 | Bias (편향) | Variance (분산) |
|------|-------------|-----------------|
| Batch Normalization | 유지 | **감소** $\downarrow$ |
| Dropout | 유지 | **감소** $\downarrow$ |
| 훈련 데이터 추가 | 유지 | **감소** $\downarrow$ |

---

## III. 과적합/과소적합 해결

### 1️⃣1️⃣ 높은 분산 (High Variance = 과적합) 해결
**문제**: What are 4 ways we can reduce the high variance problem for a very deep and large network?  
**한글**: 매우 깊고 큰 네트워크의 높은 분산 문제를 줄이는 4가지 방법은?

**답안 4가지**:

1. **더 많은 데이터 수집** (More data)
   - 훈련 데이터를 추가하여 일반화 성능 향상

2. **정규화** (Regularization)
   - L2 정규화, Dropout, Batch Normalization 등

3. **조기 종료** (Early Stopping)
   - 검증 오차가 증가하기 시작할 때 학습 중단

4. **네트워크 단순화** (NN architecture search/simplification)
   - 은닉층 수 감소, 뉴런 수 감소

---

### 1️⃣2️⃣ 높은 편향 (High Bias = 과소적합) 해결
**문제**: What are 4 ways we can reduce the high bias problem for a shallow network?  
**한글**: 얕은 네트워크의 높은 편향 문제를 줄이는 4가지 방법은?

**답안 4가지**:

1. **더 큰 네트워크** (Bigger network)
   - 은닉층 수 증가, 뉴런 수 증가

2. **더 오래 훈련** (Train longer)
   - 에폭 수 증가 또는 다른 최적화 알고리즘 사용

3. **NN 아키텍처 검색** (NN architecture search)
   - 더 적합한 네트워크 구조 탐색

4. **활성화 함수 변경**
   - 예: Sigmoid 대신 ReLU 사용

---

## IV. 역전파 (Backpropagation)

### 1️⃣3️⃣ Computation Graph 미분
**문제**: Calculate the gradient of $\frac{dJ}{dx}$ using the following computation graph.  
**한글**: 다음 계산 그래프를 사용하여 $\frac{dJ}{dx}$의 기울기를 계산하시오.

**문제 설정**:
- $y = 2x$
- $z = x + 5$
- $J = 3yz$

**풀이 과정**:

$x$는 두 경로를 통해 $J$에 영향:

1. **경로 1** ($x \to y \to J$): 
$$\frac{\partial J}{\partial y} \cdot \frac{\partial y}{\partial x} = (3z) \cdot (2) = 6z$$

2. **경로 2** ($x \to z \to J$): 
$$\frac{\partial J}{\partial z} \cdot \frac{\partial z}{\partial x} = (3y) \cdot (1) = 3y$$

**최종 답안**:
$$\frac{\partial J}{\partial x} = 6z + 3y$$

---

### 1️⃣4️⃣ Sigmoid의 지그재그 문제
**문제**: Explain the zigzagging dynamics of sigmoid and propose a solution to solve this problem.  
**한글**: 시그모이드의 지그재그 동역학을 설명하고 이를 해결할 방법을 제안하시오.

**문제점**:
- Sigmoid 출력 범위: $[0, 1]$ → **제로 중심이 아님** (not zero-centered)
- 입력 $x_i$가 모두 양수일 때:

$$\frac{\partial L}{\partial w_i} = \frac{\partial L}{\partial a} \cdot x_i$$

- 모든 가중치 기울기가 **같은 부호**
- 경사 하강법이 **비효율적인 지그재그 경로** 이동

**해결책 2가지**:
1. **tanh 사용**: 출력이 제로 중심 $[-1, 1]$
2. **ReLU 사용**: 포화(saturation) 문제 해결

---

### 1️⃣5️⃣ 2-Layer 역전파 수식
**문제**: Write the formulas for $dW^{[1]}$, $db^{[1]}$ in a two layered back propagation.  
**한글**: 2층 역전파에서 $dW^{[1]}$, $db^{[1]}$의 공식을 작성하시오.

**벡터화된 수식** (훈련 예제 $m$개):

1. **오차 전파**:
$$dZ^{[1]} = W^{[2]T} \cdot dZ^{[2]} \odot g'(Z^{[1]})$$

(※ $\odot$는 원소별 곱셈, element-wise multiplication)

2. **가중치 기울기**:
$$dW^{[1]} = \frac{1}{m} dZ^{[1]} \cdot X^T$$

3. **바이어스 기울기**:
$$db^{[1]} = \frac{1}{m} \sum_{\text{axis=1}} dZ^{[1]}$$

(keepdims=True 사용)

---

## V. CNN (Convolutional Neural Network)

### 1️⃣6️⃣ CNN 파라미터 감소 원리
**문제**: What are two properties of convolutions that enable us to reduce the number of parameters?  
**한글**: 합성곱이 파라미터 수를 줄일 수 있게 하는 두 가지 속성은?

**답안 2가지**:

1. **지역 연결성 (Local Connectivity)**
   - 각 출력 값이 입력의 작은 지역(receptive field)에만 연결
   - 전체 연결(fully connected)보다 연결 수 대폭 감소

2. **가중치 공유 (Weight Sharing)**
   - 동일한 필터(가중치 세트)를 입력의 모든 위치에서 재사용
   - 학습해야 할 파라미터 수 대폭 감소

---

### 1️⃣7️⃣ CNN 파라미터 개수 계산
**문제**: What are the number of trainable parameters including bias for a convolution that uses two $3 \times 3 \times 3$ filters?  
**한글**: $3 \times 3 \times 3$ 필터 2개를 사용하는 합성곱의 학습 가능한 파라미터 수는? (바이어스 포함)

**계산 과정**:
- 필터 1개당 가중치: $3 \times 3 \times 3 = 27$
- 필터 2개 총 가중치: $27 \times 2 = 54$
- 바이어스: 필터당 1개 $\to$ 2개
- **총 파라미터**: $54 + 2 = 56$개

**답**: **56개**

---

### 1️⃣8️⃣ Effective Receptive Field
**문제**: What is the effective receptive field of a $3 \times 3 \times 3$ filter?  
**한글**: $3 \times 3 \times 3$ 필터의 유효 수용 영역은?

**답**: **$3 \times 3$** (공간적 영역)

**쉬운 설명**:

**$3 \times 3 \times 3$ 필터의 의미:**
- 첫 번째 $3 \times 3$: 가로×세로 크기 (공간적 크기)
- 마지막 3: 깊이, 즉 입력 이미지의 채널 수 (RGB면 3)

**예시로 이해하기:**

입력 이미지가 **$32 \times 32$ 크기의 RGB 컬러 이미지**라면:
- 가로: 32픽셀
- 세로: 32픽셀  
- 깊이: 3채널 (R, G, B)

이때 $3 \times 3 \times 3$ 필터는:
- **가로×세로로는 $3 \times 3$ 영역만** 본다 (9개 픽셀)
- **깊이는 전체(3채널)를 다** 본다 (R, G, B 모두)

**핵심:**
- **Receptive field = 필터가 '한 번에 보는 영역'**
- 깊이(채널)는 항상 전체를 보지만
- **공간적으로는 $3 \times 3$만** 보기 때문에
- **Effective receptive field = $3 \times 3$**

**비유:**
창문(필터)으로 밖을 본다고 생각하면:
- 창문 크기는 $3 \times 3$ (가로×세로)
- 하지만 유리창 두께(깊이)는 상관없이 전체가 투명함
- 결국 **보이는 영역은 $3 \times 3$**

---

## VI. 최적화 (Optimization)

### 1️⃣9️⃣ Mini-batch의 장점
**문제**: What are the advantages of mini batch optimisation compared to batch optimisation?  
**한글**: 배치 최적화와 비교하여 미니 배치 최적화의 장점은?

**답안 2가지**:

1. **벡터화 이점 (Vectorization Speedup)**
   - 행렬 연산을 통한 벡터화 가능
   - GPU/CPU 병렬 처리 활용
   - 계산 속도 향상

2. **빠른 학습 진행**
   - 전체 훈련 세트를 처리하지 않고도 파라미터 업데이트
   - 더 자주 업데이트 → 빠른 수렴
   - 메모리 효율적

---

### 2️⃣0️⃣ 최소 파라미터 NN 설계
**문제**: Draw a neural network with a minimal number of training parameters for a one dimensional input where $y=0$ if $x>0$ and $y=1$ otherwise. Assume the BCE is used for the loss function and only sigmoid activation function can be used.  
**한글**: 1차원 입력에 대해 $x>0$이면 $y=0$, 그렇지 않으면 $y=1$인 최소 파라미터 신경망을 그리시오. BCE 손실 함수와 시그모이드 활성화 함수만 사용.

**답: 단일 뉴런 (로지스틱 회귀 모델)**

**신경망 구조도**:

```
     입력층        은닉층(없음)      출력층
     
       x  ────────W──────────> (z) ───σ───> ŷ
              │                          
              b (bias)                   
              │                          
              └─────────────────>        

     z = Wx + b
     ŷ = σ(z) = 1/(1 + e^(-z))
```

**또는 더 자세한 그림**:

```
                    ┌─────────────┐
     x (입력) ─────>│   뉴런 1개   │────> ŷ (출력)
                    │             │
                    │  W (가중치)  │     σ(Wx + b)
                    │  b (바이어스)│
                    │  σ (sigmoid) │
                    └─────────────┘
                    
     손실 함수: BCE Loss
```

**파라미터 상세**:
- **가중치 $W$**: 1개 (예: $W = -10$)
- **바이어스 $b$**: 1개 (예: $b = 0$)
- **총 파라미터**: **2개** (최소!)

**수식**:
$$z = Wx + b$$
$$\hat{y} = \sigma(z) = \frac{1}{1 + e^{-z}}$$

**손실 함수**:
$$\mathcal{L}(\hat{y}, y) = -y \log(\hat{y}) - (1-y) \log(1-\hat{y})$$

**조건**: 
- $W$는 음수여야 함 ($W \ll 0$)
- $b \approx 0$
- $x > 0$이면 $z < 0 \to \sigma(z) \approx 0$ ($y=0$)
- $x \le 0$이면 $z \ge 0 \to \sigma(z) \approx 1$ ($y=1$)

---