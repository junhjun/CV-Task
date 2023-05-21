# Report on CNN Practical Exercise Assignment

## 1. 실험 개요
‘Bag of Tricks for Image Classification with CNN’ 논문을 읽고 논문에서 소개된 다양한 기법들을 직접 적용해서 성능 변화를 확인 후 결과를 정리한다.
- 실험은 https://github.com/bentrevett/pytorch-image-classification의 ResNet50 모델을 기반으로 진행한다.
- Cosine Learning Rate Decay, Label Smoothing, Mixup 세 가지 기법과 이들의 조합에 대한 성능 비교를 실시한다.

<br>

## 2. 실험 환경
Google Colab, GPU NVIDIA A100-SXM4-40GB, CUDA 11.8, Pytorch 2.0.0, Torchvision 0.15.1

<br>

## 3. 실험 방법
**(1) 데이터셋 준비**
- 데이터셋은 CUB200의 2011년 버전을 사용한다. 이는 500x500 해상도의 총 200개 종의 조류 이미지로 구성된 데이터셋이다. Train Set / Valid Set / Test Set은 총 8472 / 942 / 2374개로 구성한다.
- 이미지 데이터를 pre-trained model에 맞도록 224x224 해상도로 조정하고 정규화를 시킨다. 학습 데이터의 경우 이미지에 최대 5도의 무작위 회전, 50% 확률의 좌우반전, 임의의 Crop 등의 증강 기법을 적용해 모델이 다양하게 학습할 수 있도록 한다.

**(2) 모델 선정**
- Torchvision에서 제공하는 ResNet50 pre-trained model을 사용한다. 본 모델의 학습 가능한 파라미터는 총 23,917,832개이다.
- Batch size : 64, Epoch : 5, Adam Optimizer, OneCycle Learning Rate Scheduler를 기본 값으로 사용하여 학습한다.
<p align="center"><img width="500" alt="image" src="https://github.com/junhjun/CV-task/assets/53384975/e177b77a-61b8-466e-99ec-db36630affde"></p>
<p align="center"><img width="500" alt="image" src="https://github.com/junhjun/CV-task/assets/53384975/463f9d96-9a4f-4fc4-8784-ce205d9daedb"></p>

**(3) 기법 적용**
- 해당 논문은 기본 ResNet model에 Efficient Training 테크닉들을 적용하고 ResNet 구조에 변화를 주는 Model Tweaks를 적용한다. 이에 더해서, 학습 시 적용할 수 있는 Training Refinements 기법들을 적용해보며 성능 향상을 이끌어낸다.
- 본 실험에 적용될 기법은 Training Refinements 기법들 중 Cosine Learning Rate Decay, Label Smoothing, Mixup 세 가지이며, 아래와 같은 조합으로 실험을 진행한다. <br><br>

  **a. Cosine Learning Rate Decay**
  - Cosine Decay를 적용했을 시, Learning Rate는 선형적으로 증가하는 워밍업을 거친 후, Cosine 함수를 따라 0까지 점진적으로 감소하게 된다. 이에 따라, Validation Accuracy도 부 드럽게 상승할 수 있다.
  - 적절한 Learning Rate 초기값을 찾아 설정하고, Cosine Decay로 Learning Rate를 조절한다.
  <br>
  
  **b. Label Smoothing**
  - Label Smoothing은 정답인 레이블에 확률을 몰아 주는 것이 아닌, 오답인 레이블에 대해서도 조금씩 확률을 부여하도록 Loss를 구성하는 방법이다. 즉, 이미지 분류 시 완벽하게 특정 클래 스라고 분류하는 것이 아닌 다른 클래스일 가능성도 부여하며, 확률 값을 적절히 퍼트려 주는 것이라고 할 수 있다.
  - Loss Function을 새로 정의하고, 기존의 Cross Entropy 대신 사용한다.
  <br>
  
  **c. Mixup (α = 0.1, 0.01)**
  - xup은 학습 데이터셋으로부터 두 개의 이미지와 레이블을 뽑고, 이미지와 레이블을 각각 섞어서 새로운 이미지와 레이블을 생성한다. 즉, 비율에 따라 두 이미지가 섞인 하나의 이미지 가 생성되고, 레이블 또한 비율에 따라 섞여 새로운 레이블이 생성된다.
  - 이렇게 생성된 데이터를 학습 데이터셋에 추가하면, 모델의 학습 능력과 일반화 성능을 향상 시킬 수 있다.
  - Beta(α, α) 분포에 의해 얻은 [0, 1] 값에 따라 섞인 이미지, 섞인 레이블을 새롭게 정의한다. 본 실험에는 α가 0.1, 0.01인 경우에 대해 결과를 비교한다.
  - 기존 Train 함수를 수정하여, Mixup 기법을 적용하도록 한다.
  <br>
  
  **d. Cosine Learning Rate Decay + Label Smoothing**
  <br>
  
  **e. Cosine Learning Rate Decay + Mixup**
  <br>
  
  **f. Label Smoothing + Mixup**
  <br>
  
  **g. Cosine Learning Rate Decay + Label Smoothing + Mixup**
  <br>

**(4) 결과 출력 및 성능 비교**
- 성능 비교를 위한 시각화 기법으로 TensorBoard를 이용한다. 각각의 학습 과정에서 매 epoch 마다 결과 데이터를 저장한다. 학습과 평가 과정에서 Train Loss, Top-1 Validation Accuracy, Top-5 Validation Accuracy, Learning Rate 데이터를 저장하고, TensorBoard를 이 용해 원하는 결과를 선택해 비교할 수 있도록 한다.
- 이 때, 명확한 비교를 위해 Smoothing과 Outlier 제거 기능은 해제한다. 
 
<br>

## 4. 실험 결과
**a. Cosine Learning Rate Decay**
- Base model은 빨간색으로, Cosine Decay 기법으로 학습한 model은 노란색으로 표시했다.
- Cosine Decay가 적용된 model은 초기 Learning Rate가 Base model 보다 낮은 것을 확인할 수 있다. Train Loss와 Validation Accuracy 모두 성능이 매우 낮아짐을 알 수 있다.
- 초기 Learning Rate 값 설정과 관련하여 낮은 성능의 원인이 있을 것 같다.
<p align="center"><img width="500" alt="image" src="https://github.com/junhjun/CV-task/assets/53384975/081b5b54-156d-4a12-a771-7d9f3ab4e48a"></p>
<p align="center"><img width="500" alt="image" src="https://github.com/junhjun/CV-task/assets/53384975/ad5b7c71-7847-4790-ab8f-20f209cb5067"></p>

<br>

**b. Label Smoothing**
- Base model은 빨간색으로, Label Smoothing 기법으로 학습한 model은 초록색으로 표시했다.
- Label Smoothing 기법이 적용된 경우, Top-1, Top-5 모두 Base model모다 높은 정확도를 보였지만, Train Loss는 상대적으로 느리게 감소하였다.
<p align="center"><img width="500" alt="image" src="https://github.com/junhjun/CV-task/assets/53384975/e51e3b29-99e4-44ab-a0b6-fc61757d7ac9"></p>
<p align="center"><img width="500" alt="image" src="https://github.com/junhjun/CV-task/assets/53384975/62ebeb38-1072-4d0a-a1ed-fc2198d188db"></p>

<br>

**c. Mixup**
- Base model은 빨간색으로, α = 0.1로 학습한 model은 하늘색, α = 0.01로 학습한 model은 파란색으로 표시했다.
- Mixup 기법이 적용된 경우, 두 경우 모두 Base model 보다 Train Loss가 상대적으로 느리게 감소하고, Base model에 비해 낮은 정확도를 보였다.
- α 값의 두 경우를 비교해보면 α=0.01일 때의 성능이 더 높으므로, 이후 Mixup 기법을 적용 할 때는 α = 0.01로 고정하여 사용했다.
<p align="center"><img width="500" alt="image" src="https://github.com/junhjun/CV-task/assets/53384975/f77ebccb-6c62-4009-8d18-61e5f3309502"></p>
<p align="center"><img width="500" alt="image" src="https://github.com/junhjun/CV-task/assets/53384975/d07944e5-94e0-48ad-8bc7-70e034ff75a8"></p>

<br>

**d. Cosine Learning Rate Decay + Label Smoothing**
- Base model은 빨간색으로, Cosine Decay + Label Smoothing 기법으로 학습한 model은 보라 색으로 표시했다.
- Cosine Decay 기법을 사용했을 때, 성능 감소가 워낙 크고, Label Smoothing 기법을 사용했 을 때 성능 향상이 작기 때문에, 전체적으로는 성능이 하락된 것으로 보였다. 따라서, 두 기법 을 동시에 사용했을 때 시너지 효과는 기대할 수 없었다.
<p align="center"><img width="500" alt="image" src="https://github.com/junhjun/CV-task/assets/53384975/e3492af8-bf26-4e5a-9a33-368b294d5811"></p>
<p align="center"><img width="500" alt="image" src="https://github.com/junhjun/CV-task/assets/53384975/90c39911-bbd3-4766-be42-cc8a80218694"></p>

<br>

**e. Cosine Learning Rate Decay + Mixup**
- Base model은 빨간색으로, Cosine Decay + Mixup 기법으로 학습한 model은 분홍색으로 표시했다.
- 두 기법을 각각 사용했을 때, 성능이 감소만 되었기 때문에 전체적으로 성능이 더욱 하락된 것으로 보였다. 따라서, 두 기법은 서로 보완해주는 기능이 없기에 함께 사용하지 않을 게 더 좋음을 알 수 있다.
<p align="center"><img width="500" alt="image" src="https://github.com/junhjun/CV-task/assets/53384975/49a7bbfd-e1bd-4485-9d39-87593e05f385"></p>
<p align="center"><img width="500" alt="image" src="https://github.com/junhjun/CV-task/assets/53384975/f3b3642c-b68b-4ce6-ac36-88df74c5b637"></p>

<br>

**f. Label Smoothing + Mixup**
- Base model은 빨간색으로, Labeling Smoothing + Mixup 기법으로 학습한 model은 연두색 으로 표시했다.
- 성능 하락이 매우 작게 있었던 두 기법을 함께 사용하니, 성능이 향상되는 결과를 확인할 수 있었다. Train Loss 감소는 Base model에서 더욱 잘 되지만, 정확도 관점에서는 epoch을 더 많이 주면 정확도 향상을 기대할 수 있을 것 같다.
- 성능 향상에 비해, 학습 시간은 큰 차이가 없음을 알 수 있다.
<p align="center"><img width="500" alt="image" src="https://github.com/junhjun/CV-task/assets/53384975/3c12a12e-0555-4ebe-b5d1-22aec9a98d63"></p>
<p align="center"><img width="500" alt="image" src="https://github.com/junhjun/CV-task/assets/53384975/8a732d30-b4d9-4d59-9186-f12b16a738ea"></p>

<br>

**g. Cosine Learning Rate Decay + Label Smoothing + Mixup**
- Base model은 빨간색으로, Cosine Decay + Labeling Smoothing + Mixup 기법으로 학습한 model은 흰색으로 표시했다.
- 세 가지 기법을 모두 적용한 model을 학습 시켰을 때, 성능은 하락하는 것을 확인할 수 있다. Label Smoothing + Mixup (f)의 긍정적 효과에 비해, Cosine Decay 기법의 영향이 매우 크기 때문에 이러한 결과가 나왔다.
<p align="center"><img width="500" alt="image" src="https://github.com/junhjun/CV-task/assets/53384975/261f5a9b-4ea4-4e1a-be34-7156c59775bd"></p>
<p align="center"><img width="500" alt="image" src="https://github.com/junhjun/CV-task/assets/53384975/3df6dbdf-be39-491f-9ae9-bfce268add48"></p>

<br>

## 5. 결론
전체 실험 중 실험 **b. Label Smoothing** 기법만 적용한 model이 가장 우수한 결과로 나왔다. 또한, 실험 **f. Label Smoothing + Mixup** 기법의 조합도 성능 향상을 기대할 수 있을 것으로 확인됐다. Cosine Learning Rate Decay를 적용한 model의 경우, 성능이 비정상적으로 낮게 나왔기 때문에 해당 방법이 올바르게 적용된 것인지 자세히 확인해 볼 필요가 있다. Mixup 기법의 경우, α 값이 달라짐에 따라 성능에 큰 영향을 주므로 적절한 값을 찾아 사용해야 한다. 각 기법들을 조합했을 때, 시너지를 내거나 유의미한 변화가 생기는 경우는 확인할 수 없었다. 더 나은 컴퓨팅 환경을 갖추고, epoch 수를 높인다면 각 기법의 영향을 확실하게 관찰할 수 있을 것이다.
<p align="center"><img width="500" alt="image" src="https://github.com/junhjun/CV-task/assets/53384975/519924ad-9c2b-41a4-873f-45f27f32b085"></p>

<br>

## 6. 소감
‘Bag of Tricks for Image Classification with CNN’ 논문은 모델 구조의 큰 변화 없이 비교적 사소한 트 릭만으로도 성능을 높일 수 있는 방법을 제시한다. 이미지 분류 시, 일반적으로 취하는 데이터 전처리 기법부터 소개하고 점진적으로 트릭들을 추가해가며 성능 비교를 한다.

이미지 관련 프로젝트에서 코드 작성할 때를 생각해보니, 가장 먼저 데이터를 로드 해오고, Resize, Rotation, Crop 등 몇 가지 증강 기법을 적용하는 절차를 별 생각없이 그냥 했던 것 같다. 또한, GPU 가 허용하는 한 batch size는 키우는 것이 정확도 높이기에 좋고, batch size를 키우는 만큼 epoch과 learning rate 또한 늘려주어야 한다는 것은 이유도 잘 모른 채 그저 하나의 팁으로 머릿속에 넣어 두 고 있었다. 논문을 읽으며 그런 사소한 것들이 어떻게 모델 성능 향상을 이끌 수 있었는지 이해할 수 있었다. 무언가를 배울 때 그냥 받아들였던 것들에 대해서 자세히 알게 되었다. 더 알려고 노력하지 않 았던 것 같다.

해당 논문은 실제 Task에 적용하기 좋은 실용적인 논문이다. 본 과제를 진행하며 머릿속으로 이해는 되 었지만, 실제로 구현하는데 있어서 많은 부족함을 느꼈다. 기본 ResNet 모델에 기법을 적용하는 것은 큰 어려움은 없었지만 먼저, 코드를 작성하며 비효율적으로 작성하고 있다는 느낌이 계속 들었다. 머릿 속으로는 변수만 바꾸게 하여 특정 기법만 적용해 학습을 시켜 결과를 저장하는 것까지 효율적으로 짤 수 있을 것 같았지만 뜻대로 잘 되지 않았다.

어떨 때는 학습을 마치고, 변수를 초기화 시킨 다음 새로운 학습을 진행했는데, 매우 높은 정확도로 시 작하는 오버피팅 같은 결과가 나타났다. 여러 시도를 해 보다가, Train Iterator를 초기화 하여 재학습 시키니 정상적인 결과가 나왔다. 결국은 해결이 됐지만, 왜 해결 되었는지도 잘 모르는 이런 찝찝한 경 험이 자주 생기는 것 같다.

TensorBoard를 사용한 시각화도 처음 해 보았다. 학습 과정에서 매 epoch 마다 결과값을 저장하도록 하고, 파일 형식으로 저장된 데이터를 이용해 시각화를 하는 원리로 이해 했다. Colab을 사용해 여러 학습을 돌렸고 잠깐 한 눈 팔 때마다 런타임 연결이 끊겨, 처음부터 다시 학습하는 것을 몇 번이나 반 복했다. 매 학습이 끝날 때마다, 결과값을 압축해 구글 드라이브에 저장하도록 했고, 런타임 연결이 끊 겨도 다시 학습할 일이 없게 했다.

GPU 환경에서 학습을 돌려보고 싶은 생각이 마구 들었다. Colab 환경은 유료 GPU를 사용하지 않는 한, 속도도 느리고 간단하게 사용하는 목적이 아니라면 불편한 점이 더 많은 것 같다. 특히, 데이터셋이 그렇게 크지 않아도 1 epoch에 2-3분이 소요되어, 높은 epoch으로 정확한 결과를 내기 어려웠다. ResNet 모델 자체의 파라미터 수가 굉장히 많아 어쩔 수 없는 부분도 있지만 아쉬운 마음이 컸다.

앞으로 논문을 분석하고 직접 적용해서 결과를 내야 할 일이 많을 것이다. 이걸 알기 때문에, 어렵고 마음대로 잘 안 되는 게 있었지만 결과를 내기까지 계속 앉아서 할 수 있었다. 과제를 통해 반드시 갖 추고 길러야 할 능력이 무엇인지 알았다. 조금 더 공부해서 Knowledge Distillation 기법, ResNet-D 모 델 같이 적용해보지 못한 것들도 시도해보고 싶다.
