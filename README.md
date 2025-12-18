- AlexNet style의 Convolution Neural Network의 구현
- parameter 제어를 통한 Overfitting과 Regularization 적용 model 사용
- 학습된 각 모델의 wrong, correct case 시각화

-----

>구현 환경
OS : Ubuntu 20.04 LTS
Python --version : 3.9.23
GPU : A6000

----

>실행 방법
train.py는 parameter를 통해 model을 제어할 수 있습니다.
<br>
--overfit : 활성화 시 overfitting model, 
            비활성화 시 regularization 적용 model
<br>
--lr : learning rate 명시, (default 1e-4)
<br>
--epochs : epoch 명시, (default: 40)
<br>
--batch : batch size 명시, (default: 128)
