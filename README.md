#### AlexNet style의 Convolution Neural Network의 구현
#### parameter 제어를 통한 Overfitting과 Regularization 적용 model 사용
#### 학습된 각 모델의 wrong, correct case 시각화

-----

## 구현 환경<br>
#### OS : Ubuntu 20.04 LTS
#### Python --version : 3.9.23
#### GPU : A6000

----

## train.py 실행 방법<br>
#### Parameter List of train.py
> * **--overfit** : 활성화 시 overfitting model,
> &emsp;&emsp;&emsp;&emsp;&nbsp;&nbsp;비활성화 시 regularization 적용 model
> * **--lr** : learning rate 명시, (default 1e-4)
> * **--epochs** : epoch 명시, (default: 40)
> * **--batch** : batch size 명시, (default: 128)

#### Example Command of train.py
> * ```python3 train.py --overfit``` **(Overfitting 실험)**
> * ```python3 train.py``` **(Regularization 적용 실험)**
>* ```python3 train.py --overfit --lr 1e-5 --epochs 80``` **(Overfitting 실험, learning rate: 1e-5, epoch: 80 지정)**

#### Output of train.py
>* mode : **overfit** or **regularized**
>* **result_{mode}.png** : Loss 및 Accuracy 변화 그래프
>* **model_{mode}.pth** : 학습된 모델의 가중치 파일

-----

## analysis.py 실행방법<br>
#### train.py를 통한 pth 파일이 존재해야 실행 가능합니다.
#### mode에 따른 wrong case,와 correct case를 시각화합니다.
>* ```python3 analysis.py --mode overfit```
>* ```python3 analysis.py --mode regularzied```