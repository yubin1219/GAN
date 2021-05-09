# GAN (Generative Adversarial Networks)

- DCGAN : https://arxiv.org/pdf/1511.06434.pdf
- SRGAN : https://arxiv.org/pdf/1609.04802.pdf
- ProGAN : https://arxiv.org/pdf/1710.10196.pdf
- StyleGAN : https://arxiv.org/pdf/1812.04948.pdf
- CycleGAN : https://arxiv.org/pdf/1703.10593.pdf
- pix2pix : https://arxiv.org/pdf/1611.07004.pdf
-------------------------------------------------
GAN은 생성자(generator,G)와 구분자(discriminator,D), 두 네트워크를 적대적(adversarial)으로 학습시키는 비지도 학습 기반 생성모델이다. 
G는 Zero-Mean Gaussian으로 생성된 z를 받아서 실제 데이터와 비슷한 데이터를 만들어 내도록 학습된다. D는 실제 데이터와 G가 생성한 가짜 데이터를 구별하도록 학습된다.

![캡처](https://user-images.githubusercontent.com/74402562/103651687-563d7b00-4fa5-11eb-8c88-74006ae760fb.PNG)

GAN의 목적함수
----------
![캡처1](https://user-images.githubusercontent.com/74402562/103651694-58073e80-4fa5-11eb-9075-bb52dfce7f9d.PNG)

GAN의 목적함수는 다음과 같다. D의 입장에서는 실제 데이터(x)를 입력하면 높은 확률이 나오도록 하고, 가짜 데이터(G(z))를 입력하면
확률이 낮아지도록 학습된다. G의 입장에서는 가짜 데이터(G(z))를 D에 넣었을 때 실제 데이터처럼 확률이 높게 나오도록 학습된다.


![캡처3](https://user-images.githubusercontent.com/74402562/103651697-59d10200-4fa5-11eb-9e02-daff7826cb15.PNG)

이처럼 D학습과 G학습을 번갈아 하면서 서로에게 Insight를 제공하는 형태로 학습한다.

--------------

![캡처4](https://user-images.githubusercontent.com/74402562/103651704-5b022f00-4fa5-11eb-9e66-91a71cf1060a.PNG)

G의 성능이 좋지 않을 때는 학습이 잘 되지 않는 문제를 해결하기 위해 위와 같이 수식을 조금 변형한다. 이렇게 되면 초기 G학습이 가속화된다.
