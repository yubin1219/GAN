# GAN (Cenerative Advesarial Networks)
GAN은 생성자(generator,G)와 구분자(discriminator,D), 두 네트워크를 적대적(adversarial)으로 학습시키는 비지도 학습 기반 생성모델이다. 

G는 Zero-Mean Gaussian으로 생성된 z를 받아서 실제 데이터와 비슷한 데이터를 만들어 내도록 학습된다. D는 실제 데이터와 G가 생성한 가짜 데이터를 구별하도록 학습된다.
