# ML팀 README
## 문제제기
![image](https://user-images.githubusercontent.com/79970424/126072008-b45a3e83-dca0-4c9f-991c-42fc53264902.png)

---
## 프로젝트 목적
![image](https://user-images.githubusercontent.com/79970424/126072022-8c2dd7f6-6077-40b3-85d4-387d3cc9eb49.png)
- 광고효과 발현시기 파악
- 목표판매달성을 위한 미디어 KPI설정
- 목표판매달성을 위한 최적의 채널 조합 추천

---
## MMM(Media Mix Model) 구축 히스토리
![image](https://user-images.githubusercontent.com/79970424/126072076-36dd7a28-5e59-4378-988d-7389104cb94e.png)

---
## 머신러닝 선형회귀
![image](https://user-images.githubusercontent.com/79970424/126072092-852153fd-9afa-4923-b399-34346eb20cde.png)
- 데이터가 선형 모양이 나오지 않음

---
## 머신러닝 선형회귀 (네이버 검색량 활용)
![image](https://user-images.githubusercontent.com/79970424/126072104-10c9b0d4-f5a9-4892-87fc-65f402298942.png)
![image](https://user-images.githubusercontent.com/79970424/126072110-70d107ff-6752-46fe-a858-cb695df2eea8.png)
- 데이터가 선형 모양이 나오지 않음

---
## Ad_Stock 모델 활용
![image](https://user-images.githubusercontent.com/79970424/126072122-48939c5a-10ac-4849-8146-e51ecc5fd1f8.png)
![image](https://user-images.githubusercontent.com/79970424/126072123-3ea4c371-3de7-4ede-bbf0-4f9f21ca03ef.png)
- 데이터가 선형 모양이 나오지 않음

---
## 딥러닝 MLP 모델 활용
![image](https://user-images.githubusercontent.com/79970424/126072145-bba0762a-3379-49c9-b73f-8ff915868fa6.png)
![image](https://user-images.githubusercontent.com/79970424/126072148-96a788dc-2263-4ef7-af81-be99235659f1.png)
- 데이터의 군집이 선형 모양이 나오기 시작함

---
## MMM 구축에 들어간 데이터 소개
![image](https://user-images.githubusercontent.com/79970424/126072172-b4ecb1d0-39a9-4101-9f5a-85a972d88890.png)

---
## 최종 MMM 데이터와 모델 구성
![image](https://user-images.githubusercontent.com/79970424/126072187-ed7da822-ebce-40bd-aa39-02d5ee35b95f.png)

---
## 광고효과의 발현시기 도출
![image](https://user-images.githubusercontent.com/79970424/126072212-6dcaece6-343a-470f-89e4-794f955fb6d8.png)
- 당월 광고량으로 모델이 예측한 값을 토대로 오차가 적은 달을 선택
- 5 ~ 6개월 후의 등록대수가 오차가 적음
- 오차가 적다는 의미는 당월 광고량이 5 ~ 6개월 후의 등록대수와 큰 상관관계가 있다는 뜻
- 즉 토요타의 경우 광고효과가 5 ~ 6개월 후에 나타난다고 해석해 볼 수 있음

---
## 목표판매달성을 위한 미디어KPI
![image](https://user-images.githubusercontent.com/79970424/126072271-ab402ca2-2d17-4edc-9a90-fb3f5bf6de73.png)
- 목표판매량 최솟값과 최댓값을 입력하면 모델이 예측한 값을 토대로 광고량 제공

---
## 목표판매달성을 위한 최적의 채널조합 추천
![image](https://user-images.githubusercontent.com/79970424/126072293-f7022169-cdc7-4256-a271-8907fcb67ec0.png)
- 목표판매량 최솟값과 최댓값을 입력하면 모델이 예측한 값을 토대로 채널 조합 제공
