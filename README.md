---프로그램 버전---
파이썬 버전 : 3.13.6
모듈 버전
pandas : 3.1.0
scikit-learn : 1.9.1
joblib : 1.6.0

---주요 내용---
PV=nRT의 한계를 극복하기 위해 해당 프로그램을 제작함
NIST 표준 데이터를 기반으로 해당 프로그램 제작함

nist_data_download.py
NIST의 데이터를 온도별 데이터를 추출하여, nist_data 폴더 안에 엑셀 파일로 저장함

main_train.py
nist_data 폴더 안에 있는 엑셀 파일을 기반으로, 엔탈피, 엔트로피, 열용량 등과 같은 불필요한 정보를 제거하여 preprocessed_master 파일에 저장함
이 파일에 있는 온도, 압력, 부피 세가지 데이터를 기반으로 인공신경망을 훈련시킴
훈련 이후, 훈련 데이터 및 보정치 3차원 그래프를 출력하고, 이상 기체 방정식의 오차와 신경망 오차에 대한 통계를 출력함
다시 훈련을 반복하지 않도록 신경망에 대한 데이터를 model.pkl에 저장함

predict_volume
훈련시킨 모델을 통해, 온도와 압력을 입력하면 보정된 부피를 출력함

---실행방식---
우클릭 후 터미널 실행 or shift 우클릭 후 Powershell 열기
아래 코드 입력

pip install numpy pandas scikit-learn matplotlib joblib lxml

설치 완료 후, python .\nist_data_download.py 실행
오류 없을 시, python .\main_train.py 실행
그래프가 제대로 나왔다면, 새로 생긴 python창 닫고 python .\predict_volume.py 실행
압력, 온도 입력 후, 부피 확인 가능