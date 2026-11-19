# 실제 기체 부피 시뮬레이션

이상 기체 방정식 `PV = nRT`로 계산한 몰부피와 [NIST Chemistry WebBook](https://webbook.nist.gov/chemistry/fluid/)의 실제 기체 몰부피 차이를 신경망으로 학습합니다. 온도와 압력을 입력하면 이상 기체 몰부피와 신경망으로 보정한 몰부피를 비교할 수 있습니다.

| 기체 | 폴더 | 학습 데이터 준비 |
| --- | --- | --- |
| 수소 (H₂) | `hydrogen` | 저장소에 포함된 데이터 사용 |
| 질소 (N₂) | `nitrogen` | 저장소에 포함된 데이터 사용 |
| 이산화탄소 (CO₂) | `carbondioxide` | 첫 학습 전에 NIST 데이터 다운로드 |

## 프로그램 버전

| 항목 | 기준 버전 |
| --- | --- |
| Python | 3.13.6 |
| pandas | 3.1.0 |
| scikit-learn | 1.9.1 |
| joblib | 1.6.0 |

## 주요 파일

| 파일 | 역할 |
| --- | --- |
| `nist_data_download.py` | 온도별 NIST 데이터를 `nist_data/`에 CSV로 저장합니다. |
| `main_train.py` | 학습에 필요한 열을 추려 `preprocessed_master.csv`를 생성합니다. 전체 데이터로 몰부피 보정량을 표준화해 학습하고 `model.pkl`에 저장합니다. 오차 통계와 3차원 비교 그래프를 출력합니다. |
| `predict_volume.py` | 저장된 모델을 읽어 입력한 온도와 압력에서의 보정 몰부피를 계산합니다. |

## 실행 방법

1. 저장소 루트에서 터미널 또는 PowerShell을 열고 의존성을 설치합니다.

   ```shell
   python -m pip install numpy pandas==3.1.0 scikit-learn==1.9.1 matplotlib joblib==1.6.0
   ```

2. 실행할 기체의 폴더로 이동합니다. 아래는 이산화탄소 예시입니다. 수소와 질소의 경우 각각 `cd hydrogen`, `cd nitrogen`을 실행합니다.

   ```shell
   cd carbondioxide
   ```

3. 이산화탄소는 처음 학습하기 전에 NIST 데이터를 받습니다. 수소와 질소는 저장소에 포함된 데이터로 바로 학습할 수 있습니다.

   ```shell
   python nist_data_download.py
   ```

4. 모델을 학습하거나 저장된 모델을 불러옵니다. 예전 학습 방식으로 만든 모델은 자동으로 다시 학습합니다.

   ```shell
   python main_train.py
   ```

5. 그래프 창을 닫은 뒤 같은 터미널에서 예측 프로그램을 실행합니다. 온도(K), 압력(atm) 순서로 입력하면 1몰 기준 부피(L/mol)가 표시됩니다. 종료하려면 `q`를 입력합니다.

   ```shell
   python predict_volume.py
   ```
