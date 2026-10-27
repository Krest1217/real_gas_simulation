import os
import joblib
import numpy as np


# [환경 설정] 모델 파일 경로 설정

MODEL_SAVE_PATH = "./method_b_model.pkl"

def load_trained_model():
    if not os.path.exists(MODEL_SAVE_PATH):
        raise FileNotFoundError(
            f"학습된 모델 파일('{MODEL_SAVE_PATH}')을 찾을 수 없습니다.\n"
            f"   먼저 'main_train_3.py'를 실행하여 모델을 학습시켜 주세요"
        )
    
    saved_data = joblib.load(MODEL_SAVE_PATH)
    mlp = saved_data['model']
    scaler_X = saved_data['scaler_X']
    r_const = saved_data['R_CONST']
    
    return mlp, scaler_X, r_const

def predict_volume():
    mlp, scaler_X, r_const = load_trained_model()
    n_mol = 1.0  # n = 1 mol 고정
    

    print("수소(H2) 기체 부피 예측 인터랙티브 프로그램")
    print(" 안내: 원하는 온도(K)와 압력(atm)을 입력하면 이상기체 공식과 AI 보정 모델의")
    print("       부피 예측값을 비교하여 실시간으로 출력합니다.")
    print(" 종료하려면 온도나 압력 입력창에 'q'를 입력하세요.\n")
    
    while True:
        try:
            # 사용자 입력 받기
            t_input = input("온도 입력 (K) [예: 150]: ").strip()
            if t_input.lower() == 'q':
                print("프로그램을 종료합니다.")
                break
                
            p_input = input("압력 입력 (atm) [예: 50]: ").strip()
            if p_input.lower() == 'q':
                print("프로그램을 종료합니다.")
                break
                
            # 숫자형 변환
            T_val = float(t_input)
            P_val = float(p_input)
            
            if P_val <= 0 or T_val <= 0:
                print("온도와 압력은 0보다 커야 합니다. 다시 입력해 주세요.\n")
                continue
                
            # 1. 이상기체 방정식 부피 계산 (V_ideal = RT / P)
            v_ideal = (n_mol * r_const * T_val) / P_val
            
            # 2. 인공지능 보정식 적용 (V_ai = V_ideal + MLP(P, T))
            input_features = np.array([[P_val, T_val]])
            input_scaled = scaler_X.transform(input_features)
            delta_v_pred = mlp.predict(input_scaled)[0]
            v_ai = v_ideal + delta_v_pred
            
            # 오차율 계산 (가상의 실제값 대신 반데르발스/문헌 근사 또는 이상기체와의 괴리율)
            # 여기서는 이상기체 공식 대비 AI가 보정한 부피 차이(%)를 출력합니다.
            diff_percent = ((v_ai - v_ideal) / v_ideal) * 100
            
            print("-" * 65)
            print(f"입력 조건 : 온도 T = {T_val} K, 압력 P = {P_val} atm")
            print(f" - 이상기체 방정식 부피 (V_ideal) : {v_ideal:.5f} L/mol")
            print(f" - AI 보정식 예측 부피 (V_ai)    : {v_ai:.5f} L/mol")
            print(f" - 다층신경망 보정량 (ΔV)        : {delta_v_pred:+.5f} L/mol")
            print(f" - 이상기체 대비 부피 보정률     : {diff_percent:+.2f}%")
            print("-" * 65 + "\n")
            
        except ValueError:
            print("올바른 숫자를 입력해 주세요. (종료하려면 'q' 입력)\n")
        except Exception as e:
            print(f"에러 발생: {e}\n")

if __name__ == "__main__":
    predict_volume()