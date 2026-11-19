import os
import glob
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score

# 한글 폰트 깨짐 방지 설정
plt.rcParams['font.family'] = 'Malgun Gothic' if os.name == 'nt' else 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False

# [환경 설정] 수소(H2) 기준 atm 단위 설정
R_CONST = 0.08205736  # L·atm / (mol·K)
N_MOL = 1.0           # n = 1 mol 고정

RAW_DATA_DIR = "./nist_data"                     
MASTER_CSV_PATH = "./preprocessed_master.csv"    
MODEL_SAVE_PATH = "./model.pkl"         


# STEP 1: NIST 원본 CSV 전처리 함수

def preprocess_single_nist_file(filepath):
    if os.path.getsize(filepath) < 50:
        raise ValueError("파일 용량이 너무 작거나 비어 있습니다.")

    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        first_line = f.readline()
        if "<html" in first_line.lower() or "<doctype" in first_line.lower():
            raise ValueError("NIST 서버 에러 페이지(HTML)가 다운로드된 파일입니다.")
        
        sep = '\t' if '\t' in first_line else ','

    df_raw = pd.read_csv(filepath, sep=sep)

    col_map = {}
    for col in df_raw.columns:
        c_clean = str(col).strip().lower()
        if 'temp' in c_clean:
            col_map['T_K'] = col
        elif 'press' in c_clean or c_clean == 'p':
            col_map['P_atm'] = col
        elif 'dens' in c_clean or 'rho' in c_clean:
            col_map['Density_mol_L'] = col
        elif c_clean.startswith('vol') or c_clean == 'v':
            col_map['V_real_L'] = col

    if 'T_K' not in col_map or 'P_atm' not in col_map:
        raise ValueError(f"필수 열(온도/압력)을 찾을 수 없음. 실제 컬럼: {list(df_raw.columns)}")

    df_clean = pd.DataFrame()
    df_clean['T_K'] = pd.to_numeric(df_raw[col_map['T_K']], errors='coerce')
    df_clean['P_atm'] = pd.to_numeric(df_raw[col_map['P_atm']], errors='coerce')

    if 'Density_mol_L' in col_map and 'V_real_L' in col_map:
        df_clean['Density_mol_L'] = pd.to_numeric(df_raw[col_map['Density_mol_L']], errors='coerce')
        df_clean['V_real_L'] = pd.to_numeric(df_raw[col_map['V_real_L']], errors='coerce')
    elif 'Density_mol_L' in col_map:
        df_clean['Density_mol_L'] = pd.to_numeric(df_raw[col_map['Density_mol_L']], errors='coerce')
        df_clean['V_real_L'] = 1.0 / df_clean['Density_mol_L']
    elif 'V_real_L' in col_map:
        df_clean['V_real_L'] = pd.to_numeric(df_raw[col_map['V_real_L']], errors='coerce')
        df_clean['Density_mol_L'] = 1.0 / df_clean['V_real_L']
    else:
        raise ValueError(f"밀도 또는 부피 열을 찾을 수 없음. 실제 컬럼: {list(df_raw.columns)}")

    df_clean = df_clean.dropna().copy()
    df_clean = df_clean[df_clean['P_atm'] > 0].copy()

    if len(df_clean) == 0:
        raise ValueError("숫자로 변환 가능한 유효한 데이터 행이 존재하지 않습니다.")

    df_clean['V_ideal_L'] = (N_MOL * R_CONST * df_clean['T_K']) / df_clean['P_atm']
    df_clean['delta_V'] = df_clean['V_real_L'] - df_clean['V_ideal_L']

    return df_clean


def load_and_accumulate_data():
    file_list = sorted(glob.glob(os.path.join(RAW_DATA_DIR, "*.csv")) + 
                       glob.glob(os.path.join(RAW_DATA_DIR, "*.txt")))

    if len(file_list) == 0:
        if os.path.exists(MASTER_CSV_PATH):
            return pd.read_csv(MASTER_CSV_PATH)
        raise FileNotFoundError(f"'{RAW_DATA_DIR}' 폴더에 CSV 파일이 없습니다.")
    
    print("[1단계: 데이터 전처리] NIST 파일 유연 검증 및 병합 중")

    cleaned_dfs = []
    success_count = 0
    for fpath in file_list:
        fname = os.path.basename(fpath)
        try:
            df_part = preprocess_single_nist_file(fpath)
            temps = df_part['T_K'].unique()
            print(f"성공: {fname:<20} | 온도: {temps} K | 데이터 수: {len(df_part)}개")
            cleaned_dfs.append(df_part)
            success_count += 1
        except Exception as e:
            print(f"무시됨 ({fname}): {e}")

    if success_count == 0:
        if os.path.exists(MASTER_CSV_PATH):
            return pd.read_csv(MASTER_CSV_PATH)
        raise ValueError("유효하게 읽어들인 NIST 파일이 단 하나도 없습니다.")

    combined = pd.concat(cleaned_dfs, ignore_index=True)

    if os.path.exists(MASTER_CSV_PATH):
        old_data = pd.read_csv(MASTER_CSV_PATH)
        combined = pd.concat([old_data, combined], ignore_index=True)
        combined.drop_duplicates(subset=['T_K', 'P_atm'], keep='last', inplace=True)

    combined = combined.sort_values(by=['T_K', 'P_atm']).reset_index(drop=True)
    combined.to_csv(MASTER_CSV_PATH, index=False)

    print(f"통합 전처리 파일 저장 완료: '{MASTER_CSV_PATH}'")
    print(f"총 누적 온도 구간: { sorted(combined['T_K'].unique()) } K")
    print(f"총 누적 유효 데이터 수: {len(combined)}개\n")
    return combined



# STEP 2: 모델 로드 또는 신경망 학습

def train_or_load_model(df):
    X_all = df[['P_atm', 'T_K']].values
    
    # 이미 모델 파일이 존재하면 새로 학습하지 않고 즉시 불러오기
    if os.path.exists(MODEL_SAVE_PATH):
        print(f"[캐시 로드] 기존 학습된 모델('{MODEL_SAVE_PATH}')을 불러옵니다.")

        saved_data = joblib.load(MODEL_SAVE_PATH)
        mlp = saved_data['model']
        scaler_X = saved_data['scaler_X']
        
        X_all_scaled = scaler_X.transform(X_all)
        df['delta_V_ai'] = mlp.predict(X_all_scaled)
        df['V_ai_L'] = df['V_ideal_L'] + df['delta_V_ai']
        return mlp, scaler_X, df

    print("[2단계: 신경망 학습]")

    scaler_X = StandardScaler()
    scaler_X.fit(X_all)

    mlp = MLPRegressor(
        hidden_layer_sizes=(64, 64),
        activation='tanh',
        solver='adam',
        learning_rate_init=0.003,
        max_iter=300,
        early_stopping=True,
        n_iter_no_change=10,
        tol=1e-4,
        random_state=42
    )

    unique_temps = sorted(df['T_K'].unique())
    accumulated_indices = []

    rng = np.random.default_rng(42)
    for step_idx, temp in enumerate(unique_temps, start=1):
        current_temp_indices = df[df['T_K'] == temp].index.tolist()
        accumulated_indices.extend(current_temp_indices)

        X_step = scaler_X.transform(df.loc[accumulated_indices, ['P_atm', 'T_K']].values)
        y_step = df.loc[accumulated_indices, 'delta_V'].values

        perm = rng.permutation(len(X_step))
        mlp.fit(X_step[perm], y_step[perm])

        delta_V_pred = mlp.predict(X_step)
        V_ai_step = df.loc[accumulated_indices, 'V_ideal_L'].values + delta_V_pred
        V_real_step = df.loc[accumulated_indices, 'V_real_L'].values
        mape = np.mean(np.abs((V_real_step - V_ai_step) / V_real_step)) * 100

        print(f"[Step {step_idx:02d}] 온도 {temp:5.1f} K 학습 완료 | "
              f"누적 데이터: {len(X_step):4d}개 | AI 평균 오차율: {mape:.3f}%")

    X_all_scaled = scaler_X.transform(X_all)
    df['delta_V_ai'] = mlp.predict(X_all_scaled)
    df['V_ai_L'] = df['V_ideal_L'] + df['delta_V_ai']

    joblib.dump({'model': mlp, 'scaler_X': scaler_X, 'R_CONST': R_CONST}, MODEL_SAVE_PATH)
    print(f"\n학습 완료된 모델이 '{MODEL_SAVE_PATH}'에 저장되었습니다.")

    return mlp, scaler_X, df


# STEP 3: 상세 통계 출력 후 3차원 그래프 시각화

def print_statistics_and_plot(df, mlp, scaler_X):
    # 이상기체 방정식 및 AI 보정식의 절대 오차율(%) 산출
    df['abs_error_ideal'] = np.abs((df['V_ideal_L'] - df['V_real_L']) / df['V_real_L']) * 100
    df['abs_error_ai'] = np.abs((df['V_ai_L'] - df['V_real_L']) / df['V_real_L']) * 100

    ideal_mean = df['abs_error_ideal'].mean()
    ideal_max_row = df.loc[df['abs_error_ideal'].idxmax()]
    ideal_min_row = df.loc[df['abs_error_ideal'].idxmin()]

    ai_mean = df['abs_error_ai'].mean()
    ai_max_row = df.loc[df['abs_error_ai'].idxmax()]
    ai_min_row = df.loc[df['abs_error_ai'].idxmin()]
    ai_r2 = r2_score(df['V_real_L'], df['V_ai_L'])

    # 그래프 출력 전 상세 통계 리포트 출력
    print("[최종 통계 및 상세 오차 분석 리포트]")
    print("이상기체 방정식 (PV = nRT) 통계:")
    print(f"평균 부피 오차율 : {ideal_mean:.2f}%")
    print(f"최대 오차 지점   : {ideal_max_row['abs_error_ideal']:.2f}% (온도: {ideal_max_row['T_K']} K, 압력: {ideal_max_row['P_atm']} atm)")
    print(f"최소 오차 지점   : {ideal_min_row['abs_error_ideal']:.2f}% (온도: {ideal_min_row['T_K']} K, 압력: {ideal_min_row['P_atm']} atm)\n")

    print("인공지능 보정식 (V = RT/P + MLP) 통계:")
    print(f"평균 부피 오차율 : {ai_mean:.4f}%")
    print(f"최대 오차 지점   : {ai_max_row['abs_error_ai']:.4f}% (온도: {ai_max_row['T_K']} K, 압력: {ai_max_row['P_atm']} atm)")
    print(f"최소 오차 지점   : {ai_min_row['abs_error_ai']:.4f}% (온도: {ai_min_row['T_K']} K, 압력: {ai_min_row['P_atm']} atm)")
    print(f"결정계수 (R²)    : {ai_r2:.6f}")

    # 3차원 시각화 그래프 창 띄우기
    P_min, P_max = df['P_atm'].min(), df['P_atm'].max()
    T_min, T_max = df['T_K'].min(), df['T_K'].max()

    P_space = np.linspace(P_min, P_max, 40)
    T_space = np.linspace(T_min, T_max, 40)
    P_grid, T_grid = np.meshgrid(P_space, T_space)

    V_ideal_grid = (N_MOL * R_CONST * T_grid) / P_grid
    grid_inputs = np.column_stack([P_grid.ravel(), T_grid.ravel()])
    grid_inputs_scaled = scaler_X.transform(grid_inputs)
    delta_V_grid = mlp.predict(grid_inputs_scaled).reshape(P_grid.shape)
    V_ai_grid = V_ideal_grid + delta_V_grid

    fig = plt.figure(figsize=(14, 6))

    ax1 = fig.add_subplot(121, projection='3d')
    ax1.plot_surface(P_grid, T_grid, V_ideal_grid, cmap='Blues', alpha=0.35)
    ax1.plot_surface(P_grid, T_grid, V_ai_grid, cmap='Oranges', alpha=0.75)
    ax1.scatter(df['P_atm'], df['T_K'], df['V_real_L'], color='black', s=10, label='NIST 실제 데이터')

    ax1.set_title('3D 부피 곡면: 이상기체(파랑) vs AI 보정 모델(주황)')
    ax1.set_xlabel('압력 P (atm)')
    ax1.set_ylabel('온도 T (K)')
    ax1.set_zlabel('부피 V (L/mol)')
    ax1.view_init(elev=25, azim=45)
    ax1.legend()

    ax2 = fig.add_subplot(122, projection='3d')
    ax2.plot_surface(P_grid, T_grid, delta_V_grid, cmap='coolwarm', alpha=0.85)
    ax2.set_title('다층신경망이 학습한 보정값 곡면: ΔV = MLP(P, T)')
    ax2.set_xlabel('압력 P (atm)')
    ax2.set_ylabel('온도 T (K)')
    ax2.set_zlabel('보정 부피 ΔV (L/mol)')
    ax2.view_init(elev=25, azim=135)

    plt.tight_layout()
    print("3차원 시각화 그래프 창을 띄웁니다.")
    plt.show()


if __name__ == "__main__":
    df_master = load_and_accumulate_data()
    trained_mlp, trained_scaler, df_result = train_or_load_model(df_master)
    print_statistics_and_plot(df_result, trained_mlp, trained_scaler)
