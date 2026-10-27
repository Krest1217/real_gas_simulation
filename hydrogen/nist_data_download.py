import ssl
# SSL 인증서 검증 오류 무시 설정
ssl._create_default_https_context = ssl._create_unverified_context

import os
import time
import urllib.request
import io
import pandas as pd

# 저장할 폴더 생성
SAVE_DIR = "./nist_data"
os.makedirs(SAVE_DIR, exist_ok=True)

# 다운로드할 온도 리스트
temp_list = range(50, 202, 2)

# NIST 설정값 (수소 H2)
GAS_ID = "C1333740"
P_LOW = 50
P_HIGH = 200
P_INC = 2

# 403 에러 방지
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
}

print("NIST 서버에서 데이터를 다운로드하여 CSV로 저장합니다\n")

success_count = 0

for T in temp_list:
    nist_url = (
        f"https://webbook.nist.gov/cgi/fluid.cgi?"
        f"Action=Data&Wide=on&ID={GAS_ID}&Type=IsoTherm"
        f"&T={T}&PLow={P_LOW}&PHigh={P_HIGH}&PInc={P_INC}"
        f"&Digits=5&TUnit=K&PUnit=atm&DUnit=mol%2Fl"
        f"&HUnit=kJ%2Fmol&WUnit=m%2Fs&VisUnit=uPa*s&STUnit=N%2Fm&RefState=DEF"
    )
    
    csv_filename = os.path.join(SAVE_DIR, f"NIST_H2_{T}K.csv")
    
    try:
        req = urllib.request.Request(nist_url, headers=HEADERS)
        
        with urllib.request.urlopen(req) as response:
            # 서버에서 받은 바이트 데이터를 문자열로 변환
            text_data = response.read().decode('utf-8')
            
            # 텍스트 데이터를 가짜 파일처럼 읽어들여 판다스 데이터프레임으로 변환
            df = pd.read_csv(io.StringIO(text_data), sep='\t')
            
            # 정상적인 CSV 파일 이름으로 저장
            df.to_csv(csv_filename, index=False, encoding='utf-8-sig')
            
            print(f"저장 완료: NIST_H2_{T}K.csv (데이터 행 수: {len(df)})")
            success_count += 1
        
        # 서버 차단 방지를 위한 1.5초 대기
        time.sleep(1.5)
        
    except Exception as e:
        print(f"에러 발생 ({T}K): {e}")
        time.sleep(3.0)

print(f"\n총 {success_count}개의 파일이 '{SAVE_DIR}' 폴더에 정상적으로 저장되었습니다!")