from selenium import webdriver
from selenium.webdriver.common.by import By
import time
import pandas as pd

driver = webdriver.Chrome()

# 서강대학교 나무위키 접속
url = 'https://namu.wiki/w/%EC%84%9C%EA%B0%95%EB%8C%80%ED%95%99%EA%B5%90' 
driver.get(url)

print("페이지 접속 중... 7초간 대기합니다.")
time.sleep(7) 

paragraphs = driver.find_elements(By.CLASS_NAME, 'iR1C9VHd')

wiki_data = []
for p in paragraphs:
    text = p.text.strip()
    # 의미 있는 정보(30자 이상)만 리스트에 추가
    if len(text) > 30:
        wiki_data.append(text)

# 결과 확인 및 저장
if wiki_data:
    # 중복 제거
    wiki_data = list(set(wiki_data))
    
    print(f"성공! 총 {len(wiki_data)}개의 문단을 수집했습니다.")
    df = pd.DataFrame(wiki_data, columns=['Content'])
    df.to_excel("sogang_wiki.xlsx", index=False)
    print("sogang_wiki.xlsx 파일이 생성되었습니다.")

driver.quit()