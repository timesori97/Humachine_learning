import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import os
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

# Streamlit 페이지 설정
st.set_page_config(page_title="한국철도공사 역별 승하차 현황 (2023)", layout="wide")

# 한글 폰트 설정 (Matplotlib)
plt.rcParams['font.family'] = 'Malgun Gothic'  # Windows 기준
plt.rcParams['axes.unicode_minus'] = False

# 제목
st.title("한국철도공사 역별 승하차 현황 (2023)")

# 1. 데이터 로드
@st.cache_data
def load_data():
    csv_path = 'dataset/train.csv'
    if not os.path.exists(csv_path):
        st.error(f"파일 '{csv_path}'를 찾을 수 없습니다. 공공데이터포털에서 다운로드해 'dataset' 폴더에 저장하세요.")
        # 샘플 데이터 생성
        data = {
            'Station': ['서울역', '수서역', '부산역', '대전역', '광주역'],
            'Boarding': [50000, 30000, 40000, 25000, 20000],
            'Alighting': [48000, 29000, 39000, 24000, 19000]
        }
        df = pd.DataFrame(data)
    else:
        try:
            df = pd.read_csv(csv_path, encoding='cp949')  # CP949 인코딩 사용
        except Exception as e:
            st.error(f"CSV 파일 로드 중 오류: {e}. 인코딩을 'utf-8'로 시도하거나 파일 형식을 확인하세요.")
            df = pd.DataFrame()  # 빈 데이터프레임 반환
    return df

df = load_data()

# 2. 데이터 전처리
if not df.empty:
    # 열 이름 조정 (제공된 열 이름 반영)
    try:
        df = df.rename(columns={'역명': 'Station', '승차인원': 'Boarding', '하차인원': 'Alighting'})
    except KeyError:
        st.warning("CSV 파일의 열 이름을 확인하세요. 예상 열: '역명', '승차인원', '하차인원'.")

    # 누락값 처리
    df = df.dropna()
    # 총 승하차 인원 계산
    df['Total_Passengers'] = df['Boarding'] + df['Alighting']
    # 역별 데이터 집계
    station_usage = df.groupby('Station')[['Boarding', 'Alighting', 'Total_Passengers']].sum().reset_index()

    # 3. 역 선택 위젯
    st.sidebar.header("옵션")
    selected_stations = st.sidebar.multiselect(
        "시각화할 역 선택", options=station_usage['Station'].unique(), default=station_usage['Station'].unique()[:5]
    )

    # 필터링된 데이터
    filtered_usage = station_usage[station_usage['Station'].isin(selected_stations)]

    # 4. 시각화: 역별 총 승하차 인원 (막대그래프)
    st.subheader("역별 총 승하차 인원")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x='Total_Passengers', y='Station', data=filtered_usage.sort_values('Total_Passengers', ascending=False), ax=ax)
    ax.set_title('역별 총 승하차 인원 (2023)')
    ax.set_xlabel('총 승하차 인원')
    ax.set_ylabel('역 이름')
    plt.tight_layout()
    st.pyplot(fig)

    # 5. 시각화: 승차 vs 하차 (Plotly, 인터랙티브 막대그래프)
    st.subheader("역별 승차 vs 하차 인원")
    fig_plotly = px.bar(
        filtered_usage, x='Station', y=['Boarding', 'Alighting'],
        title='역별 승차 및 하차 인원 비교',
        labels={'value': '인원', 'Station': '역 이름', 'variable': '구분'},
        barmode='group'
    )
    st.plotly_chart(fig_plotly, use_container_width=True)

    # 6. 데이터 테이블 표시
    st.subheader("데이터 테이블")
    st.dataframe(filtered_usage[['Station', 'Boarding', 'Alighting', 'Total_Passengers']], use_container_width=True)

    # 7. 데이터 요약
    st.subheader("데이터 요약")
    st.write(f"선택된 역 수: {len(filtered_usage)}")
    st.write(f"총 승차 인원: {filtered_usage['Boarding'].sum():,.0f}")
    st.write(f"총 하차 인원: {filtered_usage['Alighting'].sum():,.0f}")
    st.write(f"총 승하차 인원: {filtered_usage['Total_Passengers'].sum():,.0f}")

    # 8. 분류 모델: 붐비는 역 vs 한적한 역
    st.subheader("역 분류: 붐비는 역 vs 한적한 역")
    # 중앙값을 기준으로 클래스 라벨 생성
    threshold = station_usage['Total_Passengers'].median()
    station_usage['Crowdedness'] = station_usage['Total_Passengers'].apply(
        lambda x: '붐비는 역' if x >= threshold else '한적한 역'
    )

    # 특성과 타겟 설정
    X = station_usage[['Boarding', 'Alighting', 'Total_Passengers']]
    y = station_usage['Crowdedness']

    # 데이터 스케일링
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 로지스틱 회귀 모델 학습
    model = LogisticRegression(random_state=42)
    model.fit(X_scaled, y)

    # 예측
    station_usage['Predicted_Crowdedness'] = model.predict(X_scaled)

    # 모델 성능
    accuracy = accuracy_score(y, station_usage['Predicted_Crowdedness'])
    st.write(f"모델 정확도: {accuracy:.2f}")
    st.write("분류 보고서:")
    st.text(classification_report(y, station_usage['Predicted_Crowdedness']))

    # 분류 결과 테이블
    st.subheader("분류 결과")
    st.dataframe(station_usage[['Station', 'Boarding', 'Alighting', 'Total_Passengers', 'Crowdedness', 'Predicted_Crowdedness']], use_container_width=True)

    # 분류 시각화 (Plotly)
    st.subheader("분류 시각화")
    fig_class = px.scatter(
        station_usage, x='Total_Passengers', y='Station', color='Predicted_Crowdedness',
        title='역별 분류 결과 (붐비는 역 vs 한적한 역)',
        labels={'Total_Passengers': '총 승하차 인원', 'Station': '역 이름', 'Predicted_Crowdedness': '분류'},
        color_discrete_map={'붐비는 역': 'red', '한적한 역': 'blue'}
    )
    st.plotly_chart(fig_class, use_container_width=True)

else:
    st.error("데이터를 로드하지 못해 시각화를 생성할 수 없습니다.")

