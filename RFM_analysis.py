import pandas as pd
from datetime import datetime

class RFMAnalyzer:
    def __init__(self, data):
        self.data = data
        self.today = datetime.now()
        self.rfm_df = None
        self.quantiles = None

    def preprocess_data(self):
        """날짜 형식 변환"""
        self.data['prf_ymd'] = pd.to_datetime(self.data['prf_ymd'], format='%Y%m%d')

    def calculate_rfm(self):
        """RFM 지표 계산"""
        self.rfm_df = self.data.groupby('prnts_cstmr_id').agg({
            'prf_ymd': lambda x: (self.today - x.max()).days,  # Recency
            'prnts_cstmr_id': 'count',                        # Frequency
            'ntprc_amt': 'sum'                                # Monetary
        }).rename(columns={'prf_ymd': 'Recency', 'prnts_cstmr_id': 'Frequency', 'ntprc_amt': 'Monetary'})

    def calculate_quantiles(self):
        """RFM 점수 부여를 위한 분위 계산"""
        self.quantiles = self.rfm_df.quantile(q=[0.2, 0.4, 0.6, 0.8]).to_dict()

    def r_score(self, x):
        """Recency 점수 계산"""
        if x <= self.quantiles['Recency'][0.2]:
            return 5
        elif x <= self.quantiles['Recency'][0.4]:
            return 4
        elif x <= self.quantiles['Recency'][0.6]:
            return 3
        elif x <= self.quantiles['Recency'][0.8]:
            return 2
        else:
            return 1

    def fm_score(self, x, col):
        """Frequency, Monetary 점수 계산"""
        if x <= self.quantiles[col][0.2]:
            return 1
        elif x <= self.quantiles[col][0.4]:
            return 2
        elif x <= self.quantiles[col][0.6]:
            return 3
        elif x <= self.quantiles[col][0.8]:
            return 4
        else:
            return 5

    def assign_rfm_scores(self):
        """RFM 점수 부여"""
        self.rfm_df['R'] = self.rfm_df['Recency'].apply(self.r_score)
        self.rfm_df['F'] = self.rfm_df['Frequency'].apply(self.fm_score, args=('Frequency',))
        self.rfm_df['M'] = self.rfm_df['Monetary'].apply(self.fm_score, args=('Monetary',))

    def get_strategy(self, r, f, m):
        """RFM 점수를 바탕으로 고객 분류"""
        if r == 5 and f == 5 and m == 5:
            return 'VIP 고객', '종합적으로 가장 점수가 높은 VIP 고객입니다.'
        elif r >= 4 and f >= 4 and m >= 3:
            return '충성 고객', '자주 구매하며 상당한 금액을 지출하는 고객입니다.'
        elif r == 5 and 1 <= f <= 3 and 1 <= m <= 3:
            return '최근 구매 고객', '최근에 첫 구매를 했거나, 재구매 가능성이 높은 고객입니다.'
        elif 3 <= r <= 4 and 1 <= f <= 3 and 1 <= m <= 3:
            return '잠재 고객', '구매 빈도와 금액은 낮지만, 최근에 구매한 고객입니다.'
        elif r == 1 and f <= 2 and m <= 2:
            return '관리 필요 고객', '구매를 한지 오래된 고객이며, 구매 빈도와 소비금액이 적습니다.'
        elif r >= 2 and r <= 3 and f >= 2 and f <= 3 and m >= 2 and m <= 3:
            return '일반 고객', '일반적인 소비 패턴을 보이는 평범한 고객입니다.'
        elif r <= 2 and f >= 2 and m >= 2:
            return '저활동 고객', '과거에 구매 빈도와 지출금액이 높은 편이나, 최근에 구매를 하지 않은 고객입니다.'
        elif r <= 2 and f <= 2 and m >= 2:
            return '이탈 위험 고객', '구매 빈도가 감소하여 이탈 가능성이 높은 고객입니다.'
        else:
            return '일반 고객', '일반적인 소비 패턴을 보이는 평범한 고객입니다.'

    def assign_customer_segments(self):
        """고객 분류 및 설명 추가"""
        self.rfm_df['분류'], self.rfm_df['분류고객 설명'] = zip(*self.rfm_df.apply(
            lambda row: self.get_strategy(row['R'], row['F'], row['M']), axis=1))

    def run_analysis(self):
        """RFM 분석 실행"""
        self.preprocess_data()
        self.calculate_rfm()
        self.calculate_quantiles()
        self.assign_rfm_scores()
        self.assign_customer_segments()
        return self.rfm_df

rfm_analyzer = RFMAnalyzer(series_sales)
rfm_result = rfm_analyzer.run_analysis()
display(rfm_result)
