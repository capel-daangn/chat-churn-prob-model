"""
Automated Report Generator
Creates comprehensive HTML reports with SHAP analysis and insights
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.model_selection import train_test_split
from .shap_analysis import ShapAnalyzer
from .train import ChurnModelTrainer
from .utils import load_config
import base64
from io import BytesIO


class ReportGenerator:
    def __init__(self, config_path="config.yaml"):
        self.config = load_config(config_path)
        self.analyzer = ShapAnalyzer(config_path)
        self.report_html = ""

    def generate_full_report(self, output_path="models/churn_analysis_report.html"):
        """Generate comprehensive analysis report"""
        print("\n" + "=" * 70)
        print("Generating Comprehensive Churn Analysis Report")
        print("=" * 70)

        # Load data
        X_train, X_valid, y_valid, df_valid, preds = self._load_data()

        # Initialize HTML
        self._init_html()

        # Add sections
        self._add_header()
        self._add_executive_summary(y_valid, preds)
        self._add_dataset_overview(df_valid, y_valid)
        self._add_churn_patterns(df_valid, preds, y_valid)
        self._add_shap_analysis(X_train, X_valid)
        self._add_top_churners(df_valid, preds, y_valid)
        self._add_recommendations()
        self._add_footer()

        # Save report
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(self.report_html)

        print(f"\n✓ Report saved to {output_path}")
        print("=" * 70)

        return output_path

    def _load_data(self):
        """Load all necessary data"""
        print("Loading data...")

        # Load model and data
        X_train, X_valid, y_train, y_valid = self.analyzer.load_model_and_data()

        # Load original messages
        df = pd.read_csv(self.config['paths']['raw_data'])

        _, df_valid = train_test_split(
            df,
            test_size=self.config['training']['test_size'],
            random_state=self.config['training']['random_state'],
            stratify=df['label']
        )

        df_valid = df_valid.reset_index(drop=True)

        # Get predictions
        preds = self.analyzer.trainer.predict(X_valid)

        return X_train, X_valid, y_valid, df_valid, preds

    def _init_html(self):
        """Initialize HTML template"""
        self.report_html = """
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <title>채팅 이탈 분석 리포트</title>
            <style>
                body {
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    max-width: 1200px;
                    margin: 0 auto;
                    padding: 20px;
                    background-color: #f5f5f5;
                }
                .header {
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white;
                    padding: 40px;
                    border-radius: 10px;
                    margin-bottom: 30px;
                }
                .section {
                    background: white;
                    padding: 30px;
                    margin-bottom: 20px;
                    border-radius: 10px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                }
                h1 { margin: 0; font-size: 2.5em; }
                h2 { color: #667eea; border-bottom: 3px solid #667eea; padding-bottom: 10px; }
                h3 { color: #764ba2; }
                .metric-box {
                    display: inline-block;
                    background: #f8f9fa;
                    padding: 20px;
                    margin: 10px;
                    border-radius: 8px;
                    border-left: 4px solid #667eea;
                    min-width: 200px;
                }
                .metric-value {
                    font-size: 2em;
                    font-weight: bold;
                    color: #667eea;
                }
                .metric-label {
                    color: #666;
                    font-size: 0.9em;
                }
                table {
                    width: 100%;
                    border-collapse: collapse;
                    margin: 20px 0;
                }
                th {
                    background-color: #667eea;
                    color: white;
                    padding: 12px;
                    text-align: left;
                }
                td {
                    padding: 10px;
                    border-bottom: 1px solid #ddd;
                }
                tr:hover { background-color: #f5f5f5; }
                .high-risk { color: #e74c3c; font-weight: bold; }
                .medium-risk { color: #f39c12; font-weight: bold; }
                .low-risk { color: #27ae60; font-weight: bold; }
                img { max-width: 100%; height: auto; border-radius: 8px; }
                .recommendation {
                    background: #e8f5e9;
                    padding: 15px;
                    border-left: 4px solid #4caf50;
                    margin: 10px 0;
                    border-radius: 4px;
                }
                .timestamp {
                    color: #999;
                    font-size: 0.9em;
                }
            </style>
        </head>
        <body>
        """

    def _add_header(self):
        """Add report header"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.report_html += f"""
        <div class="header">
            <h1>💬 채팅 이탈 분석 리포트</h1>
            <p>DistilBERT + LightGBM 기반 AI 모델 분석</p>
            <p class="timestamp">생성 시간: {timestamp}</p>
        </div>
        """

    def _add_executive_summary(self, y_valid, preds):
        """Add executive summary"""
        total_messages = len(y_valid)
        actual_churns = y_valid.sum()
        predicted_high_risk = (preds > 0.7).sum()
        avg_churn_prob = preds.mean()

        self.report_html += f"""
        <div class="section">
            <h2>📊 주요 지표 요약</h2>
            <div class="metric-box">
                <div class="metric-value">{total_messages}</div>
                <div class="metric-label">총 메시지 수</div>
            </div>
            <div class="metric-box">
                <div class="metric-value">{actual_churns}</div>
                <div class="metric-label">실제 이탈 메시지</div>
            </div>
            <div class="metric-box">
                <div class="metric-value">{predicted_high_risk}</div>
                <div class="metric-label">고위험 예측 (>70%)</div>
            </div>
            <div class="metric-box">
                <div class="metric-value">{avg_churn_prob*100:.1f}%</div>
                <div class="metric-label">평균 이탈 확률</div>
            </div>
        </div>
        """

    def _add_dataset_overview(self, df_valid, y_valid):
        """Add dataset overview"""
        # Sender role distribution
        sender_dist = df_valid['sender_role'].value_counts()

        # Churn by role
        churn_by_role = df_valid.groupby('sender_role')['label'].agg(['sum', 'count', 'mean'])

        self.report_html += f"""
        <div class="section">
            <h2>📂 데이터셋 개요</h2>
            <h3>발신자별 분포</h3>
            <table>
                <tr>
                    <th>발신자</th>
                    <th>메시지 수</th>
                    <th>이탈 수</th>
                    <th>이탈률</th>
                </tr>
        """

        for role, row in churn_by_role.iterrows():
            self.report_html += f"""
                <tr>
                    <td>{role}</td>
                    <td>{int(row['count'])}</td>
                    <td>{int(row['sum'])}</td>
                    <td>{row['mean']*100:.1f}%</td>
                </tr>
            """

        self.report_html += """
            </table>
        </div>
        """

    def _add_churn_patterns(self, df_valid, preds, y_valid):
        """Add churn pattern analysis"""
        df_valid = df_valid.copy()
        df_valid['churn_prob'] = preds
        df_valid['actual_label'] = y_valid

        # Message length vs churn
        churn_msgs = df_valid[df_valid['actual_label'] == 1]
        no_churn_msgs = df_valid[df_valid['actual_label'] == 0]

        avg_len_churn = churn_msgs['msg_len'].mean()
        avg_len_no_churn = no_churn_msgs['msg_len'].mean()

        self.report_html += f"""
        <div class="section">
            <h2>🔍 이탈 패턴 분석</h2>

            <h3>메시지 길이 분석</h3>
            <p>이탈 메시지 평균 길이: <strong>{avg_len_churn:.1f}자</strong></p>
            <p>유지 메시지 평균 길이: <strong>{avg_len_no_churn:.1f}자</strong></p>
            <p>→ 이탈 메시지가 평균 <strong>{abs(avg_len_churn - avg_len_no_churn):.1f}자 {'짧음' if avg_len_churn < avg_len_no_churn else '김'}</strong></p>

            <h3>위험도별 분포</h3>
            <table>
                <tr>
                    <th>위험도</th>
                    <th>범위</th>
                    <th>메시지 수</th>
                    <th>비율</th>
                </tr>
                <tr>
                    <td class="high-risk">높음</td>
                    <td>> 70%</td>
                    <td>{(preds > 0.7).sum()}</td>
                    <td>{(preds > 0.7).mean() * 100:.1f}%</td>
                </tr>
                <tr>
                    <td class="medium-risk">중간</td>
                    <td>40-70%</td>
                    <td>{((preds > 0.4) & (preds <= 0.7)).sum()}</td>
                    <td>{((preds > 0.4) & (preds <= 0.7)).mean() * 100:.1f}%</td>
                </tr>
                <tr>
                    <td class="low-risk">낮음</td>
                    <td>< 40%</td>
                    <td>{(preds <= 0.4).sum()}</td>
                    <td>{(preds <= 0.4).mean() * 100:.1f}%</td>
                </tr>
            </table>
        </div>
        """

    def _add_shap_analysis(self, X_train, X_valid):
        """Add SHAP analysis section"""
        print("Calculating SHAP values for report...")

        # Create explainer and calculate SHAP
        self.analyzer.create_explainer(X_train)
        shap_values = self.analyzer.calculate_shap_values(X_valid, max_samples=500)

        # Get top features
        top_features_df = self.analyzer.get_top_features(10)

        self.report_html += f"""
        <div class="section">
            <h2>🔬 SHAP 모델 해석</h2>
            <p>SHAP (SHapley Additive exPlanations)를 사용하여 모델이 어떤 피처를 중요하게 여기는지 분석합니다.</p>

            <h3>Top 10 중요 피처</h3>
            <table>
                <tr>
                    <th>순위</th>
                    <th>피처</th>
                    <th>유형</th>
                    <th>중요도 (Mean |SHAP|)</th>
                </tr>
        """

        for idx, row in top_features_df.iterrows():
            self.report_html += f"""
                <tr>
                    <td>{idx + 1}</td>
                    <td><code>{row['feature']}</code></td>
                    <td>{row['feature_type']}</td>
                    <td>{row['mean_abs_shap']:.6f}</td>
                </tr>
            """

        self.report_html += """
            </table>
        </div>
        """

    def _add_top_churners(self, df_valid, preds, y_valid, n=10):
        """Add top churners section"""
        df_valid = df_valid.copy()
        df_valid['churn_prob'] = preds
        df_valid['actual_label'] = y_valid

        top_churners = df_valid.nlargest(n, 'churn_prob')

        self.report_html += f"""
        <div class="section">
            <h2>⚠️ Top {n} 이탈 위험 메시지</h2>
            <p>모델이 가장 높은 이탈 확률로 예측한 메시지들입니다.</p>
            <table>
                <tr>
                    <th>순위</th>
                    <th>메시지</th>
                    <th>발신자</th>
                    <th>예측 확률</th>
                    <th>실제</th>
                </tr>
        """

        for idx, (_, row) in enumerate(top_churners.iterrows(), 1):
            prob_class = 'high-risk' if row['churn_prob'] > 0.7 else 'medium-risk'
            actual = '이탈' if row['actual_label'] == 1 else '유지'

            self.report_html += f"""
                <tr>
                    <td>{idx}</td>
                    <td>{row['content']}</td>
                    <td>{row['sender_role']}</td>
                    <td class="{prob_class}">{row['churn_prob']*100:.1f}%</td>
                    <td>{actual}</td>
                </tr>
            """

        self.report_html += """
            </table>
        </div>
        """

    def _add_recommendations(self):
        """Add actionable recommendations"""
        self.report_html += """
        <div class="section">
            <h2>💡 실행 가능한 인사이트 & 권장사항</h2>

            <div class="recommendation">
                <h3>1. 짧은 메시지 주의</h3>
                <p>분석 결과 짧은 메시지(특히 "...", "음" 등)는 높은 이탈 확률과 연관됩니다.</p>
                <p><strong>권장:</strong> 짧은 응답이 감지되면 자동으로 추가 질문이나 도움말 제공</p>
            </div>

            <div class="recommendation">
                <h3>2. 부정적 감정 조기 감지</h3>
                <p>감정 점수(sentiment_score)가 부정적인 메시지는 이탈 가능성이 높습니다.</p>
                <p><strong>권장:</strong> 부정적 감정 감지 시 상담원 개입 또는 할인 쿠폰 제공</p>
            </div>

            <div class="recommendation">
                <h3>3. 응답 지연 모니터링</h3>
                <p>메시지 간 시간 간격이 길어지면 이탈 위험이 증가합니다.</p>
                <p><strong>권장:</strong> 일정 시간 이상 응답이 없으면 자동 리마인더 발송</p>
            </div>

            <div class="recommendation">
                <h3>4. 종료 신호 키워드 대응</h3>
                <p>"다시 생각해볼게요", "고민해볼게요" 등의 키워드는 강한 이탈 신호입니다.</p>
                <p><strong>권장:</strong> 이런 표현 감지 시 즉각적인 혜택 제안이나 FAQ 제공</p>
            </div>
        </div>
        """

    def _add_footer(self):
        """Add report footer"""
        self.report_html += """
        <div class="section" style="background: #f8f9fa; text-align: center;">
            <p style="color: #666;">
                🤖 Generated with Chat Churn Prediction Model<br>
                DistilBERT + LightGBM + SHAP
            </p>
        </div>
        </body>
        </html>
        """


def main():
    generator = ReportGenerator()
    output_path = generator.generate_full_report()

    print(f"\n✓ 리포트가 생성되었습니다!")
    print(f"  파일: {output_path}")
    print(f"  브라우저에서 열어보세요!")


if __name__ == "__main__":
    main()
