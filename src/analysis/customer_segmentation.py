"""
AI驱动的商业分析平台 - 客户分类模块
功能：客户价值分类、流失预测、RFM分析、KMeans聚类
入参：客户数据和交易数据
出参：客户分群结果和可视化图表
异常处理：模型训练异常捕获
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.neighbors import KNeighborsClassifier

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

PROCESSED_DIR = 'data/processed'
VIZ_DIR = 'data/viz'
os.makedirs(VIZ_DIR, exist_ok=True)

class CustomerSegmentation:
    """
    客户分类类
    功能：客户价值分类、流失预测、RFM分析、聚类分析
    """
    
    def __init__(self):
        self.customers_df = None
        self.transactions_df = None
        self.rfm_df = None
        self.models = {}
        self.scaler = StandardScaler()
        self.label_encoders = {}
    
    def log_step(self, message, status='info'):
        """
        记录处理步骤日志
        入参：message - 日志消息, status - 状态
        出参：打印彩色日志
        """
        colors = {'info': '\033[94m', 'success': '\033[92m', 'error': '\033[91m', 'warning': '\033[93m'}
        reset = '\033[0m'
        print(f'{colors.get(status, "")}[客户分类] {message}{reset}')
    
    def load_data(self):
        """
        加载客户和交易数据
        入参：无
        出参：DataFrame
        """
        customers_path = os.path.join(PROCESSED_DIR, 'customers_processed.csv')
        transactions_path = os.path.join(PROCESSED_DIR, 'transactions_processed.csv')
        
        if os.path.exists(customers_path):
            self.customers_df = pd.read_csv(customers_path)
            self.log_step(f'加载客户数据: {len(self.customers_df)} 条', 'success')
        
        if os.path.exists(transactions_path):
            self.transactions_df = pd.read_csv(transactions_path)
            self.log_step(f'加载交易数据: {len(self.transactions_df)} 条', 'success')
        
        return self.customers_df is not None and self.transactions_df is not None
    
    def calculate_rfm(self):
        """
        计算RFM指标（最近购买时间、购买频率、消费金额）
        入参：无
        出参：RFM DataFrame
        """
        self.log_step('计算RFM指标...')
        
        if self.transactions_df is None:
            self.log_step('交易数据未加载', 'error')
            return None
        
        df = self.transactions_df.copy()
        df['transaction_date'] = pd.to_datetime(df['transaction_date'])
        
        reference_date = df['transaction_date'].max() + pd.Timedelta(days=1)
        
        rfm = df.groupby('customer_id').agg({
            'transaction_date': lambda x: (reference_date - x.max()).days,
            'transaction_id': 'count',
            'total_amount': 'sum'
        }).reset_index()
        
        rfm.columns = ['customer_id', 'Recency', 'Frequency', 'Monetary']
        
        rfm['R_Score'] = pd.qcut(rfm['Recency'], q=5, labels=[5, 4, 3, 2, 1])
        rfm['F_Score'] = pd.qcut(rfm['Frequency'].rank(method='first'), q=5, labels=[1, 2, 3, 4, 5])
        rfm['M_Score'] = pd.qcut(rfm['Monetary'].rank(method='first'), q=5, labels=[1, 2, 3, 4, 5])
        
        rfm['RFM_Score'] = rfm['R_Score'].astype(str) + rfm['F_Score'].astype(str) + rfm['M_Score'].astype(str)
        rfm['RFM_Total'] = rfm['R_Score'].astype(int) + rfm['F_Score'].astype(int) + rfm['M_Score'].astype(int)
        
        def segment_customer(row):
            if row['RFM_Total'] >= 13:
                return '高价值客户'
            elif row['RFM_Total'] >= 10:
                return '潜力客户'
            elif row['RFM_Total'] >= 7:
                return '普通客户'
            elif row['RFM_Total'] >= 4:
                return '需维护客户'
            else:
                return '流失风险客户'
        
        rfm['Customer_Segment'] = rfm.apply(segment_customer, axis=1)
        
        self.rfm_df = rfm
        self.log_step(f'RFM计算完成: {len(rfm)} 个客户', 'success')
        
        return rfm
    
    def plot_rfm_analysis(self):
        """
        绘制RFM分析图表
        入参：无
        出参：保存可视化图表
        """
        if self.rfm_df is None:
            return
        
        self.log_step('生成RFM分析图表...')
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        ax1 = axes[0, 0]
        segment_counts = self.rfm_df['Customer_Segment'].value_counts()
        colors = plt.cm.Set2(np.linspace(0, 1, len(segment_counts)))
        ax1.pie(segment_counts.values, labels=segment_counts.index, autopct='%1.1f%%', colors=colors)
        ax1.set_title('客户分群分布', fontsize=14, fontweight='bold')
        
        ax2 = axes[0, 1]
        ax2.scatter(self.rfm_df['Recency'], self.rfm_df['Monetary'], 
                   c=self.rfm_df['RFM_Total'], cmap='viridis', alpha=0.5, s=20)
        ax2.set_title('Recency vs Monetary', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Recency (天)')
        ax2.set_ylabel('Monetary (消费金额)')
        ax2.set_colorbar = plt.colorbar(ax2.collections[0], ax=ax2)
        
        ax3 = axes[0, 2]
        ax3.scatter(self.rfm_df['Frequency'], self.rfm_df['Monetary'], 
                   c=self.rfm_df['RFM_Total'], cmap='plasma', alpha=0.5, s=20)
        ax3.set_title('Frequency vs Monetary', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Frequency (购买次数)')
        ax3.set_ylabel('Monetary (消费金额)')
        
        ax4 = axes[1, 0]
        self.rfm_df['R_Score'].value_counts().sort_index().plot(kind='bar', ax=ax4, color='#2E86AB')
        ax4.set_title('R Score分布', fontsize=14, fontweight='bold')
        ax4.set_xlabel('R Score')
        ax4.set_ylabel('客户数')
        
        ax5 = axes[1, 1]
        self.rfm_df['F_Score'].value_counts().sort_index().plot(kind='bar', ax=ax5, color='#A23B72')
        ax5.set_title('F Score分布', fontsize=14, fontweight='bold')
        ax5.set_xlabel('F Score')
        ax5.set_ylabel('客户数')
        
        ax6 = axes[1, 2]
        self.rfm_df['M_Score'].value_counts().sort_index().plot(kind='bar', ax=ax6, color='#F18F01')
        ax6.set_title('M Score分布', fontsize=14, fontweight='bold')
        ax6.set_xlabel('M Score')
        ax6.set_ylabel('客户数')
        
        plt.tight_layout()
        plt.savefig(f'{VIZ_DIR}/RFM分析.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        self.log_step('RFM分析图表已保存', 'success')
    
    def perform_kmeans_clustering(self, n_clusters=5):
        """
        执行KMeans聚类分析
        入参：n_clusters - 聚类数量
        出参：聚类结果DataFrame
        """
        self.log_step('执行KMeans聚类分析...')
        
        if self.rfm_df is None:
            self.log_step('RFM数据未计算', 'error')
            return None
        
        features = ['Recency', 'Frequency', 'Monetary']
        X = self.rfm_df[features].copy()
        
        X_scaled = self.scaler.fit_transform(X)
        
        inertias = []
        K_range = range(2, 11)
        for k in K_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(X_scaled)
            inertias.append(kmeans.inertia_)
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        ax1 = axes[0]
        ax1.plot(K_range, inertias, 'bo-', linewidth=2, markersize=8)
        ax1.set_title('肘部法则 - 确定最佳聚类数', fontsize=14, fontweight='bold')
        ax1.set_xlabel('聚类数量 K')
        ax1.set_ylabel('簇内平方和 (Inertia)')
        ax1.grid(True, alpha=0.3)
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        self.rfm_df['Cluster'] = kmeans.fit_predict(X_scaled)
        
        cluster_stats = self.rfm_df.groupby('Cluster')[features].mean()
        
        ax2 = axes[1]
        cluster_counts = self.rfm_df['Cluster'].value_counts().sort_index()
        ax2.bar(cluster_counts.index, cluster_counts.values, color=plt.cm.Set2(np.linspace(0, 1, n_clusters)))
        ax2.set_title('各聚类客户数量', fontsize=14, fontweight='bold')
        ax2.set_xlabel('聚类')
        ax2.set_ylabel('客户数')
        
        plt.tight_layout()
        plt.savefig(f'{VIZ_DIR}/KMeans聚类分析.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        self.log_step(f'KMeans聚类完成: {n_clusters} 个聚类', 'success')
        
        return self.rfm_df
    
    def train_churn_prediction_model(self):
        """
        训练客户流失预测模型
        入参：无
        出参：模型评估结果
        """
        self.log_step('训练客户流失预测模型...')
        
        if self.customers_df is None:
            self.log_step('客户数据未加载', 'error')
            return None
        
        df = self.customers_df.copy()
        
        df['churn_label'] = (df['is_churn_risk'] == 1).astype(int)
        
        feature_cols = ['age', 'total_orders', 'total_spent', 'last_login_days',
                       'customer_tenure_days', 'avg_order_value', 'order_frequency']
        
        available_features = [col for col in feature_cols if col in df.columns]
        
        categorical_cols = ['gender', 'city_tier', 'income_level', 'membership_level']
        for col in categorical_cols:
            if col in df.columns:
                le = LabelEncoder()
                df[col + '_encoded'] = le.fit_transform(df[col].astype(str))
                self.label_encoders[col] = le
                available_features.append(col + '_encoded')
        
        df_clean = df.dropna(subset=available_features + ['churn_label'])
        
        X = df_clean[available_features]
        y = df_clean['churn_label']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        self.models['Logistic Regression'] = LogisticRegression(random_state=42, max_iter=1000)
        self.models['Logistic Regression'].fit(X_train_scaled, y_train)
        
        self.models['Random Forest'] = RandomForestClassifier(n_estimators=100, random_state=42)
        self.models['Random Forest'].fit(X_train, y_train)
        
        self.models['Gradient Boosting'] = GradientBoostingClassifier(n_estimators=100, random_state=42)
        self.models['Gradient Boosting'].fit(X_train, y_train)
        
        self.models['KNN'] = KNeighborsClassifier(n_neighbors=5)
        self.models['KNN'].fit(X_train_scaled, y_train)
        
        results = []
        for name, model in self.models.items():
            if name in ['Logistic Regression', 'KNN']:
                y_pred = model.predict(X_test_scaled)
                y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
            else:
                y_pred = model.predict(X_test)
                y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            report = classification_report(y_test, y_pred, output_dict=True)
            auc = roc_auc_score(y_test, y_pred_proba)
            
            results.append({
                '模型': name,
                '准确率': round(report['accuracy'], 4),
                '精确率': round(report['weighted avg']['precision'], 4),
                '召回率': round(report['weighted avg']['recall'], 4),
                'F1分数': round(report['weighted avg']['f1-score'], 4),
                'AUC': round(auc, 4)
            })
        
        results_df = pd.DataFrame(results).sort_values('AUC', ascending=False)
        
        self.log_step('流失预测模型训练完成', 'success')
        
        return results_df, X_test, y_test
    
    def plot_churn_analysis(self, X_test, y_test):
        """
        绘制流失预测分析图表
        入参：X_test, y_test - 测试数据
        出参：保存可视化图表
        """
        self.log_step('生成流失预测分析图表...')
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        
        ax1 = axes[0, 0]
        for name, model in self.models.items():
            if name in ['Logistic Regression', 'KNN']:
                scaler = StandardScaler()
                X_test_scaled = scaler.fit_transform(X_test)
                y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
            else:
                y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
            auc = roc_auc_score(y_test, y_pred_proba)
            ax1.plot(fpr, tpr, label=f'{name} (AUC={auc:.3f})')
        
        ax1.plot([0, 1], [0, 1], 'k--', label='随机猜测')
        ax1.set_title('ROC曲线对比', fontsize=14, fontweight='bold')
        ax1.set_xlabel('假正率 (FPR)')
        ax1.set_ylabel('真正率 (TPR)')
        ax1.legend(loc='lower right')
        ax1.grid(True, alpha=0.3)
        
        ax2 = axes[0, 1]
        best_model_name = max(self.models.keys(), key=lambda x: roc_auc_score(y_test, self.models[x].predict_proba(X_test)[:, 1] if x not in ['Logistic Regression', 'KNN'] else self.models[x].predict_proba(StandardScaler().fit_transform(X_test))[:, 1]))
        best_model = self.models[best_model_name]
        
        if best_model_name in ['Random Forest', 'Gradient Boosting']:
            y_pred = best_model.predict(X_test)
        else:
            scaler = StandardScaler()
            y_pred = best_model.predict(scaler.fit_transform(X_test))
        
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax2)
        ax2.set_title(f'混淆矩阵 - {best_model_name}', fontsize=14, fontweight='bold')
        ax2.set_xlabel('预测值')
        ax2.set_ylabel('真实值')
        
        ax3 = axes[1, 0]
        if 'Random Forest' in self.models:
            feature_importance = pd.DataFrame({
                '特征': X_test.columns,
                '重要性': self.models['Random Forest'].feature_importances_
            }).sort_values('重要性', ascending=True)
            
            ax3.barh(feature_import['特征'].tail(10), feature_importance['重要性'].tail(10), color='#2E86AB')
            ax3.set_title('特征重要性 (随机森林)', fontsize=14, fontweight='bold')
            ax3.set_xlabel('重要性')
        
        ax4 = axes[1, 1]
        churn_counts = pd.Series(y_test).value_counts()
        ax4.pie(churn_counts.values, labels=['非流失', '流失'], autopct='%1.1f%%', 
               colors=['#2E86AB', '#A23B72'])
        ax4.set_title('客户流失分布', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f'{VIZ_DIR}/客户流失预测.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        self.log_step('流失预测分析图表已保存', 'success')
    
    def run_full_analysis(self):
        """
        执行完整的客户分类分析
        入参：无
        出参：分析结果
        """
        print('\033[96m' + '='*60 + '\033[0m')
        print('\033[96m  AI驱动的商业分析平台 - 客户分类分析\033[0m')
        print('\033[96m' + '='*60 + '\033[0m')
        print()
        
        self.load_data()
        print()
        
        self.calculate_rfm()
        print()
        
        self.plot_rfm_analysis()
        print()
        
        self.perform_kmeans_clustering()
        print()
        
        results_df, X_test, y_test = self.train_churn_prediction_model()
        print()
        
        self.plot_churn_analysis(X_test, y_test)
        print()
        
        print('\033[92m' + '='*60 + '\033[0m')
        print('\033[92m  ✅ 客户分类分析完成！\033[0m')
        print('\033[92m' + '='*60 + '\033[0m')
        
        print('\n客户分群统计:')
        if self.rfm_df is not None:
            print(self.rfm_df['Customer_Segment'].value_counts())
        
        print('\n流失预测模型性能:')
        print(results_df.to_string(index=False))
        
        return {
            'rfm': self.rfm_df,
            'churn_metrics': results_df
        }

def main():
    segmentation = CustomerSegmentation()
    segmentation.run_full_analysis()

if __name__ == '__main__':
    main()
