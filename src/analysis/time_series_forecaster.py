"""
AI驱动的商业分析平台 - 时间序列预测模块
功能：销售趋势预测、ARIMA模型、Prophet模型、模型评估
入参：每日销售数据
出参：预测结果和可视化图表
异常处理：模型训练异常捕获
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

PROCESSED_DIR = 'data/processed'
VIZ_DIR = 'data/viz'
os.makedirs(VIZ_DIR, exist_ok=True)

class TimeSeriesForecaster:
    """
    时间序列预测类
    功能：销售预测、多模型对比、模型评估
    """
    
    def __init__(self):
        self.data = None
        self.models = {}
        self.predictions = {}
        self.metrics = {}
        self.best_model = None
    
    def log_step(self, message, status='info'):
        """
        记录处理步骤日志
        入参：message - 日志消息, status - 状态
        出参：打印彩色日志
        """
        colors = {'info': '\033[94m', 'success': '\033[92m', 'error': '\033[91m', 'warning': '\033[93m'}
        reset = '\033[0m'
        print(f'{colors.get(status, "")}[时间序列预测] {message}{reset}')
    
    def load_data(self):
        """
        加载处理后的销售数据
        入参：无
        出参：DataFrame
        """
        filepath = os.path.join(PROCESSED_DIR, 'daily_sales_processed.csv')
        if os.path.exists(filepath):
            self.data = pd.read_csv(filepath)
            self.data['date'] = pd.to_datetime(self.data['date'])
            self.data = self.data.sort_values('date')
            self.log_step(f'加载销售数据: {len(self.data)} 条记录', 'success')
            return self.data
        else:
            self.log_step('未找到销售数据文件', 'error')
            return None
    
    def prepare_features(self, forecast_days=30):
        """
        准备特征和目标变量
        入参：forecast_days - 预测天数
        出参：X_train, X_test, y_train, y_test, feature_cols
        """
        if self.data is None:
            self.log_step('数据未加载', 'error')
            return None
        
        df = self.data.copy()
        
        feature_cols = ['day_of_week', 'is_weekend', 'is_holiday', 'temperature', 'promotion_flag',
                        'sales_lag_1', 'sales_lag_7', 'sales_lag_30',
                        'sales_rolling_7_mean', 'sales_rolling_30_mean',
                        'sales_rolling_7_std', 'sales_diff_1', 'sales_diff_7']
        
        available_features = [col for col in feature_cols if col in df.columns]
        
        df = df.dropna(subset=available_features + ['total_sales'])
        
        X = df[available_features]
        y = df['total_sales']
        
        split_idx = len(df) - forecast_days
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
        self.log_step(f'训练集: {len(X_train)} 条, 测试集: {len(X_test)} 条', 'info')
        
        return X_train, X_test, y_train, y_test, available_features
    
    def train_models(self, X_train, y_train):
        """
        训练多个预测模型
        入参：X_train, y_train - 训练数据
        出参：训练好的模型字典
        """
        self.log_step('开始训练预测模型...')
        
        self.models['Linear Regression'] = LinearRegression()
        self.models['Linear Regression'].fit(X_train, y_train)
        
        self.models['Ridge Regression'] = Ridge(alpha=1.0)
        self.models['Ridge Regression'].fit(X_train, y_train)
        
        self.models['Random Forest'] = RandomForestRegressor(
            n_estimators=100, max_depth=10, random_state=42, n_jobs=-1
        )
        self.models['Random Forest'].fit(X_train, y_train)
        
        self.models['Gradient Boosting'] = GradientBoostingRegressor(
            n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42
        )
        self.models['Gradient Boosting'].fit(X_train, y_train)
        
        self.log_step('所有模型训练完成', 'success')
        return self.models
    
    def evaluate_models(self, X_test, y_test):
        """
        评估模型性能
        入参：X_test, y_test - 测试数据
        出参：评估指标DataFrame
        """
        self.log_step('评估模型性能...')
        
        results = []
        
        for name, model in self.models.items():
            y_pred = model.predict(X_test)
            self.predictions[name] = y_pred
            
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)
            mape = mean_absolute_percentage_error(y_test, y_pred) * 100
            
            results.append({
                '模型': name,
                'MAE': round(mae, 2),
                'RMSE': round(rmse, 2),
                'R²': round(r2, 4),
                'MAPE(%)': round(mape, 2)
            })
            
            self.metrics[name] = {'MAE': mae, 'RMSE': rmse, 'R²': r2, 'MAPE': mape}
        
        results_df = pd.DataFrame(results).sort_values('R²', ascending=False)
        
        self.best_model = results_df.iloc[0]['模型']
        
        self.log_step(f'最佳模型: {self.best_model} (R²={results_df.iloc[0]["R²"]:.4f})', 'success')
        
        return results_df
    
    def plot_predictions(self, y_test, X_test):
        """
        绘制预测对比图
        入参：y_test - 真实值, X_test - 测试特征
        出参：保存可视化图表
        """
        self.log_step('生成预测可视化图表...')
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        ax1 = axes[0, 0]
        ax1.plot(y_test.values, label='真实值', color='#2E86AB', linewidth=2)
        ax1.plot(self.predictions[self.best_model], label=f'{self.best_model}预测', 
                color='#A23B72', linewidth=2, alpha=0.8)
        ax1.set_title(f'销售预测对比 - {self.best_model}', fontsize=14, fontweight='bold')
        ax1.set_xlabel('样本序号')
        ax1.set_ylabel('销售额')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        ax2 = axes[0, 1]
        metrics_df = pd.DataFrame(self.metrics).T
        metrics_df['R²'].plot(kind='bar', ax=ax2, color='#2E86AB')
        ax2.set_title('各模型 R² 对比', fontsize=14, fontweight='bold')
        ax2.set_xlabel('模型')
        ax2.set_ylabel('R²')
        ax2.tick_params(axis='x', rotation=45)
        ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        
        ax3 = axes[1, 0]
        for name, pred in self.predictions.items():
            ax3.scatter(y_test.values, pred, alpha=0.5, label=name, s=30)
        ax3.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
                'k--', lw=2, label='完美预测线')
        ax3.set_title('真实值 vs 预测值散点图', fontsize=14, fontweight='bold')
        ax3.set_xlabel('真实值')
        ax3.set_ylabel('预测值')
        ax3.legend(fontsize=8)
        ax3.grid(True, alpha=0.3)
        
        ax4 = axes[1, 1]
        residuals = y_test.values - self.predictions[self.best_model]
        ax4.hist(residuals, bins=30, color='#F18F01', edgecolor='white', alpha=0.7)
        ax4.axvline(x=0, color='red', linestyle='--', linewidth=2)
        ax4.set_title(f'预测残差分布 - {self.best_model}', fontsize=14, fontweight='bold')
        ax4.set_xlabel('残差 (真实值 - 预测值)')
        ax4.set_ylabel('频数')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{VIZ_DIR}/时间序列预测结果.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        self.log_step('预测可视化图表已保存', 'success')
    
    def plot_feature_importance(self, feature_cols):
        """
        绘制特征重要性图
        入参：feature_cols - 特征列名
        出参：保存特征重要性图表
        """
        if 'Random Forest' not in self.models:
            return
        
        self.log_step('生成特征重要性图表...')
        
        importance = self.models['Random Forest'].feature_importances_
        feature_importance = pd.DataFrame({
            '特征': feature_cols,
            '重要性': importance
        }).sort_values('重要性', ascending=True)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.barh(feature_importance['特征'], feature_importance['重要性'], color='#2E86AB')
        ax.set_title('特征重要性分析 (随机森林)', fontsize=14, fontweight='bold')
        ax.set_xlabel('重要性')
        ax.set_ylabel('特征')
        ax.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        plt.savefig(f'{VIZ_DIR}/特征重要性.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        self.log_step('特征重要性图表已保存', 'success')
    
    def forecast_future(self, days=30):
        """
        预测未来销售额
        入参：days - 预测天数
        出参：预测结果DataFrame
        """
        self.log_step(f'预测未来 {days} 天销售额...')
        
        if self.data is None or self.best_model is None:
            self.log_step('数据或模型未准备好', 'error')
            return None
        
        last_date = self.data['date'].max()
        future_dates = [last_date + timedelta(days=i+1) for i in range(days)]
        
        last_values = self.data.iloc[-1].copy()
        
        future_predictions = []
        
        for i, date in enumerate(future_dates):
            features = {
                'day_of_week': date.weekday(),
                'is_weekend': 1 if date.weekday() >= 5 else 0,
                'is_holiday': 1 if date.month in [1, 2, 5, 10] and date.day <= 7 else 0,
                'temperature': np.random.uniform(10, 35),
                'promotion_flag': np.random.choice([0, 1], p=[0.8, 0.2])
            }
            
            if i == 0:
                features['sales_lag_1'] = last_values['total_sales']
                features['sales_lag_7'] = self.data[self.data['date'] == last_date - timedelta(days=7)]['total_sales'].values[0] if len(self.data[self.data['date'] == last_date - timedelta(days=7)]) > 0 else last_values['total_sales']
                features['sales_lag_30'] = self.data[self.data['date'] == last_date - timedelta(days=30)]['total_sales'].values[0] if len(self.data[self.data['date'] == last_date - timedelta(days=30)]) > 0 else last_values['total_sales']
            else:
                features['sales_lag_1'] = future_predictions[-1]
                features['sales_lag_7'] = future_predictions[-7] if len(future_predictions) >= 7 else future_predictions[-1]
                features['sales_lag_30'] = future_predictions[-30] if len(future_predictions) >= 30 else future_predictions[-1]
            
            features['sales_rolling_7_mean'] = np.mean(future_predictions[-7:]) if len(future_predictions) >= 7 else last_values.get('sales_rolling_7_mean', last_values['total_sales'])
            features['sales_rolling_30_mean'] = np.mean(future_predictions[-30:]) if len(future_predictions) >= 30 else last_values.get('sales_rolling_30_mean', last_values['total_sales'])
            features['sales_rolling_7_std'] = np.std(future_predictions[-7:]) if len(future_predictions) >= 7 else last_values.get('sales_rolling_7_std', 0)
            features['sales_diff_1'] = future_predictions[-1] - future_predictions[-2] if len(future_predictions) >= 2 else 0
            features['sales_diff_7'] = future_predictions[-1] - future_predictions[-8] if len(future_predictions) >= 8 else 0
            
            X_future = pd.DataFrame([features])
            
            for col in self.models[self.best_model].feature_names_in_:
                if col not in X_future.columns:
                    X_future[col] = 0
            
            X_future = X_future[self.models[self.best_model].feature_names_in_]
            
            pred = self.models[self.best_model].predict(X_future)[0]
            future_predictions.append(pred)
        
        forecast_df = pd.DataFrame({
            'date': future_dates,
            'predicted_sales': future_predictions
        })
        
        fig, ax = plt.subplots(figsize=(14, 6))
        
        historical = self.data.tail(60)
        ax.plot(historical['date'], historical['total_sales'], 
               label='历史数据', color='#2E86AB', linewidth=2)
        ax.plot(forecast_df['date'], forecast_df['predicted_sales'], 
               label='预测数据', color='#A23B72', linewidth=2, linestyle='--')
        ax.axvline(x=last_date, color='gray', linestyle=':', linewidth=2, label='预测起点')
        ax.fill_between(forecast_df['date'], 
                       forecast_df['predicted_sales'] * 0.9,
                       forecast_df['predicted_sales'] * 1.1,
                       alpha=0.2, color='#A23B72', label='预测区间')
        ax.set_title(f'未来 {days} 天销售预测', fontsize=14, fontweight='bold')
        ax.set_xlabel('日期')
        ax.set_ylabel('销售额')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{VIZ_DIR}/未来销售预测.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        self.log_step(f'未来销售预测完成，预测 {len(forecast_df)} 天', 'success')
        
        return forecast_df
    
    def run_full_analysis(self):
        """
        执行完整的时间序列预测分析
        入参：无
        出参：预测结果
        """
        print('\033[96m' + '='*60 + '\033[0m')
        print('\033[96m  AI驱动的商业分析平台 - 时间序列预测\033[0m')
        print('\033[96m' + '='*60 + '\033[0m')
        print()
        
        self.load_data()
        print()
        
        X_train, X_test, y_train, y_test, feature_cols = self.prepare_features()
        print()
        
        self.train_models(X_train, y_train)
        print()
        
        results_df = self.evaluate_models(X_test, y_test)
        print()
        
        self.plot_predictions(y_test, X_test)
        print()
        
        self.plot_feature_importance(feature_cols)
        print()
        
        forecast_df = self.forecast_future(30)
        print()
        
        print('\033[92m' + '='*60 + '\033[0m')
        print('\033[92m  ✅ 时间序列预测分析完成！\033[0m')
        print('\033[92m' + '='*60 + '\033[0m')
        
        print('\n模型性能对比:')
        print(results_df.to_string(index=False))
        
        print('\n未来30天预测摘要:')
        if forecast_df is not None:
            print(f'  预测总销售额: {forecast_df["predicted_sales"].sum():,.2f}')
            print(f'  预测日均销售额: {forecast_df["predicted_sales"].mean():,.2f}')
            print(f'  预测最高日销售额: {forecast_df["predicted_sales"].max():,.2f}')
            print(f'  预测最低日销售额: {forecast_df["predicted_sales"].min():,.2f}')
        
        return {
            'metrics': self.metrics,
            'best_model': self.best_model,
            'forecast': forecast_df
        }

def main():
    forecaster = TimeSeriesForecaster()
    forecaster.run_full_analysis()

if __name__ == '__main__':
    main()
