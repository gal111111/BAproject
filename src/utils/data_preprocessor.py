"""
AI驱动的商业分析平台 - 数据预处理模块
功能：数据清洗、缺失值处理、异常值检测、特征工程
入参：原始数据路径
出参：清洗后的数据保存到data/processed/
异常处理：文件读取/写入异常捕获
"""
import pandas as pd
import numpy as np
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

RAW_DIR = 'data/raw'
PROCESSED_DIR = 'data/processed'
os.makedirs(PROCESSED_DIR, exist_ok=True)

class DataPreprocessor:
    """
    数据预处理类
    功能：数据清洗、缺失值处理、异常值检测、特征工程
    """
    
    def __init__(self):
        self.processed_data = {}
        self.preprocessing_log = []
    
    def log_step(self, message, status='info'):
        """
        记录处理步骤日志
        入参：message - 日志消息, status - 状态(info/success/error)
        出参：打印彩色日志
        """
        colors = {
            'info': '\033[94m',
            'success': '\033[92m',
            'error': '\033[91m',
            'warning': '\033[93m'
        }
        reset = '\033[0m'
        print(f'{colors.get(status, "")}[数据预处理] {message}{reset}')
        self.preprocessing_log.append(f'{datetime.now().strftime("%H:%M:%S")} - {message}')
    
    def load_data(self, filename):
        """
        加载原始数据
        入参：filename - 文件名
        出参：DataFrame或None
        """
        filepath = os.path.join(RAW_DIR, filename)
        try:
            df = pd.read_csv(filepath, encoding='utf-8-sig')
            self.log_step(f'加载数据: {filename} ({len(df)} 行, {len(df.columns)} 列)', 'success')
            return df
        except FileNotFoundError:
            self.log_step(f'文件不存在: {filename}', 'error')
            return None
        except Exception as e:
            self.log_step(f'加载失败: {filename} - {str(e)}', 'error')
            return None
    
    def check_missing_values(self, df, name='数据集'):
        """
        检查缺失值
        入参：df - DataFrame, name - 数据集名称
        出参：缺失值统计DataFrame
        """
        missing = df.isnull().sum()
        missing_pct = (missing / len(df) * 100).round(2)
        missing_df = pd.DataFrame({
            '列名': df.columns,
            '缺失值数量': missing.values,
            '缺失比例(%)': missing_pct.values
        })
        missing_df = missing_df[missing_df['缺失值数量'] > 0]
        
        if len(missing_df) > 0:
            self.log_step(f'{name} 发现 {len(missing_df)} 列存在缺失值', 'warning')
        else:
            self.log_step(f'{name} 无缺失值', 'success')
        
        return missing_df
    
    def handle_missing_values(self, df, strategy='auto'):
        """
        处理缺失值
        入参：df - DataFrame, strategy - 处理策略(auto/mean/median/mode/drop)
        出参：处理后的DataFrame
        """
        df_clean = df.copy()
        
        for col in df_clean.columns:
            if df_clean[col].isnull().sum() > 0:
                missing_pct = df_clean[col].isnull().sum() / len(df_clean) * 100
                
                if missing_pct > 50:
                    df_clean = df_clean.drop(columns=[col])
                    self.log_step(f'删除缺失率>50%的列: {col}', 'warning')
                elif df_clean[col].dtype in ['int64', 'float64']:
                    if strategy == 'mean':
                        df_clean[col] = df_clean[col].fillna(df_clean[col].mean())
                    elif strategy == 'median':
                        df_clean[col] = df_clean[col].fillna(df_clean[col].median())
                    else:
                        df_clean[col] = df_clean[col].fillna(df_clean[col].median())
                    self.log_step(f'数值列 {col} 缺失值已填充(中位数)', 'info')
                else:
                    if strategy == 'mode':
                        fill_value = df_clean[col].mode()[0] if len(df_clean[col].mode()) > 0 else 'Unknown'
                    else:
                        fill_value = 'Unknown'
                    df_clean[col] = df_clean[col].fillna(fill_value)
                    self.log_step(f'分类列 {col} 缺失值已填充({fill_value})', 'info')
        
        return df_clean
    
    def detect_outliers_iqr(self, df, columns):
        """
        使用IQR方法检测异常值
        入参：df - DataFrame, columns - 需要检测的列名列表
        出参：异常值统计字典
        """
        outliers_stats = {}
        
        for col in columns:
            if col in df.columns and df[col].dtype in ['int64', 'float64']:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
                outliers_stats[col] = {
                    'count': len(outliers),
                    'percentage': round(len(outliers) / len(df) * 100, 2),
                    'lower_bound': round(lower_bound, 2),
                    'upper_bound': round(upper_bound, 2)
                }
                
                if len(outliers) > 0:
                    self.log_step(f'{col}: 发现 {len(outliers)} 个异常值 ({outliers_stats[col]["percentage"]}%)', 'warning')
        
        return outliers_stats
    
    def handle_outliers(self, df, columns, method='clip'):
        """
        处理异常值
        入参：df - DataFrame, columns - 列名列表, method - 处理方法(clip/remove)
        出参：处理后的DataFrame
        """
        df_clean = df.copy()
        
        for col in columns:
            if col in df_clean.columns and df_clean[col].dtype in ['int64', 'float64']:
                Q1 = df_clean[col].quantile(0.25)
                Q3 = df_clean[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                if method == 'clip':
                    df_clean[col] = df_clean[col].clip(lower_bound, upper_bound)
                    self.log_step(f'{col} 异常值已裁剪到 [{lower_bound:.2f}, {upper_bound:.2f}]', 'info')
                elif method == 'remove':
                    df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]
                    self.log_step(f'{col} 异常值已移除', 'info')
        
        return df_clean
    
    def feature_engineering_transactions(self, df):
        """
        交易数据特征工程
        入参：df - 交易数据DataFrame
        出参：特征增强后的DataFrame
        """
        df_feat = df.copy()
        
        df_feat['transaction_date'] = pd.to_datetime(df_feat['transaction_date'])
        df_feat['year'] = df_feat['transaction_date'].dt.year
        df_feat['month'] = df_feat['transaction_date'].dt.month
        df_feat['day'] = df_feat['transaction_date'].dt.day
        df_feat['day_of_week'] = df_feat['transaction_date'].dt.dayofweek
        df_feat['is_weekend'] = (df_feat['day_of_week'] >= 5).astype(int)
        df_feat['quarter'] = df_feat['transaction_date'].dt.quarter
        df_feat['week_of_year'] = df_feat['transaction_date'].dt.isocalendar().week
        
        df_feat['price_per_item'] = df_feat['total_amount'] / df_feat['quantity']
        
        if 'discount_rate' in df_feat.columns:
            df_feat['discount_amount'] = df_feat['unit_price'] * df_feat['discount_rate']
        else:
            df_feat['discount_amount'] = 0
            df_feat['discount_rate'] = 0
        
        if 'payment_method' not in df_feat.columns:
            df_feat['payment_method'] = '未知'
        
        if 'channel' not in df_feat.columns:
            df_feat['channel'] = '网页'
        
        if 'is_returned' not in df_feat.columns:
            df_feat['is_returned'] = 0
        
        if 'province' not in df_feat.columns:
            df_feat['province'] = '未知'
        
        if 'product_category' not in df_feat.columns:
            df_feat['product_category'] = '其他'
        
        self.log_step('交易数据特征工程完成', 'success')
        return df_feat
    
    def feature_engineering_customers(self, df):
        """
        客户数据特征工程
        入参：df - 客户数据DataFrame
        出参：特征增强后的DataFrame
        """
        df_feat = df.copy()
        
        if 'register_date' in df_feat.columns:
            df_feat['register_date'] = pd.to_datetime(df_feat['register_date'])
            df_feat['customer_tenure_days'] = (datetime.now() - df_feat['register_date']).dt.days
            df_feat['customer_tenure_months'] = df_feat['customer_tenure_days'] / 30
        else:
            df_feat['customer_tenure_days'] = 365
            df_feat['customer_tenure_months'] = 12
        
        df_feat['avg_order_value'] = df_feat['total_spent'] / (df_feat['total_orders'] + 1)
        df_feat['order_frequency'] = df_feat['total_orders'] / (df_feat['customer_tenure_months'] + 1)
        
        if 'last_login_days' in df_feat.columns:
            df_feat['is_active'] = (df_feat['last_login_days'] <= 30).astype(int)
            df_feat['is_churn_risk'] = (df_feat['last_login_days'] >= 90).astype(int)
        else:
            df_feat['is_active'] = 1
            df_feat['is_churn_risk'] = 0
            df_feat['last_login_days'] = 0
        
        if 'age' in df_feat.columns:
            age_bins = [0, 25, 35, 45, 55, 100]
            age_labels = ['18-25', '26-35', '36-45', '46-55', '55+']
            df_feat['age_group'] = pd.cut(df_feat['age'], bins=age_bins, labels=age_labels)
        else:
            df_feat['age_group'] = '未知'
            df_feat['age'] = 35
        
        if 'gender' not in df_feat.columns:
            df_feat['gender'] = '未知'
        
        if 'city_tier' not in df_feat.columns:
            df_feat['city_tier'] = '二线'
        
        if 'income_level' not in df_feat.columns:
            df_feat['income_level'] = '中'
        
        if 'membership_level' not in df_feat.columns:
            df_feat['membership_level'] = '普通'
        
        if 'preferred_category' not in df_feat.columns:
            df_feat['preferred_category'] = '其他'
        
        if 'customer_segment' not in df_feat.columns:
            df_feat['customer_segment'] = '普通客户'
        
        self.log_step('客户数据特征工程完成', 'success')
        return df_feat
    
    def feature_engineering_daily_sales(self, df):
        """
        每日销售数据特征工程
        入参：df - 每日销售DataFrame
        出参：特征增强后的DataFrame
        """
        df_feat = df.copy()
        
        df_feat['date'] = pd.to_datetime(df_feat['date'])
        df_feat = df_feat.sort_values('date')
        
        if 'total_sales' not in df_feat.columns:
            if 'sales' in df_feat.columns:
                df_feat['total_sales'] = df_feat['sales']
            else:
                df_feat['total_sales'] = df_feat['order_count'] * df_feat['avg_order_value']
        
        df_feat['sales_lag_1'] = df_feat['total_sales'].shift(1)
        df_feat['sales_lag_7'] = df_feat['total_sales'].shift(7)
        df_feat['sales_lag_30'] = df_feat['total_sales'].shift(30)
        
        df_feat['sales_rolling_7_mean'] = df_feat['total_sales'].rolling(window=7).mean()
        df_feat['sales_rolling_30_mean'] = df_feat['total_sales'].rolling(window=30).mean()
        df_feat['sales_rolling_7_std'] = df_feat['total_sales'].rolling(window=7).std()
        
        df_feat['sales_diff_1'] = df_feat['total_sales'].diff(1)
        df_feat['sales_diff_7'] = df_feat['total_sales'].diff(7)
        
        df_feat['sales_pct_change_1'] = df_feat['total_sales'].pct_change(1)
        df_feat['sales_pct_change_7'] = df_feat['total_sales'].pct_change(7)
        
        if 'year' not in df_feat.columns:
            df_feat['year'] = df_feat['date'].dt.year
        
        if 'month' not in df_feat.columns:
            df_feat['month'] = df_feat['date'].dt.month
        
        if 'day' not in df_feat.columns:
            df_feat['day'] = df_feat['date'].dt.day
        
        if 'day_of_week' not in df_feat.columns:
            df_feat['day_of_week'] = df_feat['date'].dt.dayofweek
        
        if 'is_weekend' not in df_feat.columns:
            df_feat['is_weekend'] = (df_feat['day_of_week'] >= 5).astype(int)
        
        if 'is_holiday' not in df_feat.columns:
            df_feat['is_holiday'] = 0
        
        if 'temperature' not in df_feat.columns:
            df_feat['temperature'] = 20
        
        if 'promotion_flag' not in df_feat.columns:
            df_feat['promotion_flag'] = 0
        
        if 'competitor_price' not in df_feat.columns:
            df_feat['competitor_price'] = 0
        
        df_feat = df_feat.dropna()
        
        self.log_step('每日销售数据特征工程完成', 'success')
        return df_feat
    
    def save_processed_data(self, df, filename):
        """
        保存处理后的数据
        入参：df - DataFrame, filename - 文件名
        出参：保存到data/processed/
        """
        filepath = os.path.join(PROCESSED_DIR, filename)
        df.to_csv(filepath, index=False, encoding='utf-8-sig')
        self.log_step(f'保存处理数据: {filename} ({len(df)} 行)', 'success')
    
    def process_all_data(self):
        """
        处理所有数据集
        入参：无
        出参：处理后的数据字典
        """
        self.log_step('开始数据预处理流程...', 'info')
        print()
        
        transactions_df = self.load_data('transactions.csv')
        if transactions_df is not None:
            self.check_missing_values(transactions_df, '交易数据')
            transactions_df = self.handle_missing_values(transactions_df)
            transactions_df = self.feature_engineering_transactions(transactions_df)
            self.save_processed_data(transactions_df, 'transactions_processed.csv')
            self.processed_data['transactions'] = transactions_df
        
        print()
        customers_df = self.load_data('customers.csv')
        if customers_df is not None:
            self.check_missing_values(customers_df, '客户数据')
            customers_df = self.handle_missing_values(customers_df)
            customers_df = self.feature_engineering_customers(customers_df)
            self.save_processed_data(customers_df, 'customers_processed.csv')
            self.processed_data['customers'] = customers_df
        
        print()
        daily_sales_df = self.load_data('daily_sales.csv')
        if daily_sales_df is not None:
            self.check_missing_values(daily_sales_df, '每日销售数据')
            daily_sales_df = self.handle_missing_values(daily_sales_df)
            daily_sales_df = self.feature_engineering_daily_sales(daily_sales_df)
            self.save_processed_data(daily_sales_df, 'daily_sales_processed.csv')
            self.processed_data['daily_sales'] = daily_sales_df
        
        print()
        user_behavior_df = self.load_data('user_behavior.csv')
        if user_behavior_df is not None:
            self.check_missing_values(user_behavior_df, '用户行为数据')
            user_behavior_df = self.handle_missing_values(user_behavior_df)
            self.save_processed_data(user_behavior_df, 'user_behavior_processed.csv')
            self.processed_data['user_behavior'] = user_behavior_df
        
        print()
        reviews_df = self.load_data('reviews.csv')
        if reviews_df is not None:
            self.check_missing_values(reviews_df, '评论数据')
            reviews_df = self.handle_missing_values(reviews_df)
            self.save_processed_data(reviews_df, 'reviews_processed.csv')
            self.processed_data['reviews'] = reviews_df
        
        print()
        products_df = self.load_data('products.csv')
        if products_df is not None:
            self.check_missing_values(products_df, '商品数据')
            products_df = self.handle_missing_values(products_df)
            self.save_processed_data(products_df, 'products_processed.csv')
            self.processed_data['products'] = products_df
        
        print()
        self.log_step('所有数据预处理完成！', 'success')
        
        return self.processed_data

def main():
    """
    主函数
    """
    print('\033[96m' + '='*60 + '\033[0m')
    print('\033[96m  AI驱动的商业分析平台 - 数据预处理模块\033[0m')
    print('\033[96m' + '='*60 + '\033[0m')
    print()
    
    preprocessor = DataPreprocessor()
    processed_data = preprocessor.process_all_data()
    
    print()
    print('\033[92m' + '='*60 + '\033[0m')
    print('\033[92m  ✅ 数据预处理完成！\033[0m')
    print('\033[92m' + '='*60 + '\033[0m')
    
    print('\n处理后的数据文件:')
    for f in os.listdir(PROCESSED_DIR):
        if f.endswith('.csv'):
            filepath = os.path.join(PROCESSED_DIR, f)
            size = os.path.getsize(filepath) / 1024
            print(f'  📄 {f} ({size:.1f} KB)')

if __name__ == '__main__':
    main()
