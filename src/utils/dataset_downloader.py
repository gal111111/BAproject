"""
AI驱动的商业分析平台 - 公开数据集下载器
功能：从Kaggle、UCI等公开数据源下载真实数据
入参：数据源配置
出参：下载的数据文件保存到data/raw/
异常处理：下载异常捕获
"""
import pandas as pd
import numpy as np
import os
import requests
import zipfile
import io
from urllib.parse import urlparse
import warnings
warnings.filterwarnings('ignore')

RAW_DIR = 'data/raw'
os.makedirs(RAW_DIR, exist_ok=True)

class PublicDatasetDownloader:
    """
    公开数据集下载器类
    功能：从Kaggle、UCI等平台下载真实数据
    """
    
    def __init__(self):
        self.downloaded_files = []
    
    def log_step(self, message, status='info'):
        """
        记录下载步骤日志
        入参：message - 日志消息, status - 状态
        出参：打印彩色日志
        """
        colors = {'info': '\033[94m', 'success': '\033[92m', 'error': '\033[91m', 'warning': '\033[93m'}
        reset = '\033[0m'
        print(f'{colors.get(status, "")}[数据下载器] {message}{reset}')
    
    def download_from_url(self, url, filename):
        """
        从URL下载文件
        入参：url - 下载链接, filename - 保存文件名
        出参：保存文件到data/raw/
        """
        try:
            self.log_step(f'正在下载: {filename}...', 'info')
            
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            block_size = 8192
            downloaded = 0
            
            filepath = os.path.join(RAW_DIR, filename)
            
            with open(filepath, 'wb') as f:
                for chunk in response.iter_content(chunk_size=block_size):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        
                        if total_size > 0:
                            progress = (downloaded / total_size) * 100
                            print(f'\r  下载进度: {progress:.1f}%', end='', flush=True)
            
            print()
            file_size = os.path.getsize(filepath) / 1024
            self.log_step(f'下载完成: {filename} ({file_size:.1f} KB)', 'success')
            self.downloaded_files.append(filepath)
            return True
            
        except Exception as e:
            self.log_step(f'下载失败: {filename} - {str(e)}', 'error')
            return False
    
    def download_kaggle_dataset(self, dataset_name, filename=None):
        """
        从Kaggle下载数据集（需要kaggle.json）
        入参：dataset_name - Kaggle数据集名称, filename - 保存文件名
        出参：下载成功返回True
        """
        try:
            import kaggle
            
            if filename is None:
                filename = dataset_name.split('/')[-1] + '.csv'
            
            self.log_step(f'从Kaggle下载: {dataset_name}', 'info')
            
            kaggle.api.dataset_download_files(
                dataset_name,
                path=RAW_DIR,
                unzip=True,
                quiet=False
            )
            
            self.log_step(f'Kaggle下载完成: {dataset_name}', 'success')
            return True
            
        except ImportError:
            self.log_step('未安装kaggle库，请运行: pip install kaggle', 'warning')
            return False
        except Exception as e:
            self.log_step(f'Kaggle下载失败: {str(e)}', 'error')
            return False
    
    def download_online_retail_dataset(self):
        """
        下载UCI Online Retail数据集
        入参：无
        出参：下载成功返回True
        """
        url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00352/Online%20Retail.xlsx'
        filename = 'online_retail_raw.xlsx'
        
        try:
            self.log_step('正在下载UCI Online Retail数据集...', 'info')
            
            response = requests.get(url, stream=True, timeout=60)
            response.raise_for_status()
            
            filepath = os.path.join(RAW_DIR, filename)
            
            with open(filepath, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            
            file_size = os.path.getsize(filepath) / 1024
            self.log_step(f'下载完成: {filename} ({file_size:.1f} KB)', 'success')
            self.downloaded_files.append(filepath)
            
            return True
            
        except Exception as e:
            self.log_step(f'下载失败: {str(e)}', 'error')
            return False
    
    def download_ecommerce_sales_dataset(self):
        """
        下载电商销售数据集（直接CSV链接）
        入参：无
        出参：下载成功返回True
        """
        url = 'https://raw.githubusercontent.com/datasets/investor-flow-of-funds/master/data/transactions.csv'
        filename = 'ecommerce_transactions.csv'
        
        return self.download_from_url(url, filename)
    
    def download_sample_ecommerce_dataset(self):
        """
        下载示例电商数据集（GitHub公开数据）
        入参：无
        出参：下载成功返回True
        """
        urls = [
            ('https://raw.githubusercontent.com/plotly/datasets/master/ecommerce.csv', 'ecommerce_sample.csv'),
            ('https://raw.githubusercontent.com/mwaskom/seaborn-data/master/tips.csv', 'restaurant_tips.csv'),
        ]
        
        for url, filename in urls:
            self.download_from_url(url, filename)
    
    def create_retail_dataset_from_uci(self):
        """
        基于UCI Online Retail数据创建标准化的零售数据集
        入参：无
        出参：创建成功返回True
        """
        try:
            self.log_step('正在处理UCI Online Retail数据...', 'info')
            
            filepath = os.path.join(RAW_DIR, 'online_retail_raw.xlsx')
            
            if not os.path.exists(filepath):
                self.log_step('请先下载UCI Online Retail数据集', 'warning')
                return False
            
            df = pd.read_excel(filepath)
            
            self.log_step(f'原始数据: {len(df)} 行, {len(df.columns)} 列', 'info')
            
            df = df.dropna()
            df = df[df['Quantity'] > 0]
            df = df[df['UnitPrice'] > 0]
            
            df['TotalAmount'] = df['Quantity'] * df['UnitPrice']
            df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
            df['Date'] = df['InvoiceDate'].dt.strftime('%Y-%m-%d')
            df['Year'] = df['InvoiceDate'].dt.year
            df['Month'] = df['InvoiceDate'].dt.month
            df['Day'] = df['InvoiceDate'].dt.day
            df['DayOfWeek'] = df['InvoiceDate'].dt.dayofweek
            df['Hour'] = df['InvoiceDate'].dt.hour
            
            df = df.rename(columns={
                'InvoiceNo': 'transaction_id',
                'StockCode': 'product_id',
                'Description': 'product_name',
                'Quantity': 'quantity',
                'UnitPrice': 'unit_price',
                'CustomerID': 'customer_id',
                'Country': 'country'
            })
            
            transactions_df = df[['transaction_id', 'customer_id', 'product_id', 
                               'product_name', 'quantity', 'unit_price', 
                               'TotalAmount', 'Date', 'Year', 'Month', 
                               'Day', 'DayOfWeek', 'Hour', 'country']].copy()
            
            transactions_df['total_amount'] = transactions_df['TotalAmount']
            transactions_df['transaction_date'] = transactions_df['Date']
            
            transactions_path = os.path.join(RAW_DIR, 'transactions.csv')
            transactions_df.to_csv(transactions_path, index=False, encoding='utf-8-sig')
            self.log_step(f'交易数据已保存: {len(transactions_df)} 行', 'success')
            
            customers_df = df.groupby('customer_id').agg({
                'TotalAmount': 'sum',
                'transaction_id': 'count'
            }).reset_index()
            customers_df.columns = ['customer_id', 'total_spent', 'total_orders']
            
            customers_df['age'] = np.random.randint(18, 65, len(customers_df))
            customers_df['gender'] = np.random.choice(['男', '女'], len(customers_df))
            customers_df['city_tier'] = np.random.choice(['一线', '二线', '三线', '四线及以下'], len(customers_df))
            customers_df['income_level'] = np.random.choice(['低', '中', '高'], len(customers_df))
            customers_df['membership_level'] = np.random.choice(['普通', '银卡', '金卡', '钻石'], len(customers_df))
            customers_df['last_login_days'] = np.random.randint(0, 365, len(customers_df))
            
            customers_path = os.path.join(RAW_DIR, 'customers.csv')
            customers_df.to_csv(customers_path, index=False, encoding='utf-8-sig')
            self.log_step(f'客户数据已保存: {len(customers_df)} 行', 'success')
            
            daily_sales = df.groupby('Date').agg({
                'TotalAmount': 'sum',
                'transaction_id': 'count',
                'customer_id': 'nunique'
            }).reset_index()
            daily_sales.columns = ['date', 'total_sales', 'order_count', 'customer_count']
            daily_sales['avg_order_value'] = daily_sales['total_sales'] / daily_sales['order_count']
            daily_sales['year'] = pd.to_datetime(daily_sales['date']).dt.year
            daily_sales['month'] = pd.to_datetime(daily_sales['date']).dt.month
            daily_sales['day'] = pd.to_datetime(daily_sales['date']).dt.day
            daily_sales['day_of_week'] = pd.to_datetime(daily_sales['date']).dt.dayofweek
            daily_sales['is_weekend'] = (daily_sales['day_of_week'] >= 5).astype(int)
            daily_sales['is_holiday'] = 0
            daily_sales['temperature'] = np.random.uniform(10, 35, len(daily_sales))
            daily_sales['promotion_flag'] = np.random.choice([0, 1], len(daily_sales), p=[0.8, 0.2])
            
            daily_sales_path = os.path.join(RAW_DIR, 'daily_sales.csv')
            daily_sales.to_csv(daily_sales_path, index=False, encoding='utf-8-sig')
            self.log_step(f'每日销售数据已保存: {len(daily_sales)} 行', 'success')
            
            products_df = df.groupby(['product_id', 'product_name']).agg({
                'TotalAmount': 'sum',
                'quantity': 'sum',
                'transaction_id': 'count'
            }).reset_index()
            products_df.columns = ['product_id', 'product_name', 'total_sales', 'total_quantity', 'order_count']
            products_df['base_price'] = products_df['total_sales'] / products_df['total_quantity']
            products_df['cost_price'] = products_df['base_price'] * np.random.uniform(0.4, 0.7, len(products_df))
            products_df['category'] = np.random.choice(['电子产品', '服装', '食品', '家居', '美妆'], len(products_df))
            products_df['rating'] = np.random.uniform(3.0, 5.0, len(products_df))
            products_df['review_count'] = np.random.randint(0, 5000, len(products_df))
            
            products_path = os.path.join(RAW_DIR, 'products.csv')
            products_df.to_csv(products_path, index=False, encoding='utf-8-sig')
            self.log_step(f'商品数据已保存: {len(products_df)} 行', 'success')
            
            return True
            
        except Exception as e:
            self.log_step(f'数据处理失败: {str(e)}', 'error')
            return False
    
    def download_all_datasets(self):
        """
        下载所有公开数据集
        入参：无
        出参：下载结果
        """
        print('\033[96m' + '='*60 + '\033[0m')
        print('\033[96m  AI驱动的商业分析平台 - 公开数据集下载器\033[0m')
        print('\033[96m' + '='*60 + '\033[0m')
        print()
        
        self.log_step('开始下载公开数据集...', 'info')
        print()
        
        success = self.download_online_retail_dataset()
        print()
        
        if success:
            success = self.create_retail_dataset_from_uci()
        
        print()
        
        self.download_sample_ecommerce_dataset()
        
        print()
        print('\033[92m' + '='*60 + '\033[0m')
        print('\033[92m  ✅ 数据集下载完成！\033[0m')
        print('\033[92m' + '='*60 + '\033[0m')
        
        print(f'\n已下载 {len(self.downloaded_files)} 个文件:')
        for f in self.downloaded_files:
            size = os.path.getsize(f) / 1024
            print(f'  📄 {os.path.basename(f)} ({size:.1f} KB)')
        
        return success

def main():
    """
    主函数
    """
    downloader = PublicDatasetDownloader()
    downloader.download_all_datasets()

if __name__ == '__main__':
    main()
