"""
AI驱动的商业分析平台 - 描述性统计分析模块
功能：数据概览、趋势分析、分布分析、相关性分析
入参：处理后的数据
出参：分析结果和可视化图表
异常处理：数据加载异常捕获
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

PROCESSED_DIR = 'data/processed'
VIZ_DIR = 'data/viz'
os.makedirs(VIZ_DIR, exist_ok=True)

class DescriptiveAnalyzer:
    """
    描述性统计分析类
    功能：数据概览、趋势分析、分布分析、相关性分析
    """
    
    def __init__(self):
        self.data = {}
        self.analysis_results = {}
    
    def log_step(self, message, status='info'):
        """
        记录分析步骤日志
        入参：message - 日志消息, status - 状态
        出参：打印彩色日志
        """
        colors = {'info': '\033[94m', 'success': '\033[92m', 'error': '\033[91m', 'warning': '\033[93m'}
        reset = '\033[0m'
        print(f'{colors.get(status, "")}[描述性分析] {message}{reset}')
    
    def load_data(self):
        """
        加载处理后的数据
        入参：无
        出参：数据字典
        """
        files = {
            'transactions': 'transactions_processed.csv',
            'customers': 'customers_processed.csv',
            'daily_sales': 'daily_sales_processed.csv',
            'user_behavior': 'user_behavior_processed.csv',
            'reviews': 'reviews_processed.csv',
            'products': 'products_processed.csv'
        }
        
        for name, filename in files.items():
            filepath = os.path.join(PROCESSED_DIR, filename)
            if os.path.exists(filepath):
                self.data[name] = pd.read_csv(filepath)
                self.log_step(f'加载数据: {name} ({len(self.data[name])} 行)', 'success')
        
        return self.data
    
    def generate_overview_stats(self):
        """
        生成数据概览统计
        入参：无
        出参：概览统计字典
        """
        self.log_step('生成数据概览统计...')
        
        overview = {}
        
        if 'transactions' in self.data:
            df = self.data['transactions']
            overview['transactions'] = {
                '总交易数': len(df),
                '总销售额': df['total_amount'].sum(),
                '平均订单金额': df['total_amount'].mean(),
                '客户数量': df['customer_id'].nunique(),
                '商品数量': df['product_id'].nunique(),
                '退货率': df['is_returned'].mean() * 100
            }
        
        if 'customers' in self.data:
            df = self.data['customers']
            overview['customers'] = {
                '总客户数': len(df),
                '活跃客户比例': df['is_active'].mean() * 100,
                '流失风险比例': df['is_churn_risk'].mean() * 100,
                '平均消费金额': df['total_spent'].mean(),
                '平均订单数': df['total_orders'].mean()
            }
        
        if 'daily_sales' in self.data:
            df = self.data['daily_sales']
            overview['daily_sales'] = {
                '统计天数': len(df),
                '日均销售额': df['total_sales'].mean(),
                '销售额标准差': df['total_sales'].std(),
                '最高日销售额': df['total_sales'].max(),
                '最低日销售额': df['total_sales'].min()
            }
        
        if 'reviews' in self.data:
            df = self.data['reviews']
            overview['reviews'] = {
                '总评论数': len(df),
                '平均评分': df['rating'].mean(),
                '好评率': (df['rating'] >= 4).mean() * 100,
                '差评率': (df['rating'] <= 2).mean() * 100
            }
        
        self.analysis_results['overview'] = overview
        self.log_step('数据概览统计完成', 'success')
        return overview
    
    def analyze_sales_trends(self):
        """
        分析销售趋势
        入参：无
        出参：趋势分析结果
        """
        self.log_step('分析销售趋势...')
        
        if 'daily_sales' not in self.data:
            self.log_step('缺少每日销售数据', 'error')
            return None
        
        df = self.data['daily_sales'].copy()
        df['date'] = pd.to_datetime(df['date'])
        
        monthly_sales = df.groupby(['year', 'month']).agg({
            'total_sales': 'sum',
            'order_count': 'sum',
            'customer_count': 'sum'
        }).reset_index()
        
        monthly_sales['year_month'] = monthly_sales['year'].astype(str) + '-' + monthly_sales['month'].astype(str).str.zfill(2)
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        ax1 = axes[0, 0]
        ax1.plot(df['date'], df['total_sales'], color='#2E86AB', linewidth=0.8, alpha=0.7)
        ax1.set_title('每日销售趋势', fontsize=14, fontweight='bold')
        ax1.set_xlabel('日期')
        ax1.set_ylabel('销售额')
        ax1.grid(True, alpha=0.3)
        
        ax2 = axes[0, 1]
        weekday_sales = df.groupby('day_of_week')['total_sales'].mean()
        weekday_names = ['周一', '周二', '周三', '周四', '周五', '周六', '周日']
        colors = ['#A23B72' if i < 5 else '#2E86AB' for i in range(7)]
        ax2.bar(weekday_names, weekday_sales.values, color=colors)
        ax2.set_title('各工作日平均销售额', fontsize=14, fontweight='bold')
        ax2.set_xlabel('星期')
        ax2.set_ylabel('平均销售额')
        
        ax3 = axes[1, 0]
        monthly_pivot = df.pivot_table(index='month', columns='year', values='total_sales', aggfunc='mean')
        monthly_pivot.plot(kind='bar', ax=ax3, colormap='viridis')
        ax3.set_title('各月份平均销售额对比', fontsize=14, fontweight='bold')
        ax3.set_xlabel('月份')
        ax3.set_ylabel('平均销售额')
        ax3.legend(title='年份')
        ax3.tick_params(axis='x', rotation=0)
        
        ax4 = axes[1, 1]
        quarter_sales = df.groupby('month')['total_sales'].mean()
        ax4.plot(quarter_sales.index, quarter_sales.values, marker='o', linewidth=2, markersize=8, color='#F18F01')
        ax4.fill_between(quarter_sales.index, quarter_sales.values, alpha=0.3, color='#F18F01')
        ax4.set_title('月度销售季节性', fontsize=14, fontweight='bold')
        ax4.set_xlabel('月份')
        ax4.set_ylabel('平均销售额')
        ax4.set_xticks(range(1, 13))
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{VIZ_DIR}/销售趋势分析.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        self.log_step('销售趋势分析完成，图表已保存', 'success')
        return monthly_sales
    
    def analyze_customer_segments(self):
        """
        分析客户群体
        入参：无
        出参：客户分群分析结果
        """
        self.log_step('分析客户群体...')
        
        if 'customers' not in self.data:
            self.log_step('缺少客户数据', 'error')
            return None
        
        df = self.data['customers'].copy()
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        ax1 = axes[0, 0]
        segment_counts = df['customer_segment'].value_counts()
        colors = plt.cm.Set2(np.linspace(0, 1, len(segment_counts)))
        ax1.pie(segment_counts.values, labels=segment_counts.index, autopct='%1.1f%%', colors=colors)
        ax1.set_title('客户分群分布', fontsize=14, fontweight='bold')
        
        ax2 = axes[0, 1]
        age_spending = df.groupby('age_group')['total_spent'].mean()
        age_spending.plot(kind='bar', ax=ax2, color='#2E86AB')
        ax2.set_title('各年龄段平均消费', fontsize=14, fontweight='bold')
        ax2.set_xlabel('年龄段')
        ax2.set_ylabel('平均消费金额')
        ax2.tick_params(axis='x', rotation=0)
        
        ax3 = axes[0, 2]
        membership_counts = df['membership_level'].value_counts()
        ax3.bar(membership_counts.index, membership_counts.values, color='#A23B72')
        ax3.set_title('会员等级分布', fontsize=14, fontweight='bold')
        ax3.set_xlabel('会员等级')
        ax3.set_ylabel('客户数')
        
        ax4 = axes[1, 0]
        city_spending = df.groupby('city_tier')['total_spent'].mean()
        city_spending.plot(kind='bar', ax=ax4, color='#F18F01')
        ax4.set_title('各城市等级平均消费', fontsize=14, fontweight='bold')
        ax4.set_xlabel('城市等级')
        ax4.set_ylabel('平均消费金额')
        ax4.tick_params(axis='x', rotation=0)
        
        ax5 = axes[1, 1]
        income_spending = df.groupby('income_level')['total_spent'].mean()
        income_order = ['低', '中', '高']
        income_spending = income_spending.reindex(income_order)
        income_spending.plot(kind='bar', ax=ax5, color='#3BCEAC')
        ax5.set_title('各收入水平平均消费', fontsize=14, fontweight='bold')
        ax5.set_xlabel('收入水平')
        ax5.set_ylabel('平均消费金额')
        ax5.tick_params(axis='x', rotation=0)
        
        ax6 = axes[1, 2]
        category_pref = df['preferred_category'].value_counts()
        ax6.barh(category_pref.index, category_pref.values, color='#C73E1D')
        ax6.set_title('客户偏好品类分布', fontsize=14, fontweight='bold')
        ax6.set_xlabel('客户数')
        ax6.set_ylabel('品类')
        
        plt.tight_layout()
        plt.savefig(f'{VIZ_DIR}/客户群体分析.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        self.log_step('客户群体分析完成，图表已保存', 'success')
        return df['customer_segment'].value_counts()
    
    def analyze_product_performance(self):
        """
        分析商品表现
        入参：无
        出参：商品分析结果
        """
        self.log_step('分析商品表现...')
        
        if 'transactions' not in self.data:
            self.log_step('缺少交易数据', 'error')
            return None
        
        df = self.data['transactions'].copy()
        
        product_stats = df.groupby('product_category').agg({
            'total_amount': ['sum', 'mean', 'count'],
            'quantity': 'sum',
            'is_returned': 'mean'
        }).round(2)
        
        product_stats.columns = ['总销售额', '平均订单金额', '订单数', '总销量', '退货率']
        product_stats = product_stats.sort_values('总销售额', ascending=False)
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        ax1 = axes[0, 0]
        product_stats['总销售额'].plot(kind='bar', ax=ax1, color='#2E86AB')
        ax1.set_title('各品类总销售额', fontsize=14, fontweight='bold')
        ax1.set_xlabel('品类')
        ax1.set_ylabel('销售额')
        ax1.tick_params(axis='x', rotation=45)
        
        ax2 = axes[0, 1]
        product_stats['订单数'].plot(kind='bar', ax=ax2, color='#A23B72')
        ax2.set_title('各品类订单数', fontsize=14, fontweight='bold')
        ax2.set_xlabel('品类')
        ax2.set_ylabel('订单数')
        ax2.tick_params(axis='x', rotation=45)
        
        ax3 = axes[1, 0]
        channel_sales = df.groupby('channel')['total_amount'].sum()
        ax3.pie(channel_sales.values, labels=channel_sales.index, autopct='%1.1f%%', colors=plt.cm.Set3.colors)
        ax3.set_title('各渠道销售占比', fontsize=14, fontweight='bold')
        
        ax4 = axes[1, 1]
        payment_sales = df.groupby('payment_method')['total_amount'].sum()
        ax4.bar(payment_sales.index, payment_sales.values, color='#F18F01')
        ax4.set_title('各支付方式销售额', fontsize=14, fontweight='bold')
        ax4.set_xlabel('支付方式')
        ax4.set_ylabel('销售额')
        ax4.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(f'{VIZ_DIR}/商品表现分析.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        self.log_step('商品表现分析完成，图表已保存', 'success')
        return product_stats
    
    def analyze_correlations(self):
        """
        分析特征相关性
        入参：无
        出参：相关性矩阵
        """
        self.log_step('分析特征相关性...')
        
        if 'daily_sales' not in self.data:
            self.log_step('缺少每日销售数据', 'error')
            return None
        
        df = self.data['daily_sales'].copy()
        
        numeric_cols = ['total_sales', 'order_count', 'customer_count', 'avg_order_value',
                       'temperature', 'is_weekend', 'is_holiday', 'promotion_flag']
        
        available_cols = [col for col in numeric_cols if col in df.columns]
        corr_matrix = df[available_cols].corr()
        
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                   fmt='.2f', square=True, linewidths=0.5, ax=ax)
        ax.set_title('销售数据特征相关性热力图', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f'{VIZ_DIR}/特征相关性分析.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        self.log_step('特征相关性分析完成，图表已保存', 'success')
        return corr_matrix
    
    def run_all_analysis(self):
        """
        执行所有描述性分析
        入参：无
        出参：分析结果字典
        """
        print('\033[96m' + '='*60 + '\033[0m')
        print('\033[96m  AI驱动的商业分析平台 - 描述性统计分析\033[0m')
        print('\033[96m' + '='*60 + '\033[0m')
        print()
        
        self.load_data()
        print()
        
        self.generate_overview_stats()
        print()
        
        self.analyze_sales_trends()
        print()
        
        self.analyze_customer_segments()
        print()
        
        self.analyze_product_performance()
        print()
        
        self.analyze_correlations()
        print()
        
        print('\033[92m' + '='*60 + '\033[0m')
        print('\033[92m  ✅ 描述性统计分析完成！\033[0m')
        print('\033[92m' + '='*60 + '\033[0m')
        
        print('\n生成的可视化图表:')
        for f in os.listdir(VIZ_DIR):
            if f.endswith('.png'):
                filepath = os.path.join(VIZ_DIR, f)
                size = os.path.getsize(filepath) / 1024
                print(f'  📊 {f} ({size:.1f} KB)')
        
        return self.analysis_results

def main():
    analyzer = DescriptiveAnalyzer()
    analyzer.run_all_analysis()

if __name__ == '__main__':
    main()
