"""
AI驱动的商业分析平台 - 增强的探索性数据分析（EDA）模块
功能：数据分布分析、相关性分析、交互特征分析、数据可视化
入参：处理后的数据
出参：可视化图表保存到data/viz/
异常处理：绘图异常捕获
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

PROCESSED_DIR = 'data/processed'
VIZ_DIR = 'data/viz'
os.makedirs(VIZ_DIR, exist_ok=True)

class EnhancedEDA:
    """
    增强的探索性数据分析类
    功能：数据分布分析、相关性分析、交互特征分析、数据可视化
    """
    
    def __init__(self):
        self.analysis_results = {}
        self.figures = []
    
    def log_step(self, message, status='info'):
        """
        记录分析步骤日志
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
        print(f'{colors.get(status, "")}[EDA分析] {message}{reset}')
    
    def load_processed_data(self):
        """
        加载处理后的数据
        入参：无
        出参：数据字典
        """
        data = {}
        
        try:
            data['transactions'] = pd.read_csv(os.path.join(PROCESSED_DIR, 'transactions_processed.csv'))
            self.log_step(f'加载交易数据: {len(data["transactions"])} 行', 'success')
            
            data['customers'] = pd.read_csv(os.path.join(PROCESSED_DIR, 'customers_processed.csv'))
            self.log_step(f'加载客户数据: {len(data["customers"])} 行', 'success')
            
            data['daily_sales'] = pd.read_csv(os.path.join(PROCESSED_DIR, 'daily_sales_processed.csv'))
            self.log_step(f'加载每日销售数据: {len(data["daily_sales"])} 行', 'success')
            
            data['products'] = pd.read_csv(os.path.join(PROCESSED_DIR, 'products_processed.csv'))
            self.log_step(f'加载商品数据: {len(data["products"])} 行', 'success')
            
            return data
            
        except Exception as e:
            self.log_step(f'加载数据失败: {str(e)}', 'error')
            return None
    
    def descriptive_statistics(self, df, name='数据集'):
        """
        描述性统计分析
        入参：df - DataFrame, name - 数据集名称
        出参：统计结果DataFrame
        """
        self.log_step(f'正在进行 {name} 的描述性统计分析...', 'info')
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) > 0:
            stats_df = df[numeric_cols].describe().T
            stats_df['skewness'] = df[numeric_cols].skew()
            stats_df['kurtosis'] = df[numeric_cols].kurtosis()
            
            self.analysis_results[f'{name}_descriptive_stats'] = stats_df
            
            self.log_step(f'{name} 描述性统计完成，共 {len(numeric_cols)} 个数值列', 'success')
            
            return stats_df
        else:
            self.log_step(f'{name} 没有数值列', 'warning')
            return None
    
    def plot_distribution_histogram(self, df, columns, title_prefix='数据分布'):
        """
        绘制数据分布直方图
        入参：df - DataFrame, columns - 列名列表, title_prefix - 标题前缀
        出参：保存图表到data/viz/
        """
        numeric_cols = [col for col in columns if col in df.columns and df[col].dtype in [np.int64, np.float64]]
        
        if len(numeric_cols) == 0:
            self.log_step('没有可绘制的数值列', 'warning')
            return
        
        n_cols = min(4, len(numeric_cols))
        n_rows = (len(numeric_cols) + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols*4, n_rows*3))
        if n_rows == 1 and n_cols == 1:
            axes = np.array([[axes]])
        elif n_rows == 1 or n_cols == 1:
            axes = axes.reshape(n_rows, n_cols)
        
        for idx, col in enumerate(numeric_cols):
            row = idx // n_cols
            col_idx = idx % n_cols
            ax = axes[row, col_idx]
            
            ax.hist(df[col].dropna(), bins=30, edgecolor='black', alpha=0.7)
            ax.set_title(f'{col} 分布', fontsize=10)
            ax.set_xlabel(col, fontsize=8)
            ax.set_ylabel('频数', fontsize=8)
            ax.grid(True, alpha=0.3)
            
            skewness = df[col].skew()
            ax.text(0.98, 0.95, f'偏度: {skewness:.2f}', 
                    transform=ax.transAxes, ha='right', va='top', 
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        for idx in range(len(numeric_cols), n_rows * n_cols):
            row = idx // n_cols
            col_idx = idx % n_cols
            fig.delaxes(axes[row, col_idx])
        
        plt.tight_layout()
        filename = f'{title_prefix}_直方图.png'
        filepath = os.path.join(VIZ_DIR, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.log_step(f'数据分布直方图已保存: {filename}', 'success')
        self.figures.append(filepath)
    
    def plot_boxplot(self, df, columns, title_prefix='箱线图'):
        """
        绘制箱线图检测异常值
        入参：df - DataFrame, columns - 列名列表, title_prefix - 标题前缀
        出参：保存图表到data/viz/
        """
        numeric_cols = [col for col in columns if col in df.columns and df[col].dtype in [np.int64, np.float64]]
        
        if len(numeric_cols) == 0:
            self.log_step('没有可绘制的数值列', 'warning')
            return
        
        n_cols = min(4, len(numeric_cols))
        n_rows = (len(numeric_cols) + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols*4, n_rows*3))
        if n_rows == 1 and n_cols == 1:
            axes = np.array([[axes]])
        elif n_rows == 1 or n_cols == 1:
            axes = axes.reshape(n_rows, n_cols)
        
        for idx, col in enumerate(numeric_cols):
            row = idx // n_cols
            col_idx = idx % n_cols
            ax = axes[row, col_idx]
            
            bp = ax.boxplot(df[col].dropna(), vert=True, patch_artist=True)
            bp['boxes'][0].set_facecolor('lightblue')
            bp['boxes'][0].set_alpha(0.7)
            
            ax.set_title(f'{col} 箱线图', fontsize=10)
            ax.set_ylabel(col, fontsize=8)
            ax.grid(True, alpha=0.3, axis='y')
            
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            outliers = df[(df[col] < Q1 - 1.5*IQR) | (df[col] > Q3 + 1.5*IQR)]
            
            ax.text(0.98, 0.95, f'异常值: {len(outliers)}', 
                    transform=ax.transAxes, ha='right', va='top', 
                    bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.5))
        
        for idx in range(len(numeric_cols), n_rows * n_cols):
            row = idx // n_cols
            col_idx = idx % n_cols
            fig.delaxes(axes[row, col_idx])
        
        plt.tight_layout()
        filename = f'{title_prefix}_箱线图.png'
        filepath = os.path.join(VIZ_DIR, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.log_step(f'箱线图已保存: {filename}', 'success')
        self.figures.append(filepath)
    
    def plot_correlation_heatmap(self, df, title='相关性热力图'):
        """
        绘制相关性热力图
        入参：df - DataFrame, title - 图表标题
        出参：保存图表到data/viz/
        """
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) < 2:
            self.log_step('数值列不足，无法绘制相关性热力图', 'warning')
            return
        
        corr_matrix = df[numeric_cols].corr()
        
        fig, ax = plt.subplots(figsize=(12, 10))
        
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        
        sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', 
                    cmap='coolwarm', center=0, square=True, 
                    linewidths=0.5, cbar_kws={"shrink": 0.8}, ax=ax)
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        filename = f'{title}.png'
        filepath = os.path.join(VIZ_DIR, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.log_step(f'相关性热力图已保存: {filename}', 'success')
        self.figures.append(filepath)
        
        return corr_matrix
    
    def plot_scatter_matrix(self, df, columns, title='散点图矩阵'):
        """
        绘制散点图矩阵
        入参：df - DataFrame, columns - 列名列表, title - 图表标题
        出参：保存图表到data/viz/
        """
        numeric_cols = [col for col in columns if col in df.columns and df[col].dtype in [np.int64, np.float64]]
        
        if len(numeric_cols) < 2:
            self.log_step('数值列不足，无法绘制散点图矩阵', 'warning')
            return
        
        n_cols = min(4, len(numeric_cols))
        
        fig, axes = plt.subplots(n_cols, n_cols, figsize=(n_cols*3, n_cols*3))
        
        for i in range(n_cols):
            for j in range(n_cols):
                ax = axes[i, j]
                
                if i == j:
                    ax.hist(df[numeric_cols[j]].dropna(), bins=20, edgecolor='black', alpha=0.7)
                    ax.set_title(numeric_cols[j], fontsize=8)
                else:
                    ax.scatter(df[numeric_cols[j]], df[numeric_cols[i]], alpha=0.5, s=10)
                    
                    corr = df[numeric_cols[j]].corr(df[numeric_cols[i]])
                    ax.text(0.98, 0.02, f'r={corr:.2f}', 
                            transform=ax.transAxes, ha='right', va='bottom',
                            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
                
                ax.tick_params(labelsize=6)
        
        plt.suptitle(title, fontsize=12, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        filename = f'{title}.png'
        filepath = os.path.join(VIZ_DIR, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.log_step(f'散点图矩阵已保存: {filename}', 'success')
        self.figures.append(filepath)
    
    def plot_time_series(self, df, date_col, value_cols, title='时间序列分析'):
        """
        绘制时间序列折线图
        入参：df - DataFrame, date_col - 日期列名, value_cols - 数值列列表, title - 图表标题
        出参：保存图表到data/viz/
        """
        df_plot = df.copy()
        df_plot[date_col] = pd.to_datetime(df_plot[date_col])
        df_plot = df_plot.sort_values(date_col)
        
        valid_cols = [col for col in value_cols if col in df_plot.columns]
        
        if len(valid_cols) == 0:
            self.log_step('没有有效的数值列', 'warning')
            return
        
        fig, ax = plt.subplots(figsize=(14, 6))
        
        for col in valid_cols:
            ax.plot(df_plot[date_col], df_plot[col], label=col, linewidth=1.5, alpha=0.8)
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('日期', fontsize=12)
        ax.set_ylabel('数值', fontsize=12)
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        filename = f'{title}.png'
        filepath = os.path.join(VIZ_DIR, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.log_step(f'时间序列图已保存: {filename}', 'success')
        self.figures.append(filepath)
    
    def analyze_sales_trends(self, df):
        """
        分析销售趋势
        入参：df - 每日销售数据DataFrame
        出参：趋势分析结果
        """
        self.log_step('正在进行销售趋势分析...', 'info')
        
        df['date'] = pd.to_datetime(df['date'])
        
        monthly_sales = df.groupby(df['date'].dt.to_period('M'))['total_sales'].sum()
        weekly_sales = df.groupby(df['date'].dt.to_period('W'))['total_sales'].sum()
        
        fig, axes = plt.subplots(2, 1, figsize=(14, 10))
        
        axes[0].plot(monthly_sales.index.astype(str), monthly_sales.values, 
                     marker='o', linewidth=2, markersize=6)
        axes[0].set_title('月度销售趋势', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('月份', fontsize=12)
        axes[0].set_ylabel('销售额', fontsize=12)
        axes[0].grid(True, alpha=0.3)
        axes[0].tick_params(axis='x', rotation=45)
        
        axes[1].plot(weekly_sales.index.astype(str), weekly_sales.values, 
                     marker='s', linewidth=1.5, markersize=4)
        axes[1].set_title('周度销售趋势', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('周', fontsize=12)
        axes[1].set_ylabel('销售额', fontsize=12)
        axes[1].grid(True, alpha=0.3)
        axes[1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        filename = '销售趋势分析.png'
        filepath = os.path.join(VIZ_DIR, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.log_step(f'销售趋势分析图已保存: {filename}', 'success')
        self.figures.append(filepath)
        
        trend_analysis = {
            'monthly_sales': monthly_sales.to_dict(),
            'weekly_sales': weekly_sales.to_dict(),
            'max_month': monthly_sales.idxmax(),
            'min_month': monthly_sales.idxmin(),
            'avg_monthly_sales': monthly_sales.mean()
        }
        
        self.analysis_results['sales_trends'] = trend_analysis
        
        return trend_analysis
    
    def run_full_eda(self):
        """
        运行完整的EDA分析流程
        入参：无
        出参：分析结果
        """
        print('\033[96m' + '='*60 + '\033[0m')
        print('\033[96m  AI驱动的商业分析平台 - 增强的EDA分析\033[0m')
        print('\033[96m' + '='*60 + '\033[0m')
        print()
        
        data = self.load_processed_data()
        
        if data is None:
            return None
        
        print()
        self.log_step('开始EDA分析流程...', 'info')
        print()
        
        total_steps = 6
        current_step = 0
        
        current_step += 1
        print(f'\n[{current_step}/{total_steps}] 描述性统计分析')
        self.descriptive_statistics(data['transactions'], '交易数据')
        self.descriptive_statistics(data['customers'], '客户数据')
        self.descriptive_statistics(data['daily_sales'], '每日销售数据')
        
        current_step += 1
        print(f'\n[{current_step}/{total_steps}] 数据分布分析（直方图）')
        self.plot_distribution_histogram(
            data['transactions'], 
            ['total_amount', 'quantity', 'unit_price', 'price_per_item'],
            '交易数据'
        )
        self.plot_distribution_histogram(
            data['customers'],
            ['total_spent', 'total_orders', 'age'],
            '客户数据'
        )
        
        current_step += 1
        print(f'\n[{current_step}/{total_steps}] 异常值检测（箱线图）')
        self.plot_boxplot(
            data['transactions'],
            ['total_amount', 'quantity', 'unit_price'],
            '交易数据'
        )
        self.plot_boxplot(
            data['customers'],
            ['total_spent', 'total_orders'],
            '客户数据'
        )
        
        current_step += 1
        print(f'\n[{current_step}/{total_steps}] 相关性分析（热力图）')
        self.plot_correlation_heatmap(data['daily_sales'], '每日销售相关性热力图')
        self.plot_correlation_heatmap(data['customers'], '客户数据相关性热力图')
        
        current_step += 1
        print(f'\n[{current_step}/{total_steps}] 交互特征分析（散点图矩阵）')
        self.plot_scatter_matrix(
            data['daily_sales'],
            ['total_sales', 'order_count', 'customer_count', 'avg_order_value'],
            '每日销售数据'
        )
        
        current_step += 1
        print(f'\n[{current_step}/{total_steps}] 时间序列分析')
        self.analyze_sales_trends(data['daily_sales'])
        
        print()
        print('\033[92m' + '='*60 + '\033[0m')
        print('\033[92m  ✅ EDA分析完成！\033[0m')
        print('\033[92m' + '='*60 + '\033[0m')
        
        print(f'\n已生成 {len(self.figures)} 个可视化图表:')
        for fig in self.figures:
            size = os.path.getsize(fig) / 1024
            print(f'  📊 {os.path.basename(fig)} ({size:.1f} KB)')
        
        return self.analysis_results

def main():
    """
    主函数
    """
    eda = EnhancedEDA()
    results = eda.run_full_eda()

if __name__ == '__main__':
    main()
