"""
AI驱动的商业分析平台 - 情感分析模块
功能：客户评论情感分析、关键词提取、情感趋势分析
入参：评论数据
出参：情感分析结果和可视化图表
异常处理：分析异常捕获
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import re
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

PROCESSED_DIR = 'data/processed'
VIZ_DIR = 'data/viz'
os.makedirs(VIZ_DIR, exist_ok=True)

POSITIVE_WORDS = [
    '满意', '好', '快', '棒', '喜欢', '推荐', '优秀', '完美', '超值', '划算',
    '质量好', '物流快', '服务好', '包装好', '性价比', '实惠', '漂亮', '好用',
    '舒适', '精致', '高端', '大气', '时尚', '美观', '耐用', '正品', '惊喜'
]

NEGATIVE_WORDS = [
    '差', '慢', '坏', '失望', '退货', '垃圾', '骗', '假', '次', '烂',
    '质量差', '物流慢', '服务差', '包装差', '贵', '不值', '难看', '难用',
    '不舒服', '粗糙', '低端', '小气', '过时', '丑', '易坏', '假货', '失望'
]

class SentimentAnalyzer:
    """
    情感分析类
    功能：评论情感分析、关键词提取、情感趋势分析
    """
    
    def __init__(self):
        self.reviews_df = None
        self.sentiment_results = None
    
    def log_step(self, message, status='info'):
        """
        记录处理步骤日志
        入参：message - 日志消息, status - 状态
        出参：打印彩色日志
        """
        colors = {'info': '\033[94m', 'success': '\033[92m', 'error': '\033[91m', 'warning': '\033[93m'}
        reset = '\033[0m'
        print(f'{colors.get(status, "")}[情感分析] {message}{reset}')
    
    def load_data(self):
        """
        加载评论数据
        入参：无
        出参：DataFrame
        """
        filepath = os.path.join(PROCESSED_DIR, 'reviews_processed.csv')
        if os.path.exists(filepath):
            self.reviews_df = pd.read_csv(filepath)
            self.log_step(f'加载评论数据: {len(self.reviews_df)} 条', 'success')
            return True
        else:
            self.log_step('未找到评论数据文件', 'error')
            return False
    
    def analyze_sentiment_simple(self, text):
        """
        简单情感分析（基于词典）
        入参：text - 评论文本
        出参：情感分数和标签
        """
        if pd.isna(text) or text == 'Unknown':
            return 0, '中性'
        
        text = str(text)
        
        positive_count = sum(1 for word in POSITIVE_WORDS if word in text)
        negative_count = sum(1 for word in NEGATIVE_WORDS if word in text)
        
        sentiment_score = positive_count - negative_count
        
        if sentiment_score > 0:
            sentiment_label = '正面'
        elif sentiment_score < 0:
            sentiment_label = '负面'
        else:
            sentiment_label = '中性'
        
        return sentiment_score, sentiment_label
    
    def analyze_all_reviews(self):
        """
        分析所有评论的情感
        入参：无
        出参：情感分析结果DataFrame
        """
        self.log_step('开始分析评论情感...')
        
        if self.reviews_df is None:
            self.log_step('评论数据未加载', 'error')
            return None
        
        df = self.reviews_df.copy()
        
        sentiments = df['comment'].apply(self.analyze_sentiment_simple)
        df['sentiment_score'] = sentiments.apply(lambda x: x[0])
        df['sentiment_label_v2'] = sentiments.apply(lambda x: x[1])
        
        df['sentiment_final'] = df.apply(
            lambda row: row['sentiment'] if 'sentiment' in row and pd.notna(row['sentiment']) else row['sentiment_label_v2'],
            axis=1
        )
        
        self.sentiment_results = df
        self.log_step(f'情感分析完成: {len(df)} 条评论', 'success')
        
        return df
    
    def extract_keywords(self, sentiment_type='all', top_n=20):
        """
        提取关键词
        入参：sentiment_type - 情感类型, top_n - 返回关键词数量
        出参：关键词列表
        """
        self.log_step(f'提取关键词 (情感类型: {sentiment_type})...')
        
        if self.sentiment_results is None:
            self.log_step('情感分析结果未生成', 'error')
            return None
        
        df = self.sentiment_results.copy()
        
        if sentiment_type != 'all':
            df = df[df['sentiment_final'] == sentiment_type]
        
        all_comments = ' '.join(df['comment'].dropna().astype(str).tolist())
        
        words = re.findall(r'[\u4e00-\u9fa5]{2,}', all_comments)
        
        word_freq = Counter(words)
        
        stopwords = ['非常', '有点', '比较', '还是', '这个', '那个', '但是', '因为', '所以', '而且', '或者', '以及']
        for word in stopwords:
            if word in word_freq:
                del word_freq[word]
        
        top_keywords = word_freq.most_common(top_n)
        
        self.log_step(f'关键词提取完成: {len(top_keywords)} 个', 'success')
        
        return top_keywords
    
    def analyze_sentiment_trend(self):
        """
        分析情感趋势
        入参：无
        出参：情感趋势DataFrame
        """
        self.log_step('分析情感趋势...')
        
        if self.sentiment_results is None:
            self.log_step('情感分析结果未生成', 'error')
            return None
        
        df = self.sentiment_results.copy()
        
        if 'review_date' in df.columns:
            df['review_date'] = pd.to_datetime(df['review_date'])
            df['year_month'] = df['review_date'].dt.to_period('M')
            
            trend = df.groupby(['year_month', 'sentiment_final']).size().unstack(fill_value=0)
            
            self.log_step('情感趋势分析完成', 'success')
            
            return trend
        
        return None
    
    def analyze_by_category(self):
        """
        按商品类别分析情感
        入参：无
        出参：类别情感分析结果
        """
        self.log_step('按商品类别分析情感...')
        
        if self.sentiment_results is None:
            self.log_step('情感分析结果未生成', 'error')
            return None
        
        df = self.sentiment_results.copy()
        
        if 'product_category' in df.columns:
            category_sentiment = df.groupby('product_category').agg({
                'rating': 'mean',
                'sentiment_score': 'mean',
                'review_id': 'count'
            }).round(2)
            
            category_sentiment.columns = ['平均评分', '平均情感分数', '评论数']
            category_sentiment = category_sentiment.sort_values('平均评分', ascending=False)
            
            self.log_step('类别情感分析完成', 'success')
            
            return category_sentiment
        
        return None
    
    def plot_sentiment_analysis(self):
        """
        绘制情感分析图表
        入参：无
        出参：保存可视化图表
        """
        self.log_step('生成情感分析图表...')
        
        if self.sentiment_results is None:
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        ax1 = axes[0, 0]
        sentiment_counts = self.sentiment_results['sentiment_final'].value_counts()
        colors = {'正面': '#2E86AB', '中性': '#F18F01', '负面': '#A23B72'}
        ax1.pie(sentiment_counts.values, labels=sentiment_counts.index, autopct='%1.1f%%',
               colors=[colors.get(x, '#888888') for x in sentiment_counts.index])
        ax1.set_title('情感分布', fontsize=14, fontweight='bold')
        
        ax2 = axes[0, 1]
        rating_counts = self.sentiment_results['rating'].value_counts().sort_index()
        ax2.bar(rating_counts.index, rating_counts.values, color='#2E86AB')
        ax2.set_title('评分分布', fontsize=14, fontweight='bold')
        ax2.set_xlabel('评分')
        ax2.set_ylabel('评论数')
        
        ax3 = axes[0, 2]
        positive_keywords = self.extract_keywords('正面', 10)
        if positive_keywords:
            words, counts = zip(*positive_keywords)
            ax3.barh(words, counts, color='#2E86AB')
            ax3.set_title('正面评论关键词 TOP10', fontsize=14, fontweight='bold')
            ax3.set_xlabel('出现次数')
        
        ax4 = axes[1, 0]
        negative_keywords = self.extract_keywords('负面', 10)
        if negative_keywords:
            words, counts = zip(*negative_keywords)
            ax4.barh(words, counts, color='#A23B72')
            ax4.set_title('负面评论关键词 TOP10', fontsize=14, fontweight='bold')
            ax4.set_xlabel('出现次数')
        
        ax5 = axes[1, 1]
        trend = self.analyze_sentiment_trend()
        if trend is not None:
            trend.plot(ax=ax5, marker='o', linewidth=2)
            ax5.set_title('情感趋势变化', fontsize=14, fontweight='bold')
            ax5.set_xlabel('月份')
            ax5.set_ylabel('评论数')
            ax5.legend(title='情感类型')
            ax5.tick_params(axis='x', rotation=45)
        
        ax6 = axes[1, 2]
        category_sentiment = self.analyze_by_category()
        if category_sentiment is not None:
            category_sentiment['平均评分'].plot(kind='bar', ax=ax6, color='#F18F01')
            ax6.set_title('各品类平均评分', fontsize=14, fontweight='bold')
            ax6.set_xlabel('品类')
            ax6.set_ylabel('平均评分')
            ax6.tick_params(axis='x', rotation=45)
            ax6.axhline(y=3.5, color='red', linestyle='--', alpha=0.5, label='及格线')
        
        plt.tight_layout()
        plt.savefig(f'{VIZ_DIR}/情感分析.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        self.log_step('情感分析图表已保存', 'success')
    
    def generate_wordcloud_data(self):
        """
        生成词云数据
        入参：无
        出参：词频字典
        """
        self.log_step('生成词云数据...')
        
        if self.sentiment_results is None:
            return None
        
        all_comments = ' '.join(self.sentiment_results['comment'].dropna().astype(str).tolist())
        words = re.findall(r'[\u4e00-\u9fa5]{2,}', all_comments)
        word_freq = Counter(words)
        
        return dict(word_freq.most_common(100))
    
    def run_full_analysis(self):
        """
        执行完整的情感分析
        入参：无
        出参：分析结果
        """
        print('\033[96m' + '='*60 + '\033[0m')
        print('\033[96m  AI驱动的商业分析平台 - 情感分析\033[0m')
        print('\033[96m' + '='*60 + '\033[0m')
        print()
        
        self.load_data()
        print()
        
        self.analyze_all_reviews()
        print()
        
        self.plot_sentiment_analysis()
        print()
        
        print('\033[92m' + '='*60 + '\033[0m')
        print('\033[92m  ✅ 情感分析完成！\033[0m')
        print('\033[92m' + '='*60 + '\033[0m')
        
        if self.sentiment_results is not None:
            print('\n情感分布统计:')
            print(self.sentiment_results['sentiment_final'].value_counts())
            
            print('\n评分统计:')
            print(f'  平均评分: {self.sentiment_results["rating"].mean():.2f}')
            print(f'  好评率 (>=4分): {(self.sentiment_results["rating"] >= 4).mean() * 100:.1f}%')
            print(f'  差评率 (<=2分): {(self.sentiment_results["rating"] <= 2).mean() * 100:.1f}%')
        
        return self.sentiment_results

def main():
    analyzer = SentimentAnalyzer()
    analyzer.run_full_analysis()

if __name__ == '__main__':
    main()
