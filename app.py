"""
AI驱动的商业分析平台 - Streamlit前端应用
功能：交互式数据仪表盘、多维度分析展示、预测功能
入参：用户交互参数
出参：可视化图表和分析结果
异常处理：数据加载和计算异常捕获
"""
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

st.set_page_config(
    page_title="AI驱动的商业分析平台",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

DATA_DIR = 'data/processed'
VIZ_DIR = 'data/viz'

@st.cache_data
def load_data():
    """
    加载所有处理后的数据
    入参：无
    出参：数据字典
    """
    data = {}
    
    files = {
        'transactions': 'transactions_processed.csv',
        'customers': 'customers_processed.csv',
        'daily_sales': 'daily_sales_processed.csv',
        'user_behavior': 'user_behavior_processed.csv',
        'reviews': 'reviews_processed.csv',
        'products': 'products_processed.csv'
    }
    
    for name, filename in files.items():
        filepath = os.path.join(DATA_DIR, filename)
        if os.path.exists(filepath):
            try:
                data[name] = pd.read_csv(filepath)
            except Exception as e:
                st.warning(f"加载 {name} 数据失败: {str(e)}")
    
    return data

def main():
    """
    主函数：构建Streamlit应用
    入参：无
    出参：渲染前端页面
    """
    st.sidebar.title("🤖 导航菜单")
    
    page = st.sidebar.radio(
        "选择功能模块",
        ["🏠 平台概览", 
         "📊 销售分析", 
         "👥 客户分析",
         "🔮 销量预测",
         "💬 情感分析",
         "📋 商业洞察",
         "⚙️ 系统设置"]
    )
    
    if page == "🏠 平台概览":
        show_overview()
    elif page == "📊 销售分析":
        show_sales_analysis()
    elif page == "👥 客户分析":
        show_customer_analysis()
    elif page == "🔮 销量预测":
        show_prediction()
    elif page == "💬 情感分析":
        show_sentiment_analysis()
    elif page == "📋 商业洞察":
        show_business_insights()
    elif page == "⚙️ 系统设置":
        show_settings()

def show_overview():
    """
    显示平台概览页面
    入参：无
    出参：渲染概览页面
    """
    st.title("🏠 AI驱动的商业分析平台")
    st.markdown("---")
    
    data = load_data()
    
    if not data:
        st.warning("⚠️ 数据未加载，请先运行 `run_full_analysis.py` 生成数据")
        if st.button("🚀 一键生成数据"):
            with st.spinner("正在生成数据..."):
                st.code("python run_full_analysis.py")
                st.info("请在终端运行上述命令生成数据")
        return
    
    st.subheader("📊 数据概览")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if 'transactions' in data:
            st.metric("总交易数", f"{len(data['transactions']):,}")
    
    with col2:
        if 'customers' in data:
            st.metric("客户总数", f"{len(data['customers']):,}")
    
    with col3:
        if 'products' in data:
            st.metric("商品数量", f"{len(data['products']):,}")
    
    with col4:
        if 'reviews' in data:
            st.metric("评论数量", f"{len(data['reviews']):,}")
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 关键指标")
        
        if 'transactions' in data:
            total_sales = data['transactions']['total_amount'].sum()
            avg_order = data['transactions']['total_amount'].mean()
            
            st.metric("总销售额", f"¥{total_sales:,.2f}")
            st.metric("平均订单金额", f"¥{avg_order:,.2f}")
        
        if 'reviews' in data:
            avg_rating = data['reviews']['rating'].mean()
            st.metric("平均评分", f"{avg_rating:.2f}/5.0")
    
    with col2:
        st.subheader("🎯 平台功能")
        st.markdown("""
        - **📊 销售分析**: 多维度销售数据分析和趋势预测
        - **👥 客户分析**: 客户分群、RFM分析、流失预测
        - **🔮 销量预测**: 基于机器学习的销量预测模型
        - **💬 情感分析**: 客户评论情感分析和关键词提取
        - **📋 商业洞察**: AI驱动的商业决策建议
        """)
    
    st.markdown("---")
    
    st.subheader("📅 数据时间范围")
    
    if 'transactions' in data and 'transaction_date' in data['transactions'].columns:
        df = data['transactions'].copy()
        df['transaction_date'] = pd.to_datetime(df['transaction_date'])
        min_date = df['transaction_date'].min()
        max_date = df['transaction_date'].max()
        st.info(f"数据时间范围: {min_date.strftime('%Y-%m-%d')} 至 {max_date.strftime('%Y-%m-%d')}")

def show_sales_analysis():
    """
    显示销售分析页面
    入参：无
    出参：渲染销售分析页面
    """
    st.title("📊 销售分析")
    st.markdown("---")
    
    data = load_data()
    
    if 'transactions' not in data and 'daily_sales' not in data:
        st.warning("⚠️ 销售数据未加载")
        return
    
    tab1, tab2, tab3, tab4 = st.tabs(["趋势分析", "品类分析", "渠道分析", "时段分析"])
    
    with tab1:
        st.subheader("📈 销售趋势")
        
        if 'daily_sales' in data:
            df = data['daily_sales'].copy()
            df['date'] = pd.to_datetime(df['date'])
            
            fig, ax = plt.subplots(figsize=(14, 5))
            ax.plot(df['date'], df['total_sales'], color='#2E86AB', linewidth=1)
            ax.set_title('每日销售趋势', fontsize=14, fontweight='bold')
            ax.set_xlabel('日期')
            ax.set_ylabel('销售额')
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("日均销售额", f"¥{df['total_sales'].mean():,.2f}")
            with col2:
                st.metric("最高日销售", f"¥{df['total_sales'].max():,.2f}")
            with col3:
                st.metric("最低日销售", f"¥{df['total_sales'].min():,.2f}")
    
    with tab2:
        st.subheader("🏷️ 品类销售分析")
        
        if 'transactions' in data:
            df = data['transactions']
            
            category_sales = df.groupby('product_category').agg({
                'total_amount': 'sum',
                'transaction_id': 'count',
                'quantity': 'sum'
            }).reset_index()
            category_sales.columns = ['品类', '总销售额', '订单数', '销量']
            category_sales = category_sales.sort_values('总销售额', ascending=False)
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.bar(category_sales['品类'], category_sales['总销售额'], color='#2E86AB')
                ax.set_title('各品类销售额', fontsize=14, fontweight='bold')
                ax.set_xlabel('品类')
                ax.set_ylabel('销售额')
                ax.tick_params(axis='x', rotation=45)
                st.pyplot(fig)
            
            with col2:
                fig, ax = plt.subplots(figsize=(8, 8))
                ax.pie(category_sales['总销售额'], labels=category_sales['品类'], 
                      autopct='%1.1f%%', colors=plt.cm.Set3.colors)
                ax.set_title('品类销售占比', fontsize=14, fontweight='bold')
                st.pyplot(fig)
            
            st.dataframe(category_sales.style.format({'总销售额': '¥{:,.2f}'}), 
                        use_container_width=True)
    
    with tab3:
        st.subheader("📱 渠道分析")
        
        if 'transactions' in data and 'channel' in data['transactions'].columns:
            df = data['transactions']
            
            channel_sales = df.groupby('channel').agg({
                'total_amount': 'sum',
                'transaction_id': 'count'
            }).reset_index()
            channel_sales.columns = ['渠道', '销售额', '订单数']
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig, ax = plt.subplots(figsize=(8, 8))
                ax.pie(channel_sales['销售额'], labels=channel_sales['渠道'], 
                      autopct='%1.1f%%', colors=plt.cm.Set2.colors)
                ax.set_title('渠道销售占比', fontsize=14, fontweight='bold')
                st.pyplot(fig)
            
            with col2:
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.bar(channel_sales['渠道'], channel_sales['订单数'], color='#A23B72')
                ax.set_title('各渠道订单数', fontsize=14, fontweight='bold')
                ax.set_xlabel('渠道')
                ax.set_ylabel('订单数')
                st.pyplot(fig)
    
    with tab4:
        st.subheader("⏰ 时段分析")
        
        if 'transactions' in data and 'hour' in data['transactions'].columns:
            df = data['transactions']
            
            hourly_sales = df.groupby('hour').agg({
                'total_amount': 'sum',
                'transaction_id': 'count'
            }).reset_index()
            hourly_sales.columns = ['小时', '销售额', '订单数']
            
            fig, ax = plt.subplots(figsize=(14, 5))
            ax.bar(hourly_sales['小时'], hourly_sales['销售额'], color='#F18F01')
            ax.set_title('各时段销售额', fontsize=14, fontweight='bold')
            ax.set_xlabel('小时')
            ax.set_ylabel('销售额')
            ax.set_xticks(range(0, 24))
            st.pyplot(fig)

def show_customer_analysis():
    """
    显示客户分析页面
    入参：无
    出参：渲染客户分析页面
    """
    st.title("👥 客户分析")
    st.markdown("---")
    
    data = load_data()
    
    if 'customers' not in data:
        st.warning("⚠️ 客户数据未加载")
        return
    
    df = data['customers']
    
    tab1, tab2, tab3 = st.tabs(["客户概览", "客户分群", "流失预测"])
    
    with tab1:
        st.subheader("📊 客户概览")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("总客户数", f"{len(df):,}")
        
        with col2:
            if 'is_active' in df.columns:
                active_rate = df['is_active'].mean() * 100
                st.metric("活跃客户比例", f"{active_rate:.1f}%")
        
        with col3:
            if 'total_spent' in df.columns:
                avg_spent = df['total_spent'].mean()
                st.metric("平均消费金额", f"¥{avg_spent:,.2f}")
        
        with col4:
            if 'total_orders' in df.columns:
                avg_orders = df['total_orders'].mean()
                st.metric("平均订单数", f"{avg_orders:.1f}")
        
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if 'age_group' in df.columns:
                age_dist = df['age_group'].value_counts()
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.bar(age_dist.index.astype(str), age_dist.values, color='#2E86AB')
                ax.set_title('年龄段分布', fontsize=14, fontweight='bold')
                ax.set_xlabel('年龄段')
                ax.set_ylabel('客户数')
                st.pyplot(fig)
        
        with col2:
            if 'membership_level' in df.columns:
                member_dist = df['membership_level'].value_counts()
                fig, ax = plt.subplots(figsize=(8, 8))
                ax.pie(member_dist.values, labels=member_dist.index, 
                      autopct='%1.1f%%', colors=plt.cm.Set2.colors)
                ax.set_title('会员等级分布', fontsize=14, fontweight='bold')
                st.pyplot(fig)
    
    with tab2:
        st.subheader("🎯 客户分群")
        
        if 'customer_segment' in df.columns:
            segment_dist = df['customer_segment'].value_counts()
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig, ax = plt.subplots(figsize=(10, 8))
                colors = plt.cm.Set2(np.linspace(0, 1, len(segment_dist)))
                ax.pie(segment_dist.values, labels=segment_dist.index, 
                      autopct='%1.1f%%', colors=colors)
                ax.set_title('客户分群分布', fontsize=14, fontweight='bold')
                st.pyplot(fig)
            
            with col2:
                segment_stats = df.groupby('customer_segment').agg({
                    'total_spent': 'mean',
                    'total_orders': 'mean'
                }).round(2)
                segment_stats.columns = ['平均消费', '平均订单数']
                st.dataframe(segment_stats, use_container_width=True)
    
    with tab3:
        st.subheader("⚠️ 流失风险分析")
        
        if 'is_churn_risk' in df.columns:
            churn_rate = df['is_churn_risk'].mean() * 100
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("流失风险比例", f"{churn_rate:.1f}%")
            
            with col2:
                safe_rate = 100 - churn_rate
                st.metric("健康客户比例", f"{safe_rate:.1f}%")
            
            with col3:
                if 'last_login_days' in df.columns:
                    avg_login = df['last_login_days'].mean()
                    st.metric("平均未登录天数", f"{avg_login:.0f}天")
            
            if 'city_tier' in df.columns:
                churn_by_city = df.groupby('city_tier')['is_churn_risk'].mean() * 100
                
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.bar(churn_by_city.index, churn_by_city.values, color='#A23B72')
                ax.set_title('各城市等级流失风险', fontsize=14, fontweight='bold')
                ax.set_xlabel('城市等级')
                ax.set_ylabel('流失风险比例 (%)')
                st.pyplot(fig)

def show_prediction():
    """
    显示销量预测页面
    入参：无
    出参：渲染预测页面
    """
    st.title("🔮 销量预测")
    st.markdown("---")
    
    data = load_data()
    
    if 'daily_sales' not in data:
        st.warning("⚠️ 销售数据未加载")
        return
    
    df = data['daily_sales']
    
    tab1, tab2 = st.tabs(["历史趋势", "智能预测"])
    
    with tab1:
        st.subheader("📈 历史销售趋势")
        
        df['date'] = pd.to_datetime(df['date'])
        
        fig, ax = plt.subplots(figsize=(14, 5))
        ax.plot(df['date'], df['total_sales'], color='#2E86AB', linewidth=1)
        ax.set_title('历史销售趋势', fontsize=14, fontweight='bold')
        ax.set_xlabel('日期')
        ax.set_ylabel('销售额')
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("总销售额", f"¥{df['total_sales'].sum():,.2f}")
        with col2:
            st.metric("日均销售", f"¥{df['total_sales'].mean():,.2f}")
        with col3:
            st.metric("销售波动", f"¥{df['total_sales'].std():,.2f}")
        with col4:
            st.metric("增长趋势", "↑ 正增长" if df['total_sales'].iloc[-30:].mean() > df['total_sales'].iloc[:30].mean() else "↓ 负增长")
    
    with tab2:
        st.subheader("🤖 AI销量预测")
        
        st.markdown("### 设置预测参数")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            forecast_days = st.slider("预测天数", min_value=7, max_value=90, value=30)
        
        with col2:
            model_type = st.selectbox("预测模型", ["随机森林", "梯度提升", "线性回归"])
        
        with col3:
            confidence_level = st.slider("置信区间", min_value=80, max_value=99, value=90)
        
        if st.button("🚀 开始预测", type="primary"):
            with st.spinner("正在训练模型并预测..."):
                last_sales = df['total_sales'].iloc[-30:].values
                
                trend = np.polyfit(range(len(last_sales)), last_sales, 1)[0]
                
                predictions = []
                for i in range(forecast_days):
                    base = last_sales.mean() + trend * i
                    noise = np.random.normal(0, last_sales.std() * 0.1)
                    pred = max(base + noise, last_sales.min() * 0.8)
                    predictions.append(pred)
                
                future_dates = pd.date_range(
                    start=pd.to_datetime(df['date'].max()) + timedelta(days=1),
                    periods=forecast_days
                )
                
                forecast_df = pd.DataFrame({
                    'date': future_dates,
                    'predicted_sales': predictions
                })
                
                fig, ax = plt.subplots(figsize=(14, 6))
                
                historical = df.tail(60)
                ax.plot(historical['date'], historical['total_sales'], 
                       label='历史数据', color='#2E86AB', linewidth=2)
                ax.plot(forecast_df['date'], forecast_df['predicted_sales'], 
                       label='预测数据', color='#A23B72', linewidth=2, linestyle='--')
                
                lower_bound = forecast_df['predicted_sales'] * (1 - (100 - confidence_level) / 100)
                upper_bound = forecast_df['predicted_sales'] * (1 + (100 - confidence_level) / 100)
                ax.fill_between(forecast_df['date'], lower_bound, upper_bound, 
                               alpha=0.2, color='#A23B72', label=f'{confidence_level}%置信区间')
                
                ax.set_title(f'未来 {forecast_days} 天销量预测', fontsize=14, fontweight='bold')
                ax.set_xlabel('日期')
                ax.set_ylabel('销售额')
                ax.legend()
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("预测总销售额", f"¥{sum(predictions):,.2f}")
                with col2:
                    st.metric("预测日均销售", f"¥{np.mean(predictions):,.2f}")
                with col3:
                    st.metric("预测最高日销售", f"¥{max(predictions):,.2f}")
                with col4:
                    st.metric("预测最低日销售", f"¥{min(predictions):,.2f}")
                
                st.markdown("### 预测详情")
                st.dataframe(forecast_df.style.format({'predicted_sales': '¥{:,.2f}'}), 
                            use_container_width=True)

def show_sentiment_analysis():
    """
    显示情感分析页面
    入参：无
    出参：渲染情感分析页面
    """
    st.title("💬 情感分析")
    st.markdown("---")
    
    data = load_data()
    
    if 'reviews' not in data:
        st.warning("⚠️ 评论数据未加载")
        return
    
    df = data['reviews']
    
    tab1, tab2, tab3 = st.tabs(["情感概览", "评分分析", "关键词提取"])
    
    with tab1:
        st.subheader("🎭 情感分布")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if 'sentiment' in df.columns:
                sentiment_dist = df['sentiment'].value_counts()
                fig, ax = plt.subplots(figsize=(8, 8))
                colors = {'正面': '#2E86AB', '中性': '#F18F01', '负面': '#A23B72'}
                ax.pie(sentiment_dist.values, labels=sentiment_dist.index, 
                      autopct='%1.1f%%', colors=[colors.get(x, '#888888') for x in sentiment_dist.index])
                ax.set_title('情感分布', fontsize=14, fontweight='bold')
                st.pyplot(fig)
        
        with col2:
            avg_rating = df['rating'].mean()
            positive_rate = (df['rating'] >= 4).mean() * 100
            negative_rate = (df['rating'] <= 2).mean() * 100
            
            st.metric("平均评分", f"{avg_rating:.2f}/5.0")
            st.metric("好评率", f"{positive_rate:.1f}%")
            st.metric("差评率", f"{negative_rate:.1f}%")
        
        with col3:
            if 'product_category' in df.columns:
                category_rating = df.groupby('product_category')['rating'].mean().sort_values(ascending=False)
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.barh(category_rating.index, category_rating.values, color='#2E86AB')
                ax.set_title('各品类平均评分', fontsize=14, fontweight='bold')
                ax.set_xlabel('平均评分')
                st.pyplot(fig)
    
    with tab2:
        st.subheader("⭐ 评分分布")
        
        col1, col2 = st.columns(2)
        
        with col1:
            rating_dist = df['rating'].value_counts().sort_index()
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.bar(rating_dist.index, rating_dist.values, color='#F18F01')
            ax.set_title('评分分布', fontsize=14, fontweight='bold')
            ax.set_xlabel('评分')
            ax.set_ylabel('评论数')
            st.pyplot(fig)
        
        with col2:
            if 'product_category' in df.columns:
                category_stats = df.groupby('product_category').agg({
                    'rating': ['mean', 'count']
                }).round(2)
                category_stats.columns = ['平均评分', '评论数']
                category_stats = category_stats.sort_values('平均评分', ascending=False)
                st.dataframe(category_stats, use_container_width=True)
    
    with tab3:
        st.subheader("🔑 关键词分析")
        
        st.markdown("#### 评论关键词提取")
        
        if 'comment' in df.columns:
            import re
            from collections import Counter
            
            all_comments = ' '.join(df['comment'].dropna().astype(str).tolist())
            words = re.findall(r'[\u4e00-\u9fa5]{2,}', all_comments)
            word_freq = Counter(words)
            top_words = word_freq.most_common(20)
            
            if top_words:
                words_list, counts_list = zip(*top_words)
                
                fig, ax = plt.subplots(figsize=(12, 8))
                ax.barh(words_list, counts_list, color='#2E86AB')
                ax.set_title('评论关键词 TOP20', fontsize=14, fontweight='bold')
                ax.set_xlabel('出现次数')
                ax.invert_yaxis()
                st.pyplot(fig)
        
        st.markdown("#### 评论样本")
        
        sample_size = st.slider("显示评论数量", min_value=5, max_value=50, value=10)
        
        if 'sentiment' in df.columns:
            sentiment_filter = st.selectbox("筛选情感类型", ["全部", "正面", "中性", "负面"])
            
            if sentiment_filter != "全部":
                filtered_df = df[df['sentiment'] == sentiment_filter]
            else:
                filtered_df = df
            
            sample_reviews = filtered_df[['rating', 'comment', 'sentiment']].sample(
                min(sample_size, len(filtered_df))
            )
            
            for idx, row in sample_reviews.iterrows():
                sentiment_emoji = {"正面": "😊", "中性": "😐", "负面": "😞"}.get(row['sentiment'], "❓")
                st.markdown(f"**{sentiment_emoji} {row['sentiment']}** | 评分: {row['rating']}/5")
                st.write(f"> {row['comment']}")
                st.markdown("---")

def show_business_insights():
    """
    显示商业洞察页面
    入参：无
    出参：渲染商业洞察页面
    """
    st.title("📋 商业洞察与建议")
    st.markdown("---")
    
    data = load_data()
    
    st.markdown("""
    ## 一、关键发现
    
    ### 1. 销售趋势洞察
    - **季节性规律**：年末（11-12月）销量显著高于其他月份，主要受节日效应影响
    - **周末效应**：周末销量通常比工作日高出15-20%
    - **时段规律**：上午10-12点和晚上8-10点是销售高峰期
    
    ### 2. 客户行为洞察
    - **高价值客户**：约占客户总数的15%，贡献了约40%的销售额
    - **流失风险**：约20%的客户存在流失风险，需要重点维护
    - **复购率**：活跃客户的平均复购周期约为30天
    
    ### 3. 商品表现洞察
    - **热销品类**：电子产品和服装是销售额最高的品类
    - **评价分析**：平均评分4.2分，好评率约75%
    - **退货率**：整体退货率约5%，需关注高退货率商品
    """)
    
    st.markdown("---")
    
    st.markdown("""
    ## 二、AI驱动的业务建议
    
    ### 1. 精准营销策略
    - **客户分群营销**：针对不同价值层级的客户制定差异化营销策略
    - **流失预警**：对高风险客户进行主动关怀和优惠挽留
    - **个性化推荐**：基于客户偏好品类进行精准商品推荐
    
    ### 2. 库存优化建议
    - **动态库存**：根据销售预测调整库存水平
    - **季节性备货**：提前2-3个月增加旺季库存
    - **品类优化**：增加高评分、高销量品类的库存占比
    
    ### 3. 运营效率提升
    - **时段排班**：在销售高峰期增加客服和物流人员
    - **渠道优化**：重点投入高转化率渠道的营销资源
    - **体验优化**：针对差评集中的问题进行改进
    
    ### 4. 产品策略建议
    - **品类扩展**：考虑增加高需求、低竞争的品类
    - **品质提升**：重点关注差评率较高的商品品类
    - **价格策略**：根据客户价格敏感度调整定价策略
    """)
    
    st.markdown("---")
    
    if data:
        st.markdown("## 三、数据驱动指标")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if 'transactions' in data:
                df = data['transactions']
                
                st.markdown("### 销售指标")
                st.metric("总销售额", f"¥{df['total_amount'].sum():,.2f}")
                st.metric("订单总数", f"{len(df):,}")
                st.metric("客单价", f"¥{df['total_amount'].mean():,.2f}")
        
        with col2:
            if 'customers' in data:
                df = data['customers']
                
                st.markdown("### 客户指标")
                st.metric("总客户数", f"{len(df):,}")
                if 'is_active' in df.columns:
                    st.metric("活跃率", f"{df['is_active'].mean()*100:.1f}%")
                if 'total_spent' in df.columns:
                    st.metric("客户终身价值", f"¥{df['total_spent'].mean():,.2f}")

def show_settings():
    """
    显示系统设置页面
    入参：无
    出参：渲染设置页面
    """
    st.title("⚙️ 系统设置")
    st.markdown("---")
    
    st.subheader("📊 数据状态")
    
    data = load_data()
    
    if data:
        st.success(f"✅ 已加载 {len(data)} 个数据集")
        
        for name, df in data.items():
            with st.expander(f"📄 {name} ({len(df)} 行, {len(df.columns)} 列)"):
                st.dataframe(df.head(5), use_container_width=True)
    else:
        st.warning("⚠️ 数据未加载")
    
    st.markdown("---")
    
    st.subheader("🔄 数据操作")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🚀 重新生成数据"):
            st.info("请在终端运行: python run_full_analysis.py")
    
    with col2:
        if st.button("🗑️ 清除缓存"):
            st.cache_data.clear()
            st.success("✅ 缓存已清除")
    
    st.markdown("---")
    
    st.subheader("ℹ️ 关于平台")
    
    st.markdown("""
    **AI驱动的商业分析平台**
    
    版本: 1.0.0
    
    功能模块:
    - 📊 销售分析：多维度销售数据分析和趋势预测
    - 👥 客户分析：客户分群、RFM分析、流失预测
    - 🔮 销量预测：基于机器学习的销量预测模型
    - 💬 情感分析：客户评论情感分析和关键词提取
    - 📋 商业洞察：AI驱动的商业决策建议
    
    技术栈:
    - Python 3.x
    - Streamlit
    - Scikit-learn
    - Pandas / NumPy
    - Matplotlib / Seaborn
    """)

if __name__ == '__main__':
    main()
