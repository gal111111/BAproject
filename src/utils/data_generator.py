"""
AI驱动的商业分析平台 - 数据生成器
功能：生成零售电商多维度数据集（用户行为、购买历史、客户评论等）
入参：无
出参：生成多个CSV文件到data/raw/目录
异常处理：文件写入异常捕获
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import os

np.random.seed(42)
random.seed(42)

DATA_DIR = 'data/raw'
os.makedirs(DATA_DIR, exist_ok=True)

PRODUCTS = [
    {'id': i, 'category': random.choice(['电子产品', '服装', '食品', '家居', '美妆', '运动', '图书', '母婴']),
     'name': f'商品_{i}', 'base_price': np.random.uniform(10, 2000)}
    for i in range(1, 501)
]

CUSTOMER_SEGMENTS = ['高价值客户', '普通客户', '潜在流失客户', '新客户', '沉睡客户']

COMMENTS_POSITIVE = [
    '非常满意，质量很好', '物流很快，包装完好', '性价比很高，推荐购买',
    '客服态度很好', '产品功能强大', '外观设计漂亮', '使用体验很好',
    '物超所值', '回购了很多次', '朋友推荐的，确实不错'
]

COMMENTS_NEGATIVE = [
    '质量一般，有点失望', '物流太慢了', '价格偏贵',
    '客服回复太慢', '产品有瑕疵', '与描述不符',
    '包装破损', '退货流程麻烦', '不会再买了', '体验很差'
]

COMMENTS_NEUTRAL = [
    '还行吧，凑合用', '一般般', '有待改进',
    '勉强接受', '中规中矩', '没什么特别的'
]

def generate_customers(num_customers=5000):
    """
    生成客户基础信息数据
    入参：num_customers - 客户数量
    出参：DataFrame保存到data/raw/customers.csv
    """
    print('\033[94m[数据生成器] 正在生成客户数据...\033[0m')
    
    customers = []
    for i in range(1, num_customers + 1):
        register_date = datetime(2020, 1, 1) + timedelta(days=random.randint(0, 1460))
        age = random.randint(18, 65)
        gender = random.choice(['男', '女'])
        city_tier = random.choice(['一线', '二线', '三线', '四线及以下'])
        
        income_level = random.choice(['低', '中', '高'])
        if city_tier == '一线':
            income_level = random.choices(['低', '中', '高'], weights=[0.2, 0.4, 0.4])[0]
        elif city_tier == '四线及以下':
            income_level = random.choices(['低', '中', '高'], weights=[0.5, 0.35, 0.15])[0]
        
        membership_level = random.choices(['普通', '银卡', '金卡', '钻石'], weights=[0.5, 0.25, 0.18, 0.07])[0]
        
        customers.append({
            'customer_id': f'C{str(i).zfill(6)}',
            'register_date': register_date.strftime('%Y-%m-%d'),
            'age': age,
            'gender': gender,
            'city_tier': city_tier,
            'province': random.choice(['广东', '北京', '上海', '浙江', '江苏', '四川', '湖北', '山东', '河南', '福建']),
            'income_level': income_level,
            'membership_level': membership_level,
            'preferred_category': random.choice(['电子产品', '服装', '食品', '家居', '美妆', '运动', '图书', '母婴']),
            'total_orders': random.randint(0, 200),
            'total_spent': np.random.uniform(0, 50000),
            'last_login_days': random.randint(0, 365),
            'customer_segment': random.choice(CUSTOMER_SEGMENTS)
        })
    
    df = pd.DataFrame(customers)
    df.to_csv(f'{DATA_DIR}/customers.csv', index=False, encoding='utf-8-sig')
    print(f'\033[92m[数据生成器] 客户数据已保存: {DATA_DIR}/customers.csv ({len(df)} 条记录)\033[0m')
    return df

def generate_transactions(customers_df, num_transactions=50000):
    """
    生成交易订单数据
    入参：customers_df - 客户数据DataFrame, num_transactions - 交易数量
    出参：DataFrame保存到data/raw/transactions.csv
    """
    print('\033[94m[数据生成器] 正在生成交易数据...\033[0m')
    
    transactions = []
    customer_ids = customers_df['customer_id'].tolist()
    
    for i in range(num_transactions):
        customer_id = random.choice(customer_ids)
        product = random.choice(PRODUCTS)
        
        trans_date = datetime(2022, 1, 1) + timedelta(days=random.randint(0, 730))
        quantity = random.choices([1, 2, 3, 4, 5], weights=[0.5, 0.25, 0.15, 0.07, 0.03])[0]
        
        discount_rate = random.choices([0, 0.05, 0.1, 0.15, 0.2, 0.3], weights=[0.4, 0.2, 0.15, 0.1, 0.1, 0.05])[0]
        unit_price = product['base_price'] * (1 - discount_rate)
        total_amount = unit_price * quantity
        
        payment_method = random.choice(['支付宝', '微信支付', '银行卡', '信用卡', '花呗'])
        
        is_holiday = trans_date.month in [1, 2, 6, 11, 12] and trans_date.day <= 7
        channel = random.choice(['APP', '网页', '小程序', '线下门店'])
        
        transactions.append({
            'transaction_id': f'T{str(i).zfill(8)}',
            'customer_id': customer_id,
            'product_id': product['id'],
            'product_category': product['category'],
            'transaction_date': trans_date.strftime('%Y-%m-%d'),
            'quantity': quantity,
            'unit_price': round(unit_price, 2),
            'total_amount': round(total_amount, 2),
            'discount_rate': discount_rate,
            'payment_method': payment_method,
            'channel': channel,
            'is_holiday': is_holiday,
            'is_returned': random.choices([0, 1], weights=[0.95, 0.05])[0],
            'province': random.choice(['广东', '北京', '上海', '浙江', '江苏']),
            'day_of_week': trans_date.weekday(),
            'month': trans_date.month,
            'year': trans_date.year,
            'hour': random.randint(0, 23)
        })
    
    df = pd.DataFrame(transactions)
    df.to_csv(f'{DATA_DIR}/transactions.csv', index=False, encoding='utf-8-sig')
    print(f'\033[92m[数据生成器] 交易数据已保存: {DATA_DIR}/transactions.csv ({len(df)} 条记录)\033[0m')
    return df

def generate_user_behavior(customers_df, num_behaviors=100000):
    """
    生成用户行为数据（浏览、点击、收藏等）
    入参：customers_df - 客户数据DataFrame, num_behaviors - 行为数量
    出参：DataFrame保存到data/raw/user_behavior.csv
    """
    print('\033[94m[数据生成器] 正在生成用户行为数据...\033[0m')
    
    behaviors = []
    customer_ids = customers_df['customer_id'].tolist()
    behavior_types = ['浏览', '点击', '收藏', '加购', '搜索', '分享']
    
    for i in range(num_behaviors):
        customer_id = random.choice(customer_ids)
        product = random.choice(PRODUCTS)
        behavior_date = datetime(2022, 1, 1) + timedelta(days=random.randint(0, 730), hours=random.randint(0, 23))
        
        behavior_type = random.choices(behavior_types, weights=[0.4, 0.25, 0.1, 0.15, 0.08, 0.02])[0]
        
        behaviors.append({
            'behavior_id': f'B{str(i).zfill(8)}',
            'customer_id': customer_id,
            'product_id': product['id'],
            'product_category': product['category'],
            'behavior_type': behavior_type,
            'behavior_date': behavior_date.strftime('%Y-%m-%d'),
            'behavior_hour': behavior_date.hour,
            'device_type': random.choice(['iOS', 'Android', 'PC', '平板']),
            'session_duration': random.randint(10, 3600),
            'page_views': random.randint(1, 20),
            'is_converted': 1 if behavior_type == '加购' and random.random() > 0.7 else 0
        })
    
    df = pd.DataFrame(behaviors)
    df.to_csv(f'{DATA_DIR}/user_behavior.csv', index=False, encoding='utf-8-sig')
    print(f'\033[92m[数据生成器] 用户行为数据已保存: {DATA_DIR}/user_behavior.csv ({len(df)} 条记录)\033[0m')
    return df

def generate_reviews(transactions_df, num_reviews=20000):
    """
    生成客户评论数据
    入参：transactions_df - 交易数据DataFrame, num_reviews - 评论数量
    出参：DataFrame保存到data/raw/reviews.csv
    """
    print('\033[94m[数据生成器] 正在生成客户评论数据...\033[0m')
    
    reviews = []
    transactions = transactions_df[transactions_df['is_returned'] == 0].sample(min(num_reviews, len(transactions_df)))
    
    for idx, trans in transactions.iterrows():
        rating = random.choices([1, 2, 3, 4, 5], weights=[0.05, 0.08, 0.15, 0.32, 0.40])[0]
        
        if rating >= 4:
            comment = random.choice(COMMENTS_POSITIVE)
            sentiment = '正面'
        elif rating <= 2:
            comment = random.choice(COMMENTS_NEGATIVE)
            sentiment = '负面'
        else:
            comment = random.choice(COMMENTS_NEUTRAL)
            sentiment = '中性'
        
        review_date = datetime.strptime(trans['transaction_date'], '%Y-%m-%d') + timedelta(days=random.randint(1, 14))
        
        reviews.append({
            'review_id': f'R{str(len(reviews)).zfill(8)}',
            'transaction_id': trans['transaction_id'],
            'customer_id': trans['customer_id'],
            'product_id': trans['product_id'],
            'product_category': trans['product_category'],
            'rating': rating,
            'comment': comment,
            'sentiment': sentiment,
            'review_date': review_date.strftime('%Y-%m-%d'),
            'is_verified_purchase': 1,
            'helpful_count': random.randint(0, 100),
            'has_image': random.choices([0, 1], weights=[0.7, 0.3])[0]
        })
    
    df = pd.DataFrame(reviews)
    df.to_csv(f'{DATA_DIR}/reviews.csv', index=False, encoding='utf-8-sig')
    print(f'\033[92m[数据生成器] 客户评论数据已保存: {DATA_DIR}/reviews.csv ({len(df)} 条记录)\033[0m')
    return df

def generate_daily_sales(num_days=730):
    """
    生成每日销售汇总数据（用于时间序列预测）
    入参：num_days - 天数
    出参：DataFrame保存到data/raw/daily_sales.csv
    """
    print('\033[94m[数据生成器] 正在生成每日销售汇总数据...\033[0m')
    
    daily_sales = []
    start_date = datetime(2022, 1, 1)
    
    base_sales = 50000
    trend = 50
    
    for i in range(num_days):
        date = start_date + timedelta(days=i)
        
        seasonal = 5000 * np.sin(2 * np.pi * i / 365)
        weekly = 3000 if date.weekday() >= 5 else 0
        
        if date.month in [11, 12]:
            holiday_effect = 15000
        elif date.month in [6, 7]:
            holiday_effect = 8000
        else:
            holiday_effect = 0
        
        noise = np.random.normal(0, 2000)
        
        sales = base_sales + trend * i + seasonal + weekly + holiday_effect + noise
        sales = max(sales, 10000)
        
        daily_sales.append({
            'date': date.strftime('%Y-%m-%d'),
            'total_sales': round(sales, 2),
            'order_count': int(sales / np.random.uniform(80, 150)),
            'customer_count': int(sales / np.random.uniform(100, 200)),
            'avg_order_value': round(np.random.uniform(80, 150), 2),
            'year': date.year,
            'month': date.month,
            'day': date.day,
            'day_of_week': date.weekday(),
            'is_weekend': 1 if date.weekday() >= 5 else 0,
            'is_holiday': 1 if date.month in [1, 2, 5, 10] and date.day <= 7 else 0,
            'temperature': np.random.uniform(10, 35),
            'promotion_flag': random.choices([0, 1], weights=[0.8, 0.2])[0]
        })
    
    df = pd.DataFrame(daily_sales)
    df.to_csv(f'{DATA_DIR}/daily_sales.csv', index=False, encoding='utf-8-sig')
    print(f'\033[92m[数据生成器] 每日销售数据已保存: {DATA_DIR}/daily_sales.csv ({len(df)} 条记录)\033[0m')
    return df

def generate_products():
    """
    生成商品基础信息数据
    入参：无
    出参：DataFrame保存到data/raw/products.csv
    """
    print('\033[94m[数据生成器] 正在生成商品数据...\033[0m')
    
    products_data = []
    for p in PRODUCTS:
        products_data.append({
            'product_id': p['id'],
            'product_name': p['name'],
            'category': p['category'],
            'base_price': round(p['base_price'], 2),
            'cost_price': round(p['base_price'] * np.random.uniform(0.4, 0.7), 2),
            'stock_quantity': random.randint(0, 1000),
            'supplier': f'供应商_{random.randint(1, 50)}',
            'brand': f'品牌_{random.randint(1, 30)}',
            'rating': round(np.random.uniform(3.0, 5.0), 1),
            'review_count': random.randint(0, 5000),
            'is_active': random.choices([0, 1], weights=[0.1, 0.9])[0],
            'create_date': (datetime(2020, 1, 1) + timedelta(days=random.randint(0, 1000))).strftime('%Y-%m-%d')
        })
    
    df = pd.DataFrame(products_data)
    df.to_csv(f'{DATA_DIR}/products.csv', index=False, encoding='utf-8-sig')
    print(f'\033[92m[数据生成器] 商品数据已保存: {DATA_DIR}/products.csv ({len(df)} 条记录)\033[0m')
    return df

def main():
    """
    主函数：生成所有数据集
    入参：无
    出参：生成所有数据文件
    """
    print('\033[96m' + '='*60 + '\033[0m')
    print('\033[96m  AI驱动的商业分析平台 - 数据生成器\033[0m')
    print('\033[96m' + '='*60 + '\033[0m')
    
    customers_df = generate_customers(5000)
    transactions_df = generate_transactions(customers_df, 50000)
    generate_user_behavior(customers_df, 100000)
    generate_reviews(transactions_df, 20000)
    generate_daily_sales(730)
    generate_products()
    
    print('\033[92m' + '='*60 + '\033[0m')
    print('\033[92m  ✅ 所有数据集生成完成！\033[0m')
    print('\033[92m' + '='*60 + '\033[0m')
    print(f'\n数据文件列表:')
    for f in os.listdir(DATA_DIR):
        if f.endswith('.csv'):
            filepath = os.path.join(DATA_DIR, f)
            size = os.path.getsize(filepath) / 1024
            print(f'  📄 {f} ({size:.1f} KB)')

if __name__ == '__main__':
    main()
