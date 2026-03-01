"""
AI驱动的商业分析平台 - 主程序入口
功能：一键运行全流程分析
入参：无
出参：生成所有分析结果和可视化图表
异常处理：各模块异常捕获
"""
import os
import sys
import time
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def print_banner():
    """
    打印程序横幅
    入参：无
    出参：打印横幅
    """
    print('\033[96m' + '='*70 + '\033[0m')
    print('\033[96m' + '   AI驱动的商业分析平台 - 一键运行全流程分析\033[0m')
    print('\033[96m' + '='*70 + '\033[0m')
    print(f'\n\033[93m  启动时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\033[0m\n')

def print_step(step_num, total_steps, message):
    """
    打印步骤信息
    入参：step_num - 步骤编号, total_steps - 总步骤数, message - 步骤信息
    出参：打印步骤
    """
    print(f'\n\033[95m[{step_num}/{total_steps}] {message}\033[0m')
    print('\033[90m' + '-'*60 + '\033[0m')

def run_full_pipeline():
    """
    运行完整分析流程
    入参：无
    出参：分析结果
    """
    start_time = time.time()
    
    print_banner()
    
    total_steps = 6
    
    try:
        print_step(1, total_steps, '生成数据集')
        from src.utils.data_generator import main as generate_data
        generate_data()
    except Exception as e:
        print(f'\033[91m[错误] 数据生成失败: {str(e)}\033[0m')
    
    try:
        print_step(2, total_steps, '数据预处理')
        from src.utils.data_preprocessor import main as preprocess_data
        preprocess_data()
    except Exception as e:
        print(f'\033[91m[错误] 数据预处理失败: {str(e)}\033[0m')
    
    try:
        print_step(3, total_steps, '描述性统计分析')
        from src.analysis.descriptive_analyzer import main as descriptive_analysis
        descriptive_analysis()
    except Exception as e:
        print(f'\033[91m[错误] 描述性分析失败: {str(e)}\033[0m')
    
    try:
        print_step(4, total_steps, '时间序列预测')
        from src.analysis.time_series_forecaster import main as time_series_analysis
        time_series_analysis()
    except Exception as e:
        print(f'\033[91m[错误] 时间序列预测失败: {str(e)}\033[0m')
    
    try:
        print_step(5, total_steps, '客户分类分析')
        from src.analysis.customer_segmentation import main as customer_analysis
        customer_analysis()
    except Exception as e:
        print(f'\033[91m[错误] 客户分类分析失败: {str(e)}\033[0m')
    
    try:
        print_step(6, total_steps, '情感分析')
        from src.analysis.sentiment_analyzer import main as sentiment_analysis
        sentiment_analysis()
    except Exception as e:
        print(f'\033[91m[错误] 情感分析失败: {str(e)}\033[0m')
    
    elapsed_time = time.time() - start_time
    
    print('\n\033[92m' + '='*70 + '\033[0m')
    print('\033[92m  ✅ 全流程分析完成！\033[0m')
    print('\033[92m' + '='*70 + '\033[0m')
    print(f'\n\033[93m  总耗时: {elapsed_time:.2f} 秒\033[0m')
    print(f'\033[93m  完成时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\033[0m')
    
    print('\n\033[94m生成的数据文件:\033[0m')
    for directory in ['data/raw', 'data/processed']:
        if os.path.exists(directory):
            for f in os.listdir(directory):
                if f.endswith('.csv'):
                    filepath = os.path.join(directory, f)
                    size = os.path.getsize(filepath) / 1024
                    print(f'  📄 {directory}/{f} ({size:.1f} KB)')
    
    print('\n\033[94m生成的可视化图表:\033[0m')
    if os.path.exists('data/viz'):
        for f in os.listdir('data/viz'):
            if f.endswith('.png'):
                filepath = os.path.join('data/viz', f)
                size = os.path.getsize(filepath) / 1024
                print(f'  📊 data/viz/{f} ({size:.1f} KB)')
    
    print('\n\033[93m  启动前端: streamlit run app.py --server.port 8502\033[0m')

if __name__ == '__main__':
    run_full_pipeline()
