"""
AI驱动的商业分析平台 - 增强的机器学习模型构建模块
功能：模型选择、数据分割、超参数调优、模型评估、过拟合检测
入参：处理后的数据
出参：训练好的模型和评估结果
异常处理：模型训练异常捕获
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report
from sklearn.metrics import confusion_matrix
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.decomposition import PCA
import xgboost as xgb
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

PROCESSED_DIR = 'data/processed'
VIZ_DIR = 'data/viz'
MODELS_DIR = 'data/models'
os.makedirs(VIZ_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

class EnhancedMLModelBuilder:
    """
    增强的机器学习模型构建类
    功能：模型选择、数据分割、超参数调优、模型评估、过拟合检测
    """
    
    def __init__(self):
        self.models = {}
        self.evaluation_results = {}
        self.best_params = {}
        self.scalers = {}
        self.label_encoders = {}
    
    def log_step(self, message, status='info'):
        """
        记录训练步骤日志
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
        print(f'{colors.get(status, "")}[ML建模] {message}{reset}')
    
    def prepare_data_for_regression(self, df, target_col, test_size=0.2):
        """
        准备回归任务数据
        入参：df - DataFrame, target_col - 目标列名, test_size - 测试集比例
        出参：X_train, X_test, y_train, y_test
        """
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        categorical_cols = X.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if col not in self.label_encoders:
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
                self.label_encoders[col] = le
            else:
                X[col] = self.label_encoders[col].transform(X[col].astype(str))
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        self.scalers['regression'] = scaler
        
        self.log_step(f'回归数据准备完成: 训练集{len(X_train)}条, 测试集{len(X_test)}条', 'success')
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def prepare_data_for_classification(self, df, target_col, test_size=0.2, balance_method='smote'):
        """
        准备分类任务数据
        入参：df - DataFrame, target_col - 目标列名, test_size - 测试集比例, balance_method - 平衡方法
        出参：X_train, X_test, y_train, y_test
        """
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        categorical_cols = X.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if col not in self.label_encoders:
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
                self.label_encoders[col] = le
            else:
                X[col] = self.label_encoders[col].transform(X[col].astype(str))
        
        if y.dtype == 'object':
            if target_col not in self.label_encoders:
                le = LabelEncoder()
                y = le.fit_transform(y.astype(str))
                self.label_encoders[target_col] = le
            else:
                y = self.label_encoders[target_col].transform(y.astype(str))
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        if balance_method == 'smote':
            smote = SMOTE(random_state=42)
            X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
            self.log_step(f'使用SMOTE过采样: {len(y_train)} -> {len(y_train_res)}', 'info')
        elif balance_method == 'undersample':
            undersampler = RandomUnderSampler(random_state=42)
            X_train_res, y_train_res = undersampler.fit_resample(X_train, y_train)
            self.log_step(f'使用欠采样: {len(y_train)} -> {len(y_train_res)}', 'info')
        else:
            X_train_res, y_train_res = X_train, y_train
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_res)
        X_test_scaled = scaler.transform(X_test)
        
        self.scalers['classification'] = scaler
        
        self.log_step(f'分类数据准备完成: 训练集{len(X_train_scaled)}条, 测试集{len(X_test_scaled)}条', 'success')
        
        return X_train_scaled, X_test_scaled, y_train_res, y_test
    
    def build_regression_models(self, X_train, X_test, y_train, y_test):
        """
        构建回归模型
        入参：X_train, X_test, y_train, y_test
        出参：模型评估结果
        """
        self.log_step('开始构建回归模型...', 'info')
        
        models = {
            'Linear Regression': LinearRegression(),
            'Ridge Regression': Ridge(random_state=42),
            'Lasso Regression': Lasso(random_state=42),
            'Random Forest': RandomForestRegressor(random_state=42, n_estimators=100),
            'XGBoost': xgb.XGBRegressor(random_state=42, n_estimators=100, verbosity=0),
            'LightGBM': lgb.LGBMRegressor(random_state=42, n_estimators=100, verbose=-1)
        }
        
        results = {}
        
        for name, model in models.items():
            self.log_step(f'训练 {name}...', 'info')
            
            model.fit(X_train, y_train)
            
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            
            train_mae = mean_absolute_error(y_train, y_train_pred)
            test_mae = mean_absolute_error(y_test, y_test_pred)
            train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
            test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
            train_r2 = r2_score(y_train, y_train_pred)
            test_r2 = r2_score(y_test, y_test_pred)
            
            results[name] = {
                'train_mae': train_mae,
                'test_mae': test_mae,
                'train_rmse': train_rmse,
                'test_rmse': test_rmse,
                'train_r2': train_r2,
                'test_r2': test_r2,
                'overfitting': train_r2 - test_r2
            }
            
            self.models[f'regression_{name}'] = model
            
            self.log_step(f'{name} - Test R²: {test_r2:.4f}, Test RMSE: {test_rmse:.2f}', 'success')
        
        self.evaluation_results['regression'] = results
        return results
    
    def build_classification_models(self, X_train, X_test, y_train, y_test):
        """
        构建分类模型
        入参：X_train, X_test, y_train, y_test
        出参：模型评估结果
        """
        self.log_step('开始构建分类模型...', 'info')
        
        models = {
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100),
            'XGBoost': xgb.XGBClassifier(random_state=42, n_estimators=100, verbosity=0),
            'LightGBM': lgb.LGBMClassifier(random_state=42, n_estimators=100, verbose=-1)
        }
        
        results = {}
        
        for name, model in models.items():
            self.log_step(f'训练 {name}...', 'info')
            
            model.fit(X_train, y_train)
            
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            y_test_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
            
            train_acc = accuracy_score(y_train, y_train_pred)
            test_acc = accuracy_score(y_test, y_test_pred)
            train_precision = precision_score(y_train, y_train_pred, average='weighted', zero_division=0)
            test_precision = precision_score(y_test, y_test_pred, average='weighted', zero_division=0)
            train_recall = recall_score(y_train, y_train_pred, average='weighted', zero_division=0)
            test_recall = recall_score(y_test, y_test_pred, average='weighted', zero_division=0)
            train_f1 = f1_score(y_train, y_train_pred, average='weighted', zero_division=0)
            test_f1 = f1_score(y_test, y_test_pred, average='weighted', zero_division=0)
            
            results[name] = {
                'train_accuracy': train_acc,
                'test_accuracy': test_acc,
                'train_precision': train_precision,
                'test_precision': test_precision,
                'train_recall': train_recall,
                'test_recall': test_recall,
                'train_f1': train_f1,
                'test_f1': test_f1,
                'overfitting': train_acc - test_acc
            }
            
            if y_test_proba is not None:
                test_auc = roc_auc_score(y_test, y_test_proba)
                results[name]['test_auc'] = test_auc
            
            self.models[f'classification_{name}'] = model
            
            self.log_step(f'{name} - Test Accuracy: {test_acc:.4f}, Test F1: {test_f1:.4f}', 'success')
        
        self.evaluation_results['classification'] = results
        return results
    
    def hyperparameter_tuning(self, X_train, y_train, model_type='regression', method='grid'):
        """
        超参数调优
        入参：X_train, y_train, model_type - 模型类型, method - 调优方法(grid/random)
        出参：最优参数
        """
        self.log_step(f'开始超参数调优 ({method})...', 'info')
        
        if model_type == 'regression':
            base_model = xgb.XGBRegressor(random_state=42, verbosity=0)
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [4, 6, 8],
                'learning_rate': [0.01, 0.1, 0.2],
                'subsample': [0.8, 0.9, 1.0],
                'colsample_bytree': [0.8, 0.9, 1.0]
            }
            scoring = 'neg_mean_squared_error'
        else:
            base_model = xgb.XGBClassifier(random_state=42, verbosity=0)
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [4, 6, 8],
                'learning_rate': [0.01, 0.1, 0.2],
                'subsample': [0.8, 0.9, 1.0],
                'colsample_bytree': [0.8, 0.9, 1.0]
            }
            scoring = 'accuracy'
        
        if method == 'grid':
            search = GridSearchCV(base_model, param_grid, cv=5, scoring=scoring, n_jobs=-1, verbose=0)
        else:
            search = RandomizedSearchCV(base_model, param_grid, cv=5, scoring=scoring, 
                                     n_iter=50, n_jobs=-1, verbose=0, random_state=42)
        
        search.fit(X_train, y_train)
        
        self.best_params[model_type] = search.best_params_
        
        self.log_step(f'最优参数: {search.best_params_}', 'success')
        self.log_step(f'最优分数: {search.best_score_:.4f}', 'success')
        
        return search.best_params_, search.best_estimator_
    
    def detect_overfitting(self, results, model_type='regression'):
        """
        检测过拟合和欠拟合
        入参：results - 评估结果, model_type - 模型类型
        出参：过拟合分析结果
        """
        self.log_step(f'检测{model_type}模型的过拟合/欠拟合...', 'info')
        
        overfitting_analysis = {}
        
        for model_name, metrics in results.items():
            if model_type == 'regression':
                train_score = metrics['train_r2']
                test_score = metrics['test_r2']
                gap = train_score - test_score
            else:
                train_score = metrics['train_accuracy']
                test_score = metrics['test_accuracy']
                gap = train_score - test_score
            
            if gap > 0.1:
                status = '严重过拟合'
                suggestion = '增加正则化、减少模型复杂度、增加训练数据'
            elif gap > 0.05:
                status = '轻微过拟合'
                suggestion = '轻微增加正则化或减少模型复杂度'
            elif gap < -0.05:
                status = '欠拟合'
                suggestion = '增加模型复杂度、添加更多特征'
            else:
                status = '拟合良好'
                suggestion = '模型表现良好'
            
            overfitting_analysis[model_name] = {
                'train_score': train_score,
                'test_score': test_score,
                'gap': gap,
                'status': status,
                'suggestion': suggestion
            }
            
            self.log_step(f'{model_name}: {status} (差距: {gap:.4f})', 'warning' if gap > 0.05 else 'success')
        
        return overfitting_analysis
    
    def plot_model_comparison(self, results, model_type='regression'):
        """
        绘制模型对比图
        入参：results - 评估结果, model_type - 模型类型
        出参：保存图表到data/viz/
        """
        if model_type == 'regression':
            models = list(results.keys())
            test_r2 = [results[m]['test_r2'] for m in models]
            test_rmse = [results[m]['test_rmse'] for m in models]
            
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            
            axes[0].bar(models, test_r2, color='steelblue', alpha=0.8)
            axes[0].set_title('模型R²对比', fontsize=12, fontweight='bold')
            axes[0].set_ylabel('R² Score', fontsize=10)
            axes[0].set_ylim(0, 1)
            axes[0].tick_params(axis='x', rotation=45)
            axes[0].grid(True, alpha=0.3, axis='y')
            
            for i, v in enumerate(test_r2):
                axes[0].text(i, v + 0.02, f'{v:.3f}', ha='center', va='bottom', fontsize=9)
            
            axes[1].bar(models, test_rmse, color='coral', alpha=0.8)
            axes[1].set_title('模型RMSE对比', fontsize=12, fontweight='bold')
            axes[1].set_ylabel('RMSE', fontsize=10)
            axes[1].tick_params(axis='x', rotation=45)
            axes[1].grid(True, alpha=0.3, axis='y')
            
            for i, v in enumerate(test_rmse):
                axes[1].text(i, v + v*0.02, f'{v:.0f}', ha='center', va='bottom', fontsize=9)
            
            plt.tight_layout()
            filename = f'回归模型对比.png'
        else:
            models = list(results.keys())
            test_acc = [results[m]['test_accuracy'] for m in models]
            test_f1 = [results[m]['test_f1'] for m in models]
            
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            
            axes[0].bar(models, test_acc, color='steelblue', alpha=0.8)
            axes[0].set_title('模型准确率对比', fontsize=12, fontweight='bold')
            axes[0].set_ylabel('Accuracy', fontsize=10)
            axes[0].set_ylim(0, 1)
            axes[0].tick_params(axis='x', rotation=45)
            axes[0].grid(True, alpha=0.3, axis='y')
            
            for i, v in enumerate(test_acc):
                axes[0].text(i, v + 0.02, f'{v:.3f}', ha='center', va='bottom', fontsize=9)
            
            axes[1].bar(models, test_f1, color='coral', alpha=0.8)
            axes[1].set_title('模型F1分数对比', fontsize=12, fontweight='bold')
            axes[1].set_ylabel('F1 Score', fontsize=10)
            axes[1].set_ylim(0, 1)
            axes[1].tick_params(axis='x', rotation=45)
            axes[1].grid(True, alpha=0.3, axis='y')
            
            for i, v in enumerate(test_f1):
                axes[1].text(i, v + 0.02, f'{v:.3f}', ha='center', va='bottom', fontsize=9)
            
            plt.tight_layout()
            filename = f'分类模型对比.png'
        
        filepath = os.path.join(VIZ_DIR, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.log_step(f'模型对比图已保存: {filename}', 'success')
    
    def plot_actual_vs_predicted(self, y_true, y_pred, model_name='Model'):
        """
        绘制预测值vs实际值对比图
        入参：y_true - 真实值, y_pred - 预测值, model_name - 模型名称
        出参：保存图表到data/viz/
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        axes[0].scatter(y_true, y_pred, alpha=0.5, s=20)
        axes[0].plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 
                     'r--', lw=2, label='完美预测')
        axes[0].set_xlabel('实际值', fontsize=12)
        axes[0].set_ylabel('预测值', fontsize=12)
        axes[0].set_title(f'{model_name} - 预测vs实际', fontsize=12, fontweight='bold')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        r2 = r2_score(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        
        textstr = f'R² = {r2:.4f}\nRMSE = {rmse:.2f}\nMAE = {mae:.2f}'
        axes[0].text(0.05, 0.95, textstr, transform=axes[0].transAxes,
                    fontsize=10, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        residuals = y_true - y_pred
        axes[1].scatter(y_pred, residuals, alpha=0.5, s=20)
        axes[1].axhline(y=0, color='r', linestyle='--', lw=2)
        axes[1].set_xlabel('预测值', fontsize=12)
        axes[1].set_ylabel('残差', fontsize=12)
        axes[1].set_title(f'{model_name} - 残差图', fontsize=12, fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        filename = f'{model_name}_预测vs实际.png'
        filepath = os.path.join(VIZ_DIR, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.log_step(f'预测vs实际图已保存: {filename}', 'success')
    
    def run_full_ml_pipeline(self, df, target_col, task_type='regression'):
        """
        运行完整的机器学习建模流程
        入参：df - DataFrame, target_col - 目标列名, task_type - 任务类型(regression/classification)
        出参：评估结果
        """
        print('\033[96m' + '='*60 + '\033[0m')
        print('\033[96m  AI驱动的商业分析平台 - 增强的ML建模\033[0m')
        print('\033[96m' + '='*60 + '\033[0m')
        print()
        
        self.log_step(f'开始{task_type}任务建模流程...', 'info')
        print()
        
        total_steps = 5
        current_step = 0
        
        current_step += 1
        print(f'\n[{current_step}/{total_steps}] 数据准备')
        if task_type == 'regression':
            X_train, X_test, y_train, y_test = self.prepare_data_for_regression(df, target_col)
        else:
            X_train, X_test, y_train, y_test = self.prepare_data_for_classification(df, target_col)
        
        current_step += 1
        print(f'\n[{current_step}/{total_steps}] 模型训练')
        if task_type == 'regression':
            results = self.build_regression_models(X_train, X_test, y_train, y_test)
        else:
            results = self.build_classification_models(X_train, X_test, y_train, y_test)
        
        current_step += 1
        print(f'\n[{current_step}/{total_steps}] 超参数调优')
        best_params, best_model = self.hyperparameter_tuning(X_train, y_train, task_type, method='random')
        
        current_step += 1
        print(f'\n[{current_step}/{total_steps}] 过拟合检测')
        overfitting_analysis = self.detect_overfitting(results, task_type)
        
        current_step += 1
        print(f'\n[{current_step}/{total_steps}] 模型评估与可视化')
        self.plot_model_comparison(results, task_type)
        
        if task_type == 'regression':
            y_pred = best_model.predict(X_test)
            self.plot_actual_vs_predicted(y_test, y_pred, '最佳模型')
        
        print()
        print('\033[92m' + '='*60 + '\033[0m')
        print('\033[92m  ✅ ML建模完成！\033[0m')
        print('\033[92m' + '='*60 + '\033[0m')
        
        return {
            'results': results,
            'best_params': best_params,
            'overfitting_analysis': overfitting_analysis
        }

def main():
    """
    主函数
    """
    ml_builder = EnhancedMLModelBuilder()
    
    daily_sales = pd.read_csv(os.path.join(PROCESSED_DIR, 'daily_sales_processed.csv'))
    
    numeric_cols = daily_sales.select_dtypes(include=[np.number]).columns.tolist()
    if 'total_sales' in numeric_cols:
        results = ml_builder.run_full_ml_pipeline(daily_sales, 'total_sales', 'regression')

if __name__ == '__main__':
    main()
