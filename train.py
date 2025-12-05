import sys
import os
os.environ['PYTHONIOENCODING'] = 'utf-8'
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import pandas as pd
import matplotlib.pyplot as plt
import datetime
from utils.log import Logger
from utils.common import data_preprocessing
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import seaborn as sns
import numpy as np

plt.rcParams['font.family'] = 'SimHei'
plt.rcParams['font.size'] = 15


class PowerLoadModel:
    def __init__(self):

        logfile_name = 'train_' + datetime.datetime.now().strftime('%Y%m%d%H%M%S')

        self.logfile = Logger('../', logfile_name).get_logger()

        self.logfile.info('开始创建 电力负荷模型的对象')
  
        self.data_source = data_preprocessing()

        self.model = None
#准备特征工程
    def prepare_features(self, data):

        # self.logfile.info('开始准备特征数据')
        
        # 确保 time 列为 datetime 类型
        data['time'] = pd.to_datetime(data['time'])
        
        # 按时间排序
        data = data.sort_values('time').reset_index(drop=True)
        
        # 提取时间特征
        data['hour'] = data['time'].dt.hour
        data['weekday'] = data['time'].dt.weekday
        data['month'] = data['time'].dt.month
        data['day'] = data['time'].dt.day  # 添加日期特征
        data['year'] = data['time'].dt.year  # 添加年份特征
        data['is_weekend'] = (data['weekday'] >= 5).astype(int)
        
        # 添加季节性特征
        def get_season(month):
            if month in [3, 4, 5]:
                return 1  # 春季
            elif month in [6, 7, 8]:
                return 2  # 夏季
            elif month in [9, 10, 11]:
                return 3  # 秋季
            else:
                return 4  # 冬季
        data['season'] = data['month'].apply(get_season)
        
        # 时间段特征
        def get_time_period(hour):
            if 6 <= hour < 12:
                return 1  # 上午
            elif 12 <= hour < 18:
                return 2  # 下午
            elif 18 <= hour < 24:
                return 3  # 晚上
            else:
                return 4  # 凌晨
        data['time_period'] = data['hour'].apply(get_time_period)
        
        # 滞后特征（这里没用）
        # data['load_lag1'] = data['power_load'].shift(1)      # 上一小时负荷
        # data['load_lag24'] = data['power_load'].shift(24)    # 上一天同一小时负荷
        # data['load_lag168'] = data['power_load'].shift(168)  # 上一周同一小时负荷
        
        # 负荷变化率特征
        data['load_change_rate'] = data['power_load'].pct_change(1)  # 相比于前一小时的变化率
        data['load_change_rate_24h'] = data['power_load'].pct_change(24)  # 相比于前一天的变化率
        
        # 删除包含NaN的行（由于滞后特征产生）
        original_length = len(data)
        data = data.dropna().reset_index(drop=True)
        new_length = len(data)

        def get_historical_same_period(row, data_df):
            target_month = row['month']
            target_day = row['day']
            target_hour = row['hour']
            current_year = row['year']
            
            # 查找历史同期数据（排除当年数据）
            historical_data = data_df[
                (data_df['month'] == target_month) & 
                (data_df['day'] == target_day) & 
                (data_df['hour'] == target_hour) & 
                (data_df['year'] < current_year)
            ]
            
            if len(historical_data) > 0:
                # 返回最近一年的同期数据
                return historical_data.iloc[-1]['power_load']
            else:
                # 如果没有找到历史同期数据，返回全局平均值
                return data_df['power_load'].mean()
                
        # 计算历史同期特征（在删除NaN之后计算）
        data['load_same_period_last_year'] = data.apply(lambda row: get_historical_same_period(row, data), axis=1)
        
        # 选择平衡后的特征列
        feature_columns = [
            'hour', 'weekday', 'month', 'season', 'is_weekend', 'time_period',
            # 'load_lag1', 'load_lag24', 'load_lag168', 
            'load_same_period_last_year',
            'load_change_rate', 'load_change_rate_24h'  # 关键的峰值预测特征
        ]
        
        X = data[feature_columns]
        y = data['power_load']
        
        self.logfile.info(f'特征维度: {X.shape}')
        print(f'训练特征数量: {len(feature_columns)}')
        print(f'特征列: {feature_columns}')
        return X, y, data
    
    def load_test_data(self):

        self.logfile.info('加载测试数据')
        # 使用绝对路径加载测试数据
        test_file_path = os.path.join(project_root, 'data', 'train.csv')
        if not os.path.exists(test_file_path):
            self.logfile.error(f'测试数据文件不存在: {test_file_path}')
            return None
        else:
            test_data = pd.read_csv(test_file_path)
            test_data['time'] = pd.to_datetime(test_data['time'])
            return test_data
    
    def train_model(self, X, y):

        self.logfile.info('开始训练XGBoost模型')
        print('开始训练XGBoost模型')
        
        # 加载测试数据用于最终评估
        test_data = self.load_test_data()
        use_split_data = False
        
        if test_data is not None:
            X_test, y_test, _ = self.prepare_features(test_data)
            # 检查测试数据是否为空
            if len(X_test) == 0 or len(y_test) == 0:
                self.logfile.warning('测试数据处理后为空，使用训练数据划分')
                use_split_data = True
            else:
                print("使用独立的测试数据集")
        else:
            use_split_data = True
            
        if use_split_data:
            # 如果没有测试数据，从训练数据中划分一部分作为测试集
            split_idx = int(len(X) * 0.8)
            X_train_part = X[:split_idx]
            y_train_part = y[:split_idx]
            X_test = X[split_idx:]
            y_test = y[split_idx:]
            print("使用训练数据后20%作为测试集")
        
        # 使用全部训练数据进行训练
        X_train, y_train = X, y
        
        # 调整参数以提高模型性能
        param_grid = {
            'n_estimators': [100, 200],  # 增加树的数量
            'max_depth': [3,5],     # 增加树的深度
            'learning_rate': [0.1, 0.15],  # 提高学习率
            'subsample': [0.8, 0.9],       # 增加样本采样比例
            'colsample_bytree': [0.8, 0.9],# 增加特征采样比例
            'reg_alpha': [0, 0.01],        # L1正则化
            'reg_lambda': [0, 0.01],       # L2正则化
            'min_child_weight': [1, 3]       # 最小叶子节点权重
        }
        
        # 创建XGBoost回归器
        xgb = XGBRegressor(random_state=42)
        
        # 网格搜索寻找最佳参数
        self.logfile.info('开始网格搜索最佳参数')
        print('开始网格搜索最佳参数')
        grid_search = GridSearchCV(
            estimator=xgb,
            param_grid=param_grid,
            cv=3,
            scoring='neg_mean_squared_error',
            n_jobs=1  # 改为1，禁用并行处理避免编码问题
        )
        
        # 训练模型
        grid_search.fit(X_train, y_train)
        
        # 保存最佳模型
        self.model = grid_search.best_estimator_
        
        # 输出最佳参数
        self.logfile.info(f'最佳参数: {grid_search.best_params_}')
        print(f'最佳参数: {grid_search.best_params_}')
        
        # 在测试集上评估（仅当测试集非空时）
        if len(X_test) > 0 and len(y_test) > 0:
            y_test_pred = self.model.predict(X_test)
            test_mse = mean_squared_error(y_test, y_test_pred)
            test_mae = mean_absolute_error(y_test, y_test_pred)
            test_r2 = r2_score(y_test, y_test_pred)
            
            self.logfile.info(f'测试集评估结果 - MSE: {test_mse:.2f}, MAE: {test_mae:.2f}, R2: {test_r2:.4f}')
            print(f'测试集评估结果 - MSE: {test_mse:.2f}, MAE: {test_mae:.2f}, R2: {test_r2:.4f}')
            
            # 可视化预测结果
            self.visualize_predictions(y_test, y_test_pred, "测试集")
        else:
            self.logfile.warning('测试集为空，跳过评估')
            print('警告: 测试集为空，跳过评估')
        
        # 特征重要性分析
        self.analyze_feature_importance(X_train.columns)
        # print('输出y_test')
        # print(y_test)
        # print('输出y_test_pred')
        # print(y_test_pred)
        return self.model

    def visualize_predictions(self, y_true, y_pred, title_suffix=""):

        plt.figure(figsize=(12, 5))
        
        # 散点图
        plt.subplot(1, 2, 1)
        plt.scatter(y_true, y_pred, alpha=0.5)
        plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r-', lw=2)
        plt.xlabel('实际负荷')
        plt.ylabel('预测负荷')
        plt.title(f'实际 vs 预测 ({title_suffix})')
        
        # 残差图
        plt.subplot(1, 2, 2)
        residuals = y_true - y_pred
        plt.scatter(y_pred, residuals, alpha=0.5)
        plt.axhline(y=0, color='r', linestyle='-')
        plt.xlabel('预测负荷')
        plt.ylabel('残差')
        plt.title(f'残差分布 ({title_suffix})')
        
        plt.tight_layout()
        plt.savefig(f'picture/prediction_analysis_{title_suffix.replace(" ", "_")}.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
    
    def analyze_feature_importance(self, feature_names):

        if self.model is not None:
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            plt.figure(figsize=(10, 8))
            sns.barplot(data=importance_df.head(10), x='importance', y='feature')
            plt.title('特征重要性分析（前10个）')
            plt.xlabel('重要性')
            plt.tight_layout()
            plt.savefig('picture/feature_importance.png', dpi=300, bbox_inches='tight')
            plt.show()
            
            print("特征重要性排名:")
            print(importance_df)
    
    def save_model(self, filepath=None):

        if filepath is None:
            # 使用绝对路径保存模型
            filepath = os.path.join(project_root, 'model', 'xgb_model.pkl')
        
        if self.model is not None:
            # 确保模型目录存在
            model_dir = os.path.dirname(filepath)
            if not os.path.exists(model_dir):
                os.makedirs(model_dir)
            
            joblib.dump(self.model, filepath)
            self.logfile.info(f'模型已保存至: {filepath}')
            print(f'模型已保存至: {filepath}')
        else:
            self.logfile.error('模型尚未训练，无法保存')
            print('模型尚未训练，无法保存')

    def predict_load(self, year, month, day, hour):

        if self.model is None:
            print("模型尚未训练，请先训练模型")
            return None
            

        prediction_time = pd.Timestamp(year=year, month=month, day=day, hour=hour)
        

        features = {
            'hour': hour,
            'weekday': prediction_time.weekday(),
            'month': month,
            'season': self._get_season(month),
            'is_weekend': 1 if prediction_time.weekday() >= 5 else 0,
            'day_of_month': day,
            'day_of_year': prediction_time.dayofyear,
            'time_period': self._get_time_period(hour)
        }

        return features
    
    def _get_season(self, month):
        if month in [3, 4, 5]:
            return 1  # 春季
        elif month in [6, 7, 8]:
            return 2  # 夏季
        elif month in [9, 10, 11]:
            return 3  # 秋季
        else:
            return 4  # 冬季
    
    def _get_time_period(self, hour):
        if 6 <= hour < 12:
            return 1  # 上午
        elif 12 <= hour < 18:
            return 2  # 下午
        elif 18 <= hour < 24:
            return 3  # 晚上
        else:
            return 4  # 凌晨


def visualize_time_features(data):


    # 确保 time 列为 datetime 类型
    data['time'] = pd.to_datetime(data['time'])

    # 提取时间特征
    data['hour'] = data['time'].dt.hour
    data['weekday'] = data['time'].dt.weekday
    data['month'] = data['time'].dt.month
    data['is_weekend'] = (data['weekday'] >= 5).astype(int)

    def get_season(month):
        if month in [3, 4, 5]:
            return 1  # 春季
        elif month in [6, 7, 8]:
            return 2  # 夏季
        elif month in [9, 10, 11]:
            return 3  # 秋季
        else:
            return 4  # 冬季
    
    data['season'] = data['month'].apply(get_season)

    def get_time_period(hour):
        if 6 <= hour < 12:
            return 1  # 上午
        elif 12 <= hour < 18:
            return 2  # 下午
        elif 18 <= hour < 24:
            return 3  # 晚上
        else:
            return 4  # 凌晨
    
    data['time_period'] = data['hour'].apply(get_time_period)

    fig = plt.figure(figsize=(20, 15))

    # 1. 小时负荷分布
    ax1 = fig.add_subplot(3, 3, 1)
    hourly_load = data.groupby('hour')['power_load'].mean()
    ax1.bar(hourly_load.index, hourly_load.values, color='skyblue')
    ax1.set_title('各小时平均负荷')
    ax1.set_xlabel('小时')
    ax1.set_ylabel('平均负荷(MW)')

    # 2. 星期负荷分布
    ax2 = fig.add_subplot(3, 3, 2)
    weekday_load = data.groupby('weekday')['power_load'].mean()
    ax2.bar(weekday_load.index, weekday_load.values, color='lightgreen')
    ax2.set_title('星期几平均负荷')
    ax2.set_xlabel('星期(0=周一)')
    ax2.set_ylabel('平均负荷(MW)')

    # 3. 月份负荷分布
    ax3 = fig.add_subplot(3, 3, 3)
    monthly_load = data.groupby('month')['power_load'].mean()
    ax3.bar(monthly_load.index, monthly_load.values, color='salmon')
    ax3.set_title('各月份平均负荷')
    ax3.set_xlabel('月份')
    ax3.set_ylabel('平均负荷(MW)')

    # 4. 季节负荷分布
    ax4 = fig.add_subplot(3, 3, 4)
    seasonal_load = data.groupby('season')['power_load'].mean()
    ax4.bar(seasonal_load.index, seasonal_load.values, color='gold')
    ax4.set_title('季节平均负荷')
    ax4.set_xlabel('季节(1=春,2=夏,3=秋,4=冬)')
    ax4.set_ylabel('平均负荷(MW)')

    # 5. 工作日vs周末负荷对比
    ax5 = fig.add_subplot(3, 3, 5)
    workday_load = data.groupby('is_weekend')['power_load'].mean()
    ax5.bar(['工作日', '周末'], workday_load.values, color=['lightblue', 'pink'])
    ax5.set_title('工作日vs周末负荷')
    ax5.set_ylabel('平均负荷(MW)')

    # 6. 时段负荷分布
    ax6 = fig.add_subplot(3, 3, 6)
    period_load = data.groupby('time_period')['power_load'].mean()
    ax6.bar(period_load.index, period_load.values, color='orchid')
    ax6.set_title('不同时段平均负荷')
    ax6.set_xlabel('时段(1=上午,2=下午,3=晚上,4=凌晨)')
    ax6.set_ylabel('平均负荷(MW)')

    plt.tight_layout()
    plt.savefig('picture/time_features_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

    return data

# 这是热力图，但是没卵用
# def plot_time_heatmap(data):
#
#     # 确保必要的时间特征存在
#     if 'hour' not in data.columns or 'weekday' not in data.columns:
#         data['time'] = pd.to_datetime(data['time'])
#         data['hour'] = data['time'].dt.hour
#         data['weekday'] = data['time'].dt.weekday
#
#     # 创建透视表
#     pivot_data = data.pivot_table(
#         values='power_load',
#         index='weekday',
#         columns='hour',
#         aggfunc='mean'
#     )
#
#     plt.figure(figsize=(12, 8))
#     sns.heatmap(pivot_data, annot=False, cmap='YlOrRd', cbar_kws={'label': '负荷(MW)'})
#     plt.title('小时-星期负荷热力图')
#     plt.xlabel('小时')
#     plt.ylabel('星期几(0=周一)')
#     plt.savefig('picture/hour_weekday_heatmap.png', dpi=300, bbox_inches='tight')
#     plt.show()

#分析时间特征与负荷的相关性
def analyze_time_correlations(data):
    time_features = ['hour', 'weekday', 'month']

    # 检查并添加缺失的时间特征
    if 'hour' not in data.columns:
        data['time'] = pd.to_datetime(data['time'])
        data['hour'] = data['time'].dt.hour
        data['weekday'] = data['time'].dt.weekday
        data['month'] = data['time'].dt.month
        data['is_weekend'] = (data['weekday'] >= 5).astype(int)

        def get_season(month):
            if month in [3, 4, 5]:
                return 1
            elif month in [6, 7, 8]:
                return 2
            elif month in [9, 10, 11]:
                return 3
            else:
                return 4

        data['season'] = data['month'].apply(get_season)

        def get_time_period(hour):
            if 6 <= hour < 12:
                return 1
            elif 12 <= hour < 18:
                return 2
            elif 18 <= hour < 24:
                return 3
            else:
                return 4

        data['time_period'] = data['hour'].apply(get_time_period)

    # 选择时间特征和目标变量
    time_features = ['hour', 'weekday', 'month', 'season', 'is_weekend', 'time_period']
    correlation_data = data[time_features + ['power_load']].corr()

    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_data, annot=True, cmap='coolwarm', center=0,
                square=True, linewidths=0.5, fmt='.2f')
    plt.title('时间特征与负荷相关性分析')
    plt.savefig('picture/time_correlation_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

    return correlation_data

if __name__ == '__main__':
    # 创建模型实例
    pm = PowerLoadModel()

    # 进行时间特征分析
    data_with_features = visualize_time_features(pm.data_source.copy())
    # plot_time_heatmap(data_with_features)
    correlation_data = analyze_time_correlations(data_with_features)
    
    # 准备特征和标签
    X, y, processed_data = pm.prepare_features(pm.data_source.copy())
    
    # 训练模型
    model = pm.train_model(X, y)
    
    # 保存模型
    pm.save_model() 
