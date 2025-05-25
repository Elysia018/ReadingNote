import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, GridSearchCV, RandomizedSearchCV
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import seaborn as sns
import pandas as pd
from scipy.stats import uniform

import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体显示中文
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 1. 数据准备
data = load_breast_cancer()
X, y = data.data, data.target

# 2. 数据预处理
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 3. 交叉验证实现
## 3.1 k折交叉验证对比实验
def compare_k_fold():
    models = {
        'SVM': SVC(kernel='linear', random_state=42),
        'Random Forest': RandomForestClassifier(random_state=42)
    }
    
    k_values = [5, 10]
    results = []
    
    for model_name, model in models.items():
        for k in k_values:
            scores = cross_val_score(model, X_train, y_train, cv=k, scoring='accuracy')
            results.append({
                'Model': model_name,
                'k': k,
                'Mean Accuracy': scores.mean(),
                'Std Accuracy': scores.std()
            })
    
    results_df = pd.DataFrame(results)
    print("\n交叉验证对比结果:")
    print(results_df)
    
    # 可视化
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Model', y='Mean Accuracy', hue='k', data=results_df)
    plt.title('不同k值下交叉验证准确率比较')
    plt.show()
    
    return results_df

k_fold_results = compare_k_fold()

## 3.2 分层交叉验证示例
def stratified_cv_example():
    model = SVC(kernel='linear', random_state=42)
    stratified_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, X_train, y_train, cv=stratified_cv)
    print(f"\n分层交叉验证平均准确率：{scores.mean():.2f} (±{scores.std():.2f})")

stratified_cv_example()

# 4. 模型选择与超参数优化
## 4.1 网格搜索调参 - 随机森林
def grid_search_rf():
    param_grid = {
        'n_estimators': [50, 100, 150],
        'max_depth': [5, 10, None],
        'min_samples_split': [2, 5, 10]
    }
    
    rf = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    print("\n随机森林网格搜索结果:")
    print("最优参数组合：", grid_search.best_params_)
    print("最优模型验证准确率：", grid_search.best_score_)
    
    # 测试集评估
    best_rf = grid_search.best_estimator_
    y_pred = best_rf.predict(X_test)
    print(f"测试集准确率：{accuracy_score(y_test, y_pred):.2f}")
    print("\n分类报告:")
    print(classification_report(y_test, y_pred))
    
    return grid_search

rf_grid_search = grid_search_rf()

## 4.2 网格搜索调参 - SVM
def grid_search_svm():
    param_grid = {
        'C': [0.1, 1, 10],
        'gamma': [0.01, 0.1, 1],
        'kernel': ['rbf', 'linear']
    }
    
    svm = SVC(random_state=42)
    grid_search = GridSearchCV(svm, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    print("\nSVM网格搜索结果:")
    print("最优参数组合：", grid_search.best_params_)
    print("最优模型验证准确率：", grid_search.best_score_)
    
    # 测试集评估
    best_svm = grid_search.best_estimator_
    y_pred = best_svm.predict(X_test)
    print(f"测试集准确率：{accuracy_score(y_test, y_pred):.2f}")
    print("\n分类报告:")
    print(classification_report(y_test, y_pred))
    
    return grid_search

svm_grid_search = grid_search_svm()

# 5. 扩展任务
## 5.1 随机搜索对比
def random_search_svm():
    param_dist = {
        'C': uniform(0.1, 10),
        'gamma': uniform(0.01, 1),
        'kernel': ['rbf', 'linear']
    }
    
    svm = SVC(random_state=42)
    random_search = RandomizedSearchCV(svm, param_dist, n_iter=20, cv=5, 
                                     scoring='accuracy', random_state=42, n_jobs=-1)
    random_search.fit(X_train, y_train)
    
    print("\nSVM随机搜索结果:")
    print("最优参数组合：", random_search.best_params_)
    print("最优模型验证准确率：", random_search.best_score_)
    
    # 测试集评估
    best_svm = random_search.best_estimator_
    y_pred = best_svm.predict(X_test)
    print(f"测试集准确率：{accuracy_score(y_test, y_pred):.2f}")
    
    return random_search

svm_random_search = random_search_svm()

## 5.2 多模型对比
def compare_models():
    models = {
        'SVM': SVC(kernel='linear', C=1, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42),
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42)
    }
    
    results = []
    for model_name, model in models.items():
        scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        test_acc = accuracy_score(y_test, y_pred)
        
        results.append({
            'Model': model_name,
            'CV Mean Accuracy': scores.mean(),
            'CV Std Accuracy': scores.std(),
            'Test Accuracy': test_acc
        })
    
    results_df = pd.DataFrame(results)
    print("\n多模型对比结果:")
    print(results_df)
    
    # 可视化
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Model', y='CV Mean Accuracy', data=results_df)
    plt.title('不同模型交叉验证平均准确率比较')
    plt.show()
    
    return results_df

model_comparison = compare_models()