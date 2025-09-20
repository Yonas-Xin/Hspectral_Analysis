 # 如果要使用增强数据进行训练与预测，提前将原始影像进行增强后保存
import sys, os
sys.path.append('.')
from core import Hyperspectral_Image
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, cohen_kappa_score
import pickle

def train_random_forest(X_train, y_train, X_test, y_test, config_name = 'rf_model', use_grid_search=False):
    if use_grid_search:
        print("正在进行网格搜索寻找最佳超参数...")
        
        # 定义超参数网格
        param_grid = {
            'n_estimators': [50, 100, 200],          # 树的数量
            'max_depth': [None, 10, 20, 30],         # 树的最大深度
            'min_samples_split': [2, 5, 10],         # 内部节点再划分所需最小样本数
            'min_samples_leaf': [1, 2, 4],           # 叶子节点最少样本数
            'max_features': ['sqrt', 'log2', None]   # 每次分割时考虑的特征数量
        }
        base_rf = RandomForestClassifier(random_state=42, n_jobs=-1)
        grid_search = GridSearchCV(
            estimator=base_rf,
            param_grid=param_grid,
            cv=3,               # 3折交叉验证
            scoring='accuracy', # 使用准确率作为评估指标
            n_jobs=-1,          # 使用所有可用的CPU核心
            verbose=1           # 显示进度
        )
        grid_search.fit(X_train, y_train)
        print("最佳超参数:", grid_search.best_params_)
        print("最佳交叉验证分数:", grid_search.best_score_)
        clf = grid_search.best_estimator_
        
    else:
        # 使用默认参数
        print("使用默认参数训练模型...")
        clf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
        clf.fit(X_train, y_train)
        
    # Make predictions and calculate metrics
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, digits=4)
    matrix = confusion_matrix(y_test, y_pred)
    kappa = cohen_kappa_score(y_test, y_pred)

    # Save model to pickle file
    pkl_name = config_name + '.pkl'
    with open(pkl_name, 'wb') as f:
        pickle.dump(clf, f)

    # Save results to txt file
    txt_name = config_name + '.txt'
    with open(txt_name, 'w') as f:
        f.write(f"Test Accuracy: {acc:.4f}\n")
        f.write(f"Cohen's Kappa: {kappa:.4f}\n")
        f.write("Classification Report:\n")
        f.write(report + "\n\n")
        f.write("Confusion Matrix:\n")
        f.write(np.array2string(matrix, separator=', '))

    # Print results to console (optional)
    print("Results saved to rf_results.txt")
    print("Test Accuracy:", acc)
    print(f"Cohen's Kappa: {kappa:.4f}")
    print("Classification Report:\n", report)
    print("Confusion Matrix:\n", matrix)

    return clf

def train_svm_model(X_train, y_train, X_test, y_test, config_name = 'svm_model', use_grid_search=False):
    if use_grid_search:
        print("正在进行SVM网格搜索寻找最佳超参数...")
        param_grid = {
            'C': [0.1, 1, 10, 100],                  # 正则化参数
            'gamma': ['scale', 'auto', 0.01, 0.1, 1], # 核函数系数
            'kernel': ['rbf', 'linear', 'poly'],     # 核函数类型
            'degree': [2, 3, 4],                     # 多项式核的度数（仅对poly核有效）
            'class_weight': [None, 'balanced']       # 类别权重
        }
        # 对于大数据集，可以简化参数网格以减少计算时间
        if X_train.shape[0] > 10000:
            print("数据量较大，使用简化的参数网格...")
            param_grid = {
                'C': [0.1, 1, 10],
                'gamma': ['scale', 'auto', 0.1],
                'kernel': ['rbf', 'linear']
            }
        base_svm = SVC(random_state=42)
        grid_search = GridSearchCV(
            estimator=base_svm,
            param_grid=param_grid,
            cv=3,               # 3折交叉验证
            scoring='accuracy', # 使用准确率作为评估指标
            n_jobs=-1,          # 使用所有可用的CPU核心
            verbose=1,          # 显示进度
            error_score='raise' # 遇到错误时抛出异常
        )
        grid_search.fit(X_train, y_train)
        print("最佳超参数:", grid_search.best_params_)
        print("最佳交叉验证分数:", grid_search.best_score_)
        clf = grid_search.best_estimator_
    else:
        # 使用默认参数
        print("使用默认参数训练SVM模型...")
        clf = SVC(kernel='rbf', C=10, gamma='scale', random_state=42)
        clf.fit(X_train, y_train)
    # Make predictions and calculate metrics
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, digits=4)
    matrix = confusion_matrix(y_test, y_pred)
    kappa = cohen_kappa_score(y_test, y_pred)

    # Save model to pickle file
    pkl_name = config_name + '.pkl'
    with open(pkl_name, 'wb') as f:
        pickle.dump(clf, f)

    # Save results to txt file
    txt_name = config_name + '.txt'
    with open(txt_name, 'w') as f:
        f.write(f"Test Accuracy: {acc:.4f}\n")
        f.write(f"Cohen's Kappa: {kappa:.4f}\n")
        f.write("Classification Report:\n")
        f.write(report + "\n\n")
        f.write("Confusion Matrix:\n")
        f.write(np.array2string(matrix, separator=', '))

    # Print results to console (optional)
    print("Results saved to rf_results.txt")
    print("Test Accuracy:", acc)
    print(f"Cohen's Kappa: {kappa:.4f}")
    print("Classification Report:\n", report)
    print("Confusion Matrix:\n", matrix)

    return clf

input_tif = r'C:\Users\85002\OneDrive - cugb.edu.cn\项目数据\张川铀资源\ZY_result\Image\research_area1.dat'
train_shp_dir = r'c:\Users\85002\OneDrive\文档\小论文\dataset\dataset_50'
test_shp_dir = r'c:\Users\85002\OneDrive\文档\小论文\dataset\dataset_100测试集'
output_path = r'svm_model_new'

func = "SVM" # 'RF' or 'SVM'
use_grid_search = True # 是否使用网格搜索
if __name__ == '__main__':
    img = Hyperspectral_Image()
    img.init(input_tif, init_fig=False)
    train_position = img.create_mask_from_mutivector(train_shp_dir)
    X_train, y_train = img.read_dataset_from_mask(train_position)

    test_position = img.create_mask_from_mutivector(test_shp_dir)
    X_test, y_test = img.read_dataset_from_mask(test_position)
    
    if func == "SVM":
        clf_rf = train_svm_model(X_train, y_train, X_test, y_test, config_name=output_path, use_grid_search=use_grid_search)
    elif func == "RF":
        clf_rf = train_random_forest(X_train, y_train, X_test, y_test, config_name=output_path, use_grid_search=use_grid_search)
    else:
        raise ValueError("Unsupported model type. Choose 'RF' or 'SVM'.")