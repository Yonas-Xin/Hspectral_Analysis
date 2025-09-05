import sys, os
base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_path)
from core import Hyperspectral_Image
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
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

input_tif = r'C:\Users\85002\OneDrive - cugb.edu.cn\项目数据\张川铀资源\ZY_result\Image\research_area1.dat'
train_shp_dir = r'c:\Users\85002\OneDrive\文档\小论文\dataset\dataset_50'
test_shp_dir = r'c:\Users\85002\OneDrive\文档\小论文\dataset\dataset_100测试集'
if __name__ == '__main__':
    config_name = 'rf_model' # 配置输出名称
    img = Hyperspectral_Image()
    img.init(input_tif)
    # img.image_enhance(f='PCA', n_components = 24) # PCA敬畏==降维
    train_position = img.create_mask_from_mutivector(train_shp_dir)
    data = img.get_dataset().transpose(1,2,0) # or img.enhance_data
    X_train = data[train_position > 0]
    y_train = train_position[train_position > 0] - 1 # 让标签从0开始

    test_position = img.create_mask_from_mutivector(test_shp_dir)
    X_test = data[test_position > 0]
    y_test = test_position[test_position > 0] - 1 # 让标签从0开始
    
    clf_rf = train_random_forest(X_train, y_train, X_test, y_test, use_grid_search=True)