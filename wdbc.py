import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np

#1.讀入數據集
df = pd.read_csv('\Breast Cancer Wisconsin 02\wdbc.data',header=None)

#2.將後面30個特徵指派NumPy的陣列X。使用一個LabelEncoder物件，將原來類別標籤所使用的字串表示法(M和B)轉換成整數，M=「惡性」，B=「良性」。
X = df.loc[:, 2:].values
y = df.loc[:, 1].values
le = LabelEncoder()
y = le.fit_transform(y)
le.classes_
np.array(['B','M'],dtype=object)

#3.惡性腫瘤標示為1類，良性腫瘤標示為0類，呼叫LabelEncoder的transform方法，再次檢查「類別標籤字串」轉換成「對應數字」的工作是否正常完成
le.transform(['M','B'])
np.array([1,0])


#4.訓練數據集80%的數據，測試數據集20%的數據，建立第一個模型管線。
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = \
    train_test_split(X, y,
                test_size=0.20,
                stratify=y,
                random_state=1)

#5.結合轉換器和估計器到管線中
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

#6.用pipe_lr管線執行fit方法的時候，StandardScaler首先會對「訓練數據集」套用fit與transform函數
pipe_lr = make_pipeline(StandardScaler(),
                        PCA(n_components=2),
                        LogisticRegression(random_state=1,solver
                        ='lbfgs'))
pipe_lr.fit(X_train, y_train)
y_pred = pipe_lr.predict(X_test)
print('Test Accuracy: %.3f' % pipe_lr.score(X_test, y_test))
#Test Accuracy: 0.956

#7.分層K折交叉驗證法，設定n_fold參數，來指定「折」的數目。
import numpy as np
from sklearn.model_selection import StratifiedKFold
kfold = StratifiedKFold(n_splits=10).split(X_train, y_train)
scores = []
for k, (train, test) in enumerate(kfold):
    pipe_lr.fit(X_train[test], y_train[test])
    score = pipe_lr.score(X_train[test], y_train[test])
    scores.append(score)
    print('Fold: %2d, Class dist.: %s, Acc: %.3f' % (k+1, np.bincount(y_train[train]), score))
#Fold:  1, Class dist.: [256 153], Acc: 0.891
#Fold:  2, Class dist.: [256 153], Acc: 1.000
#Fold:  3, Class dist.: [256 153], Acc: 0.978
#Fold:  4, Class dist.: [256 153], Acc: 0.978
#Fold:  5, Class dist.: [256 153], Acc: 0.978
#Fold:  6, Class dist.: [257 153], Acc: 0.933
#Fold:  7, Class dist.: [257 153], Acc: 0.978
#Fold:  8, Class dist.: [257 153], Acc: 0.933
#Fold:  9, Class dist.: [257 153], Acc: 0.933
#Fold: 10, Class dist.: [257 153], Acc: 0.978

print('\nCV accuracy: %.3f +/- %.3f' %
        (np.mean(scores), np.std(scores)))
#CV accuracy: 0.958 +/- 0.032

#8.kfold，使用「計分器」方便利用分層k折交叉驗證法來評估我們的模型。方法是cross_val_score
from sklearn.model_selection import cross_val_score
scores = cross_val_score(estimator=pipe_lr,
                        X=X_train,
                        y=y_train,
                        cv=10,
                        n_jobs=1)
print('CV accuracy scores: %s' % scores)
#CV accuracy: 0.958 +/- 0.032
#CV accuracy scores: [0.93478261 0.93478261 0.95652174 
#                     0.95652174 0.93478261 0.95555556
#                     0.97777778 0.93333333 0.95555556 
#                     0.95555556]
print('CV accuracy scores: %.3f +/- %.3f' % (np.mean(scores),
    np.std(scores)))