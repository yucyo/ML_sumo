import matplotlib.pyplot as plt
import japanize_matplotlib
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

y = df["real_rent"]
X = df.drop(['real_rent',"name"], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.33, random_state=0)

lgb_train = lgb.Dataset(X_train, y_train)
lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)

lgbm_params = {
        'objective': 'regression',
        'metric': 'rmse',
        'num_leaves':80
}

model = lgb.train(lgbm_params, lgb_train, valid_sets=lgb_eval, verbose_eval=-1)

y_pred = model.predict(X_test, num_iteration=model.best_iteration)

print(r2_score(y_test, y_pred)  )
lgb.plot_importance(model, figsize=(12, 6))
plt.show()
