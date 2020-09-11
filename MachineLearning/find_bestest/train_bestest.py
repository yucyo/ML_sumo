# coding: utf-8
import matplotlib.pyplot as plt
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

y = df["賃料料+管理費"]
X = df.drop(['賃料料+管理費',"マンション名"], axis=1)

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

pred = list(model.predict(X, num_iteration=model.best_iteration))
pred = pd.Series(pred, name="予測値")
diff = pd.Series(df["賃料料+管理費"]-pred,name="予測値との差")
df_search = pd.concat([df_for_search,diff,pred], axis=1)
df_search = df_search.sort_values("予測値との差")
df_search = df_search[["マンション名",'賃料料+管理費', '予測値',  '予測値との差', '詳細URL']]
df_search.to_csv('otoku.csv', sep = '\t',encoding='utf-16')
