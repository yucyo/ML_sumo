pred = list(model.predict(X, num_iteration=model.best_iteration))
pred = pd.Series(pred, name="予測値")
diff = pd.Series(df["賃料料+管理費"]-pred,name="予測値との差")
df_search = pd.concat([df_for_search,diff,pred], axis=1)
df_search = df_search.sort_values("予測値との差")
df_search = df_search[["マンション名",'賃料料+管理費', '予測値',  '予測値との差', '詳細URL']]
df_search.to_csv('otoku.csv', sep = '\t',encoding='utf-16')
