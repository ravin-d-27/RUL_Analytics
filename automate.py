
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
def find_best_n_estimators(X_train, y_train, X_test, y_test, start, stop, step = 1, rs = 0):
    lst = []
    n_est = []
    for i in range(start, stop, step):
        model = RandomForestClassifier(criterion='entropy',n_estimators=i,random_state=rs)
        model.fit(X_train,y_train.ravel())
        print("Checking n_estimators = {} for Random Forest Classifier".format(i))

        y_pred = model.predict(X_test)
        lst.append(accuracy_score(y_test,y_pred))
        n_est.append(i)

    maxi = max(lst)
    ind = 0
    for i in range(len(lst)):
        if (lst[i]==maxi):
           ind = i
           break
    
    return [maxi, n_est[ind]]
     

