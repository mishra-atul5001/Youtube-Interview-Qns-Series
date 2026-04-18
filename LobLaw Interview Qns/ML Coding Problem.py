import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def binary_classification_ml_problem(df: pd.DataFrame):
    
    X = df.drop('target',axis = 1)
    y = df['target']
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2)
    
    classifier = RandomForestClassifier(n_estimators=50, max_depth=2)
    classifier.fit(X_train, y_train)
    
    y_pred = classifier.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {accuracy * 100:.2f}%')
    
    return classifier, accuracy
    
    
    