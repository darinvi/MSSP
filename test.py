import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from mssp import MSSP

if __name__ == '__main__':
    data = load_diabetes()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['target'] = data.target
    X_train, X_test, y_train, y_test = train_test_split(df.drop(columns=['target']).values, df['target'].values, test_size=0.2, random_state=42)
    model = MSSP(
        X_train, 
        y_train, 
        n_children=3, 
        n_levels=10, 
        n_solutions=[100] * 2 + [25] * 2 + [10] * 2 + [5] * 2 + [2] * 2,  # as the complexity gets higher, consider less and less solutions
        allow_synergies=True, 
        n_jobs=-1, 
        early_stopping=True
    )
    model.fit()
    print(model.evaluate(X_test, y_test, 'mape'))