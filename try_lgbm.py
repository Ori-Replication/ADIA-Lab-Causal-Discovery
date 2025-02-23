import marimo

__generated_with = "0.9.1"
app = marimo.App()


@app.cell
def __():
    import pandas as pd
    import featuretools as ft
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import LabelEncoder
    from sklearn.model_selection import GroupKFold, cross_val_score
    import numpy as np
    from tpot import TPOTClassifier
    from boruta import BorutaPy
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.metrics import balanced_accuracy_score
    import joblib
    return (
        BorutaPy,
        GroupKFold,
        LabelEncoder,
        PolynomialFeatures,
        RandomForestClassifier,
        TPOTClassifier,
        balanced_accuracy_score,
        cross_val_score,
        ft,
        joblib,
        np,
        pd,
        train_test_split,
    )


@app.cell
def __(LabelEncoder, display, np, pd):
    X_y_group_train = pd.read_csv('mid_data/X_y_group_train_updated_v10.2_copula.csv')

    print("Adding numeric labels y")
    le = LabelEncoder()
    X_y_group_train["y"] = le.fit_transform(X_y_group_train["label"])
    # reordering columns:
    X_y_group_train = X_y_group_train[["dataset", "variable"] + X_y_group_train.columns.drop(["dataset", "variable", "label", "y"]).tolist() + ["label", "y"]]


    blacklist = ["ttest(v,X)", "pvalue(ttest(v,X))<=0.05", "ttest(v,Y)", "pvalue(ttest(v,Y))<=0.05", "ttest(X,Y)", "pvalue(ttest(X,Y))<=0.05"]
    columns_to_drop = [col for col in blacklist if col in X_y_group_train.columns]
    X_y_group_train = X_y_group_train.drop(columns=columns_to_drop)

    #去掉所有含'dcor'的列
    X_y_group_train = X_y_group_train.drop(columns=[col for col in X_y_group_train.columns if 'dcor_' in col])


    numeric_columns = X_y_group_train.select_dtypes(include=[np.number]).columns
    X_y_group_train[numeric_columns] = X_y_group_train[numeric_columns].fillna(X_y_group_train[numeric_columns].mean())

    display(X_y_group_train)

    print("Extracting X_train, y_train, and group")
    X_train = X_y_group_train.drop(["variable", "dataset", "label", "y"], axis="columns")

    y_train = X_y_group_train["y"]
    group_train = X_y_group_train["dataset"]
    return (
        X_train,
        X_y_group_train,
        blacklist,
        columns_to_drop,
        group_train,
        le,
        numeric_columns,
        y_train,
    )


@app.cell
def __(X_train, clean_feature_names, train_test_split, y_train):
    import re

    def _clean_feature_names(X):

        def clean_name(name):
            name = re.sub('[^\\w\\s-]', '_', name)
            if name[0].isdigit():
                name = 'f_' + name
            return name
        X.columns = [clean_name(col) for col in X.columns]
        return X
    X_train_1 = clean_feature_names(X_train)
    (X_train_1, X_test, y_train_1, y_test) = train_test_split(X_train_1, y_train, test_size=0.2, random_state=42)
    return X_test, X_train_1, re, y_test, y_train_1


@app.cell
def __(mo):
    mo.md(
        r"""
        Best LGBM Pharamter now:
        ```python
        model = LGBMClassifier(
            n_estimators=2000,
            learning_rate=0.05,
            max_depth=7,
            num_leaves=29,
            min_child_samples=20,
            subsample=0.7,
            colsample_bytree=0.7,
            reg_alpha=0.1,
            reg_lambda=0.1,
            random_state=42,
            n_jobs=-1,
            class_weight='balanced',
            device='gpu',
            gpu_platform_id=1,
            gpu_device_id=0,
        )
        ```

        Score: 0.5935


        """
    )
    return


@app.cell
def __(
    X_test,
    X_train_1,
    balanced_accuracy_score_1,
    display,
    model,
    y_test,
    y_train_1,
    y_train_pred,
):
    import lightgbm as lgb
    from sklearn.metrics import balanced_accuracy_score, classification_report
    from lightgbm import LGBMClassifier
    class_weights = {0: 10, 1: 10, 2: 20, 3: 15, 4: 10, 5: 10, 6: 10, 7: 20}
    _model = LGBMClassifier(n_estimators=2000, learning_rate=0.05, max_depth=7, num_leaves=29, min_child_samples=20, subsample=0.7, colsample_bytree=0.7, reg_alpha=0.1, reg_lambda=0.1, random_state=42, n_jobs=-1, class_weight='balanced', device='cpu')
    display(_model)
    _callbacks = [lgb.log_evaluation(period=1), lgb.early_stopping(stopping_rounds=10)]
    _model.fit(X_train_1, y_train_1, callbacks=_callbacks, eval_set=[(X_test, y_test)])
    _y_train_pred = model.predict(X_train_1)
    y_test_pred = model.predict(X_test)
    _train_score = balanced_accuracy_score_1(y_train_1, y_train_pred)
    _test_score = balanced_accuracy_score_1(y_test, y_test_pred)
    print(f'Train balanced accuracy: {_train_score:.4f}')
    print(f'Test balanced accuracy: {_test_score:.4f}')
    print('\nClassification Report for Test Set:')
    print(classification_report(y_test, y_test_pred))
    return (
        LGBMClassifier,
        balanced_accuracy_score,
        class_weights,
        classification_report,
        lgb,
        y_test_pred,
    )


@app.cell
def __(classification_report_1, le, pd, y_test, y_test_pred):
    from sklearn.metrics import confusion_matrix, classification_report
    import seaborn as sns
    import matplotlib.pyplot as plt
    y_true = y_test
    y_pred = y_test_pred
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()
    report = classification_report_1(y_true, y_pred, target_names=le.classes_, output_dict=True)
    df_report = pd.DataFrame(report).transpose()
    plt.figure(figsize=(12, 8))
    sns.heatmap(df_report.iloc[:-3, :-1].astype(float), annot=True, cmap='YlGnBu')
    plt.title('Classification Report Heatmap')
    plt.show()
    misclassified = y_true != y_pred
    error_df = pd.DataFrame({'True': le.inverse_transform(y_true[misclassified]), 'Predicted': le.inverse_transform(y_pred[misclassified])})
    error_counts = error_df.groupby(['True', 'Predicted']).size().unstack(fill_value=0)
    plt.figure(figsize=(12, 8))
    sns.heatmap(error_counts, annot=True, fmt='d', cmap='YlOrRd')
    plt.title('Misclassification Heatmap')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()
    print('Most common misclassifications:')
    print(error_df.groupby(['True', 'Predicted']).size().sort_values(ascending=False).head(10))
    return (
        classification_report,
        cm,
        confusion_matrix,
        df_report,
        error_counts,
        error_df,
        misclassified,
        plt,
        report,
        sns,
        y_pred,
        y_true,
    )


@app.cell
def __(X_train_new, clean_feature_names, re):
    def _clean_feature_names(X):

        def clean_name(name):
            name = re.sub('[^\\w\\s-]', '_', name)
            if name[0].isdigit():
                name = 'f_' + name
            return name
        X.columns = [clean_name(col) for col in X.columns]
        return X
    X_train_new = clean_feature_names(X_train_new)
    return (X_train_new,)


@app.cell
def __(
    LGBMClassifier,
    X_train_new,
    balanced_accuracy_score_1,
    classification_report_1,
    display,
    lgb,
    model,
    train_test_split,
    y_train_1,
    y_train_pred,
):
    _model = LGBMClassifier(n_estimators=2000, learning_rate=0.05, max_depth=7, num_leaves=29, min_child_samples=20, subsample=0.7, colsample_bytree=0.7, reg_alpha=0.1, reg_lambda=0.1, random_state=42, n_jobs=-1, class_weight='balanced', device='gpu', gpu_platform_id=1, gpu_device_id=0)
    display(_model)
    (X_train_new_1, X_test_1, y_train_2, y_test_1) = train_test_split(X_train_new, y_train_1, test_size=0.2, random_state=42)
    _callbacks = [lgb.log_evaluation(period=1), lgb.early_stopping(stopping_rounds=10)]
    _model.fit(X_train_new_1, y_train_2, callbacks=_callbacks, eval_set=[(X_test_1, y_test_1)])
    _y_train_pred = model.predict(X_train_new_1)
    y_test_pred_1 = model.predict(X_test_1)
    _train_score = balanced_accuracy_score_1(y_train_2, y_train_pred)
    _test_score = balanced_accuracy_score_1(y_test_1, y_test_pred_1)
    print(f'Train balanced accuracy: {_train_score:.4f}')
    print(f'Test balanced accuracy: {_test_score:.4f}')
    print('\nClassification Report for Test Set:')
    print(classification_report_1(y_test_1, y_test_pred_1))
    return X_test_1, X_train_new_1, y_test_1, y_test_pred_1, y_train_2


@app.cell
def __():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()

