"""
Treina e compara 3 modelos de classificacao para prever churn, usando as
features sem vazamento de dados (ver feature_engineering.py), e salva o
melhor em models/modelo_churn.pkl.
"""
import os

import joblib
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report, f1_score,
                              roc_auc_score)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler

BASE = os.path.dirname(os.path.dirname(__file__))
MODELS = os.path.join(BASE, "models")
os.makedirs(MODELS, exist_ok=True)

COLS_NUM = ["total_compras", "qtd_compras", "ticket_medio", "dias_desde_ultima_compra",
            "dias_desde_primeira_compra", "categorias_distintas", "dias_desde_cadastro"]


def treinar(features: pd.DataFrame):
    X = features.drop(columns=["cliente_id", "churn"])
    y = features["churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    # Scaler ajustado SO no treino (diferente do P13/P14) - evita que
    # estatisticas do conjunto de teste vazem para o pre-processamento.
    scaler = StandardScaler()
    X_train = X_train.copy()
    X_test = X_test.copy()
    X_train[COLS_NUM] = scaler.fit_transform(X_train[COLS_NUM])
    X_test[COLS_NUM] = scaler.transform(X_test[COLS_NUM])

    modelos = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=200, random_state=42),
    }

    resultados = []
    relatorios = {}
    for nome, modelo in modelos.items():
        cv_f1 = cross_val_score(modelo, X_train, y_train, cv=5, scoring="f1").mean()
        modelo.fit(X_train, y_train)
        y_pred = modelo.predict(X_test)
        y_proba = modelo.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba)

        resultados.append({"modelo": nome, "accuracy": acc, "f1": f1, "roc_auc": auc, "cv_f1_mean": cv_f1})
        relatorios[nome] = classification_report(y_test, y_pred)

        print("=" * 60)
        print(nome)
        print(f"accuracy={acc:.4f}  f1={f1:.4f}  roc_auc={auc:.4f}  cv_f1_mean={cv_f1:.4f}")
        print(relatorios[nome])

    df_res = pd.DataFrame(resultados).sort_values("f1", ascending=False).reset_index(drop=True)
    print("RANKING DE MODELOS (por F1):")
    print(df_res.to_string(index=False))

    melhor_nome = df_res.iloc[0]["modelo"]
    melhor_modelo = modelos[melhor_nome]

    joblib.dump({"modelo": melhor_modelo, "scaler": scaler, "colunas": list(X.columns), "cols_num": COLS_NUM},
                os.path.join(MODELS, "modelo_churn.pkl"))
    print(f"\nModelo salvo: {melhor_nome} (F1={df_res.iloc[0]['f1']:.4f}) -> models/modelo_churn.pkl")

    return df_res, relatorios, (X_test, y_test)


if __name__ == "__main__":
    features = pd.read_csv(os.path.join(BASE, "data", "features.csv"))
    treinar(features)
