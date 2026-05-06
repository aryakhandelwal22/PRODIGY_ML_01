import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.impute import SimpleImputer

# 1. Load Data
df = pd.read_csv("train.csv")

# 2. Feature Engineering
df["TotalBath"] = df["FullBath"] + 0.5 * df["HalfBath"]
df["HouseAge"]  = df["YrSold"] - df["YearBuilt"]
df["TotalPorch"] = (
    df["OpenPorchSF"] + df["EnclosedPorch"] +
    df["3SsnPorch"]  + df["ScreenPorch"]
)

features = [
    "GrLivArea",
    "TotalBsmtSF",
    "1stFlrSF",
    "BedroomAbvGr",
    "TotalBath",
    "GarageCars",
    "OverallQual",
    "HouseAge",
    "TotalPorch",
    "LotArea",
]

X = df[features].copy()
y = df["SalePrice"]

# 3. Handle Missing Values
imputer = SimpleImputer(strategy="median")
X = pd.DataFrame(imputer.fit_transform(X), columns=features)

# 4. Train / Test Split + Scaling
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

# 5. Train Models
lr    = LinearRegression()
ridge = Ridge(alpha=10)

lr.fit(X_train_sc, y_train)
ridge.fit(X_train_sc, y_train)

y_pred_lr    = lr.predict(X_test_sc)
y_pred_ridge = ridge.predict(X_test_sc)

# 6. Evaluation
def evaluate(name, y_true, y_pred):
    r2  = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    print("\n" + "-"*40)
    print(" " + name)
    print("-"*40)
    print("  R2 Score : {:.4f}".format(r2))
    print("  MAE      : ${:,.2f}".format(mae))

evaluate("Linear Regression", y_test, y_pred_lr)
evaluate("Ridge Regression",  y_test, y_pred_ridge)

# Cross-validation
cv_scores = cross_val_score(ridge, X_train_sc, y_train, cv=5, scoring="r2")
print("\n  Ridge 5-Fold CV R2: {:.4f} +/- {:.4f}".format(cv_scores.mean(), cv_scores.std()))

# 7. Custom House Prediction
custom = {
    "GrLivArea": 2000, "TotalBsmtSF": 800, "1stFlrSF": 1000,
    "BedroomAbvGr": 3, "TotalBath": 2.0,   "GarageCars": 2,
    "OverallQual": 7,  "HouseAge": 10,     "TotalPorch": 100,
    "LotArea": 8000,
}
custom_df = pd.DataFrame([custom])
custom_sc = scaler.transform(custom_df)
predicted  = ridge.predict(custom_sc)[0]
print("\n  Predicted Price (custom house): ${:,.2f}".format(predicted))

# 8. Plots
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("Task-01 Improved: House Price Prediction", fontsize=14, fontweight="bold")

# Plot 1 - Actual vs Predicted
axes[0].scatter(y_test, y_pred_ridge, alpha=0.5, color="steelblue", edgecolors="white", linewidth=0.3)
axes[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--", lw=2)
axes[0].set_xlabel("Actual Price")
axes[0].set_ylabel("Predicted Price")
axes[0].set_title("Actual vs Predicted (Ridge)")

# Plot 2 - Residuals
residuals = y_test.values - y_pred_ridge
axes[1].scatter(y_pred_ridge, residuals, alpha=0.5, color="coral", edgecolors="white", linewidth=0.3)
axes[1].axhline(0, color="black", lw=1.5, linestyle="--")
axes[1].set_xlabel("Predicted Price")
axes[1].set_ylabel("Residuals")
axes[1].set_title("Residual Plot")

# Plot 3 - Feature Importance
coefs = pd.Series(np.abs(ridge.coef_), index=features).sort_values()
coefs.plot(kind="barh", ax=axes[2], color="mediumseagreen")
axes[2].set_title("Feature Importance (|coef|)")
axes[2].set_xlabel("Absolute Coefficient")

plt.tight_layout()
plt.savefig("task1_improved_plots.png", dpi=150)
plt.show()
print("\nPlot saved as task1_improved_plots.png")