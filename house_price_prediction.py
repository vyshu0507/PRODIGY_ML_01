import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────
# If you have the Kaggle CSV, place train.csv in the same folder and
# the script will load it automatically. Otherwise synthetic data is used.
try:
    df = pd.read_csv('train.csv')
    print("✅ Loaded Kaggle dataset")

    # Select relevant columns for this task
    features = ['GrLivArea', 'BedroomAbvGr', 'FullBath', 'HalfBath']
    target   = 'SalePrice'

    df = df[features + [target]].dropna()

    # Rename for clarity
    df.rename(columns={
        'GrLivArea':    'SquareFootage',
        'BedroomAbvGr': 'Bedrooms',
        'FullBath':     'FullBathrooms',
        'HalfBath':     'HalfBathrooms',
    }, inplace=True)
    df['Bathrooms'] = df['FullBathrooms'] + 0.5 * df['HalfBathrooms']
    df.drop(columns=['FullBathrooms', 'HalfBathrooms'], inplace=True)

except FileNotFoundError:
    print("ℹ️  train.csv not found – using synthetic data for demonstration")
    np.random.seed(42)
    n = 1000
    sqft     = np.random.randint(500, 5000, n)
    bedrooms = np.random.randint(1, 6, n)
    baths    = np.random.choice([1, 1.5, 2, 2.5, 3, 3.5], n)
    noise    = np.random.normal(0, 30000, n)
    price    = (sqft * 120 + bedrooms * 8000 + baths * 12000 + 50000 + noise).clip(50000)
    df = pd.DataFrame({
        'SquareFootage': sqft,
        'Bedrooms':      bedrooms,
        'Bathrooms':     baths,
        'SalePrice':     price
    })

print(f"\n📊 Dataset shape: {df.shape}")
print(df.describe().round(2))

# ─────────────────────────────────────────────
# 2. EXPLORATORY DATA ANALYSIS (EDA)
# ─────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
fig.suptitle('Feature Distributions', fontsize=14, fontweight='bold')

for ax, col in zip(axes, ['SquareFootage', 'Bedrooms', 'Bathrooms']):
    ax.hist(df[col], bins=30, color='steelblue', edgecolor='white', alpha=0.85)
    ax.set_title(col)
    ax.set_xlabel(col)
    ax.set_ylabel('Count')

plt.tight_layout()
plt.savefig('/mnt/user-data/outputs/01_feature_distributions.png', dpi=150, bbox_inches='tight')
plt.close()

# Scatter plots: each feature vs SalePrice
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
fig.suptitle('Features vs Sale Price', fontsize=14, fontweight='bold')

colors = ['steelblue', 'darkorange', 'seagreen']
for ax, col, c in zip(axes, ['SquareFootage', 'Bedrooms', 'Bathrooms'], colors):
    ax.scatter(df[col], df['SalePrice'], alpha=0.3, s=15, color=c)
    ax.set_xlabel(col)
    ax.set_ylabel('Sale Price ($)')
    ax.set_title(f'{col} vs Sale Price')
    # Trend line
    z = np.polyfit(df[col], df['SalePrice'], 1)
    p = np.poly1d(z)
    xline = np.linspace(df[col].min(), df[col].max(), 100)
    ax.plot(xline, p(xline), 'r--', lw=1.5, label='Trend')
    ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig('/mnt/user-data/outputs/02_feature_vs_price.png', dpi=150, bbox_inches='tight')
plt.close()

# Correlation heatmap
fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(df.corr(), annot=True, fmt='.2f', cmap='coolwarm', center=0,
            square=True, ax=ax, linewidths=0.5)
ax.set_title('Correlation Matrix', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('/mnt/user-data/outputs/03_correlation_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()

# ─────────────────────────────────────────────
# 3. PREPARE DATA
# ─────────────────────────────────────────────
X = df[['SquareFootage', 'Bedrooms', 'Bathrooms']]
y = df['SalePrice']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

print(f"\n✂️  Train size: {X_train.shape[0]}  |  Test size: {X_test.shape[0]}")

# ─────────────────────────────────────────────
# 4. TRAIN MODEL
# ─────────────────────────────────────────────
model = LinearRegression()
model.fit(X_train_sc, y_train)

print("\n📐 Model Coefficients:")
for feat, coef in zip(X.columns, model.coef_):
    print(f"   {feat:<20} {coef:>12,.2f}")
print(f"   {'Intercept':<20} {model.intercept_:>12,.2f}")

# ─────────────────────────────────────────────
# 5. EVALUATE MODEL
# ─────────────────────────────────────────────
y_pred = model.predict(X_test_sc)

mae  = mean_absolute_error(y_test, y_pred)
mse  = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2   = r2_score(y_test, y_pred)

print(f"\n📈 Model Performance:")
print(f"   MAE  : ${mae:>12,.2f}")
print(f"   RMSE : ${rmse:>12,.2f}")
print(f"   R²   :  {r2:>12.4f}  ({r2*100:.1f}% variance explained)")

# ─────────────────────────────────────────────
# 6. VISUALISE RESULTS
# ─────────────────────────────────────────────

# 6a. Actual vs Predicted
fig, ax = plt.subplots(figsize=(7, 6))
ax.scatter(y_test, y_pred, alpha=0.4, s=20, color='steelblue', label='Predictions')
lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
ax.plot(lims, lims, 'r--', lw=1.5, label='Perfect fit')
ax.set_xlabel('Actual Price ($)')
ax.set_ylabel('Predicted Price ($)')
ax.set_title('Actual vs Predicted House Prices', fontsize=13, fontweight='bold')
ax.legend()
ax.text(0.05, 0.92, f'R² = {r2:.3f}', transform=ax.transAxes,
        fontsize=10, color='darkred',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.8))
plt.tight_layout()
plt.savefig('/mnt/user-data/outputs/04_actual_vs_predicted.png', dpi=150, bbox_inches='tight')
plt.close()

# 6b. Residual plot
residuals = y_test - y_pred
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

axes[0].scatter(y_pred, residuals, alpha=0.4, s=18, color='darkorange')
axes[0].axhline(0, color='red', linestyle='--', lw=1.5)
axes[0].set_xlabel('Predicted Price ($)')
axes[0].set_ylabel('Residual ($)')
axes[0].set_title('Residuals vs Predicted', fontsize=12, fontweight='bold')

axes[1].hist(residuals, bins=40, color='darkorange', edgecolor='white', alpha=0.85)
axes[1].axvline(0, color='red', linestyle='--', lw=1.5)
axes[1].set_xlabel('Residual ($)')
axes[1].set_ylabel('Count')
axes[1].set_title('Residual Distribution', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig('/mnt/user-data/outputs/05_residual_analysis.png', dpi=150, bbox_inches='tight')
plt.close()

# 6c. Feature importance (coefficients)
fig, ax = plt.subplots(figsize=(7, 4))
coef_series = pd.Series(model.coef_, index=X.columns).sort_values()
colors_bar  = ['steelblue' if c > 0 else 'tomato' for c in coef_series]
coef_series.plot(kind='barh', ax=ax, color=colors_bar, edgecolor='white')
ax.axvline(0, color='black', lw=0.8)
ax.set_xlabel('Standardised Coefficient')
ax.set_title('Feature Importance (Standardised)', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('/mnt/user-data/outputs/06_feature_importance.png', dpi=150, bbox_inches='tight')
plt.close()

# ─────────────────────────────────────────────
# 7. PREDICT NEW HOUSES
# ─────────────────────────────────────────────
new_houses = pd.DataFrame({
    'SquareFootage': [1500, 2500, 3500],
    'Bedrooms':      [3,    4,    5   ],
    'Bathrooms':     [2,    2.5,  3   ],
})

new_sc     = scaler.transform(new_houses)
new_prices = model.predict(new_sc)

print("\n🏠 Predictions for new houses:")
print("-" * 55)
for i, (_, row) in enumerate(new_houses.iterrows()):
    print(f"   {int(row.SquareFootage)} sqft | {int(row.Bedrooms)} bed | "
          f"{row.Bathrooms} bath  →  ${new_prices[i]:,.0f}")
print("-" * 55)
print("\n✅ All plots saved to /mnt/user-data/outputs/")
