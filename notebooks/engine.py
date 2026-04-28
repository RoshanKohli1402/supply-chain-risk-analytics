import pandas as pd

df = pd.read_csv('DataCoSupplyChainDataset.csv', encoding='latin-1')

# See all columns without truncation
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)
pd.set_option('display.width', 1000)
pd.set_option('display.colheader_justify', 'left')

# Check shape first
print(df.shape)
print(df.columns.tolist())
cols_to_drop = [
    'Customer Email',      # PII - useless for analysis
    'Customer Password',   # PII - should never be in a dataset
    'Customer Fname',      # PII
    'Customer Lname',      # PII
    'Customer Street',     # too granular
    'Customer Zipcode',    # too granular
    'Order Zipcode',       # too granular
    'Product Description', # unstructured text, not needed
    'Product Image',       # URL, useless
    'Product Card Id',     # duplicate of Product Category Id
    'Product Category Id', # duplicate of Category Id
    'Order Item Cardprod Id', # internal ID, redundant
    'Order Customer Id',   # duplicate of Customer Id
    'Latitude',            # too granular, we have region/country
    'Longitude',           # same
]

df.drop(columns=cols_to_drop, inplace=True)
print(df.shape)  # should be (39683, 38)
df.rename(columns={
    'Type': 'payment_type',
    'Days for shipping (real)': 'days_shipping_real',
    'Days for shipment (scheduled)': 'days_shipping_scheduled',
    'Benefit per order': 'benefit_per_order',
    'Sales per customer': 'sales_per_customer',
    'Delivery Status': 'delivery_status',
    'Late_delivery_risk': 'late_delivery_risk',
    'Category Id': 'category_id',
    'Category Name': 'category_name',
    'Customer City': 'customer_city',
    'Customer Country': 'customer_country',
    'Customer Id': 'customer_id',
    'Customer Segment': 'customer_segment',
    'Customer State': 'customer_state',
    'Department Id': 'department_id',
    'Department Name': 'department_name',
    'Market': 'market',
    'Order City': 'order_city',
    'Order Country': 'order_country',
    'order date (DateOrders)': 'order_date',
    'Order Id': 'order_id',
    'Order Item Discount': 'item_discount',
    'Order Item Discount Rate': 'item_discount_rate',
    'Order Item Id': 'order_item_id',
    'Order Item Product Price': 'item_product_price',
    'Order Item Profit Ratio': 'item_profit_ratio',
    'Order Item Quantity': 'item_quantity',
    'Sales': 'sales',
    'Order Item Total': 'item_total',
    'Order Profit Per Order': 'profit_per_order',
    'Order Region': 'order_region',
    'Order State': 'order_state',
    'Order Status': 'order_status',
    'Product Name': 'product_name',
    'Product Price': 'product_price',
    'Product Status': 'product_status',
    'shipping date (DateOrders)': 'shipping_date',
    'Shipping Mode': 'shipping_mode',
}, inplace=True)

print(df.columns.tolist())
# Null check
print("=== NULL CHECK ===")
print(df.isnull().sum()[df.isnull().sum() > 0])

# Data types
print("\n=== DATA TYPES ===")
print(df.dtypes)

# Quick sanity check on dates
print("\n=== DATE SAMPLES ===")
print(df['order_date'].head(3))
print(df['shipping_date'].head(3))
# 1. Drop the single null rows (only 1 each, not worth imputing)
df.dropna(inplace=True)
print(f"Rows after null drop: {len(df)}")  # should be ~39682

# 2. Fix date columns
df['order_date'] = pd.to_datetime(df['order_date'], format='%m/%d/%Y %H:%M')
df['shipping_date'] = pd.to_datetime(df['shipping_date'], format='%m/%d/%Y %H:%M')

# 3. Fix ID columns (should be int not float)
df['order_id'] = df['order_id'].astype(int)
df['order_item_id'] = df['order_item_id'].astype(int)
df['item_quantity'] = df['item_quantity'].astype(int)
df['product_status'] = df['product_status'].astype(int)

# 4. Extract useful date features
df['order_year'] = df['order_date'].dt.year
df['order_month'] = df['order_date'].dt.month
df['order_day'] = df['order_date'].dt.day
df['order_dayofweek'] = df['order_date'].dt.dayofweek

# 5. Create delay column (core of our analysis)
df['shipping_delay'] = df['days_shipping_real'] - df['days_shipping_scheduled']
# Positive = late, 0 = on time, Negative = early

# 6. Final check
print("\n=== FINAL DTYPES ===")
print(df.dtypes)
print(f"\n=== DELAY COLUMN SAMPLE ===")
print(df['shipping_delay'].describe())
import matplotlib.pyplot as plt
import seaborn as sns

# === 1. DELIVERY STATUS DISTRIBUTION ===
print("=== DELIVERY STATUS ===")
print(df['delivery_status'].value_counts())
print(f"\nLate delivery risk rate: {df['late_delivery_risk'].mean()*100:.1f}%")

# === 2. DELAY BY SHIPPING MODE ===
print("\n=== AVG DELAY BY SHIPPING MODE ===")
print(df.groupby('shipping_mode')['shipping_delay'].mean().sort_values(ascending=False))

# === 3. DELAY BY MARKET/REGION ===
print("\n=== AVG DELAY BY MARKET ===")
print(df.groupby('market')['shipping_delay'].mean().sort_values(ascending=False))

print("\n=== AVG DELAY BY ORDER REGION ===")
print(df.groupby('order_region')['shipping_delay'].mean().sort_values(ascending=False))

# === 4. DELAY BY DEPARTMENT ===
print("\n=== AVG DELAY BY DEPARTMENT ===")
print(df.groupby('department_name')['shipping_delay'].mean().sort_values(ascending=False))

# === 5. ORDER STATUS DISTRIBUTION ===
print("\n=== ORDER STATUS ===")
print(df['order_status'].value_counts())

# === 6. PROFIT ANALYSIS ===
print("\n=== PROFIT SUMMARY ===")
print(df['profit_per_order'].describe())
print(f"Orders with negative profit: {(df['profit_per_order'] < 0).sum()}")
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('Supply Chain Risk - EDA Overview', fontsize=16, fontweight='bold')

# 1. Delivery Status
delivery_counts = df['delivery_status'].value_counts()
axes[0,0].pie(delivery_counts, labels=delivery_counts.index, autopct='%1.1f%%',
              colors=['#e74c3c','#2ecc71','#3498db','#95a5a6'])
axes[0,0].set_title('Delivery Status Distribution')

# 2. Delay by Shipping Mode
shipping_delay = df.groupby('shipping_mode')['shipping_delay'].mean().sort_values(ascending=False)
axes[0,1].bar(shipping_delay.index, shipping_delay.values, color=['#e74c3c','#e67e22','#f1c40f','#2ecc71'])
axes[0,1].set_title('Avg Delay by Shipping Mode')
axes[0,1].set_ylabel('Days Delayed')
axes[0,1].tick_params(axis='x', rotation=15)

# 3. Delay by Market
market_delay = df.groupby('market')['shipping_delay'].mean().sort_values(ascending=False)
axes[0,2].barh(market_delay.index, market_delay.values, color='#3498db')
axes[0,2].set_title('Avg Delay by Market')
axes[0,2].set_xlabel('Days Delayed')

# 4. Top 5 Delayed Regions
region_delay = df.groupby('order_region')['shipping_delay'].mean().sort_values(ascending=False).head(5)
axes[1,0].barh(region_delay.index, region_delay.values, color='#e74c3c')
axes[1,0].set_title('Top 5 High Risk Regions')
axes[1,0].set_xlabel('Avg Delay (Days)')

# 5. Delay by Department
dept_delay = df.groupby('department_name')['shipping_delay'].mean().sort_values(ascending=False)
axes[1,1].barh(dept_delay.index, dept_delay.values, color='#9b59b6')
axes[1,1].set_title('Avg Delay by Department')
axes[1,1].set_xlabel('Avg Delay (Days)')

# 6. Profit Distribution
axes[1,2].hist(df['profit_per_order'], bins=50, color='#2ecc71', edgecolor='black')
axes[1,2].axvline(x=0, color='red', linestyle='--', linewidth=2, label='Break-even')
axes[1,2].set_title('Profit Per Order Distribution')
axes[1,2].set_xlabel('Profit')
axes[1,2].set_ylabel('Count')
axes[1,2].legend()

plt.tight_layout()
plt.savefig('eda_overview.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved as eda_overview.png")
# === FEATURE ENGINEERING ===

# 1. Late delivery binary flag (target variable)
df['is_late'] = (df['delivery_status'] == 'Late delivery').astype(int)

# 2. Delay ratio — how late vs scheduled
df['delay_ratio'] = df['shipping_delay'] / (df['days_shipping_scheduled'] + 1)

# 3. Discount impact flag
df['high_discount'] = (df['item_discount_rate'] > 0.2).astype(int)

# 4. Profit flag
df['is_loss'] = (df['profit_per_order'] < 0).astype(int)

# 5. Encode shipping mode risk
shipping_risk = {
    'Second Class': 3,
    'First Class': 2,
    'Same Day': 1,
    'Standard Class': 0
}
df['shipping_risk_score'] = df['shipping_mode'].map(shipping_risk)

# 6. Encode market
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['market_encoded'] = le.fit_transform(df['market'])
df['customer_segment_encoded'] = le.fit_transform(df['customer_segment'])
df['department_encoded'] = le.fit_transform(df['department_name'])
df['order_status_encoded'] = le.fit_transform(df['order_status'])

# 7. Order value tier
df['order_value_tier'] = pd.cut(df['sales'],
                                 bins=[0, 50, 150, 300, 10000],
                                 labels=[0, 1, 2, 3])
df['order_value_tier'] = df['order_value_tier'].astype(int)

# Final feature set
print("=== ENGINEERED FEATURES ===")
print(df[['is_late', 'delay_ratio', 'high_discount', 'is_loss',
          'shipping_risk_score', 'market_encoded',
          'order_value_tier', 'shipping_delay']].describe())

print(f"\nLate delivery rate in target: {df['is_late'].mean()*100:.1f}%")
# Clean feature set — no leakage
features = [
    'days_shipping_scheduled',  # what was PLANNED (ok to use)
    'shipping_risk_score',       # shipping mode risk
    'high_discount',             # discount flag
    'is_loss',                   # profit flag
    'market_encoded',            # market
    'customer_segment_encoded',  # customer segment
    'department_encoded',        # department
    'order_value_tier',          # order size
    'item_quantity',             # quantity
    'item_discount_rate',        # discount rate
    'item_profit_ratio',         # profit ratio
    'order_month',               # seasonality
    'order_dayofweek'            # day pattern
]

X = df[features]
y = df['is_late']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    auc = roc_auc_score(y_test, y_prob)
    results[name] = {'model': model, 'auc': auc, 'y_pred': y_pred}

    print(f"\n=== {name} ===")
    print(f"AUC-ROC: {auc:.4f}")
    print(classification_report(y_test, y_pred))
import shap

# Use XGBoost as final model
best_model = results['XGBoost']['model']

# SHAP explainer
explainer = shap.TreeExplainer(best_model)
shap_values = explainer.shap_values(X_test)

# === 1. FEATURE IMPORTANCE (SHAP) ===
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, X_test, plot_type="bar",
                  feature_names=features, show=False)
plt.title('Feature Importance - XGBoost (SHAP)', fontweight='bold')
plt.tight_layout()
plt.savefig('shap_importance.png', dpi=150, bbox_inches='tight')
plt.show()

# === 2. SHAP BEE SWARM ===
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, X_test,
                  feature_names=features, show=False)
plt.title('SHAP Value Distribution', fontweight='bold')
plt.tight_layout()
plt.savefig('shap_beeswarm.png', dpi=150, bbox_inches='tight')
plt.show()

print("Saved shap_importance.png and shap_beeswarm.png")
# === SUPPLIER RISK SCORING SYSTEM ===

# 1. Create supplier-level aggregations
supplier_risk = df.groupby('department_name').agg(
    total_orders=('order_id', 'count'),
    late_orders=('is_late', 'sum'),
    avg_delay=('shipping_delay', 'mean'),
    avg_profit=('profit_per_order', 'mean'),
    loss_orders=('is_loss', 'sum'),
    avg_discount_rate=('item_discount_rate', 'mean'),
    total_sales=('sales', 'sum')
).reset_index()

# 2. Calculate base metrics
supplier_risk['late_rate'] = supplier_risk['late_orders'] / supplier_risk['total_orders']
supplier_risk['loss_rate'] = supplier_risk['loss_orders'] / supplier_risk['total_orders']

# 3. Normalize each metric 0-1
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

supplier_risk['delay_score'] = scaler.fit_transform(supplier_risk[['avg_delay']])
supplier_risk['late_rate_score'] = scaler.fit_transform(supplier_risk[['late_rate']])
supplier_risk['loss_rate_score'] = scaler.fit_transform(supplier_risk[['loss_rate']])
supplier_risk['discount_score'] = scaler.fit_transform(supplier_risk[['avg_discount_rate']])

# 4. Composite risk score (weighted)
supplier_risk['risk_score'] = (
    supplier_risk['late_rate_score'] * 0.40 +   # biggest weight
    supplier_risk['delay_score'] * 0.30 +         # second biggest
    supplier_risk['loss_rate_score'] * 0.20 +     # financial risk
    supplier_risk['discount_score'] * 0.10        # discount pressure
) * 100  # scale to 0-100

# 5. Risk tier
supplier_risk['risk_tier'] = pd.cut(
    supplier_risk['risk_score'],
    bins=[0, 33, 66, 100],
    labels=['Low Risk', 'Medium Risk', 'High Risk']
)

# 6. Sort by risk
supplier_risk = supplier_risk.sort_values('risk_score', ascending=False)

print("=== SUPPLIER RISK SCORECARD ===")
print(supplier_risk[['department_name', 'total_orders', 'late_rate',
                       'avg_delay', 'risk_score', 'risk_tier']].to_string(index=False))
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle('Supplier Risk Scorecard', fontsize=16, fontweight='bold')

# 1. Risk Score Bar Chart
colors = ['#e74c3c' if t == 'High Risk'
          else '#f39c12' if t == 'Medium Risk'
          else '#2ecc71' for t in supplier_risk['risk_tier']]

axes[0].barh(supplier_risk['department_name'],
             supplier_risk['risk_score'], color=colors)
axes[0].set_xlabel('Risk Score (0-100)')
axes[0].set_title('Department Risk Score')
axes[0].axvline(x=66, color='red', linestyle='--', alpha=0.5, label='High Risk threshold')
axes[0].axvline(x=33, color='orange', linestyle='--', alpha=0.5, label='Medium Risk threshold')
axes[0].legend()

# 2. Late Rate vs Avg Delay Scatter
scatter_colors = ['#e74c3c' if t == 'High Risk'
                  else '#f39c12' if t == 'Medium Risk'
                  else '#2ecc71' for t in supplier_risk['risk_tier']]

scatter = axes[1].scatter(supplier_risk['late_rate'],
                           supplier_risk['avg_delay'],
                           c=scatter_colors, s=supplier_risk['total_orders']/10,
                           alpha=0.7, edgecolors='black')

for _, row in supplier_risk.iterrows():
    axes[1].annotate(row['department_name'],
                     (row['late_rate'], row['avg_delay']),
                     fontsize=8, ha='left', va='bottom')

axes[1].set_xlabel('Late Delivery Rate')
axes[1].set_ylabel('Avg Delay (Days)')
axes[1].set_title('Late Rate vs Delay (bubble = order volume)')

# Legend
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor='#e74c3c', label='High Risk'),
                   Patch(facecolor='#f39c12', label='Medium Risk'),
                   Patch(facecolor='#2ecc71', label='Low Risk')]
axes[1].legend(handles=legend_elements)

plt.tight_layout()
plt.savefig('supplier_risk_scorecard.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved supplier_risk_scorecard.png")
from prophet import Prophet

# === DEMAND FORECASTING ===
# Aggregate daily order volume
daily_orders = df.groupby('order_date').agg(
    y=('order_id', 'count'),
    revenue=('sales', 'sum')
).reset_index()

daily_orders.rename(columns={'order_date': 'ds'}, inplace=True)
daily_orders = daily_orders.sort_values('ds')

print(f"Date range: {daily_orders['ds'].min()} to {daily_orders['ds'].max()}")
print(f"Total days: {len(daily_orders)}")
print(daily_orders.head())

# === TRAIN PROPHET ===
model_prophet = Prophet(
    yearly_seasonality=True,
    weekly_seasonality=True,
    daily_seasonality=False,
    changepoint_prior_scale=0.05
)

model_prophet.fit(daily_orders[['ds', 'y']])

# === FORECAST 90 DAYS AHEAD ===
future = model_prophet.make_future_dataframe(periods=90)
forecast = model_prophet.predict(future)

# === PLOT ===
fig1 = model_prophet.plot(forecast)
plt.title('Demand Forecast - Next 90 Days', fontweight='bold')
plt.xlabel('Date')
plt.ylabel('Daily Orders')
plt.savefig('demand_forecast.png', dpi=150, bbox_inches='tight')
plt.show()

# === SEASONALITY ===
fig2 = model_prophet.plot_components(forecast)
plt.savefig('forecast_components.png', dpi=150, bbox_inches='tight')
plt.show()

print("Saved demand_forecast.png and forecast_components.png")
# Fix: strip time, keep date only
df['order_date_only'] = df['order_date'].dt.date

daily_orders = df.groupby('order_date_only').agg(
    y=('order_id', 'nunique'),  # unique orders per day
    revenue=('sales', 'sum')
).reset_index()

daily_orders.rename(columns={'order_date_only': 'ds'}, inplace=True)
daily_orders['ds'] = pd.to_datetime(daily_orders['ds'])
daily_orders = daily_orders.sort_values('ds')

print(f"Date range: {daily_orders['ds'].min()} to {daily_orders['ds'].max()}")
print(f"Total days: {len(daily_orders)}")
print(f"Avg daily orders: {daily_orders['y'].mean():.0f}")
print(daily_orders.head())

# Retrain Prophet
model_prophet = Prophet(
    yearly_seasonality=True,
    weekly_seasonality=True,
    daily_seasonality=False,
    changepoint_prior_scale=0.05
)

model_prophet.fit(daily_orders[['ds', 'y']])

future = model_prophet.make_future_dataframe(periods=90)
forecast = model_prophet.predict(future)

fig1 = model_prophet.plot(forecast)
plt.title('Demand Forecast - Next 90 Days', fontweight='bold')
plt.xlabel('Date')
plt.ylabel('Daily Orders')
plt.savefig('demand_forecast.png', dpi=150, bbox_inches='tight')
plt.show()

fig2 = model_prophet.plot_components(forecast)
plt.savefig('forecast_components.png', dpi=150, bbox_inches='tight')
plt.show()

print("Done")
# Run this at the bottom of your notebook
import joblib

# Save the model
joblib.dump(best_model, 'xgb_risk_model.pkl')
joblib.dump(scaler, 'minmax_scaler.pkl')

# Save clean dataset
df.to_csv('supply_chain_cleaned.csv', index=False)

# Save supplier risk scorecard
supplier_risk.to_csv('supplier_risk_scorecard.csv', index=False)

# Save forecast
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].to_csv('demand_forecast.csv', index=False)

print("All files saved ✅")
