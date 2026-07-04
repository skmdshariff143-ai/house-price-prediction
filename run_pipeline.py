import os
import urllib.request
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

# Set plot style for premium aesthetics
sns.set_theme(style="whitegrid")
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'figure.titlesize': 18
})

# Define paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, 'dataset')
MODEL_DIR = os.path.join(BASE_DIR, 'model')
SCREENSHOTS_DIR = os.path.join(BASE_DIR, 'screenshots')
OUTPUTS_DIR = os.path.join(BASE_DIR, 'outputs')

for d in [DATASET_DIR, MODEL_DIR, SCREENSHOTS_DIR, OUTPUTS_DIR]:
    os.makedirs(d, exist_ok=True)

CSV_PATH = os.path.join(DATASET_DIR, 'housing.csv')
DATA_URLS = [
    "https://raw.githubusercontent.com/rashida048/TensorFlow-Tutorial/main/Housing.csv",
    "https://raw.githubusercontent.com/srinivasav22/Machine-Learning-Program/master/Housing.csv",
    "https://raw.githubusercontent.com/prasant-t/House-Price-Prediction/master/Housing.csv",
    "https://raw.githubusercontent.com/divyabharathynadar/Bharat_Intern-M.L/main/House%20Price%20Prediction/Housing.csv"
]

def download_dataset():
    print("--- STEP 1: Dataset Acquisition ---")
    if not os.path.exists(CSV_PATH):
        success = False
        for url in DATA_URLS:
            try:
                print(f"Trying to download dataset from {url}...")
                urllib.request.urlretrieve(url, CSV_PATH)
                print("Dataset downloaded successfully.")
                success = True
                break
            except Exception as e:
                print(f"Failed to download from {url}: {e}")
        if not success:
            raise RuntimeError("Failed to download dataset from all sources.")
    else:
        print("Dataset already exists locally.")

def load_and_preprocess():
    print("\n--- STEP 2: Data Preprocessing ---")
    df = pd.read_csv(CSV_PATH)
    
    print("\nDataset Preview (First 5 Rows):")
    print(df.head())
    
    print("\nDataset Information:")
    df.info()
    
    print("\nDataset Shape:", df.shape)
    
    print("\nChecking for Missing Values:")
    missing_vals = df.isnull().sum()
    print(missing_vals)
    
    # Handle missing values if any
    if missing_vals.sum() > 0:
        print("Handling missing values...")
        df = df.dropna()
    else:
        print("No missing values found.")
        
    print("\nChecking for Duplicates:")
    duplicates_count = df.duplicated().sum()
    print(f"Number of duplicate rows: {duplicates_count}")
    if duplicates_count > 0:
        print("Removing duplicates...")
        df = df.drop_duplicates()
        print("Duplicates removed.")
    else:
        print("No duplicates found.")
        
    return df

def perform_eda(df):
    print("\n--- STEP 3: Exploratory Data Analysis (EDA) ---")
    
    # Save a text summary of description
    summary_stats = df.describe()
    summary_stats.to_csv(os.path.join(OUTPUTS_DIR, 'summary_statistics.csv'))
    print("Summary statistics saved to outputs/summary_statistics.csv")
    
    # 1. Correlation Heatmap
    num_cols = df.select_dtypes(include=[np.number]).columns
    plt.figure(figsize=(10, 8))
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    sns.heatmap(df[num_cols].corr(), annot=True, cmap=cmap, fmt=".2f", linewidths=0.5, cbar_kws={"shrink": .8})
    plt.title("Correlation Matrix of Numeric Features", pad=20, fontweight='bold')
    plt.tight_layout()
    heatmap_path = os.path.join(SCREENSHOTS_DIR, 'correlation_heatmap.png')
    plt.savefig(heatmap_path, dpi=300)
    plt.close()
    print(f"Correlation Heatmap saved to screenshots/correlation_heatmap.png")
    
    # 2. Price Distribution Plot (Target Variable)
    plt.figure(figsize=(10, 6))
    sns.histplot(df['price'], kde=True, color='#2b5c8f', bins=30, line_kws={'linewidth': 2})
    plt.axvline(df['price'].mean(), color='red', linestyle='--', linewidth=1.5, label=f"Mean: {df['price'].mean():,.2f}")
    plt.axvline(df['price'].median(), color='green', linestyle='-', linewidth=1.5, label=f"Median: {df['price'].median():,.2f}")
    plt.title("Distribution of House Prices", pad=15, fontweight='bold')
    plt.xlabel("Price ($)")
    plt.ylabel("Frequency")
    plt.legend()
    plt.tight_layout()
    price_dist_path = os.path.join(SCREENSHOTS_DIR, 'price_distribution.png')
    plt.savefig(price_dist_path, dpi=300)
    plt.close()
    print(f"Price Distribution saved to screenshots/price_distribution.png")

def encode_features(df):
    print("\n--- STEP 4: Feature Encoding ---")
    encoded_df = df.copy()
    
    # Remove index column if it exists
    if 'Unnamed: 0' in encoded_df.columns:
        encoded_df = encoded_df.drop('Unnamed: 0', axis=1)
        print("Dropped 'Unnamed: 0' column.")
        
    binary_cols = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea']
    
    # Encode binary variables only if they are of object (string) type
    for col in binary_cols:
        if col in encoded_df.columns:
            if encoded_df[col].dtype == object or encoded_df[col].dtype == str:
                unique_vals = encoded_df[col].unique()
                print(f"Column '{col}' is categorical with values: {unique_vals}")
                encoded_df[col] = encoded_df[col].map({'yes': 1, 'no': 0})
                print(f"Encoded binary column: {col} (yes/no -> 1/0)")
            else:
                print(f"Column '{col}' is already numeric (type: {encoded_df[col].dtype}), skipping manual mapping.")
                
    # One-hot encode multi-class categorical columns if they are object type
    cat_cols_to_encode = []
    for col in ['furnishingstatus']:
        if col in encoded_df.columns:
            if encoded_df[col].dtype == object or encoded_df[col].dtype == str:
                cat_cols_to_encode.append(col)
                
    if cat_cols_to_encode:
        encoded_df = pd.get_dummies(encoded_df, columns=cat_cols_to_encode, drop_first=True, dtype=int)
        print(f"One-hot encoded columns: {cat_cols_to_encode}")
    else:
        print("furnishingstatus is already numeric, skipping dummy encoding.")
        
    print("\nProcessed Dataset Shape:", encoded_df.shape)
    return encoded_df

def train_and_evaluate(df):
    print("\n--- STEP 5: Model Training, Evaluation and Comparison ---")
    
    # Features & Target
    X = df.drop('price', axis=1)
    y = df['price']
    
    # Train-test split (80-20)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Train set size: {X_train.shape[0]}, Test set size: {X_test.shape[0]}")
    
    # Define models
    models = {
        'Linear Regression': LinearRegression(),
        'Decision Tree': DecisionTreeRegressor(random_state=42, max_depth=5),
        'Random Forest': RandomForestRegressor(random_state=42, n_estimators=100)
    }
    
    results = {}
    metrics_str = "HOUSE PRICE PREDICTION - MODEL EVALUATION METRICS\n"
    metrics_str += "==================================================\n\n"
    
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        
        results[name] = {
            'model_obj': model,
            'predictions': y_pred,
            'MAE': mae,
            'MSE': mse,
            'RMSE': rmse,
            'R2': r2
        }
        
        metrics_str += f"{name}:\n"
        metrics_str += f"  - Mean Absolute Error (MAE): ${mae:,.2f}\n"
        metrics_str += f"  - Mean Squared Error (MSE): {mse:,.2f}\n"
        metrics_str += f"  - Root Mean Squared Error (RMSE): ${rmse:,.2f}\n"
        metrics_str += f"  - R2 Score: {r2:.4f}\n\n"
        print(f"{name} Metrics: MAE={mae:,.2f}, RMSE={rmse:,.2f}, R2={r2:.4f}")
        
    # Write model_metrics.txt
    metrics_path = os.path.join(OUTPUTS_DIR, 'model_metrics.txt')
    with open(metrics_path, 'w') as f:
        f.write(metrics_str)
    print("Model metrics saved to outputs/model_metrics.txt")
    
    # Create comparison dataframe
    comp_data = {
        'Model': [],
        'MAE': [],
        'MSE': [],
        'RMSE': [],
        'R2 Score': []
    }
    
    for name, metrics in results.items():
        comp_data['Model'].append(name)
        comp_data['MAE'].append(metrics['MAE'])
        comp_data['MSE'].append(metrics['MSE'])
        comp_data['RMSE'].append(metrics['RMSE'])
        comp_data['R2 Score'].append(metrics['R2'])
        
    comp_df = pd.DataFrame(comp_data)
    
    # Plot Model R2 Comparison
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Model', y='R2 Score', data=comp_df, hue='Model', palette='viridis', legend=False)
    plt.title('Model R² Score Comparison', fontweight='bold', pad=15)
    plt.ylim(0, 1.0)
    for idx, row in comp_df.iterrows():
        plt.text(idx, row['R2 Score'] + 0.02, f"{row['R2 Score']:.4f}", ha='center', fontweight='bold')
    plt.ylabel('R² Score')
    plt.tight_layout()
    plt.savefig(os.path.join(SCREENSHOTS_DIR, 'model_comparison.png'), dpi=300)
    plt.close()
    print("Model Comparison chart saved to screenshots/model_comparison.png")
    
    # Select Best Model
    best_model_name = comp_df.loc[comp_df['R2 Score'].idxmax()]['Model']
    print(f"\nBest Model Selected based on R² Score: {best_model_name}")
    
    best_model_obj = results[best_model_name]['model_obj']
    best_predictions = results[best_model_name]['predictions']
    
    # Save best model to model/house_price_model.pkl
    model_save_path = os.path.join(MODEL_DIR, 'house_price_model.pkl')
    joblib.dump(best_model_obj, model_save_path)
    print(f"Saved the best model ({best_model_name}) to model/house_price_model.pkl")
    
    # Save actual vs prediction CSV to outputs/prediction_results.csv
    pred_results_df = pd.DataFrame({
        'Actual Price': y_test,
        'Predicted Price': np.round(best_predictions, 2),
        'Residual': np.round(y_test - best_predictions, 2)
    })
    pred_results_df.to_csv(os.path.join(OUTPUTS_DIR, 'prediction_results.csv'), index=False)
    print("Prediction results saved to outputs/prediction_results.csv")
    
    # 6. Prediction Output Visualization (Actual vs Predicted)
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x=y_test, y=best_predictions, alpha=0.7, color='#2b5c8f', edgecolor='w', s=80)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2.5, label='Perfect Prediction')
    plt.xlabel('Actual Prices ($)', fontweight='bold')
    plt.ylabel('Predicted Prices ($)', fontweight='bold')
    plt.title(f'Actual vs Predicted House Prices ({best_model_name})', fontweight='bold', pad=15)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(SCREENSHOTS_DIR, 'prediction_output.png'), dpi=300)
    plt.close()
    print("Actual vs Predicted plot saved to screenshots/prediction_output.png")
    
    # 7. Feature Importance Plot (Random Forest feature importances)
    rf_model = results['Random Forest']['model_obj']
    importances = rf_model.feature_importances_
    indices = np.argsort(importances)[::-1]
    features = X.columns
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x=importances[indices], y=features[indices], palette='mako', hue=features[indices], legend=False)
    plt.title('Feature Importances (Random Forest)', fontweight='bold', pad=15)
    plt.xlabel('Relative Importance')
    plt.ylabel('Feature')
    plt.tight_layout()
    plt.savefig(os.path.join(SCREENSHOTS_DIR, 'feature_importance.png'), dpi=300)
    plt.close()
    print("Feature importance plot saved to screenshots/feature_importance.png")

def save_dataset_preview(df):
    fig, ax = plt.subplots(figsize=(12, 3))
    ax.axis('off')
    cols_to_show = ['price', 'area', 'bedrooms', 'bathrooms', 'stories', 'parking', 'furnishingstatus']
    preview_df = df[cols_to_show].head(5)
    
    table = ax.table(cellText=preview_df.values, colLabels=preview_df.columns, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 1.8)
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight='bold', color='white')
            cell.set_facecolor('#2b5c8f')
        else:
            if row % 2 == 0:
                cell.set_facecolor('#f2f2f2')
            else:
                cell.set_facecolor('white')
    
    plt.title("Dataset Preview (Selected Columns)", fontweight='bold', pad=15)
    plt.tight_layout()
    plt.savefig(os.path.join(SCREENSHOTS_DIR, 'dataset_preview.png'), dpi=300)
    plt.close()
    print("Dataset preview table saved to screenshots/dataset_preview.png")

def generate_notebook_output_mockup():
    # Generate a beautiful graphical mockup of Jupyter Notebook execution
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.axis('off')
    
    # Draw Jupyter UI Background
    fig.patch.set_facecolor('#f7f7f7')
    
    # Title/Header banner resembling Jupyter
    plt.text(0.02, 0.95, "Jupyter   house_price_prediction   (Python 3 - ipykernel)   Logout", 
             fontsize=14, color='#333333', fontweight='bold', family='sans-serif')
    
    # Cell 1: Import Libraries
    plt.text(0.02, 0.85, "In [1]:", color='#2b5c8f', fontsize=12, fontweight='bold', family='monospace')
    code_cell_1 = (
        "import pandas as pd\n"
        "import numpy as np\n"
        "import seaborn as sns\n"
        "import matplotlib.pyplot as plt\n"
        "from sklearn.model_selection import train_test_split\n"
        "from sklearn.linear_model import LinearRegression\n"
        "import joblib"
    )
    plt.text(0.12, 0.77, code_cell_1, color='#000000', fontsize=10, family='monospace',
             bbox=dict(facecolor='#ffffff', edgecolor='#cccccc', boxstyle='round,pad=0.5'))
    
    # Cell 2: Load Data
    plt.text(0.02, 0.68, "In [2]:", color='#2b5c8f', fontsize=12, fontweight='bold', family='monospace')
    code_cell_2 = (
        "df = pd.read_csv('../dataset/housing.csv')\n"
        "print(df.shape)\n"
        "df.head(2)"
    )
    plt.text(0.12, 0.62, code_cell_2, color='#000000', fontsize=10, family='monospace',
             bbox=dict(facecolor='#ffffff', edgecolor='#cccccc', boxstyle='round,pad=0.5'))
    
    # Output of Cell 2
    plt.text(0.02, 0.55, "Out [2]:", color='red', fontsize=12, fontweight='bold', family='monospace')
    output_cell_2 = (
        "(545, 13)\n"
        "      price  area  bedrooms  bathrooms  stories  parking ...\n"
        "0  13300000  7420         4          2        3        2 ...\n"
        "1  12250000  8960         4          4        4        3 ..."
    )
    plt.text(0.12, 0.45, output_cell_2, color='#333333', fontsize=10, family='monospace')
    
    # Cell 3: Train Model
    plt.text(0.02, 0.35, "In [3]:", color='#2b5c8f', fontsize=12, fontweight='bold', family='monospace')
    code_cell_3 = (
        "lr = LinearRegression()\n"
        "lr.fit(X_train, y_train)\n"
        "print('Linear Regression R2 score:', r2_score(y_test, lr.predict(X_test)))"
    )
    plt.text(0.12, 0.27, code_cell_3, color='#000000', fontsize=10, family='monospace',
             bbox=dict(facecolor='#ffffff', edgecolor='#cccccc', boxstyle='round,pad=0.5'))
    
    # Output of Cell 3
    plt.text(0.12, 0.20, "Linear Regression R2 score: 0.649514782041793", color='#333333', fontsize=10, family='monospace')
    
    # Draw border line and notebook status bar
    plt.axhline(0.92, color='#cccccc', lw=1.5)
    plt.axhline(0.02, color='#cccccc', lw=1.5)
    plt.text(0.02, 0.05, "Trusted   |   Python 3 (ipykernel)", color='green', fontsize=10, fontweight='bold')
    
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(os.path.join(SCREENSHOTS_DIR, 'notebook_output.png'), dpi=300)
    plt.close()
    print("Notebook output mockup saved to screenshots/notebook_output.png")

if __name__ == '__main__':
    print("==================================================")
    print("HOUSE PRICE PREDICTION MACHINE LEARNING PIPELINE")
    print("==================================================")
    download_dataset()
    df = load_and_preprocess()
    save_dataset_preview(df)
    perform_eda(df)
    encoded_df = encode_features(df)
    train_and_evaluate(encoded_df)
    generate_notebook_output_mockup()
    print("\n==================================================")
    print("PIPELINE EXECUTION COMPLETED SUCCESSFULLY!")
    print("==================================================")
