"""
Baseline Models for Hotel Price Prediction
==========================================

This script implements baseline machine learning models for hotel price prediction:
1. Linear Regression (with regularization)
2. Random Forest Regression

These models serve as benchmarks for more advanced models and provide
interpretable results for understanding feature importance.

Models use:
- Log-transformed target (log_price_cad) for better distribution
- Preprocessed features with categorical encoding
- Cross-validation for robust evaluation
- Multiple metrics: RMSE, MAE, MAPE, R²

Author: ML Pipeline - Baseline Models
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import warnings
warnings.filterwarnings('ignore')

class BaselineModels:
    def __init__(self):
        """Initialize baseline models."""
        self.models = {}
        self.results = {}
        self.preprocessor = None
        
        print("🏗️ BASELINE MODELS FOR HOTEL PRICE PREDICTION")
        print("=" * 55)
        
    def load_splits(self):
        """Load train/validation/test splits."""
        print("📂 Loading train/validation/test splits...")
        
        try:
            # Load split data
            train_df = pd.read_csv('Data/train_split.csv')
            val_df = pd.read_csv('Data/validation_split.csv')
            test_df = pd.read_csv('Data/test_split.csv')
            
            print(f"   ✅ Loaded splits:")
            print(f"     • Train: {len(train_df):,} samples")
            print(f"     • Validation: {len(val_df):,} samples")
            print(f"     • Test: {len(test_df):,} samples")
            
            # Load split metadata
            import json
            with open('Data/split_info.json', 'r') as f:
                split_info = json.load(f)
            
            feature_columns = split_info['feature_columns']
            print(f"   📊 Features: {len(feature_columns)} columns")
            
            # Prepare data arrays
            X_train = train_df[feature_columns]
            y_train = train_df['log_price_cad']
            y_train_orig = train_df['price_cad']
            
            X_val = val_df[feature_columns]
            y_val = val_df['log_price_cad']
            y_val_orig = val_df['price_cad']
            
            X_test = test_df[feature_columns]
            y_test = test_df['log_price_cad']
            y_test_orig = test_df['price_cad']
            
            # Store data
            self.data = {
                'X_train': X_train, 'y_train': y_train, 'y_train_orig': y_train_orig,
                'X_val': X_val, 'y_val': y_val, 'y_val_orig': y_val_orig,
                'X_test': X_test, 'y_test': y_test, 'y_test_orig': y_test_orig,
                'feature_columns': feature_columns
            }
            
            print(f"   💰 Target: log_price_cad (log-transformed)")
            print(f"   📈 Target range: {y_train.min():.3f} to {y_train.max():.3f}")
            
            return True
        except Exception as e:
            print(f"   ❌ Error loading splits: {e}")
            return False
    
    def create_preprocessor(self):
        """Create preprocessing pipeline for features."""
        print(f"\n⚙️ CREATING PREPROCESSING PIPELINE")
        print("-" * 40)
        
        # Identify numeric and categorical columns
        X_sample = self.data['X_train']
        
        numeric_features = []
        categorical_features = []
        
        for col in X_sample.columns:
            if X_sample[col].dtype in ['int64', 'float64']:
                numeric_features.append(col)
            else:
                categorical_features.append(col)
        
        print(f"   📊 Feature Types:")
        print(f"     • Numeric features: {len(numeric_features)}")
        print(f"     • Categorical features: {len(categorical_features)}")
        
        # Create preprocessing steps
        numeric_transformer = StandardScaler()
        categorical_transformer = OneHotEncoder(drop='first', handle_unknown='ignore')
        
        # Combine preprocessors
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ],
            remainder='drop'
        )
        
        print(f"   ✅ Preprocessing pipeline created:")
        print(f"     • Numeric: StandardScaler")
        print(f"     • Categorical: OneHotEncoder (drop_first=True)")
        
        # Store feature info
        self.numeric_features = numeric_features
        self.categorical_features = categorical_features
    
    def build_models(self):
        """Build baseline model pipelines."""
        print(f"\n🏗️ BUILDING BASELINE MODELS")
        print("-" * 35)
        
        # Model configurations
        models_config = {
            'linear_regression': {
                'name': 'Linear Regression',
                'model': LinearRegression(),
                'description': 'Basic linear regression (OLS)'
            },
            'ridge_regression': {
                'name': 'Ridge Regression',
                'model': Ridge(alpha=1.0, random_state=42),
                'description': 'L2 regularized linear regression'
            },
            'lasso_regression': {
                'name': 'Lasso Regression', 
                'model': Lasso(alpha=0.1, random_state=42, max_iter=2000),
                'description': 'L1 regularized linear regression'
            },
            'elastic_net': {
                'name': 'ElasticNet Regression',
                'model': ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42, max_iter=2000),
                'description': 'L1 + L2 regularized linear regression'
            },
            'random_forest': {
                'name': 'Random Forest',
                'model': RandomForestRegressor(
                    n_estimators=50,  # Reduced for faster training
                    max_depth=10,     # Reduced depth
                    min_samples_split=10,
                    min_samples_leaf=5,
                    random_state=42,
                    n_jobs=-1
                ),
                'description': 'Ensemble of decision trees'
            }
        }
        
        # Create pipelines
        for model_key, config in models_config.items():
            pipeline = Pipeline([
                ('preprocessor', self.preprocessor),
                ('regressor', config['model'])
            ])
            
            self.models[model_key] = {
                'pipeline': pipeline,
                'name': config['name'],
                'description': config['description']
            }
            
            print(f"   ✅ {config['name']}: {config['description']}")
        
        print(f"\n   📋 Total models: {len(self.models)}")
    
    def calculate_metrics(self, y_true, y_pred, y_true_orig, y_pred_orig):
        """Calculate comprehensive evaluation metrics."""
        
        # Log-scale metrics
        rmse_log = np.sqrt(mean_squared_error(y_true, y_pred))
        mae_log = mean_absolute_error(y_true, y_pred)
        r2_log = r2_score(y_true, y_pred)
        
        # Original scale metrics
        rmse_orig = np.sqrt(mean_squared_error(y_true_orig, y_pred_orig))
        mae_orig = mean_absolute_error(y_true_orig, y_pred_orig)
        r2_orig = r2_score(y_true_orig, y_pred_orig)
        
        # MAPE (Mean Absolute Percentage Error)
        mape = np.mean(np.abs((y_true_orig - y_pred_orig) / y_true_orig)) * 100
        
        return {
            'rmse_log': rmse_log,
            'mae_log': mae_log,
            'r2_log': r2_log,
            'rmse_orig': rmse_orig,
            'mae_orig': mae_orig,
            'r2_orig': r2_orig,
            'mape': mape
        }
    
    def train_and_evaluate(self):
        """Train and evaluate all baseline models."""
        print(f"\n🎯 TRAINING AND EVALUATION")
        print("-" * 30)
        
        for model_key, model_info in self.models.items():
            print(f"\n   🔧 Training {model_info['name']}...")
            
            pipeline = model_info['pipeline']
            
            # Train the model
            pipeline.fit(self.data['X_train'], self.data['y_train'])
            
            # Make predictions
            y_train_pred = pipeline.predict(self.data['X_train'])
            y_val_pred = pipeline.predict(self.data['X_val'])
            
            # Convert back to original scale
            y_train_pred_orig = np.expm1(y_train_pred)
            y_val_pred_orig = np.expm1(y_val_pred)
            
            # Calculate metrics
            train_metrics = self.calculate_metrics(
                self.data['y_train'], y_train_pred,
                self.data['y_train_orig'], y_train_pred_orig
            )
            
            val_metrics = self.calculate_metrics(
                self.data['y_val'], y_val_pred,
                self.data['y_val_orig'], y_val_pred_orig
            )
            
            # Store results
            self.results[model_key] = {
                'name': model_info['name'],
                'pipeline': pipeline,
                'train_metrics': train_metrics,
                'val_metrics': val_metrics,
                'train_pred': y_train_pred,
                'val_pred': y_val_pred,
                'train_pred_orig': y_train_pred_orig,
                'val_pred_orig': y_val_pred_orig
            }
            
            print(f"     ✅ Training completed")
            print(f"     📊 Validation RMSE: ${val_metrics['rmse_orig']:.2f} CAD")
            print(f"     📊 Validation MAPE: {val_metrics['mape']:.2f}%")
            print(f"     📊 Validation R²: {val_metrics['r2_orig']:.3f}")
    
    def cross_validate_best_models(self):
        """Perform cross-validation on the best performing models."""
        print(f"\n🔄 CROSS-VALIDATION OF TOP MODELS")
        print("-" * 40)
        
        # Find top 3 models by validation R²
        val_r2_scores = {k: v['val_metrics']['r2_orig'] for k, v in self.results.items()}
        top_models = sorted(val_r2_scores.items(), key=lambda x: x[1], reverse=True)[:3]
        
        print(f"   🏆 Top 3 models by validation R²:")
        for i, (model_key, r2_score) in enumerate(top_models, 1):
            print(f"     {i}. {self.results[model_key]['name']}: {r2_score:.3f}")
        
        # Perform 5-fold cross-validation
        cv_results = {}
        
        for model_key, _ in top_models:
            print(f"\n   🔄 Cross-validating {self.results[model_key]['name']}...")
            
            pipeline = self.results[model_key]['pipeline']
            
            # Combine train and validation for CV
            X_combined = pd.concat([self.data['X_train'], self.data['X_val']], ignore_index=True)
            y_combined = pd.concat([self.data['y_train'], self.data['y_val']], ignore_index=True)
            
            # Perform cross-validation (reduced to 3-fold for speed)
            cv_scores = cross_val_score(
                pipeline, X_combined, y_combined,
                cv=3, scoring='r2', n_jobs=-1
            )
            
            cv_results[model_key] = {
                'mean_cv_score': cv_scores.mean(),
                'std_cv_score': cv_scores.std(),
                'cv_scores': cv_scores
            }
            
            print(f"     📊 CV R² Score: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
        
        self.cv_results = cv_results
    
    def analyze_feature_importance(self):
        """Analyze feature importance for tree-based models."""
        print(f"\n🔍 FEATURE IMPORTANCE ANALYSIS")
        print("-" * 35)
        
        # Find Random Forest model
        rf_model = None
        for model_key, model_info in self.results.items():
            if 'random_forest' in model_key:
                rf_model = model_info
                break
        
        if rf_model is None:
            print(f"   ⚠️ No Random Forest model found for feature importance")
            return
        
        # Get feature importance
        pipeline = rf_model['pipeline']
        
        # Get feature names after preprocessing
        preprocessor = pipeline.named_steps['preprocessor']
        
        # Fit preprocessor to get feature names
        preprocessor.fit(self.data['X_train'])
        
        # Get feature names
        try:
            feature_names = preprocessor.get_feature_names_out()
        except:
            # Fallback for older sklearn versions
            feature_names = [f'feature_{i}' for i in range(len(pipeline.named_steps['regressor'].feature_importances_))]
        
        # Get importance scores
        importance_scores = pipeline.named_steps['regressor'].feature_importances_
        
        # Create importance DataFrame
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance_scores
        }).sort_values('importance', ascending=False)
        
        print(f"   🏆 Top 10 Most Important Features:")
        for i, (_, row) in enumerate(importance_df.head(10).iterrows(), 1):
            print(f"     {i:2d}. {row['feature']:<30}: {row['importance']:.4f}")
        
        # Store for visualization
        self.feature_importance = importance_df
        
        return importance_df
    
    def create_performance_comparison(self):
        """Create comprehensive performance comparison."""
        print(f"\n📊 CREATING PERFORMANCE COMPARISON")
        print("-" * 40)
        
        # Create comparison DataFrame
        comparison_data = []
        
        for model_key, result in self.results.items():
            train_metrics = result['train_metrics']
            val_metrics = result['val_metrics']
            
            comparison_data.append({
                'Model': result['name'],
                'Train_RMSE_CAD': train_metrics['rmse_orig'],
                'Val_RMSE_CAD': val_metrics['rmse_orig'],
                'Train_MAPE_%': train_metrics['mape'],
                'Val_MAPE_%': val_metrics['mape'],
                'Train_R2': train_metrics['r2_orig'],
                'Val_R2': val_metrics['r2_orig'],
                'Overfit_Score': train_metrics['r2_orig'] - val_metrics['r2_orig']
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.sort_values('Val_R2', ascending=False)
        
        print(f"   📋 Model Performance Summary:")
        print(f"   {comparison_df.to_string(index=False, float_format='%.3f')}")
        
        # Create visualization
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Baseline Models Performance Comparison', fontsize=16, fontweight='bold')
        
        # Plot 1: RMSE Comparison
        x_pos = np.arange(len(comparison_df))
        width = 0.35
        
        axes[0, 0].bar(x_pos - width/2, comparison_df['Train_RMSE_CAD'], width, 
                      label='Train', alpha=0.8, color='skyblue')
        axes[0, 0].bar(x_pos + width/2, comparison_df['Val_RMSE_CAD'], width,
                      label='Validation', alpha=0.8, color='orange')
        
        axes[0, 0].set_xlabel('Models')
        axes[0, 0].set_ylabel('RMSE (CAD)')
        axes[0, 0].set_title('RMSE Comparison')
        axes[0, 0].set_xticks(x_pos)
        axes[0, 0].set_xticklabels(comparison_df['Model'], rotation=45, ha='right')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: MAPE Comparison
        axes[0, 1].bar(x_pos - width/2, comparison_df['Train_MAPE_%'], width,
                      label='Train', alpha=0.8, color='lightgreen')
        axes[0, 1].bar(x_pos + width/2, comparison_df['Val_MAPE_%'], width,
                      label='Validation', alpha=0.8, color='salmon')
        
        axes[0, 1].set_xlabel('Models')
        axes[0, 1].set_ylabel('MAPE (%)')
        axes[0, 1].set_title('MAPE Comparison')
        axes[0, 1].set_xticks(x_pos)
        axes[0, 1].set_xticklabels(comparison_df['Model'], rotation=45, ha='right')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: R² Comparison
        axes[1, 0].bar(x_pos - width/2, comparison_df['Train_R2'], width,
                      label='Train', alpha=0.8, color='gold')
        axes[1, 0].bar(x_pos + width/2, comparison_df['Val_R2'], width,
                      label='Validation', alpha=0.8, color='purple')
        
        axes[1, 0].set_xlabel('Models')
        axes[1, 0].set_ylabel('R² Score')
        axes[1, 0].set_title('R² Score Comparison')
        axes[1, 0].set_xticks(x_pos)
        axes[1, 0].set_xticklabels(comparison_df['Model'], rotation=45, ha='right')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Feature Importance (if available)
        if hasattr(self, 'feature_importance'):
            top_features = self.feature_importance.head(10)
            axes[1, 1].barh(range(len(top_features)), top_features['importance'], 
                           color='steelblue', alpha=0.7)
            axes[1, 1].set_yticks(range(len(top_features)))
            axes[1, 1].set_yticklabels(top_features['feature'], fontsize=9)
            axes[1, 1].set_xlabel('Importance Score')
            axes[1, 1].set_title('Top 10 Feature Importance (Random Forest)')
            axes[1, 1].grid(True, alpha=0.3)
        else:
            axes[1, 1].text(0.5, 0.5, 'Feature Importance\nNot Available', 
                           ha='center', va='center', transform=axes[1, 1].transAxes,
                           fontsize=12)
            axes[1, 1].set_title('Feature Importance')
        
        plt.tight_layout()
        plt.savefig('Data/baseline_models_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"   ✅ Saved comparison plots to: Data/baseline_models_comparison.png")
        
        # Store comparison
        self.comparison_df = comparison_df
        
        return comparison_df
    
    def save_models(self):
        """Save trained models and results."""
        print(f"\n💾 SAVING MODELS AND RESULTS")
        print("-" * 35)
        
        try:
            # Save best model
            best_model_key = self.comparison_df.iloc[0]['Model'].lower().replace(' ', '_')
            best_pipeline = None
            
            for model_key, result in self.results.items():
                if best_model_key in model_key:
                    best_pipeline = result['pipeline']
                    break
            
            if best_pipeline:
                joblib.dump(best_pipeline, 'Data/best_baseline_model.pkl')
                print(f"   ✅ Saved best model: {self.comparison_df.iloc[0]['Model']}")
            
            # Save all results
            results_summary = {
                'comparison': self.comparison_df.to_dict('records'),
                'feature_columns': self.data['feature_columns']
            }
            
            if hasattr(self, 'cv_results'):
                results_summary['cross_validation'] = self.cv_results
            
            if hasattr(self, 'feature_importance'):
                results_summary['feature_importance'] = self.feature_importance.to_dict('records')
            
            import json
            with open('Data/baseline_results.json', 'w') as f:
                json.dump(results_summary, f, indent=2, default=str)
            
            print(f"   ✅ Saved results summary: Data/baseline_results.json")
            
            return True
        except Exception as e:
            print(f"   ❌ Error saving models: {e}")
            return False
    
    def run_baseline_evaluation(self):
        """Run the complete baseline model evaluation."""
        print(f"🚀 Starting Baseline Model Evaluation...")
        print()
        
        # Load data
        if not self.load_splits():
            return None
        
        # Create preprocessor
        self.create_preprocessor()
        
        # Build models
        self.build_models()
        
        # Train and evaluate
        self.train_and_evaluate()
        
        # Cross-validate best models
        self.cross_validate_best_models()
        
        # Analyze feature importance
        self.analyze_feature_importance()
        
        # Create performance comparison
        comparison_df = self.create_performance_comparison()
        
        # Save models and results
        if self.save_models():
            print(f"\n🎉 BASELINE EVALUATION COMPLETED!")
            print(f"   🏆 Best Model: {comparison_df.iloc[0]['Model']}")
            print(f"   📊 Best Validation R²: {comparison_df.iloc[0]['Val_R2']:.3f}")
            print(f"   💰 Best Validation RMSE: ${comparison_df.iloc[0]['Val_RMSE_CAD']:.2f} CAD")
            print(f"   📈 Best Validation MAPE: {comparison_df.iloc[0]['Val_MAPE_%']:.2f}%")
            
            return self.results, comparison_df
        else:
            return None, None


def main():
    """Run baseline model evaluation."""
    
    # Initialize baseline models
    baseline = BaselineModels()
    
    # Run evaluation
    results, comparison = baseline.run_baseline_evaluation()
    
    if results and comparison is not None:
        print(f"\n✅ Baseline models evaluation completed successfully!")
        print(f"📈 Ready for advanced model development")
        return baseline, results, comparison
    else:
        print(f"\n❌ Baseline evaluation failed!")
        return None, None, None


if __name__ == "__main__":
    main()