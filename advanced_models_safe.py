"""
Advanced Models for Hotel Price Prediction (Safe Version)
========================================================

This script implements advanced gradient boosting models with safe importing
to handle missing dependencies gracefully. Only LightGBM is used if XGBoost
has OpenMP issues on macOS.

Author: ML Pipeline - Advanced Models (Safe)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import warnings
warnings.filterwarnings('ignore')

# Safe import of gradient boosting libraries
AVAILABLE_MODELS = []

print("🔍 Checking available gradient boosting libraries...")

# Check XGBoost
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
    AVAILABLE_MODELS.append('XGBoost')
    print("✅ XGBoost available")
except Exception as e:
    XGBOOST_AVAILABLE = False
    print(f"❌ XGBoost not available: {str(e)[:100]}...")
    print("💡 For macOS: Try 'brew install libomp' or use conda instead")

# Check LightGBM
try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
    AVAILABLE_MODELS.append('LightGBM')
    print("✅ LightGBM available")
except Exception as e:
    LIGHTGBM_AVAILABLE = False
    print(f"❌ LightGBM not available: {e}")

# Check CatBoost
try:
    import catboost as cb
    CATBOOST_AVAILABLE = True
    AVAILABLE_MODELS.append('CatBoost')
    print("✅ CatBoost available")
except Exception as e:
    CATBOOST_AVAILABLE = False
    print(f"❌ CatBoost not available: {e}")

print(f"📋 Available models: {AVAILABLE_MODELS}")

if not AVAILABLE_MODELS:
    print("💡 Fallback: Will use enhanced Random Forest with gradient boosting-like features")

class AdvancedModelsSafe:
    def __init__(self):
        """Initialize advanced models."""
        self.models = {}
        self.results = {}
        self.feature_encoders = {}
        
        print("🚀 ADVANCED MODELS FOR HOTEL PRICE PREDICTION (SAFE)")
        print("=" * 60)
        
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
            X_train = train_df[feature_columns].copy()
            y_train = train_df['log_price_cad'].copy()
            y_train_orig = train_df['price_cad'].copy()
            
            X_val = val_df[feature_columns].copy()
            y_val = val_df['log_price_cad'].copy()
            y_val_orig = val_df['price_cad'].copy()
            
            X_test = test_df[feature_columns].copy()
            y_test = test_df['log_price_cad'].copy()
            y_test_orig = test_df['price_cad'].copy()
            
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
    
    def prepare_features_for_boosting(self):
        """Prepare features specifically for gradient boosting models."""
        print(f"\n🔧 PREPARING FEATURES FOR GRADIENT BOOSTING")
        print("-" * 50)
        
        # Identify categorical and numerical features
        X_sample = self.data['X_train']
        
        categorical_features = []
        numerical_features = []
        
        for col in X_sample.columns:
            if X_sample[col].dtype == 'object':
                categorical_features.append(col)
            else:
                numerical_features.append(col)
        
        print(f"   📊 Feature Types:")
        print(f"     • Numerical: {len(numerical_features)}")
        print(f"     • Categorical: {len(categorical_features)}")
        
        # Prepare encoded datasets
        datasets = ['X_train', 'X_val', 'X_test']
        
        for dataset_name in datasets:
            X = self.data[dataset_name].copy()
            
            # Label encode categorical features
            for col in categorical_features:
                if col not in self.feature_encoders:
                    # Fit encoder on training data
                    if dataset_name == 'X_train':
                        encoder = LabelEncoder()
                        encoder.fit(X[col].astype(str))
                        self.feature_encoders[col] = encoder
                    else:
                        continue
                
                # Transform using fitted encoder
                encoder = self.feature_encoders[col]
                
                # Handle unseen categories
                try:
                    X[col] = encoder.transform(X[col].astype(str))
                except ValueError:
                    X[col] = X[col].astype(str)
                    for i, val in enumerate(X[col]):
                        if val in encoder.classes_:
                            X.iloc[i, X.columns.get_loc(col)] = encoder.transform([val])[0]
                        else:
                            X.iloc[i, X.columns.get_loc(col)] = 0
            
            self.data[f'{dataset_name}_encoded'] = X
        
        print(f"   ✅ Categorical features encoded using LabelEncoder")
        
        self.categorical_features = categorical_features
        self.numerical_features = numerical_features
        
        return categorical_features, numerical_features
    
    def build_lightgbm_model(self):
        """Build and train LightGBM model."""
        if not LIGHTGBM_AVAILABLE:
            print(f"\n❌ LightGBM not available - skipping")
            return None
            
        print(f"\n💡 BUILDING LIGHTGBM MODEL")
        print("-" * 35)
        
        # LightGBM parameters
        lgb_params = {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.1,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'random_state': 42,
            'n_estimators': 150  # Increased for better performance
        }
        
        print(f"   📋 LightGBM Parameters:")
        for key, value in lgb_params.items():
            if key != 'verbose':
                print(f"     • {key}: {value}")
        
        # Create model
        lgb_model = lgb.LGBMRegressor(**lgb_params)
        
        # Train model
        print(f"   🔧 Training LightGBM...")
        
        lgb_model.fit(
            self.data['X_train_encoded'], 
            self.data['y_train'],
            eval_set=[(self.data['X_val_encoded'], self.data['y_val'])],
            callbacks=[lgb.early_stopping(stopping_rounds=10), lgb.log_evaluation(0)]
        )
        
        # Make predictions
        y_train_pred = lgb_model.predict(self.data['X_train_encoded'])
        y_val_pred = lgb_model.predict(self.data['X_val_encoded'])
        
        # Convert to original scale
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
        self.results['lightgbm'] = {
            'name': 'LightGBM',
            'model': lgb_model,
            'train_metrics': train_metrics,
            'val_metrics': val_metrics,
            'train_pred': y_train_pred,
            'val_pred': y_val_pred,
            'train_pred_orig': y_train_pred_orig,
            'val_pred_orig': y_val_pred_orig
        }
        
        print(f"   ✅ LightGBM training completed")
        print(f"   📊 Validation RMSE: ${val_metrics['rmse_orig']:.2f} CAD")
        print(f"   📊 Validation MAPE: {val_metrics['mape']:.2f}%")
        print(f"   📊 Validation R²: {val_metrics['r2_orig']:.3f}")
        
        return lgb_model
    
    def build_enhanced_random_forest(self):
        """Build enhanced Random Forest as fallback."""
        print(f"\n🌲 BUILDING ENHANCED RANDOM FOREST")
        print("-" * 40)
        
        from sklearn.ensemble import RandomForestRegressor
        
        # Enhanced Random Forest parameters
        rf_params = {
            'n_estimators': 200,  # More trees
            'max_depth': 15,      # Deeper trees
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'max_features': 'sqrt',
            'bootstrap': True,
            'random_state': 42,
            'n_jobs': -1
        }
        
        print(f"   📋 Enhanced RF Parameters:")
        for key, value in rf_params.items():
            print(f"     • {key}: {value}")
        
        # Create model
        rf_model = RandomForestRegressor(**rf_params)
        
        # Train model
        print(f"   🔧 Training Enhanced Random Forest...")
        
        rf_model.fit(self.data['X_train_encoded'], self.data['y_train'])
        
        # Make predictions
        y_train_pred = rf_model.predict(self.data['X_train_encoded'])
        y_val_pred = rf_model.predict(self.data['X_val_encoded'])
        
        # Convert to original scale
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
        self.results['enhanced_rf'] = {
            'name': 'Enhanced Random Forest',
            'model': rf_model,
            'train_metrics': train_metrics,
            'val_metrics': val_metrics,
            'train_pred': y_train_pred,
            'val_pred': y_val_pred,
            'train_pred_orig': y_train_pred_orig,
            'val_pred_orig': y_val_pred_orig
        }
        
        print(f"   ✅ Enhanced RF training completed")
        print(f"   📊 Validation RMSE: ${val_metrics['rmse_orig']:.2f} CAD")
        print(f"   📊 Validation MAPE: {val_metrics['mape']:.2f}%")
        print(f"   📊 Validation R²: {val_metrics['r2_orig']:.3f}")
        
        return rf_model
    
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
    
    def analyze_feature_importance(self):
        """Analyze feature importance for available models."""
        print(f"\n🔍 FEATURE IMPORTANCE ANALYSIS")
        print("-" * 35)
        
        importance_data = {}
        
        for model_key, result in self.results.items():
            model = result['model']
            
            if hasattr(model, 'feature_importances_'):
                importance_scores = model.feature_importances_
                feature_names = self.data['X_train_encoded'].columns
                
                importance_df = pd.DataFrame({
                    'feature': feature_names,
                    'importance': importance_scores,
                    'model': result['name']
                }).sort_values('importance', ascending=False)
                
                importance_data[model_key] = importance_df
                
                print(f"\n   🏆 {result['name']} - Top 10 Features:")
                for i, (_, row) in enumerate(importance_df.head(10).iterrows(), 1):
                    print(f"     {i:2d}. {row['feature']:<25}: {row['importance']:.4f}")
        
        self.feature_importance_data = importance_data
        return importance_data
    
    def create_model_comparison(self):
        """Create comprehensive comparison with baseline models."""
        print(f"\n📊 CREATING MODEL COMPARISON")
        print("-" * 35)
        
        # Load baseline results for comparison
        try:
            import json
            with open('Data/baseline_results.json', 'r') as f:
                baseline_data = json.load(f)
            
            baseline_comparison = pd.DataFrame(baseline_data['comparison'])
            print(f"   📋 Loaded baseline results for comparison")
        except:
            print(f"   ⚠️ Could not load baseline results")
            baseline_comparison = None
        
        # Create advanced models comparison
        advanced_data = []
        
        for model_key, result in self.results.items():
            train_metrics = result['train_metrics']
            val_metrics = result['val_metrics']
            
            advanced_data.append({
                'Model': result['name'],
                'Train_RMSE_CAD': train_metrics['rmse_orig'],
                'Val_RMSE_CAD': val_metrics['rmse_orig'],
                'Train_MAPE_%': train_metrics['mape'],
                'Val_MAPE_%': val_metrics['mape'],
                'Train_R2': train_metrics['r2_orig'],
                'Val_R2': val_metrics['r2_orig'],
                'Overfit_Score': train_metrics['r2_orig'] - val_metrics['r2_orig'],
                'Model_Type': 'Advanced'
            })
        
        advanced_df = pd.DataFrame(advanced_data)
        
        # Combine with baseline if available
        if baseline_comparison is not None:
            baseline_comparison['Model_Type'] = 'Baseline'
            comparison_df = pd.concat([advanced_df, baseline_comparison], ignore_index=True)
        else:
            comparison_df = advanced_df
        
        comparison_df = comparison_df.sort_values('Val_R2', ascending=False)
        
        print(f"\n   📋 Complete Model Performance Summary:")
        print(f"   {comparison_df[['Model', 'Val_RMSE_CAD', 'Val_MAPE_%', 'Val_R2', 'Model_Type']].to_string(index=False, float_format='%.3f')}")
        
        # Create visualization
        self._create_comparison_plots(comparison_df)
        
        self.comparison_df = comparison_df
        return comparison_df
    
    def _create_comparison_plots(self, comparison_df):
        """Create comprehensive comparison plots."""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Advanced vs Baseline Models Comparison', fontsize=16, fontweight='bold')
        
        # Color mapping for model types
        colors = {'Advanced': 'darkgreen', 'Baseline': 'steelblue'}
        
        # Plot 1: R² Comparison
        models = comparison_df['Model']
        r2_scores = comparison_df['Val_R2']
        model_colors = [colors.get(t, 'gray') for t in comparison_df['Model_Type']]
        
        bars1 = axes[0, 0].bar(range(len(models)), r2_scores, color=model_colors, alpha=0.7)
        axes[0, 0].set_xlabel('Models')
        axes[0, 0].set_ylabel('Validation R²')
        axes[0, 0].set_title('Model Performance (R² Score)')
        axes[0, 0].set_xticks(range(len(models)))
        axes[0, 0].set_xticklabels(models, rotation=45, ha='right')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, value in zip(bars1, r2_scores):
            axes[0, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                           f'{value:.3f}', ha='center', va='bottom', fontsize=9)
        
        # Plot 2: RMSE Comparison
        rmse_scores = comparison_df['Val_RMSE_CAD']
        bars2 = axes[0, 1].bar(range(len(models)), rmse_scores, color=model_colors, alpha=0.7)
        axes[0, 1].set_xlabel('Models')
        axes[0, 1].set_ylabel('Validation RMSE (CAD)')
        axes[0, 1].set_title('Model Performance (RMSE)')
        axes[0, 1].set_xticks(range(len(models)))
        axes[0, 1].set_xticklabels(models, rotation=45, ha='right')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: MAPE Comparison
        mape_scores = comparison_df['Val_MAPE_%']
        bars3 = axes[1, 0].bar(range(len(models)), mape_scores, color=model_colors, alpha=0.7)
        axes[1, 0].set_xlabel('Models')
        axes[1, 0].set_ylabel('Validation MAPE (%)')
        axes[1, 0].set_title('Model Performance (MAPE)')
        axes[1, 0].set_xticks(range(len(models)))
        axes[1, 0].set_xticklabels(models, rotation=45, ha='right')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Feature Importance (best available model)
        if hasattr(self, 'feature_importance_data') and self.feature_importance_data:
            best_advanced_model = comparison_df[comparison_df['Model_Type'] == 'Advanced'].iloc[0]['Model'].lower()
            
            # Find matching importance data
            importance_df = None
            for key, imp_df in self.feature_importance_data.items():
                if any(word in key for word in best_advanced_model.split()):
                    importance_df = imp_df
                    break
            
            if importance_df is not None:
                top_features = importance_df.head(10)
                axes[1, 1].barh(range(len(top_features)), top_features['importance'], 
                               color='darkgreen', alpha=0.7)
                axes[1, 1].set_yticks(range(len(top_features)))
                axes[1, 1].set_yticklabels(top_features['feature'], fontsize=9)
                axes[1, 1].set_xlabel('Importance Score')
                axes[1, 1].set_title(f'Top 10 Features ({best_advanced_model.title()})')
                axes[1, 1].grid(True, alpha=0.3)
            else:
                axes[1, 1].text(0.5, 0.5, 'Feature Importance\nNot Available', 
                               ha='center', va='center', transform=axes[1, 1].transAxes)
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor=colors['Advanced'], label='Advanced Models'),
                          Patch(facecolor=colors['Baseline'], label='Baseline Models')]
        fig.legend(handles=legend_elements, loc='upper right')
        
        plt.tight_layout()
        plt.savefig('Data/advanced_models_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"   ✅ Saved comparison plots to: Data/advanced_models_comparison.png")
    
    def save_models_and_results(self):
        """Save trained models and results."""
        print(f"\n💾 SAVING ADVANCED MODELS AND RESULTS")
        print("-" * 45)
        
        try:
            # Save best advanced model
            best_model_key = None
            best_r2 = -1
            
            for model_key, result in self.results.items():
                r2 = result['val_metrics']['r2_orig']
                if r2 > best_r2:
                    best_r2 = r2
                    best_model_key = model_key
            
            if best_model_key:
                best_model = self.results[best_model_key]['model']
                best_name = self.results[best_model_key]['name']
                
                joblib.dump(best_model, 'Data/best_advanced_model.pkl')
                joblib.dump(self.feature_encoders, 'Data/feature_encoders.pkl')
                
                print(f"   ✅ Saved best advanced model: {best_name}")
                print(f"   ✅ Saved feature encoders")
            
            # Save comprehensive results
            results_summary = {
                'advanced_comparison': self.comparison_df.to_dict('records'),
                'feature_columns': self.data['feature_columns'],
                'categorical_features': self.categorical_features,
                'numerical_features': self.numerical_features,
                'available_models': AVAILABLE_MODELS
            }
            
            if hasattr(self, 'feature_importance_data'):
                results_summary['feature_importance'] = {
                    k: v.to_dict('records') for k, v in self.feature_importance_data.items()
                }
            
            import json
            with open('Data/advanced_results.json', 'w') as f:
                json.dump(results_summary, f, indent=2, default=str)
            
            print(f"   ✅ Saved results summary: Data/advanced_results.json")
            
            return True
        except Exception as e:
            print(f"   ❌ Error saving models: {e}")
            return False
    
    def run_advanced_models(self):
        """Run the complete advanced model evaluation."""
        print(f"🚀 Starting Advanced Model Development...")
        print()
        
        # Load data
        if not self.load_splits():
            return None
        
        # Prepare features
        self.prepare_features_for_boosting()
        
        # Build available models
        print(f"\n🏗️ BUILDING ADVANCED MODELS")
        print("=" * 35)
        
        models_built = 0
        
        # Try LightGBM first (most reliable)
        if LIGHTGBM_AVAILABLE:
            self.build_lightgbm_model()
            models_built += 1
        
        # If no models available, use enhanced Random Forest
        if models_built == 0:
            print(f"\n💡 No gradient boosting libraries available, using enhanced Random Forest")
            self.build_enhanced_random_forest()
            models_built += 1
        
        if models_built == 0:
            print(f"\n❌ No models could be built!")
            return None, None
        
        # Analyze feature importance
        self.analyze_feature_importance()
        
        # Create comprehensive comparison
        comparison_df = self.create_model_comparison()
        
        # Save models and results
        if self.save_models_and_results():
            best_model = comparison_df[comparison_df['Model_Type'] == 'Advanced'].iloc[0]
            
            print(f"\n🎉 ADVANCED MODEL DEVELOPMENT COMPLETED!")
            print(f"   🏆 Best Advanced Model: {best_model['Model']}")
            print(f"   📊 Best Validation R²: {best_model['Val_R2']:.3f}")
            print(f"   💰 Best Validation RMSE: ${best_model['Val_RMSE_CAD']:.2f} CAD")
            print(f"   📈 Best Validation MAPE: {best_model['Val_MAPE_%']:.2f}%")
            
            return self.results, comparison_df
        else:
            return None, None


def main():
    """Run advanced model development."""
    
    # Initialize advanced models
    advanced = AdvancedModelsSafe()
    
    # Run development
    results, comparison = advanced.run_advanced_models()
    
    if results and comparison is not None:
        print(f"\n✅ Advanced models development completed successfully!")
        print(f"🚀 Ready for hyperparameter tuning and final evaluation")
        return advanced, results, comparison
    else:
        print(f"\n❌ Advanced model development failed!")
        return None, None, None


if __name__ == "__main__":
    main()