"""
Feature Engineering Pipeline for Hotel Price Prediction
======================================================

This script creates a comprehensive feature engineering pipeline based on EDA findings.
Key focus areas:
1. Target transformation (log1p of price_cad)
2. Categorical encoding for high-impact features
3. Feature creation and selection
4. Data preprocessing for ML models

Based on EDA insights:
- Distance from attraction: strongest predictor (-0.432 correlation)
- District, room_category: high categorical impact
- Weekend/holiday flags: significant price differences
- Target needs log transformation (skewness: 2.188 → 0.072)

Author: ML Pipeline
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')

class HotelFeatureEngineer:
    def __init__(self, data_file='Data/toronto_hotels_transformed.csv'):
        """Initialize feature engineering pipeline."""
        self.data_file = data_file
        self.df = None
        self.feature_pipeline = None
        self.target_scaler = None
        
        print("🔧 HOTEL PRICE PREDICTION - FEATURE ENGINEERING")
        print("=" * 55)
        
    def load_data(self):
        """Load the transformed dataset."""
        print("📂 Loading transformed dataset...")
        
        try:
            self.df = pd.read_csv(self.data_file)
            print(f"   ✅ Loaded {len(self.df):,} rows × {len(self.df.columns)} columns")
            
            # Check for price_cad column
            if 'price_cad' not in self.df.columns:
                print(f"   ❌ price_cad column not found!")
                return False
                
            print(f"   💰 Target column: price_cad")
            print(f"   📊 Price range: ${self.df['price_cad'].min():.2f} - ${self.df['price_cad'].max():.2f} CAD")
            
            return True
        except Exception as e:
            print(f"   ❌ Error loading data: {e}")
            return False
    
    def create_target_transformation(self):
        """Apply log transformation to target variable based on EDA recommendations."""
        print(f"\n🎯 TARGET TRANSFORMATION")
        print("-" * 30)
        
        if 'price_cad' not in self.df.columns:
            print(f"   ❌ price_cad column not found!")
            return
        
        # Remove missing values
        original_count = len(self.df)
        self.df = self.df.dropna(subset=['price_cad'])
        removed_count = original_count - len(self.df)
        
        if removed_count > 0:
            print(f"   🧹 Removed {removed_count} rows with missing price_cad")
        
        # Original statistics
        original_price = self.df['price_cad']
        print(f"   📊 Original price_cad statistics:")
        print(f"     • Mean: ${original_price.mean():.2f}")
        print(f"     • Median: ${original_price.median():.2f}")
        print(f"     • Skewness: {original_price.skew():.3f}")
        
        # Apply log transformation
        self.df['log_price_cad'] = np.log1p(self.df['price_cad'])
        
        # Transformed statistics
        log_price = self.df['log_price_cad']
        print(f"\n   ✨ Log-transformed target statistics:")
        print(f"     • Mean: {log_price.mean():.3f}")
        print(f"     • Median: {log_price.median():.3f}")
        print(f"     • Skewness: {log_price.skew():.3f}")
        
        skew_improvement = abs(original_price.skew()) - abs(log_price.skew())
        print(f"     • Skewness improvement: {skew_improvement:.3f}")
        
        if skew_improvement > 0:
            print(f"     ✅ Log transformation improved distribution!")
        else:
            print(f"     ⚠️ Log transformation didn't improve skewness")
        
        print(f"   📝 Implementation: target = np.log1p(price_cad)")
        print(f"   📝 Inverse transform: price = np.expm1(prediction)")
    
    def engineer_features(self):
        """Create engineered features based on EDA insights."""
        print(f"\n🔨 FEATURE ENGINEERING")
        print("-" * 25)
        
        # 1. Location-based features (strongest predictor from EDA)
        print(f"   🗺️ Location Features:")
        if 'Distance from attraction' in self.df.columns:
            # Distance categories (based on EDA analysis)
            self.df['distance_category'] = pd.cut(
                self.df['Distance from attraction'],
                bins=[-1, 0.5, 1.0, 2.0, 5.0, float('inf')],
                labels=['very_close', 'close', 'nearby', 'moderate', 'far']
            )
            print(f"     ✅ Created distance_category (5 bins)")
        
        # 2. Temporal features
        print(f"   📅 Temporal Features:")
        if 'Check-in Date' in self.df.columns:
            self.df['checkin_date'] = pd.to_datetime(self.df['Check-in Date'])
            self.df['checkin_month'] = self.df['checkin_date'].dt.month
            self.df['checkin_day_of_week'] = self.df['checkin_date'].dt.dayofweek
            self.df['checkin_quarter'] = self.df['checkin_date'].dt.quarter
            print(f"     ✅ Created checkin_month, checkin_day_of_week, checkin_quarter")
        
        # 3. Booking behavior features
        print(f"   📋 Booking Features:")
        if 'booking_lead_time' in self.df.columns:
            # Lead time categories
            self.df['lead_time_category'] = pd.cut(
                self.df['booking_lead_time'],
                bins=[-1, 0, 7, 30, 90, float('inf')],
                labels=['same_day', 'week', 'month', 'quarter', 'long_term']
            )
            print(f"     ✅ Created lead_time_category")
        
        if 'length_of_stay' in self.df.columns:
            # Stay duration categories
            self.df['stay_category'] = pd.cut(
                self.df['length_of_stay'],
                bins=[0, 1, 3, 7, 14, float('inf')],
                labels=['overnight', 'short', 'week', 'extended', 'long_term']
            )
            print(f"     ✅ Created stay_category")
        
        # 4. Event impact features
        print(f"   🎉 Event Features:")
        if 'events_total_score' in self.df.columns:
            # Event intensity levels
            self.df['event_intensity'] = pd.cut(
                self.df['events_total_score'],
                bins=[-1, 0, 50, 100, 200, float('inf')],
                labels=['no_events', 'low', 'medium', 'high', 'peak']
            )
            print(f"     ✅ Created event_intensity")
        
        # 5. Combined temporal features
        print(f"   🔄 Combined Features:")
        
        # Weekend + Holiday combination
        if 'is_weekend' in self.df.columns and 'is_holiday' in self.df.columns:
            self.df['weekend_holiday'] = self.df['is_weekend'].astype(str) + '_' + self.df['is_holiday'].astype(str)
            print(f"     ✅ Created weekend_holiday combination")
        
        # District + Room category (high-impact combinations)
        if 'district' in self.df.columns and 'room_category' in self.df.columns:
            # Create simplified district-room combinations for top districts only
            top_districts = self.df['district'].value_counts().head(5).index
            self.df['district_simple'] = self.df['district'].apply(
                lambda x: x if x in top_districts else 'other'
            )
            self.df['district_room_combo'] = (
                self.df['district_simple'] + '_' + 
                self.df['room_category'].astype(str)
            )
            print(f"     ✅ Created district_room_combo")
        
        # 6. Weather interaction features
        if 'MEAN_TEMPERATURE' in self.df.columns:
            # Temperature categories
            self.df['temp_category'] = pd.cut(
                self.df['MEAN_TEMPERATURE'],
                bins=[-50, 0, 10, 20, 30, 50],
                labels=['very_cold', 'cold', 'mild', 'warm', 'hot']
            )
            print(f"     ✅ Created temp_category")
        
        print(f"   ✅ Feature engineering completed!")
    
    def select_features(self):
        """Select final feature set based on EDA findings and business logic."""
        print(f"\n🎯 FEATURE SELECTION")
        print("-" * 20)
        
        # Define feature groups based on EDA importance
        
        # High-impact numerical features (from correlation analysis)
        numerical_features = []
        high_impact_numerical = ['Distance from attraction', 'booking_lead_time', 
                               'events_total_score', 'length_of_stay', 'MEAN_TEMPERATURE']
        
        for feature in high_impact_numerical:
            if feature in self.df.columns:
                numerical_features.append(feature)
        
        # High-impact categorical features (from categorical analysis)
        categorical_features = []
        high_impact_categorical = ['district', 'room_category', 'is_weekend', 
                                 'is_holiday', 'has_major_event']
        
        for feature in high_impact_categorical:
            if feature in self.df.columns:
                categorical_features.append(feature)
        
        # Engineered categorical features
        engineered_categorical = ['distance_category', 'lead_time_category', 
                                'stay_category', 'event_intensity', 'temp_category',
                                'weekend_holiday', 'district_simple']
        
        for feature in engineered_categorical:
            if feature in self.df.columns:
                categorical_features.append(feature)
        
        # Additional numerical features
        additional_numerical = ['checkin_month', 'checkin_day_of_week', 'checkin_quarter',
                              'events_count', 'events_max_score', 'week_of_year']
        
        for feature in additional_numerical:
            if feature in self.df.columns:
                numerical_features.append(feature)
        
        # Remove duplicates and store
        self.numerical_features = list(set(numerical_features))
        self.categorical_features = list(set(categorical_features))
        
        print(f"   📊 Selected Features Summary:")
        print(f"     • Numerical features: {len(self.numerical_features)}")
        print(f"     • Categorical features: {len(self.categorical_features)}")
        print(f"     • Total features: {len(self.numerical_features) + len(self.categorical_features)}")
        
        print(f"\n   🔢 Numerical Features:")
        for i, feature in enumerate(self.numerical_features, 1):
            print(f"     {i:2d}. {feature}")
        
        print(f"\n   🏷️ Categorical Features:")
        for i, feature in enumerate(self.categorical_features, 1):
            print(f"     {i:2d}. {feature}")
        
        # Verify all features exist
        missing_features = []
        for feature in self.numerical_features + self.categorical_features:
            if feature not in self.df.columns:
                missing_features.append(feature)
        
        if missing_features:
            print(f"\n   ⚠️ Missing features: {missing_features}")
            # Remove missing features
            self.numerical_features = [f for f in self.numerical_features if f in self.df.columns]
            self.categorical_features = [f for f in self.categorical_features if f in self.df.columns]
            print(f"   🔧 Adjusted to available features")
    
    def create_preprocessing_pipeline(self):
        """Create sklearn preprocessing pipeline."""
        print(f"\n⚙️ PREPROCESSING PIPELINE")
        print("-" * 30)
        
        # Numerical preprocessing: StandardScaler
        numerical_transformer = StandardScaler()
        
        # Categorical preprocessing: OneHotEncoder with handling for unknown categories
        categorical_transformer = OneHotEncoder(
            drop='first',  # Avoid multicollinearity
            sparse_output=False,
            handle_unknown='ignore'  # Handle new categories in test data
        )
        
        # Combine preprocessors
        self.feature_pipeline = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, self.numerical_features),
                ('cat', categorical_transformer, self.categorical_features)
            ],
            remainder='drop'  # Drop any features not specified
        )
        
        print(f"   ✅ Created preprocessing pipeline:")
        print(f"     • Numerical: StandardScaler for {len(self.numerical_features)} features")
        print(f"     • Categorical: OneHotEncoder for {len(self.categorical_features)} features")
        print(f"     • Unknown handling: Enabled for robust predictions")
    
    def prepare_model_data(self):
        """Prepare final dataset for modeling."""
        print(f"\n📋 PREPARING MODEL DATA")
        print("-" * 25)
        
        # Select features and target
        feature_columns = self.numerical_features + self.categorical_features
        
        # Create feature matrix
        X = self.df[feature_columns].copy()
        y = self.df['log_price_cad'].copy()  # Log-transformed target
        
        # Remove any remaining missing values
        initial_shape = X.shape
        
        # Create mask for complete cases
        complete_mask = ~(X.isnull().any(axis=1) | y.isnull())
        
        X_clean = X[complete_mask]
        y_clean = y[complete_mask]
        
        removed_rows = initial_shape[0] - X_clean.shape[0]
        if removed_rows > 0:
            print(f"   🧹 Removed {removed_rows} rows with missing values")
        
        print(f"   📊 Final dataset shape:")
        print(f"     • Features (X): {X_clean.shape[0]:,} rows × {X_clean.shape[1]} features")
        print(f"     • Target (y): {len(y_clean):,} values")
        print(f"     • Target range: {y_clean.min():.3f} to {y_clean.max():.3f} (log scale)")
        
        # Store clean data
        self.X = X_clean
        self.y = y_clean
        self.feature_names = feature_columns
        
        # Also store original prices for evaluation
        self.original_prices = self.df.loc[complete_mask, 'price_cad']
        
        print(f"   ✅ Model data preparation completed!")
        
        return X_clean, y_clean
    
    def get_feature_info(self):
        """Get information about engineered features."""
        print(f"\n📈 FEATURE ENGINEERING SUMMARY")
        print("=" * 35)
        
        if hasattr(self, 'X') and hasattr(self, 'y'):
            print(f"✅ Ready for modeling:")
            print(f"   • Dataset size: {len(self.X):,} samples")
            print(f"   • Feature count: {self.X.shape[1]}")
            print(f"   • Target: log_price_cad (log1p transformed)")
            print(f"   • Original target: price_cad")
            
            print(f"\n🎯 Target distribution:")
            print(f"   • Log target mean: {self.y.mean():.3f}")
            print(f"   • Log target std: {self.y.std():.3f}")
            print(f"   • Original price mean: ${self.original_prices.mean():.2f}")
            print(f"   • Original price std: ${self.original_prices.std():.2f}")
            
            print(f"\n🔧 Pipeline components:")
            print(f"   • Feature preprocessing: ColumnTransformer")
            print(f"   • Numerical: StandardScaler ({len(self.numerical_features)} features)")
            print(f"   • Categorical: OneHotEncoder ({len(self.categorical_features)} features)")
            
        else:
            print(f"❌ Data not prepared yet. Run prepare_model_data() first.")
    
    def run_feature_engineering(self):
        """Run the complete feature engineering pipeline."""
        print(f"🚀 Starting Feature Engineering Pipeline...")
        print()
        
        # Load data
        if not self.load_data():
            return None, None
        
        # Transform target
        self.create_target_transformation()
        
        # Engineer features
        self.engineer_features()
        
        # Select features
        self.select_features()
        
        # Create preprocessing pipeline
        self.create_preprocessing_pipeline()
        
        # Prepare model data
        X, y = self.prepare_model_data()
        
        # Summary
        self.get_feature_info()
        
        return X, y
    
    def transform_new_data(self, new_data):
        """Transform new data using the fitted pipeline."""
        if self.feature_pipeline is None:
            raise ValueError("Pipeline not fitted. Run run_feature_engineering() first.")
        
        # Apply same feature engineering to new data
        # (This would need to be implemented based on the specific transformations)
        # For now, return the preprocessed features
        
        feature_columns = self.numerical_features + self.categorical_features
        X_new = new_data[feature_columns]
        
        return self.feature_pipeline.transform(X_new)
    
    def inverse_transform_target(self, predictions):
        """Convert log predictions back to original price scale."""
        return np.expm1(predictions)


def main():
    """Run the feature engineering pipeline."""
    
    # Initialize feature engineer
    engineer = HotelFeatureEngineer('Data/toronto_hotels_transformed.csv')
    
    # Run complete pipeline
    X, y = engineer.run_feature_engineering()
    
    if X is not None and y is not None:
        print(f"\n🎉 FEATURE ENGINEERING COMPLETE!")
        print(f"📊 Ready for model training with {X.shape[0]:,} samples and {X.shape[1]} features")
        
        # Save processed data for model training
        processed_data = engineer.df.copy()
        processed_data.to_csv('Data/hotel_features_engineered.csv', index=False)
        print(f"💾 Saved engineered features to: Data/hotel_features_engineered.csv")
        
        return engineer, X, y
    else:
        print(f"❌ Feature engineering failed!")
        return None, None, None


if __name__ == "__main__":
    main()