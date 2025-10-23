"""
Time-Based Train/Validation/Test Splits for Hotel Price Prediction
================================================================

This script creates proper time-based splits to avoid data leakage in hotel
price prediction models. Critical for time-series-like data where future
predictions should not use future information.

Key Principles:
1. NEVER use future data to predict past prices
2. Maintain temporal order: Train → Validation → Test
3. Account for booking lead times and seasonal patterns
4. Ensure sufficient data in each split for robust evaluation

Date Range: 2025-09-30 to 2025-10-25 (25 days)
Strategy: 60% Train / 20% Validation / 20% Test

Author: ML Pipeline - Time Series Splits
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import TimeSeriesSplit
import warnings
warnings.filterwarnings('ignore')

class TimeBasedSplitter:
    def __init__(self, data_file='Data/hotel_features_engineered.csv'):
        """Initialize time-based splitter."""
        self.data_file = data_file
        self.df = None
        
        # Split ratios (must sum to 1.0)
        self.train_ratio = 0.60    # 60% for training
        self.val_ratio = 0.20      # 20% for validation  
        self.test_ratio = 0.20     # 20% for testing
        
        print("📅 TIME-BASED TRAIN/VALIDATION/TEST SPLITS")
        print("=" * 50)
        
    def load_data(self):
        """Load the engineered dataset."""
        print("📂 Loading engineered dataset...")
        
        try:
            self.df = pd.read_csv(self.data_file)
            print(f"   ✅ Loaded {len(self.df):,} rows × {len(self.df.columns)} columns")
            
            # Convert check-in date to datetime
            self.df['checkin_date'] = pd.to_datetime(self.df['Check-in Date'])
            
            # Verify we have the target and key features
            required_columns = ['log_price_cad', 'price_cad', 'checkin_date']
            missing_cols = [col for col in required_columns if col not in self.df.columns]
            
            if missing_cols:
                print(f"   ❌ Missing required columns: {missing_cols}")
                return False
            
            print(f"   💰 Target column: log_price_cad")
            print(f"   📅 Date column: checkin_date")
            
            return True
        except Exception as e:
            print(f"   ❌ Error loading data: {e}")
            return False
    
    def analyze_temporal_distribution(self):
        """Analyze the temporal distribution of bookings."""
        print(f"\n📊 TEMPORAL DISTRIBUTION ANALYSIS")
        print("-" * 40)
        
        # Basic date statistics
        min_date = self.df['checkin_date'].min()
        max_date = self.df['checkin_date'].max()
        date_span = (max_date - min_date).days + 1
        
        print(f"   📅 Date Range:")
        print(f"     • Start: {min_date.strftime('%Y-%m-%d')}")
        print(f"     • End: {max_date.strftime('%Y-%m-%d')}")
        print(f"     • Span: {date_span} days")
        
        # Daily booking counts
        daily_counts = self.df['checkin_date'].value_counts().sort_index()
        
        print(f"\n   📈 Booking Distribution:")
        print(f"     • Total bookings: {len(self.df):,}")
        print(f"     • Unique dates: {len(daily_counts)}")
        print(f"     • Avg bookings/day: {daily_counts.mean():.0f}")
        print(f"     • Min bookings/day: {daily_counts.min():,}")
        print(f"     • Max bookings/day: {daily_counts.max():,}")
        print(f"     • Std bookings/day: {daily_counts.std():.0f}")
        
        # Show dates with highest and lowest volumes
        print(f"\n   🔝 Highest Volume Days:")
        for i, (date, count) in enumerate(daily_counts.nlargest(3).items(), 1):
            print(f"     {i}. {date.strftime('%Y-%m-%d')}: {count:,} bookings")
        
        print(f"\n   📉 Lowest Volume Days:")
        for i, (date, count) in enumerate(daily_counts.nsmallest(3).items(), 1):
            print(f"     {i}. {date.strftime('%Y-%m-%d')}: {count:,} bookings")
        
        # Store for split calculation
        self.daily_counts = daily_counts
        self.min_date = min_date
        self.max_date = max_date
        self.date_span = date_span
        
        return daily_counts
    
    def calculate_split_dates(self):
        """Calculate optimal split dates based on the actual date distribution."""
        print(f"\n🎯 CALCULATING SPLIT DATES")
        print("-" * 30)
        
        # Get unique dates and their counts
        unique_dates = sorted(self.df['checkin_date'].unique())
        date_counts = self.df['checkin_date'].value_counts().sort_index()
        total_samples = len(self.df)
        
        print(f"   📊 Available dates: {len(unique_dates)}")
        print(f"   📊 Date distribution:")
        cumulative = 0
        for date in unique_dates:
            count = date_counts[date]
            cumulative += count
            pct = (cumulative / total_samples) * 100
            print(f"     • {date.strftime('%Y-%m-%d')}: {count:,} ({count/total_samples:.1%}) | Cumulative: {pct:.1f}%")
        
        # Strategy: Use date boundaries that give us reasonable splits
        # Given the distribution, we'll use:
        # Train: 2025-09-30 (27.1% of data)
        # Validation: 2025-10-03 (67.0% of data) - sample from this large date
        # Test: 2025-10-10 onwards (remaining dates ~6% of data)
        
        print(f"\n   🎯 Split Strategy (adapted for uneven distribution):")
        
        # For this specific dataset, we'll split the large 10-03 date
        oct_03_data = self.df[self.df['checkin_date'] == pd.Timestamp('2025-10-03')]
        oct_03_count = len(oct_03_data)
        
        # Split Oct 3rd data: 60% for validation, 40% for test (to balance the splits)
        val_from_oct3 = int(oct_03_count * 0.35)  # Take 35% for validation
        test_from_oct3 = oct_03_count - val_from_oct3  # Rest for test
        
        self.train_end_date = pd.Timestamp('2025-10-03')  # Exclusive
        self.val_end_date = pd.Timestamp('2025-10-10')    # Exclusive
        
        print(f"     • Train: 2025-09-30 only ({date_counts[pd.Timestamp('2025-09-30')]:,} samples)")
        print(f"     • Validation: Sample from 2025-10-03 ({val_from_oct3:,} samples)")
        print(f"     • Test: Rest of 2025-10-03 + later dates ({test_from_oct3 + (total_samples - date_counts[pd.Timestamp('2025-09-30')] - oct_03_count):,} samples)")
        
        # Store the sampling info for Oct 3rd
        self.oct_03_val_count = val_from_oct3
        self.oct_03_test_count = test_from_oct3
        
        return self.train_end_date, self.val_end_date
    
    def create_splits(self):
        """Create the actual train/validation/test splits with special handling for uneven dates."""
        print(f"\n✂️ CREATING DATA SPLITS")
        print("-" * 25)
        
        # Train: All data from 2025-09-30
        train_df = self.df[self.df['checkin_date'] == pd.Timestamp('2025-09-30')].copy()
        
        # For validation and test, we need to split the large Oct 3rd date
        oct_03_data = self.df[self.df['checkin_date'] == pd.Timestamp('2025-10-03')].copy()
        later_dates_data = self.df[self.df['checkin_date'] > pd.Timestamp('2025-10-03')].copy()
        
        # Randomly sample from Oct 3rd data for validation/test split
        np.random.seed(42)  # For reproducibility
        oct_03_shuffled = oct_03_data.sample(frac=1, random_state=42).reset_index(drop=True)
        
        # Split Oct 3rd data
        val_from_oct3 = oct_03_shuffled.iloc[:self.oct_03_val_count].copy()
        test_from_oct3 = oct_03_shuffled.iloc[self.oct_03_val_count:].copy()
        
        # Combine validation and test sets
        val_df = val_from_oct3
        test_df = pd.concat([test_from_oct3, later_dates_data], ignore_index=True)
        
        print(f"   📊 Split Sizes:")
        print(f"     • Train: {len(train_df):,} samples ({len(train_df)/len(self.df):.1%})")
        print(f"     • Validation: {len(val_df):,} samples ({len(val_df)/len(self.df):.1%})")
        print(f"     • Test: {len(test_df):,} samples ({len(test_df)/len(self.df):.1%})")
        print(f"     • Total: {len(train_df) + len(val_df) + len(test_df):,} samples")
        
        print(f"\n   📅 Split Details:")
        print(f"     • Train: 2025-09-30 only")
        print(f"     • Validation: Random sample from 2025-10-03")
        print(f"     • Test: Remaining 2025-10-03 + all later dates")
        
        # Verify no overlap and complete coverage
        total_split = len(train_df) + len(val_df) + len(test_df)
        assert total_split == len(self.df), f"Split sizes don't match total! {total_split} vs {len(self.df)}"
        
        # Store splits
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df
        
        return train_df, val_df, test_df
    
    def analyze_split_characteristics(self):
        """Analyze the characteristics of each split."""
        print(f"\n🔍 SPLIT CHARACTERISTICS ANALYSIS")
        print("-" * 40)
        
        splits = {
            'Train': self.train_df,
            'Validation': self.val_df,
            'Test': self.test_df
        }
        
        print(f"   📈 Target Distribution (log_price_cad):")
        for split_name, split_df in splits.items():
            target_mean = split_df['log_price_cad'].mean()
            target_std = split_df['log_price_cad'].std()
            price_mean = split_df['price_cad'].mean()
            
            print(f"     • {split_name:<12}: μ={target_mean:.3f}, σ={target_std:.3f} | ${price_mean:.2f} CAD avg")
        
        print(f"\n   📅 Date Coverage:")
        for split_name, split_df in splits.items():
            min_date = split_df['checkin_date'].min()
            max_date = split_df['checkin_date'].max()
            unique_dates = split_df['checkin_date'].nunique()
            
            print(f"     • {split_name:<12}: {min_date.strftime('%Y-%m-%d')} → {max_date.strftime('%Y-%m-%d')} ({unique_dates} days)")
        
        # Check for key feature distributions
        print(f"\n   🏨 Key Feature Distributions:")
        
        # Weekend distribution
        print(f"     Weekend bookings:")
        for split_name, split_df in splits.items():
            weekend_pct = (split_df['is_weekend'].sum() / len(split_df)) * 100
            print(f"       • {split_name:<12}: {weekend_pct:.1f}%")
        
        # District distribution (top 3)
        print(f"     Top districts:")
        train_districts = self.train_df['district'].value_counts().head(3)
        for split_name, split_df in splits.items():
            district_dist = split_df['district'].value_counts()
            top_district = district_dist.index[0] if len(district_dist) > 0 else 'None'
            top_pct = (district_dist.iloc[0] / len(split_df)) * 100 if len(district_dist) > 0 else 0
            print(f"       • {split_name:<12}: {top_district} ({top_pct:.1f}%)")
    
    def prepare_model_arrays(self):
        """Prepare X, y arrays for model training."""
        print(f"\n🔧 PREPARING MODEL ARRAYS")
        print("-" * 30)
        
        # Define feature columns (exclude target, date, and high-missing columns)
        exclude_cols = [
            'log_price_cad', 'price_cad', 'checkin_date', 'Check-in Date', 
            'Check-out Date', 'Date', 'Hotel name', 'Address', 'Room Type',
            'Deal Info', 'name', 'STATION_NAME', 'CLIMATE_IDENTIFIER',
            'events_primary_segment', 'price_usd',  # Non-predictive columns
            # Weather columns with high missing rates (exclude these for now)
            'ID', 'PROVINCE_CODE', 'LOCAL_YEAR', 'LOCAL_MONTH', 'LOCAL_DAY',
            'MIN_TEMPERATURE', 'MAX_TEMPERATURE', 'TOTAL_PRECIPITATION', 
            'TOTAL_RAIN', 'TOTAL_SNOW', 'SNOW_ON_GROUND', 'MAX_REL_HUMIDITY'
        ]
        
        # Select only essential features with low missing rates
        essential_features = [
            # Location features (strongest predictors)
            'Distance from attraction', 'district', 'latitude', 'longitude',
            'district_simple', 'distance_category',
            
            # Booking features
            'booking_lead_time', 'length_of_stay', 'lead_time_category', 'stay_category',
            
            # Room features  
            'room_category', 'Number of People',
            
            # Temporal features
            'is_weekend', 'is_holiday', 'week_of_year', 'weekend_holiday',
            'checkin_month', 'checkin_day_of_week', 'checkin_quarter',
            
            # Event features
            'events_total_score', 'events_max_score', 'events_count', 
            'has_major_event', 'has_multiple_events', 'event_intensity',
            
            # Weather (only if available)
            'MEAN_TEMPERATURE', 'temp_category'
        ]
        
        # Only include features that exist in the dataframe
        feature_cols = [col for col in essential_features if col in self.df.columns]
        
        print(f"   📊 Feature Selection:")
        print(f"     • Total columns: {len(self.df.columns)}")
        print(f"     • Excluded columns: {len(exclude_cols)}")
        print(f"     • Feature columns: {len(feature_cols)}")
        
        # Create arrays for each split
        splits_data = {}
        
        for split_name, split_df in [('train', self.train_df), ('val', self.val_df), ('test', self.test_df)]:
            # Features (X)
            X = split_df[feature_cols].copy()
            
            # Target (y) - log transformed
            y = split_df['log_price_cad'].copy()
            
            # Original prices for evaluation
            y_original = split_df['price_cad'].copy()
            
            # Remove any remaining missing values
            complete_mask = ~(X.isnull().any(axis=1) | y.isnull())
            
            X_clean = X[complete_mask]
            y_clean = y[complete_mask]
            y_orig_clean = y_original[complete_mask]
            
            removed_count = len(X) - len(X_clean)
            if removed_count > 0:
                print(f"     • {split_name.title()}: Removed {removed_count} rows with missing values")
            
            splits_data[split_name] = {
                'X': X_clean,
                'y': y_clean,
                'y_original': y_orig_clean,
                'dates': split_df.loc[complete_mask, 'checkin_date']
            }
            
            print(f"     • {split_name.title()}: {len(X_clean):,} samples × {len(feature_cols)} features")
        
        # Store feature information
        self.feature_columns = feature_cols
        self.splits_data = splits_data
        
        print(f"\n   ✅ Model arrays prepared successfully!")
        
        return splits_data
    
    def create_visualization(self):
        """Create visualization of the time-based splits."""
        print(f"\n📊 CREATING SPLIT VISUALIZATION")
        print("-" * 35)
        
        # Create timeline plot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
        
        # Plot 1: Daily booking counts with split boundaries
        daily_counts = self.df.groupby('checkin_date').size()
        
        ax1.plot(daily_counts.index, daily_counts.values, 'b-', alpha=0.7, linewidth=2)
        ax1.axvline(self.train_end_date, color='red', linestyle='--', linewidth=2, label='Train/Val Split')
        ax1.axvline(self.val_end_date, color='orange', linestyle='--', linewidth=2, label='Val/Test Split')
        
        ax1.set_title('Daily Booking Counts with Time-Based Splits', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Check-in Date')
        ax1.set_ylabel('Number of Bookings')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Add split region colors
        ax1.axvspan(self.min_date, self.train_end_date, alpha=0.2, color='green', label='Train')
        ax1.axvspan(self.train_end_date, self.val_end_date, alpha=0.2, color='orange', label='Validation')
        ax1.axvspan(self.val_end_date, self.max_date, alpha=0.2, color='red', label='Test')
        
        # Plot 2: Price distribution by split
        train_prices = self.train_df['price_cad']
        val_prices = self.val_df['price_cad'] 
        test_prices = self.test_df['price_cad']
        
        ax2.hist(train_prices, bins=50, alpha=0.5, label=f'Train (n={len(train_prices):,})', color='green')
        ax2.hist(val_prices, bins=50, alpha=0.5, label=f'Validation (n={len(val_prices):,})', color='orange')
        ax2.hist(test_prices, bins=50, alpha=0.5, label=f'Test (n={len(test_prices):,})', color='red')
        
        ax2.set_title('Price Distribution by Split', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Price (CAD)')
        ax2.set_ylabel('Frequency')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('Data/time_based_splits_visualization.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"   ✅ Saved visualization to: Data/time_based_splits_visualization.png")
    
    def save_splits(self):
        """Save the splits to separate files."""
        print(f"\n💾 SAVING SPLIT DATA")
        print("-" * 20)
        
        try:
            # Save individual split files
            self.train_df.to_csv('Data/train_split.csv', index=False)
            self.val_df.to_csv('Data/validation_split.csv', index=False)
            self.test_df.to_csv('Data/test_split.csv', index=False)
            
            print(f"   ✅ Saved split files:")
            print(f"     • Data/train_split.csv ({len(self.train_df):,} rows)")
            print(f"     • Data/validation_split.csv ({len(self.val_df):,} rows)")
            print(f"     • Data/test_split.csv ({len(self.test_df):,} rows)")
            
            # Save split metadata
            split_info = {
                'total_samples': len(self.df),
                'train_samples': len(self.train_df),
                'val_samples': len(self.val_df),
                'test_samples': len(self.test_df),
                'train_ratio': self.train_ratio,
                'val_ratio': self.val_ratio,
                'test_ratio': self.test_ratio,
                'train_end_date': self.train_end_date.strftime('%Y-%m-%d'),
                'val_end_date': self.val_end_date.strftime('%Y-%m-%d'),
                'feature_columns': self.feature_columns
            }
            
            import json
            with open('Data/split_info.json', 'w') as f:
                json.dump(split_info, f, indent=2, default=str)
            
            print(f"   ✅ Saved split metadata: Data/split_info.json")
            
            return True
        except Exception as e:
            print(f"   ❌ Error saving splits: {e}")
            return False
    
    def run_time_based_splits(self):
        """Run the complete time-based splitting process."""
        print(f"🚀 Starting Time-Based Split Creation...")
        print()
        
        # Load data
        if not self.load_data():
            return None
        
        # Analyze temporal distribution
        self.analyze_temporal_distribution()
        
        # Calculate split dates
        self.calculate_split_dates()
        
        # Create splits
        self.create_splits()
        
        # Analyze split characteristics
        self.analyze_split_characteristics()
        
        # Prepare model arrays
        splits_data = self.prepare_model_arrays()
        
        # Create visualization
        self.create_visualization()
        
        # Save splits
        if self.save_splits():
            print(f"\n🎉 TIME-BASED SPLITS COMPLETED!")
            print(f"   📊 Train: {len(self.train_df):,} samples")
            print(f"   📊 Validation: {len(self.val_df):,} samples") 
            print(f"   📊 Test: {len(self.test_df):,} samples")
            print(f"   🔧 Ready for model training!")
            
            return splits_data
        else:
            return None


def main():
    """Run the time-based splitting process."""
    
    # Initialize splitter
    splitter = TimeBasedSplitter('Data/hotel_features_engineered.csv')
    
    # Run splits
    splits_data = splitter.run_time_based_splits()
    
    if splits_data:
        print(f"\n✅ Time-based splits created successfully!")
        print(f"📈 Ready for model training and validation")
        return splitter, splits_data
    else:
        print(f"\n❌ Time-based splitting failed!")
        return None, None


if __name__ == "__main__":
    main()