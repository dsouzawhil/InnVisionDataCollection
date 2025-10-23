"""
Random Train/Validation/Test Splits for Hotel Price Prediction
============================================================

This script creates proper random splits for hotel price prediction models.
Since this is a price prediction task (not time-series forecasting) and 
booking_lead_time already captures temporal relationships, random splits 
are more appropriate than time-based splits.

Benefits of Random Splits for this use case:
1. Better balance across weekend/weekday bookings
2. More representative price distributions in each split
3. Reduced domain shift between train/validation/test
4. Better generalization to diverse booking scenarios

Split Strategy: 60% Train / 20% Validation / 20% Test
Random seed: 42 (for reproducibility)

Author: ML Pipeline - Random Splits
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

class RandomSplitter:
    def __init__(self, data_file='Data/hotel_features_engineered.csv'):
        """Initialize random splitter."""
        self.data_file = data_file
        self.df = None
        
        # Split ratios (must sum to 1.0)
        self.train_ratio = 0.60    # 60% for training
        self.val_ratio = 0.20      # 20% for validation  
        self.test_ratio = 0.20     # 20% for testing
        
        # Random seed for reproducibility
        self.random_state = 42
        
        print("🎲 RANDOM TRAIN/VALIDATION/TEST SPLITS")
        print("=" * 45)
        
    def load_data(self):
        """Load the engineered dataset."""
        print("📂 Loading engineered dataset...")
        
        try:
            self.df = pd.read_csv(self.data_file)
            print(f"   ✅ Loaded {len(self.df):,} rows × {len(self.df.columns)} columns")
            
            # Verify we have the target and key features
            required_columns = ['log_price_cad', 'price_cad']
            missing_cols = [col for col in required_columns if col not in self.df.columns]
            
            if missing_cols:
                print(f"   ❌ Missing required columns: {missing_cols}")
                return False
            
            print(f"   💰 Target column: log_price_cad")
            print(f"   📊 Target range: {self.df['log_price_cad'].min():.3f} to {self.df['log_price_cad'].max():.3f} (log scale)")
            print(f"   💵 Price range: ${self.df['price_cad'].min():.2f} to ${self.df['price_cad'].max():.2f} CAD")
            
            return True
        except Exception as e:
            print(f"   ❌ Error loading data: {e}")
            return False
    
    def analyze_data_distribution(self):
        """Analyze the distribution of key features before splitting."""
        print(f"\n📊 DATA DISTRIBUTION ANALYSIS")
        print("-" * 35)
        
        total_samples = len(self.df)
        
        print(f"   📈 Dataset Overview:")
        print(f"     • Total samples: {total_samples:,}")
        print(f"     • Total features: {len(self.df.columns)}")
        
        # Target distribution
        target_mean = self.df['log_price_cad'].mean()
        target_std = self.df['log_price_cad'].std()
        price_mean = self.df['price_cad'].mean()
        price_std = self.df['price_cad'].std()
        
        print(f"\n   🎯 Target Distribution:")
        print(f"     • Log target: μ={target_mean:.3f}, σ={target_std:.3f}")
        print(f"     • Original prices: μ=${price_mean:.2f}, σ=${price_std:.2f} CAD")
        
        # Key categorical distributions
        if 'is_weekend' in self.df.columns:
            weekend_pct = (self.df['is_weekend'].sum() / total_samples) * 100
            print(f"     • Weekend bookings: {weekend_pct:.1f}%")
        
        if 'is_holiday' in self.df.columns:
            holiday_pct = (self.df['is_holiday'].sum() / total_samples) * 100
            print(f"     • Holiday bookings: {holiday_pct:.1f}%")
        
        if 'district' in self.df.columns:
            top_district = self.df['district'].value_counts().index[0]
            top_district_pct = (self.df['district'].value_counts().iloc[0] / total_samples) * 100
            print(f"     • Top district: {top_district} ({top_district_pct:.1f}%)")
        
        if 'room_category' in self.df.columns:
            top_room = self.df['room_category'].value_counts().index[0]
            top_room_pct = (self.df['room_category'].value_counts().iloc[0] / total_samples) * 100
            print(f"     • Top room category: {top_room} ({top_room_pct:.1f}%)")
    
    def prepare_features(self):
        """Prepare feature set for modeling."""
        print(f"\n🔧 FEATURE PREPARATION")
        print("-" * 25)
        
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
            
            # Weather (only if available and not too many missing values)
            'MEAN_TEMPERATURE', 'temp_category'
        ]
        
        # Only include features that exist in the dataframe
        available_features = [col for col in essential_features if col in self.df.columns]
        
        print(f"   📊 Feature Selection:")
        print(f"     • Total available features: {len(available_features)}")
        
        # Check missing values for selected features
        missing_info = []
        for feature in available_features:
            missing_count = self.df[feature].isnull().sum()
            missing_pct = (missing_count / len(self.df)) * 100
            missing_info.append((feature, missing_count, missing_pct))
        
        # Show features with missing values
        high_missing = [(f, c, p) for f, c, p in missing_info if p > 5]
        if high_missing:
            print(f"   ⚠️ Features with >5% missing values:")
            for feature, count, pct in high_missing:
                print(f"     • {feature}: {count:,} ({pct:.1f}%)")
        
        # Remove features with too many missing values (>15%)
        good_features = [f for f, c, p in missing_info if p <= 15]
        removed_features = [f for f, c, p in missing_info if p > 15]
        
        if removed_features:
            print(f"   🗑️ Removing features with >15% missing: {removed_features}")
        
        print(f"   ✅ Final feature count: {len(good_features)}")
        
        # Store feature information
        self.feature_columns = good_features
        
        return good_features
    
    def create_random_splits(self):
        """Create random train/validation/test splits."""
        print(f"\n✂️ CREATING RANDOM SPLITS")
        print("-" * 30)
        
        # Prepare the data
        X = self.df[self.feature_columns].copy()
        y = self.df['log_price_cad'].copy()
        y_original = self.df['price_cad'].copy()
        
        # Remove missing values
        complete_mask = ~(X.isnull().any(axis=1) | y.isnull())
        X_clean = X[complete_mask]
        y_clean = y[complete_mask]
        y_orig_clean = y_original[complete_mask]
        
        removed_count = len(X) - len(X_clean)
        if removed_count > 0:
            print(f"   🧹 Removed {removed_count:,} rows with missing values")
        
        print(f"   📊 Clean dataset: {len(X_clean):,} samples × {len(self.feature_columns)} features")
        
        # First split: separate train from temp (val + test)
        X_train, X_temp, y_train, y_temp, y_orig_train, y_orig_temp = train_test_split(
            X_clean, y_clean, y_orig_clean,
            test_size=(self.val_ratio + self.test_ratio),  # 40% for val + test
            random_state=self.random_state,
            stratify=None  # We could stratify by price bins if needed
        )
        
        # Second split: separate validation from test
        X_val, X_test, y_val, y_test, y_orig_val, y_orig_test = train_test_split(
            X_temp, y_temp, y_orig_temp,
            test_size=self.test_ratio / (self.val_ratio + self.test_ratio),  # 50% of temp = 20% of total
            random_state=self.random_state + 1  # Different seed for second split
        )
        
        print(f"   📊 Split Sizes:")
        print(f"     • Train: {len(X_train):,} samples ({len(X_train)/len(X_clean):.1%})")
        print(f"     • Validation: {len(X_val):,} samples ({len(X_val)/len(X_clean):.1%})")
        print(f"     • Test: {len(X_test):,} samples ({len(X_test)/len(X_clean):.1%})")
        print(f"     • Total: {len(X_train) + len(X_val) + len(X_test):,} samples")
        
        # Store splits
        self.splits_data = {
            'train': {
                'X': X_train,
                'y': y_train,
                'y_original': y_orig_train
            },
            'val': {
                'X': X_val,
                'y': y_val,
                'y_original': y_orig_val
            },
            'test': {
                'X': X_test,
                'y': y_test,
                'y_original': y_orig_test
            }
        }
        
        return self.splits_data
    
    def analyze_split_balance(self):
        """Analyze the balance and distributions across splits."""
        print(f"\n🔍 SPLIT BALANCE ANALYSIS")
        print("-" * 30)
        
        # Target distribution analysis
        print(f"   📈 Target Distribution (log_price_cad):")
        for split_name, split_data in self.splits_data.items():
            y = split_data['y']
            y_orig = split_data['y_original']
            
            log_mean = y.mean()
            log_std = y.std()
            price_mean = y_orig.mean()
            price_std = y_orig.std()
            
            print(f"     • {split_name.title():<12}: μ={log_mean:.3f}, σ={log_std:.3f} | ${price_mean:.2f}±${price_std:.2f} CAD")
        
        # Feature distribution analysis for key categorical features
        categorical_features = ['is_weekend', 'is_holiday', 'district', 'room_category']
        
        for feature in categorical_features:
            if feature in self.feature_columns:
                print(f"\n   🏷️ {feature.replace('_', ' ').title()} Distribution:")
                
                for split_name, split_data in self.splits_data.items():
                    X = split_data['X']
                    
                    if feature == 'is_weekend' or feature == 'is_holiday':
                        # Boolean features
                        true_pct = (X[feature].sum() / len(X)) * 100
                        print(f"     • {split_name.title():<12}: {true_pct:.1f}% True")
                    else:
                        # Categorical features - show top category
                        if len(X) > 0:
                            top_cat = X[feature].value_counts().index[0]
                            top_pct = (X[feature].value_counts().iloc[0] / len(X)) * 100
                            print(f"     • {split_name.title():<12}: {top_cat} ({top_pct:.1f}%)")
    
    def create_visualization(self):
        """Create visualization of the random splits."""
        print(f"\n📊 CREATING SPLIT VISUALIZATION")
        print("-" * 35)
        
        # Create comparison plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Random Splits Analysis', fontsize=16, fontweight='bold')
        
        # Plot 1: Price distribution by split
        train_prices = self.splits_data['train']['y_original']
        val_prices = self.splits_data['val']['y_original']
        test_prices = self.splits_data['test']['y_original']
        
        axes[0, 0].hist(train_prices, bins=50, alpha=0.7, label=f'Train (n={len(train_prices):,})', color='green')
        axes[0, 0].hist(val_prices, bins=50, alpha=0.7, label=f'Validation (n={len(val_prices):,})', color='orange')
        axes[0, 0].hist(test_prices, bins=50, alpha=0.7, label=f'Test (n={len(test_prices):,})', color='red')
        
        axes[0, 0].set_title('Price Distribution by Split')
        axes[0, 0].set_xlabel('Price (CAD)')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Log price distribution by split
        train_log = self.splits_data['train']['y']
        val_log = self.splits_data['val']['y']
        test_log = self.splits_data['test']['y']
        
        axes[0, 1].hist(train_log, bins=50, alpha=0.7, label='Train', color='green')
        axes[0, 1].hist(val_log, bins=50, alpha=0.7, label='Validation', color='orange')
        axes[0, 1].hist(test_log, bins=50, alpha=0.7, label='Test', color='red')
        
        axes[0, 1].set_title('Log Price Distribution by Split')
        axes[0, 1].set_xlabel('Log(Price + 1)')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Weekend distribution by split
        if 'is_weekend' in self.feature_columns:
            weekend_data = []
            split_names = []
            
            for split_name, split_data in self.splits_data.items():
                weekend_pct = (split_data['X']['is_weekend'].sum() / len(split_data['X'])) * 100
                weekend_data.append(weekend_pct)
                split_names.append(split_name.title())
            
            bars = axes[1, 0].bar(split_names, weekend_data, color=['green', 'orange', 'red'], alpha=0.7)
            axes[1, 0].set_title('Weekend Booking Distribution')
            axes[1, 0].set_ylabel('Percentage (%)')
            axes[1, 0].grid(True, alpha=0.3)
            
            # Add value labels on bars
            for bar, value in zip(bars, weekend_data):
                axes[1, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                               f'{value:.1f}%', ha='center', va='bottom')
        
        # Plot 4: Distance distribution by split
        if 'Distance from attraction' in self.feature_columns:
            train_dist = self.splits_data['train']['X']['Distance from attraction']
            val_dist = self.splits_data['val']['X']['Distance from attraction']
            test_dist = self.splits_data['test']['X']['Distance from attraction']
            
            axes[1, 1].boxplot([train_dist, val_dist, test_dist], 
                              labels=['Train', 'Validation', 'Test'],
                              patch_artist=True)
            
            axes[1, 1].set_title('Distance from Attraction Distribution')
            axes[1, 1].set_ylabel('Distance (miles)')
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('Data/random_splits_visualization.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"   ✅ Saved visualization to: Data/random_splits_visualization.png")
    
    def save_splits(self):
        """Save the splits to separate files."""
        print(f"\n💾 SAVING SPLIT DATA")
        print("-" * 20)
        
        try:
            # Create DataFrames for each split
            train_df = self.splits_data['train']['X'].copy()
            train_df['log_price_cad'] = self.splits_data['train']['y']
            train_df['price_cad'] = self.splits_data['train']['y_original']
            
            val_df = self.splits_data['val']['X'].copy()
            val_df['log_price_cad'] = self.splits_data['val']['y']
            val_df['price_cad'] = self.splits_data['val']['y_original']
            
            test_df = self.splits_data['test']['X'].copy()
            test_df['log_price_cad'] = self.splits_data['test']['y']
            test_df['price_cad'] = self.splits_data['test']['y_original']
            
            # Save individual split files
            train_df.to_csv('Data/train_split.csv', index=False)
            val_df.to_csv('Data/validation_split.csv', index=False)
            test_df.to_csv('Data/test_split.csv', index=False)
            
            print(f"   ✅ Saved split files:")
            print(f"     • Data/train_split.csv ({len(train_df):,} rows)")
            print(f"     • Data/validation_split.csv ({len(val_df):,} rows)")
            print(f"     • Data/test_split.csv ({len(test_df):,} rows)")
            
            # Save split metadata
            split_info = {
                'split_type': 'random',
                'random_state': self.random_state,
                'total_samples': sum(len(split_data['X']) for split_data in self.splits_data.values()),
                'train_samples': len(self.splits_data['train']['X']),
                'val_samples': len(self.splits_data['val']['X']),
                'test_samples': len(self.splits_data['test']['X']),
                'train_ratio': self.train_ratio,
                'val_ratio': self.val_ratio,
                'test_ratio': self.test_ratio,
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
    
    def run_random_splits(self):
        """Run the complete random splitting process."""
        print(f"🚀 Starting Random Split Creation...")
        print()
        
        # Load data
        if not self.load_data():
            return None
        
        # Analyze data distribution
        self.analyze_data_distribution()
        
        # Prepare features
        self.prepare_features()
        
        # Create splits
        splits_data = self.create_random_splits()
        
        # Analyze split balance
        self.analyze_split_balance()
        
        # Create visualization
        self.create_visualization()
        
        # Save splits
        if self.save_splits():
            print(f"\n🎉 RANDOM SPLITS COMPLETED!")
            print(f"   📊 Train: {len(self.splits_data['train']['X']):,} samples")
            print(f"   📊 Validation: {len(self.splits_data['val']['X']):,} samples") 
            print(f"   📊 Test: {len(self.splits_data['test']['X']):,} samples")
            print(f"   🎲 Random seed: {self.random_state}")
            print(f"   🔧 Ready for model training!")
            
            return splits_data
        else:
            return None


def main():
    """Run the random splitting process."""
    
    # Initialize splitter
    splitter = RandomSplitter('Data/hotel_features_engineered.csv')
    
    # Run splits
    splits_data = splitter.run_random_splits()
    
    if splits_data:
        print(f"\n✅ Random splits created successfully!")
        print(f"📈 Ready for model training and validation")
        print(f"🎯 Better balance and generalization expected vs time-based splits")
        return splitter, splits_data
    else:
        print(f"\n❌ Random splitting failed!")
        return None, None


if __name__ == "__main__":
    main()