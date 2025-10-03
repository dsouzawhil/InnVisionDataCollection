"""
Exploratory Data Analysis for Toronto Hotel Price Prediction
==========================================================

This script performs comprehensive EDA on the toronto_unified_hotel_analysis.csv dataset
to understand patterns and relationships that will inform a hotel price prediction model.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set up plotting style
plt.style.use('default')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)

def load_and_examine_data():
    """Load the dataset and examine basic structure."""
    print("🏨 TORONTO HOTEL PRICE PREDICTION - EXPLORATORY DATA ANALYSIS")
    print("=" * 70)
    
    # Load the dataset
    df = pd.read_csv('Data/toronto_unified_hotel_analysis.csv')
    
    print(f"\n📊 DATASET OVERVIEW")
    print(f"Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
    
    # Basic info
    print(f"\n📋 COLUMN INFORMATION")
    print(df.info())
    
    # Show first few rows
    print(f"\n👀 FIRST 5 ROWS")
    print(df.head())
    
    # Data types summary
    print(f"\n🔢 DATA TYPES SUMMARY")
    dtype_counts = df.dtypes.value_counts()
    for dtype, count in dtype_counts.items():
        print(f"   {dtype}: {count} columns")
    
    return df

def analyze_target_variable(df):
    """Analyze the target variable (price) distribution."""
    print(f"\n💰 TARGET VARIABLE ANALYSIS (PRICE)")
    print("=" * 50)
    
    # Focus on CAD prices as the target
    target_col = 'price_cad'
    
    # Basic statistics
    print(f"\n📊 PRICE STATISTICS (CAD)")
    price_stats = df[target_col].describe()
    print(price_stats)
    
    # Additional statistics
    print(f"\nAdditional metrics:")
    print(f"   Skewness: {df[target_col].skew():.3f}")
    print(f"   Kurtosis: {df[target_col].kurtosis():.3f}")
    print(f"   Missing values: {df[target_col].isnull().sum()}")
    print(f"   Unique values: {df[target_col].nunique()}")
    
    # Price ranges
    print(f"\n💵 PRICE RANGES")
    price_ranges = [
        ("Budget", 0, 100),
        ("Mid-range", 100, 300),
        ("Upscale", 300, 600),
        ("Luxury", 600, float('inf'))
    ]
    
    for category, min_price, max_price in price_ranges:
        if max_price == float('inf'):
            count = (df[target_col] >= min_price).sum()
            pct = (count / len(df)) * 100
            print(f"   {category} (${min_price}+): {count:,} hotels ({pct:.1f}%)")
        else:
            count = ((df[target_col] >= min_price) & (df[target_col] < max_price)).sum()
            pct = (count / len(df)) * 100
            print(f"   {category} (${min_price}-${max_price}): {count:,} hotels ({pct:.1f}%)")
    
    # Identify outliers
    Q1 = df[target_col].quantile(0.25)
    Q3 = df[target_col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers = df[(df[target_col] < lower_bound) | (df[target_col] > upper_bound)]
    print(f"\n🚨 OUTLIERS (IQR Method)")
    print(f"   Lower bound: ${lower_bound:.2f}")
    print(f"   Upper bound: ${upper_bound:.2f}")
    print(f"   Outliers: {len(outliers):,} ({len(outliers)/len(df)*100:.1f}%)")
    
    # Visualize price distribution
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Price Distribution Analysis', fontsize=16, fontweight='bold')
    
    # Histogram
    axes[0, 0].hist(df[target_col], bins=50, edgecolor='black', alpha=0.7)
    axes[0, 0].set_title('Price Distribution (All Data)')
    axes[0, 0].set_xlabel('Price (CAD)')
    axes[0, 0].set_ylabel('Frequency')
    
    # Log scale histogram
    axes[0, 1].hist(np.log1p(df[target_col]), bins=50, edgecolor='black', alpha=0.7)
    axes[0, 1].set_title('Price Distribution (Log Scale)')
    axes[0, 1].set_xlabel('Log(Price + 1)')
    axes[0, 1].set_ylabel('Frequency')
    
    # Box plot
    axes[1, 0].boxplot(df[target_col])
    axes[1, 0].set_title('Price Box Plot')
    axes[1, 0].set_ylabel('Price (CAD)')
    
    # Price without extreme outliers
    price_filtered = df[(df[target_col] >= Q1 - 3*IQR) & (df[target_col] <= Q3 + 3*IQR)][target_col]
    axes[1, 1].hist(price_filtered, bins=50, edgecolor='black', alpha=0.7)
    axes[1, 1].set_title('Price Distribution (Extreme Outliers Removed)')
    axes[1, 1].set_xlabel('Price (CAD)')
    axes[1, 1].set_ylabel('Frequency')
    
    plt.tight_layout()
    plt.savefig('hotel_price_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return df

def analyze_categorical_features(df):
    """Analyze categorical variables and their relationship with price."""
    print(f"\n🏷️ CATEGORICAL FEATURES ANALYSIS")
    print("=" * 50)
    
    # Identify categorical columns
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    # Add some numerical columns that are actually categorical
    categorical_cols.extend(['Number of People', 'length_of_stay', 'is_weekend', 
                           'has_major_event', 'has_multiple_events'])
    
    # Remove non-relevant columns
    exclude_cols = ['Date', 'Check-in Date', 'Check-out Date', 'Hotel name', 'Address']
    categorical_cols = [col for col in categorical_cols if col in df.columns and col not in exclude_cols]
    
    print(f"\n📋 CATEGORICAL FEATURES: {len(categorical_cols)}")
    for col in categorical_cols:
        print(f"   • {col}")
    
    # Analyze each categorical feature
    for col in categorical_cols[:8]:  # Limit to first 8 for readability
        print(f"\n🔍 {col.upper()}")
        print("-" * (len(col) + 4))
        
        # Value counts
        value_counts = df[col].value_counts()
        print(f"Unique values: {df[col].nunique()}")
        print(f"Top 5 values:")
        for idx, (value, count) in enumerate(value_counts.head().items()):
            pct = (count / len(df)) * 100
            print(f"   {idx+1}. {value}: {count:,} ({pct:.1f}%)")
        
        # Price analysis by category
        if df[col].nunique() <= 20:  # Only for features with reasonable number of categories
            price_by_category = df.groupby(col)['price_cad'].agg(['mean', 'median', 'std', 'count']).round(2)
            price_by_category = price_by_category.sort_values('mean', ascending=False)
            print(f"\nPrice by {col}:")
            print(price_by_category.head())

def analyze_numerical_features(df):
    """Analyze numerical features and their correlations with price."""
    print(f"\n🔢 NUMERICAL FEATURES ANALYSIS")
    print("=" * 50)
    
    # Identify numerical columns
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Remove price_usd to focus on price_cad as target
    if 'price_usd' in numerical_cols:
        numerical_cols.remove('price_usd')
    
    print(f"\n📊 NUMERICAL FEATURES: {len(numerical_cols)}")
    for col in numerical_cols:
        print(f"   • {col}")
    
    # Correlation with target variable
    print(f"\n🎯 CORRELATION WITH PRICE (CAD)")
    correlations = df[numerical_cols].corr()['price_cad'].sort_values(key=abs, ascending=False)
    
    print("\nTop correlations (absolute value):")
    for feature, corr in correlations.items():
        if feature != 'price_cad':
            print(f"   {feature}: {corr:+.3f}")
    
    # Correlation matrix visualization
    plt.figure(figsize=(14, 12))
    
    # Select top correlated features for visualization
    top_features = correlations.abs().sort_values(ascending=False).head(15).index.tolist()
    corr_matrix = df[top_features].corr()
    
    # Create heatmap
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, 
                mask=mask,
                annot=True, 
                cmap='RdBu_r', 
                center=0,
                square=True,
                fmt='.2f',
                cbar_kws={"shrink": .8})
    
    plt.title('Feature Correlation Matrix (Top 15 Features)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('correlation_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Statistical summary of key numerical features
    key_features = ['price_cad', 'booking_lead_time', 'length_of_stay', 
                   'events_total_score', 'events_count', 'MEAN_TEMPERATURE']
    key_features = [col for col in key_features if col in df.columns]
    
    print(f"\n📈 KEY NUMERICAL FEATURES SUMMARY")
    print(df[key_features].describe().round(2))

def analyze_temporal_patterns(df):
    """Analyze temporal patterns in the data."""
    print(f"\n📅 TEMPORAL PATTERNS ANALYSIS")
    print("=" * 50)
    
    # Convert date columns
    df['Check-in Date'] = pd.to_datetime(df['Check-in Date'])
    df['Check-out Date'] = pd.to_datetime(df['Check-out Date'])
    
    # Extract temporal features for analysis
    df['checkin_month'] = df['Check-in Date'].dt.month
    df['checkin_day_of_week'] = df['Check-in Date'].dt.day_name()
    df['checkin_week'] = df['Check-in Date'].dt.isocalendar().week
    
    # Monthly price patterns
    print(f"\n📊 MONTHLY PRICE PATTERNS")
    monthly_prices = df.groupby('checkin_month')['price_cad'].agg(['mean', 'median', 'count']).round(2)
    monthly_prices.index = pd.to_datetime(monthly_prices.index, format='%m').strftime('%B')
    print(monthly_prices)
    
    # Day of week patterns
    print(f"\n📊 DAY OF WEEK PATTERNS")
    dow_prices = df.groupby('checkin_day_of_week')['price_cad'].agg(['mean', 'median', 'count']).round(2)
    print(dow_prices)
    
    # Weekend vs weekday analysis
    print(f"\n🎉 WEEKEND VS WEEKDAY ANALYSIS")
    weekend_analysis = df.groupby('is_weekend')['price_cad'].agg(['mean', 'median', 'std', 'count']).round(2)
    weekend_analysis.index = ['Weekday', 'Weekend']
    print(weekend_analysis)
    
    # Booking lead time analysis
    print(f"\n⏰ BOOKING LEAD TIME ANALYSIS")
    if 'booking_lead_time' in df.columns:
        lead_time_ranges = [
            ("Last minute", 0, 7),
            ("1-2 weeks", 7, 14),
            ("2-4 weeks", 14, 28),
            ("1-2 months", 28, 60),
            ("2+ months", 60, float('inf'))
        ]
        
        for category, min_days, max_days in lead_time_ranges:
            if max_days == float('inf'):
                mask = df['booking_lead_time'] >= min_days
            else:
                mask = (df['booking_lead_time'] >= min_days) & (df['booking_lead_time'] < max_days)
            
            subset = df[mask]
            if len(subset) > 0:
                avg_price = subset['price_cad'].mean()
                count = len(subset)
                pct = (count / len(df)) * 100
                print(f"   {category}: ${avg_price:.2f} avg ({count:,} bookings, {pct:.1f}%)")

def analyze_event_impact(df):
    """Analyze the impact of events on hotel pricing."""
    print(f"\n🎭 EVENT IMPACT ANALYSIS")
    print("=" * 50)
    
    # Event features analysis
    event_cols = [col for col in df.columns if col.startswith('events_') or col.startswith('has_')]
    
    print(f"\n📊 EVENT FEATURES SUMMARY")
    for col in event_cols:
        if col in df.columns:
            non_zero = (df[col] > 0).sum()
            pct = (non_zero / len(df)) * 100
            print(f"   {col}: {non_zero:,} non-zero values ({pct:.1f}%)")
    
    # Hotels with vs without events
    print(f"\n🎯 HOTELS WITH VS WITHOUT EVENTS")
    
    has_events = df['events_count'] > 0
    
    with_events = df[has_events]['price_cad']
    without_events = df[~has_events]['price_cad']
    
    print(f"Hotels with events during stay:")
    print(f"   Count: {len(with_events):,}")
    print(f"   Average price: ${with_events.mean():.2f}")
    print(f"   Median price: ${with_events.median():.2f}")
    
    print(f"\nHotels without events during stay:")
    print(f"   Count: {len(without_events):,}")
    print(f"   Average price: ${without_events.mean():.2f}")
    print(f"   Median price: ${without_events.median():.2f}")
    
    if len(with_events) > 0 and len(without_events) > 0:
        price_diff = with_events.mean() - without_events.mean()
        price_diff_pct = (price_diff / without_events.mean()) * 100
        print(f"\nPrice difference: ${price_diff:.2f} ({price_diff_pct:+.1f}%)")
    
    # Event score impact
    if 'events_total_score' in df.columns:
        print(f"\n📈 EVENT SCORE IMPACT")
        
        # Create event score ranges
        event_score_ranges = [
            ("No events", 0, 0),
            ("Low impact", 0, 10),
            ("Medium impact", 10, 50),
            ("High impact", 50, float('inf'))
        ]
        
        for category, min_score, max_score in event_score_ranges:
            if category == "No events":
                mask = df['events_total_score'] == 0
            elif max_score == float('inf'):
                mask = df['events_total_score'] > min_score
            else:
                mask = (df['events_total_score'] > min_score) & (df['events_total_score'] <= max_score)
            
            subset = df[mask]
            if len(subset) > 0:
                avg_price = subset['price_cad'].mean()
                count = len(subset)
                pct = (count / len(df)) * 100
                print(f"   {category}: ${avg_price:.2f} avg ({count:,} hotels, {pct:.1f}%)")

def analyze_weather_effects(df):
    """Analyze weather effects on pricing."""
    print(f"\n🌤️ WEATHER EFFECTS ANALYSIS")
    print("=" * 50)
    
    weather_cols = [col for col in df.columns if col.isupper() and col.startswith(('MEAN_', 'MIN_', 'MAX_', 'TOTAL_', 'SNOW_', 'SPEED_'))]
    
    if not weather_cols:
        print("   ⚠️ No weather data available")
        return
    
    print(f"\n📊 WEATHER FEATURES SUMMARY")
    weather_summary = df[weather_cols].describe().round(2)
    print(weather_summary)
    
    # Temperature impact on pricing
    if 'MEAN_TEMPERATURE' in df.columns:
        print(f"\n🌡️ TEMPERATURE IMPACT ON PRICING")
        
        # Create temperature ranges
        temp_ranges = [
            ("Very Cold", -float('inf'), -10),
            ("Cold", -10, 0),
            ("Cool", 0, 10),
            ("Mild", 10, 20),
            ("Warm", 20, 30),
            ("Hot", 30, float('inf'))
        ]
        
        for category, min_temp, max_temp in temp_ranges:
            if min_temp == -float('inf'):
                mask = df['MEAN_TEMPERATURE'] < max_temp
            elif max_temp == float('inf'):
                mask = df['MEAN_TEMPERATURE'] >= min_temp
            else:
                mask = (df['MEAN_TEMPERATURE'] >= min_temp) & (df['MEAN_TEMPERATURE'] < max_temp)
            
            subset = df[mask]
            if len(subset) > 0:
                avg_price = subset['price_cad'].mean()
                avg_temp = subset['MEAN_TEMPERATURE'].mean()
                count = len(subset)
                pct = (count / len(df)) * 100
                print(f"   {category} ({avg_temp:.1f}°C avg): ${avg_price:.2f} avg ({count:,} bookings, {pct:.1f}%)")

def analyze_missing_data(df):
    """Analyze missing data patterns."""
    print(f"\n❓ MISSING DATA ANALYSIS")
    print("=" * 50)
    
    # Calculate missing data
    missing_data = df.isnull().sum()
    missing_pct = (missing_data / len(df)) * 100
    
    missing_df = pd.DataFrame({
        'Column': missing_data.index,
        'Missing Count': missing_data.values,
        'Missing Percentage': missing_pct.values
    })
    
    missing_df = missing_df[missing_df['Missing Count'] > 0].sort_values('Missing Percentage', ascending=False)
    
    if len(missing_df) > 0:
        print(f"\n📊 COLUMNS WITH MISSING DATA")
        print(missing_df.to_string(index=False))
        
        # Visualize missing data
        if len(missing_df) > 0:
            plt.figure(figsize=(12, 6))
            plt.barh(missing_df['Column'], missing_df['Missing Percentage'])
            plt.xlabel('Missing Percentage (%)')
            plt.title('Missing Data by Column')
            plt.tight_layout()
            plt.savefig('missing_data.png', dpi=300, bbox_inches='tight')
            plt.show()
    else:
        print("   ✅ No missing data found!")

def generate_modeling_recommendations(df):
    """Generate recommendations for model building."""
    print(f"\n🤖 MODELING RECOMMENDATIONS")
    print("=" * 70)
    
    print(f"\n🎯 TARGET VARIABLE")
    print(f"   • Use 'price_cad' as the target variable")
    print(f"   • Consider log transformation due to right skewness")
    print(f"   • Handle outliers (consider capping at 99th percentile)")
    
    print(f"\n🔧 FEATURE ENGINEERING")
    print(f"   • Create price categories (Budget/Mid-range/Upscale/Luxury)")
    print(f"   • Extract more temporal features (season, holiday indicators)")
    print(f"   • Create interaction features (events × weekend, temperature × season)")
    print(f"   • One-hot encode categorical variables with high cardinality")
    print(f"   • Consider aggregating rare categories in categorical features")
    
    print(f"\n📊 DATA PREPROCESSING")
    print(f"   • Scale numerical features (StandardScaler or RobustScaler)")
    print(f"   • Handle missing data (imputation or dropping)")
    print(f"   • Remove or cap extreme outliers")
    print(f"   • Consider feature selection based on correlation analysis")
    
    print(f"\n🧠 MODEL SELECTION")
    print(f"   • Start with Linear Regression as baseline")
    print(f"   • Try Random Forest for feature importance insights")
    print(f"   • Consider Gradient Boosting (XGBoost, LightGBM) for performance")
    print(f"   • Use cross-validation for model evaluation")
    
    print(f"\n✅ EVALUATION METRICS")
    print(f"   • Primary: RMSE (Root Mean Square Error)")
    print(f"   • Secondary: MAE (Mean Absolute Error)")
    print(f"   • Business metric: MAPE (Mean Absolute Percentage Error)")
    print(f"   • R² for explained variance")

def main():
    """Main function to run the complete EDA."""
    # Load and examine data
    df = load_and_examine_data()
    
    # Analyze target variable
    df = analyze_target_variable(df)
    
    # Analyze categorical features
    analyze_categorical_features(df)
    
    # Analyze numerical features
    analyze_numerical_features(df)
    
    # Analyze temporal patterns
    analyze_temporal_patterns(df)
    
    # Analyze event impact
    analyze_event_impact(df)
    
    # Analyze weather effects
    analyze_weather_effects(df)
    
    # Analyze missing data
    analyze_missing_data(df)
    
    # Generate modeling recommendations
    generate_modeling_recommendations(df)
    
    print(f"\n🎉 EXPLORATORY DATA ANALYSIS COMPLETE!")
    print(f"📁 Visualizations saved: hotel_price_distribution.png, correlation_matrix.png, missing_data.png")

if __name__ == "__main__":
    main()