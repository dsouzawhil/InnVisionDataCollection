"""
Exploratory Data Analysis (EDA) for Toronto Hotel Price Prediction
================================================================

This script performs comprehensive EDA on the transformed hotel dataset,
with special focus on the target variable (price_usd) and key features
that influence hotel pricing.

Analysis Steps:
1. Target Variable Analysis (price_usd)
   - Distribution analysis (histogram, box plot)
   - Skewness and normality tests
   - Log transformation evaluation
2. Feature Analysis
3. Correlation Analysis
4. Outlier Detection
5. Recommendations for ML modeling

Author: Data Analysis Pipeline
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import skew, kurtosis, normaltest
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class HotelEDA:
    def __init__(self, data_file='Data/toronto_hotels_transformed.csv'):
        """Initialize EDA with the transformed dataset."""
        self.data_file = data_file
        self.df = None
        
        print("🔍 HOTEL PRICE PREDICTION - EXPLORATORY DATA ANALYSIS")
        print("=" * 60)
        
    def load_data(self):
        """Load the transformed dataset."""
        print("📂 Loading transformed dataset...")
        
        try:
            self.df = pd.read_csv(self.data_file)
            print(f"   ✅ Loaded {len(self.df):,} rows × {len(self.df.columns)} columns")
            
            # Basic info
            print(f"   📊 Date range: {self.df['Check-in Date'].min()} to {self.df['Check-in Date'].max()}")
            print(f"   🏨 Unique hotels: {self.df['Hotel name'].nunique()}")
            print(f"   💰 Price column: price_usd")
            
            return True
        except Exception as e:
            print(f"   ❌ Error loading data: {e}")
            return False
    
    def analyze_target_variable(self):
        """
        A. Analyze Target Variable (price_usd)
        This is the most critical analysis for ML model success.
        """
        print("\n🎯 A. TARGET VARIABLE ANALYSIS (price_usd)")
        print("-" * 50)
        
        if 'price_usd' not in self.df.columns:
            print("   ❌ price_usd column not found!")
            return
        
        # Basic statistics
        price_data = self.df['price_usd'].dropna()
        
        print(f"📊 Basic Statistics:")
        print(f"   • Total records: {len(price_data):,}")
        print(f"   • Missing values: {self.df['price_usd'].isna().sum():,}")
        print(f"   • Min price: ${price_data.min():.2f}")
        print(f"   • Max price: ${price_data.max():.2f}")
        print(f"   • Mean price: ${price_data.mean():.2f}")
        print(f"   • Median price: ${price_data.median():.2f}")
        print(f"   • Standard deviation: ${price_data.std():.2f}")
        
        # Distribution characteristics
        price_skewness = skew(price_data)
        price_kurtosis = kurtosis(price_data)
        
        print(f"\n📈 Distribution Characteristics:")
        print(f"   • Skewness: {price_skewness:.3f}")
        if price_skewness > 1:
            skew_interpretation = "Highly right-skewed (long tail of expensive prices)"
        elif price_skewness > 0.5:
            skew_interpretation = "Moderately right-skewed"
        elif price_skewness > -0.5:
            skew_interpretation = "Approximately symmetric"
        else:
            skew_interpretation = "Left-skewed"
        print(f"     → {skew_interpretation}")
        
        print(f"   • Kurtosis: {price_kurtosis:.3f}")
        if price_kurtosis > 3:
            kurt_interpretation = "Heavy-tailed (more outliers than normal)"
        elif price_kurtosis < -1:
            kurt_interpretation = "Light-tailed (fewer outliers)"
        else:
            kurt_interpretation = "Normal tail behavior"
        print(f"     → {kurt_interpretation}")
        
        # Normality test
        stat, p_value = normaltest(price_data)
        print(f"   • Normality test p-value: {p_value:.2e}")
        if p_value < 0.05:
            print(f"     → Distribution is NOT normal (p < 0.05)")
        else:
            print(f"     → Distribution appears normal (p ≥ 0.05)")
        
        # Quartiles and IQR
        q1 = price_data.quantile(0.25)
        q3 = price_data.quantile(0.75)
        iqr = q3 - q1
        
        print(f"\n📦 Quartile Analysis:")
        print(f"   • Q1 (25th percentile): ${q1:.2f}")
        print(f"   • Q3 (75th percentile): ${q3:.2f}")
        print(f"   • IQR: ${iqr:.2f}")
        print(f"   • Outlier boundaries:")
        print(f"     - Lower: ${q1 - 1.5*iqr:.2f}")
        print(f"     - Upper: ${q3 + 1.5*iqr:.2f}")
        
        outliers = price_data[(price_data < q1 - 1.5*iqr) | (price_data > q3 + 1.5*iqr)]
        print(f"   • Outliers: {len(outliers):,} ({len(outliers)/len(price_data)*100:.1f}%)")
        
        # Create visualizations
        self._plot_price_distribution()
        
        # Log transformation analysis
        self._analyze_log_transformation()
        
        # Price recommendations
        self._price_modeling_recommendations()
    
    def _plot_price_distribution(self):
        """Create comprehensive price distribution plots."""
        print(f"\n📊 Creating price distribution visualizations...")
        
        price_data = self.df['price_usd'].dropna()
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Hotel Price Distribution Analysis', fontsize=16, fontweight='bold')
        
        # 1. Histogram
        axes[0, 0].hist(price_data, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, 0].axvline(price_data.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: ${price_data.mean():.2f}')
        axes[0, 0].axvline(price_data.median(), color='green', linestyle='--', linewidth=2, label=f'Median: ${price_data.median():.2f}')
        axes[0, 0].set_xlabel('Price (USD)')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Price Distribution (Histogram)')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Box plot
        bp = axes[0, 1].boxplot(price_data, patch_artist=True, vert=True)
        bp['boxes'][0].set_facecolor('lightblue')
        axes[0, 1].set_ylabel('Price (USD)')
        axes[0, 1].set_title('Price Distribution (Box Plot)')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Q-Q plot (Quantile-Quantile plot for normality)
        stats.probplot(price_data, dist="norm", plot=axes[1, 0])
        axes[1, 0].set_title('Q-Q Plot (Normal Distribution)')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Density plot with normal overlay
        axes[1, 1].hist(price_data, bins=50, density=True, alpha=0.7, color='skyblue', label='Actual Distribution')
        
        # Overlay normal distribution
        mu, sigma = price_data.mean(), price_data.std()
        x = np.linspace(price_data.min(), price_data.max(), 100)
        normal_curve = stats.norm.pdf(x, mu, sigma)
        axes[1, 1].plot(x, normal_curve, 'r-', linewidth=2, label='Normal Distribution')
        
        axes[1, 1].set_xlabel('Price (USD)')
        axes[1, 1].set_ylabel('Density')
        axes[1, 1].set_title('Price Density vs Normal Distribution')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('Data/price_distribution_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"   ✅ Saved price distribution plots to: Data/price_distribution_analysis.png")
    
    def _analyze_log_transformation(self):
        """Analyze log transformation effectiveness."""
        print(f"\n🔄 LOG TRANSFORMATION ANALYSIS")
        print("-" * 40)
        
        price_data = self.df['price_usd'].dropna()
        
        # Apply log transformation (log1p handles zeros safely)
        log_price = np.log1p(price_data)
        
        # Compare statistics
        original_skew = skew(price_data)
        log_skew = skew(log_price)
        
        original_kurt = kurtosis(price_data)
        log_kurt = kurtosis(log_price)
        
        print(f"📊 Transformation Comparison:")
        print(f"   Original price:")
        print(f"     • Skewness: {original_skew:.3f}")
        print(f"     • Kurtosis: {log_kurt:.3f}")
        
        print(f"   Log-transformed price:")
        print(f"     • Skewness: {log_skew:.3f}")
        print(f"     • Kurtosis: {log_kurt:.3f}")
        
        # Improvement metrics
        skew_improvement = abs(original_skew) - abs(log_skew)
        print(f"\n✨ Transformation Benefits:")
        print(f"   • Skewness improvement: {skew_improvement:.3f}")
        if skew_improvement > 0:
            print(f"     → Log transformation REDUCES skewness ✅")
        else:
            print(f"     → Log transformation increases skewness ❌")
        
        # Normality test on transformed data
        log_stat, log_p = normaltest(log_price)
        print(f"   • Log-price normality p-value: {log_p:.2e}")
        
        # Create comparison plot
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle('Original vs Log-Transformed Price Distribution', fontsize=14, fontweight='bold')
        
        # Original distribution
        axes[0].hist(price_data, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0].set_xlabel('Price (USD)')
        axes[0].set_ylabel('Frequency')
        axes[0].set_title(f'Original Price (Skew: {original_skew:.2f})')
        axes[0].grid(True, alpha=0.3)
        
        # Log-transformed distribution
        axes[1].hist(log_price, bins=50, alpha=0.7, color='lightgreen', edgecolor='black')
        axes[1].set_xlabel('Log(Price + 1)')
        axes[1].set_ylabel('Frequency')
        axes[1].set_title(f'Log-Transformed Price (Skew: {log_skew:.2f})')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('Data/log_transformation_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"   ✅ Saved transformation comparison to: Data/log_transformation_comparison.png")
        
        # Recommendation
        print(f"\n💡 TRANSFORMATION RECOMMENDATION:")
        if abs(log_skew) < abs(original_skew) and abs(log_skew) < 0.5:
            print(f"   ✅ RECOMMENDED: Use log transformation")
            print(f"   📝 Implementation: target = np.log1p(df['price_usd'])")
            print(f"   📝 Inverse transform: price = np.expm1(prediction)")
        else:
            print(f"   ⚠️ Log transformation shows minimal improvement")
            print(f"   📝 Consider other transformations or use original scale")
    
    def _price_modeling_recommendations(self):
        """Provide ML modeling recommendations based on price analysis."""
        print(f"\n🎯 ML MODELING RECOMMENDATIONS")
        print("-" * 40)
        
        price_data = self.df['price_usd'].dropna()
        price_skewness = skew(price_data)
        
        print(f"📋 Based on price distribution analysis:")
        
        # Skewness recommendations
        if abs(price_skewness) > 1:
            print(f"   1. TARGET TRANSFORMATION:")
            print(f"      ✅ HIGHLY RECOMMENDED: Log transformation")
            print(f"      • Current skewness: {price_skewness:.3f} (heavily skewed)")
            print(f"      • Use: np.log1p(price_usd) as target")
        elif abs(price_skewness) > 0.5:
            print(f"   1. TARGET TRANSFORMATION:")
            print(f"      ⚠️ CONSIDER: Log transformation")
            print(f"      • Current skewness: {price_skewness:.3f} (moderately skewed)")
        else:
            print(f"   1. TARGET TRANSFORMATION:")
            print(f"      ✅ OPTIONAL: Original scale is acceptable")
            print(f"      • Current skewness: {price_skewness:.3f} (approximately normal)")
        
        # Model recommendations
        print(f"\n   2. MODEL RECOMMENDATIONS:")
        print(f"      ✅ Tree-based models: Handle skewness well")
        print(f"         • Random Forest, XGBoost, LightGBM")
        print(f"      ⚠️ Linear models: May need transformation")
        print(f"         • Linear Regression, Ridge, Lasso")
        
        # Outlier handling
        q1 = price_data.quantile(0.25)
        q3 = price_data.quantile(0.75)
        iqr = q3 - q1
        outliers = price_data[(price_data < q1 - 1.5*iqr) | (price_data > q3 + 1.5*iqr)]
        outlier_pct = len(outliers) / len(price_data) * 100
        
        print(f"\n   3. OUTLIER STRATEGY:")
        if outlier_pct > 5:
            print(f"      ⚠️ HIGH OUTLIERS: {outlier_pct:.1f}% of data")
            print(f"      • Consider robust models (Random Forest)")
            print(f"      • Or winsorize extreme values")
        else:
            print(f"      ✅ MANAGEABLE OUTLIERS: {outlier_pct:.1f}% of data")
            print(f"      • Standard models should handle well")
        
        print(f"\n   4. VALIDATION STRATEGY:")
        print(f"      ✅ Use time-based splits (avoid data leakage)")
        print(f"      ✅ Monitor MAPE and RMSE metrics")
        print(f"      ✅ Validate on different price ranges")
    
    def analyze_bivariate_relationships(self):
        """
        B. Analyze Relationships with Target (Bivariate Analysis)
        This is where you find the gold - features with strongest price influence.
        """
        print(f"\n💎 B. BIVARIATE ANALYSIS - RELATIONSHIPS WITH PRICE")
        print("-" * 60)
        
        # Key numerical features for bivariate analysis
        key_numerical_features = [
            'booking_lead_time',      # How far in advance booking was made
            'events_total_score',     # Event activity level
            'MEAN_TEMPERATURE',       # Weather impact
            'Distance from attraction', # Location convenience
            'length_of_stay',         # Stay duration
            'events_count',           # Number of events
            'events_max_score',       # Peak event intensity
            'week_of_year'            # Seasonality
        ]
        
        # Key categorical features
        key_categorical_features = [
            'district',               # Location premium
            'room_category',          # Room type impact
            'is_weekend',             # Weekend premium
            'is_holiday',             # Holiday premium
            'has_major_event',        # Event premium
            'has_multiple_events'     # Multiple events impact
        ]
        
        self._analyze_numeric_vs_price(key_numerical_features)
        self._analyze_categorical_vs_price(key_categorical_features)
        self._create_price_relationship_dashboard()
    
    def _analyze_numeric_vs_price(self, features):
        """Analyze numeric features vs price with scatter plots and correlation."""
        print(f"\n📊 NUMERIC FEATURES vs PRICE")
        print("-" * 40)
        
        available_features = [f for f in features if f in self.df.columns]
        
        if not available_features:
            print(f"   ⚠️ No numeric features available")
            return
        
        # Calculate correlations and significance
        price_data = self.df['price_usd'].dropna()
        correlations = []
        
        for feature in available_features:
            if feature in self.df.columns:
                # Remove rows where either feature or price is missing
                clean_data = self.df[[feature, 'price_usd']].dropna()
                
                if len(clean_data) > 10:  # Need sufficient data
                    correlation = clean_data[feature].corr(clean_data['price_usd'])
                    correlations.append((feature, correlation, len(clean_data)))
        
        # Sort by absolute correlation strength
        correlations.sort(key=lambda x: abs(x[1]), reverse=True)
        
        print(f"   💰 Feature-Price Correlations (strongest first):")
        for feature, corr, count in correlations:
            if abs(corr) > 0.3:
                strength = "🔥 STRONG"
                importance = "HIGH IMPACT"
            elif abs(corr) > 0.15:
                strength = "⚡ MODERATE"  
                importance = "MEDIUM IMPACT"
            elif abs(corr) > 0.05:
                strength = "💫 WEAK"
                importance = "LOW IMPACT"
            else:
                strength = "❄️ MINIMAL"
                importance = "NEGLIGIBLE"
            
            direction = "↗️ POSITIVE" if corr > 0 else "↘️ NEGATIVE"
            print(f"     • {feature:<25}: {corr:>7.3f} | {strength} | {direction} | ({count:,} records)")
        
        # Create detailed scatter plots for top relationships
        self._create_scatter_plots(correlations[:6])  # Top 6 relationships
        
        # Insights for key business relationships
        self._analyze_key_business_relationships()
    
    def _create_scatter_plots(self, top_correlations):
        """Create scatter plots for top correlating features."""
        if not top_correlations:
            return
            
        print(f"\n📈 Creating scatter plots for top {len(top_correlations)} relationships...")
        
        # Determine grid size
        n_plots = len(top_correlations)
        n_cols = 3
        n_rows = (n_plots + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 6*n_rows))
        fig.suptitle('Price vs Key Features - Scatter Plot Analysis', fontsize=16, fontweight='bold')
        
        # Flatten axes for easier indexing
        if n_rows == 1:
            axes = [axes] if n_cols == 1 else axes
        else:
            axes = axes.flatten()
        
        for i, (feature, correlation, count) in enumerate(top_correlations):
            if i >= len(axes):
                break
                
            # Get clean data
            clean_data = self.df[[feature, 'price_usd']].dropna()
            
            # Create scatter plot
            axes[i].scatter(clean_data[feature], clean_data['price_usd'], 
                          alpha=0.6, s=30, color='steelblue')
            
            # Add trend line
            z = np.polyfit(clean_data[feature], clean_data['price_usd'], 1)
            p = np.poly1d(z)
            axes[i].plot(clean_data[feature], p(clean_data[feature]), "r--", alpha=0.8, linewidth=2)
            
            # Formatting
            axes[i].set_xlabel(feature.replace('_', ' ').title())
            axes[i].set_ylabel('Price (USD)')
            axes[i].set_title(f'{feature.replace("_", " ").title()}\nCorrelation: {correlation:.3f}')
            axes[i].grid(True, alpha=0.3)
        
        # Hide unused subplots
        for i in range(len(top_correlations), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.savefig('Data/price_feature_scatter_plots.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"   ✅ Saved scatter plots to: Data/price_feature_scatter_plots.png")
    
    def _analyze_key_business_relationships(self):
        """Analyze key business relationships with detailed insights."""
        print(f"\n🔍 KEY BUSINESS RELATIONSHIPS ANALYSIS")
        print("-" * 45)
        
        # 1. Booking Lead Time vs Price
        if 'booking_lead_time' in self.df.columns:
            self._analyze_booking_lead_time()
        
        # 2. Events vs Price  
        if 'events_total_score' in self.df.columns:
            self._analyze_events_impact()
            
        # 3. Weather vs Price
        if 'MEAN_TEMPERATURE' in self.df.columns:
            self._analyze_weather_impact()
            
        # 4. Distance vs Price
        if 'Distance from attraction' in self.df.columns:
            self._analyze_distance_impact()
    
    def _analyze_booking_lead_time(self):
        """Analyze booking lead time impact on price."""
        print(f"\n   📅 BOOKING LEAD TIME vs PRICE:")
        
        clean_data = self.df[['booking_lead_time', 'price_usd']].dropna()
        
        if len(clean_data) < 10:
            print(f"     ⚠️ Insufficient data")
            return
        
        # Bin lead times for analysis
        clean_data['lead_time_category'] = pd.cut(clean_data['booking_lead_time'], 
                                                bins=[-1, 0, 7, 30, 90, float('inf')],
                                                labels=['Same Day', '1-7 days', '1-4 weeks', '1-3 months', '3+ months'])
        
        lead_time_analysis = clean_data.groupby('lead_time_category')['price_usd'].agg(['mean', 'median', 'count'])
        
        for category, row in lead_time_analysis.iterrows():
            print(f"     • {category:<12}: ${row['mean']:>6.2f} avg | ${row['median']:>6.2f} median | {row['count']:>4} bookings")
        
        correlation = clean_data['booking_lead_time'].corr(clean_data['price_usd'])
        print(f"     💡 Insight: Lead time correlation = {correlation:.3f}")
        
        if correlation < -0.1:
            print(f"     📈 Pattern: Prices tend to INCREASE as check-in approaches (last-minute premium)")
        elif correlation > 0.1:
            print(f"     📈 Pattern: Prices tend to DECREASE as check-in approaches (early bird discounts)")
        else:
            print(f"     📈 Pattern: No clear lead time pricing pattern")
    
    def _analyze_events_impact(self):
        """Analyze events impact on price."""
        print(f"\n   🎉 EVENTS vs PRICE:")
        
        clean_data = self.df[['events_total_score', 'price_usd', 'has_major_event']].dropna()
        
        if len(clean_data) < 10:
            print(f"     ⚠️ Insufficient data")
            return
        
        # Events score bins
        clean_data['event_intensity'] = pd.cut(clean_data['events_total_score'],
                                             bins=[-1, 0, 50, 100, 200, float('inf')],
                                             labels=['No Events', 'Low Activity', 'Medium Activity', 'High Activity', 'Peak Activity'])
        
        event_analysis = clean_data.groupby('event_intensity')['price_usd'].agg(['mean', 'median', 'count'])
        
        for category, row in event_analysis.iterrows():
            print(f"     • {category:<15}: ${row['mean']:>6.2f} avg | ${row['median']:>6.2f} median | {row['count']:>4} records")
        
        correlation = clean_data['events_total_score'].corr(clean_data['price_usd'])
        print(f"     💡 Events correlation = {correlation:.3f}")
        
        # Major events impact
        if 'has_major_event' in clean_data.columns:
            major_event_prices = clean_data.groupby('has_major_event')['price_usd'].mean()
            if len(major_event_prices) == 2:
                premium = major_event_prices[True] - major_event_prices[False]
                premium_pct = (premium / major_event_prices[False]) * 100
                print(f"     🎪 Major Event Premium: ${premium:.2f} ({premium_pct:.1f}% increase)")
    
    def _analyze_weather_impact(self):
        """Analyze weather impact on price."""
        print(f"\n   🌤️ WEATHER vs PRICE:")
        
        clean_data = self.df[['MEAN_TEMPERATURE', 'price_usd']].dropna()
        
        if len(clean_data) < 10:
            print(f"     ⚠️ Insufficient data")
            return
        
        # Temperature bins
        clean_data['temp_category'] = pd.cut(clean_data['MEAN_TEMPERATURE'],
                                           bins=[-50, 0, 10, 20, 30, 50],
                                           labels=['Very Cold (<0°C)', 'Cold (0-10°C)', 'Mild (10-20°C)', 'Warm (20-30°C)', 'Hot (>30°C)'])
        
        temp_analysis = clean_data.groupby('temp_category')['price_usd'].agg(['mean', 'median', 'count'])
        
        for category, row in temp_analysis.iterrows():
            if pd.notna(category):
                print(f"     • {category:<18}: ${row['mean']:>6.2f} avg | ${row['median']:>6.2f} median | {row['count']:>4} records")
        
        correlation = clean_data['MEAN_TEMPERATURE'].corr(clean_data['price_usd'])
        print(f"     💡 Temperature correlation = {correlation:.3f}")
        
        if correlation > 0.1:
            print(f"     ☀️ Pattern: Warmer weather = higher prices")
        elif correlation < -0.1:
            print(f"     ❄️ Pattern: Colder weather = higher prices") 
        else:
            print(f"     🌡️ Pattern: No clear temperature pricing pattern")
    
    def _analyze_distance_impact(self):
        """Analyze distance from attraction impact on price."""
        print(f"\n   📍 DISTANCE FROM ATTRACTION vs PRICE:")
        
        clean_data = self.df[['Distance from attraction', 'price_usd']].dropna()
        
        if len(clean_data) < 10:
            print(f"     ⚠️ Insufficient data")
            return
        
        # Distance bins
        clean_data['distance_category'] = pd.cut(clean_data['Distance from attraction'],
                                               bins=[-1, 0.5, 1.0, 2.0, 5.0, float('inf')],
                                               labels=['Very Close (<0.5mi)', 'Close (0.5-1mi)', 'Nearby (1-2mi)', 'Moderate (2-5mi)', 'Far (>5mi)'])
        
        distance_analysis = clean_data.groupby('distance_category')['price_usd'].agg(['mean', 'median', 'count'])
        
        for category, row in distance_analysis.iterrows():
            if pd.notna(category):
                print(f"     • {category:<20}: ${row['mean']:>6.2f} avg | ${row['median']:>6.2f} median | {row['count']:>4} records")
        
        correlation = clean_data['Distance from attraction'].corr(clean_data['price_usd'])
        print(f"     💡 Distance correlation = {correlation:.3f}")
        
        if correlation < -0.1:
            print(f"     🏨 Pattern: Closer to attractions = higher prices (location premium)")
        elif correlation > 0.1:
            print(f"     🏨 Pattern: Further from attractions = higher prices (unexpected!)")
        else:
            print(f"     🏨 Pattern: Distance has minimal price impact")
    
    def _analyze_categorical_vs_price(self, features):
        """Analyze categorical features vs price with box plots."""
        print(f"\n📋 CATEGORICAL FEATURES vs PRICE")
        print("-" * 40)
        
        available_features = [f for f in features if f in self.df.columns]
        
        if not available_features:
            print(f"   ⚠️ No categorical features available")
            return
        
        for feature in available_features:
            print(f"\n   🏷️ {feature.upper().replace('_', ' ')}:")
            
            # Calculate price statistics by category
            price_by_category = self.df.groupby(feature)['price_usd'].agg(['mean', 'median', 'std', 'count']).round(2)
            price_by_category = price_by_category.sort_values('mean', ascending=False)
            
            # Show top categories
            print(f"     📊 Price by category (top 5):")
            for i, (category, row) in enumerate(price_by_category.head(5).iterrows()):
                if pd.notna(category) and row['count'] >= 5:  # Only show categories with enough data
                    print(f"     {i+1}. {str(category):<20}: ${row['mean']:>6.2f} avg | ${row['median']:>6.2f} median | {row['count']:>4} records")
            
            # Calculate price range (highest vs lowest)
            if len(price_by_category) >= 2:
                price_range = price_by_category['mean'].iloc[0] - price_by_category['mean'].iloc[-1]
                range_pct = (price_range / price_by_category['mean'].iloc[-1]) * 100
                print(f"     💰 Price Range: ${price_range:.2f} ({range_pct:.1f}% difference between highest and lowest)")
    
    def _create_price_relationship_dashboard(self):
        """Create a comprehensive dashboard of price relationships."""
        print(f"\n📊 Creating comprehensive price relationship dashboard...")
        
        # Key categorical features for box plots
        categorical_features = ['district', 'room_category', 'is_weekend', 'is_holiday']
        available_categorical = [f for f in categorical_features if f in self.df.columns]
        
        if len(available_categorical) < 2:
            print(f"   ⚠️ Insufficient categorical features for dashboard")
            return
        
        # Create box plots
        n_plots = len(available_categorical)
        n_cols = 2
        n_rows = (n_plots + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 6*n_rows))
        fig.suptitle('Price Distribution by Categorical Features', fontsize=16, fontweight='bold')
        
        # Flatten axes for easier indexing
        if n_rows == 1:
            axes = [axes] if n_cols == 1 else axes
        else:
            axes = axes.flatten()
        
        for i, feature in enumerate(available_categorical):
            if i >= len(axes):
                break
            
            # Create box plot
            clean_data = self.df[[feature, 'price_usd']].dropna()
            
            if len(clean_data) > 0:
                # Limit categories to top 10 to avoid overcrowding
                top_categories = clean_data[feature].value_counts().head(10).index
                plot_data = clean_data[clean_data[feature].isin(top_categories)]
                
                sns.boxplot(data=plot_data, x=feature, y='price_usd', ax=axes[i])
                axes[i].set_title(f'Price by {feature.replace("_", " ").title()}')
                axes[i].set_xlabel(feature.replace('_', ' ').title())
                axes[i].set_ylabel('Price (USD)')
                axes[i].tick_params(axis='x', rotation=45)
                axes[i].grid(True, alpha=0.3)
        
        # Hide unused subplots
        for i in range(len(available_categorical), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.savefig('Data/price_categorical_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"   ✅ Saved categorical analysis to: Data/price_categorical_analysis.png")
    
    def analyze_key_features(self):
        """Analyze key features that influence hotel pricing."""
        print(f"\n🔧 C. SUPPLEMENTARY FEATURE ANALYSIS")
        print("-" * 50)
        
        # Numerical features
        numerical_features = [
            'Distance from attraction', 'length_of_stay', 'booking_lead_time',
            'MEAN_TEMPERATURE', 'events_total_score', 'events_count'
        ]
        
        # Categorical features
        categorical_features = [
            'district', 'room_category', 'is_weekend', 'is_holiday',
            'has_major_event'
        ]
        
        self._analyze_numerical_features(numerical_features)
        self._analyze_categorical_features(categorical_features)
    
    def _analyze_numerical_features(self, features):
        """Analyze numerical features."""
        print(f"\n📊 Numerical Features Analysis:")
        
        available_features = [f for f in features if f in self.df.columns]
        
        if not available_features:
            print(f"   ⚠️ No numerical features available for analysis")
            return
        
        # Create correlation matrix
        corr_data = self.df[['price_usd'] + available_features].corr()['price_usd'].drop('price_usd')
        
        print(f"   💰 Correlation with price_usd:")
        for feature in corr_data.abs().sort_values(ascending=False).index:
            corr_val = corr_data[feature]
            if abs(corr_val) > 0.3:
                strength = "Strong"
            elif abs(corr_val) > 0.1:
                strength = "Moderate"
            else:
                strength = "Weak"
            
            print(f"     • {feature:<25}: {corr_val:>6.3f} ({strength})")
        
        # Create correlation heatmap
        plt.figure(figsize=(10, 8))
        correlation_matrix = self.df[['price_usd'] + available_features].corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                    square=True, linewidths=0.5, cbar_kws={"shrink": .8})
        plt.title('Feature Correlation Matrix')
        plt.tight_layout()
        plt.savefig('Data/feature_correlation_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"   ✅ Saved correlation matrix to: Data/feature_correlation_matrix.png")
    
    def _analyze_categorical_features(self, features):
        """Analyze categorical features."""
        print(f"\n📋 Categorical Features Analysis:")
        
        available_features = [f for f in features if f in self.df.columns]
        
        for feature in available_features:
            print(f"\n   🏷️ {feature}:")
            
            # Value counts
            value_counts = self.df[feature].value_counts()
            print(f"     • Unique values: {len(value_counts)}")
            print(f"     • Most common: {value_counts.index[0]} ({value_counts.iloc[0]} records)")
            
            # Price by category
            price_by_category = self.df.groupby(feature)['price_usd'].agg(['mean', 'median', 'count'])
            price_by_category = price_by_category.sort_values('mean', ascending=False)
            
            print(f"     • Price impact (top 3 highest avg prices):")
            for i, (category, row) in enumerate(price_by_category.head(3).iterrows()):
                print(f"       {i+1}. {category}: ${row['mean']:.2f} avg ({row['count']} records)")
    
    def run_complete_eda(self):
        """Run the complete EDA analysis."""
        print(f"🚀 Starting Comprehensive EDA Analysis...")
        print()
        
        # Load data
        if not self.load_data():
            return
        
        # A. Target variable analysis (most important)
        self.analyze_target_variable()
        
        # B. Bivariate analysis - relationships with price (THE GOLD!)
        self.analyze_bivariate_relationships()
        
        # C. Supplementary feature analysis
        self.analyze_key_features()
        
        print(f"\n🎉 EDA ANALYSIS COMPLETE!")
        print("=" * 60)
        print(f"📊 Generated visualizations:")
        print(f"   • Data/price_distribution_analysis.png")
        print(f"   • Data/log_transformation_comparison.png")
        print(f"   • Data/price_feature_scatter_plots.png")
        print(f"   • Data/price_categorical_analysis.png")
        print(f"   • Data/feature_correlation_matrix.png")
        print()
        print(f"💡 Next steps:")
        print(f"   1. Review price distribution recommendations")
        print(f"   2. Analyze bivariate relationships findings")
        print(f"   3. Decide on target transformation strategy")
        print(f"   4. Select features based on correlation analysis")
        print(f"   5. Proceed with ML model development")


def main():
    """Run the complete EDA analysis."""
    
    # Initialize EDA
    eda = HotelEDA('Data/toronto_hotels_transformed.csv')
    
    # Run complete analysis
    eda.run_complete_eda()


if __name__ == "__main__":
    main()