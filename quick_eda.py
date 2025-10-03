import pandas as pd
import numpy as np

def main():
    print('🏨 TORONTO HOTEL PRICE PREDICTION - KEY INSIGHTS')
    print('=' * 55)
    
    # Load data
    df = pd.read_csv('Data/toronto_unified_hotel_analysis.csv')
    print(f'\n📊 Dataset: {df.shape[0]:,} rows × {df.shape[1]} columns')
    
    # Target variable analysis
    print(f'\n💰 PRICE ANALYSIS (CAD)')
    print('-' * 25)
    price_stats = df['price_cad'].describe()
    print(f'Mean: ${price_stats["mean"]:.2f}')
    print(f'Median: ${price_stats["50%"]:.2f}')
    print(f'Std Dev: ${price_stats["std"]:.2f}')
    print(f'Min: ${price_stats["min"]:.2f}')
    print(f'Max: ${price_stats["max"]:.2f}')
    print(f'Skewness: {df["price_cad"].skew():.3f}')
    print(f'Missing: {df["price_cad"].isnull().sum()} ({df["price_cad"].isnull().mean()*100:.1f}%)')
    
    # Price distribution
    print(f'\n💵 PRICE SEGMENTS')
    print('-' * 20)
    price_valid = df['price_cad'].dropna()
    budget = (price_valid < 150).sum()
    mid = ((price_valid >= 150) & (price_valid < 350)).sum()
    upscale = ((price_valid >= 350) & (price_valid < 600)).sum()
    luxury = (price_valid >= 600).sum()
    
    total = len(price_valid)
    print(f'Budget (<$150): {budget:,} ({budget/total*100:.1f}%)')
    print(f'Mid-range ($150-350): {mid:,} ({mid/total*100:.1f}%)')
    print(f'Upscale ($350-600): {upscale:,} ({upscale/total*100:.1f}%)')
    print(f'Luxury ($600+): {luxury:,} ({luxury/total*100:.1f}%)')
    
    # Top correlations
    print(f'\n🔗 TOP CORRELATIONS WITH PRICE')
    print('-' * 35)
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if 'price_usd' in numerical_cols:
        numerical_cols.remove('price_usd')
    
    # Calculate correlations only for columns with valid price data
    price_mask = df['price_cad'].notna()
    correlations = df[price_mask][numerical_cols].corr()['price_cad'].abs().sort_values(ascending=False)
    
    print('Strongest correlations (absolute):')
    for feature, corr in correlations.head(8).items():
        if feature != 'price_cad' and not pd.isna(corr):
            direction = '+' if df[price_mask][feature].corr(df[price_mask]['price_cad']) > 0 else '-'
            print(f'  {feature}: {direction}{corr:.3f}')
    
    # District analysis
    print(f'\n🏙️ PRICING BY DISTRICT')
    print('-' * 25)
    district_analysis = df.groupby('district')['price_cad'].agg(['mean', 'count']).round(2)
    district_analysis = district_analysis[district_analysis['count'] >= 20]
    district_analysis = district_analysis.sort_values('mean', ascending=False)
    
    print('Top districts by average price (≥20 hotels):')
    for district, row in district_analysis.head(8).iterrows():
        print(f'  {district}: ${row["mean"]:.2f} ({int(row["count"]):,} hotels)')
    
    # Weekend analysis
    print(f'\n🎉 WEEKEND EFFECT')
    print('-' * 18)
    weekend_analysis = df.groupby('is_weekend')['price_cad'].agg(['mean', 'median', 'count']).round(2)
    
    weekday_avg = weekend_analysis.loc[0, 'mean'] if 0 in weekend_analysis.index else 0
    weekend_avg = weekend_analysis.loc[1, 'mean'] if 1 in weekend_analysis.index else 0
    
    if weekday_avg > 0 and weekend_avg > 0:
        difference = weekend_avg - weekday_avg
        pct_diff = (difference / weekday_avg) * 100
        
        print(f'Weekday average: ${weekday_avg:.2f}')
        print(f'Weekend average: ${weekend_avg:.2f}')
        print(f'Weekend premium: ${difference:+.2f} ({pct_diff:+.1f}%)')
    
    # Event impact
    print(f'\n🎭 EVENT IMPACT')
    print('-' * 15)
    
    # Check events data availability
    events_cols = [col for col in df.columns if 'events_' in col]
    if events_cols and 'events_count' in df.columns:
        has_events_mask = df['events_count'] > 0
        no_events_mask = df['events_count'] == 0
        
        if has_events_mask.any() and no_events_mask.any():
            with_events_price = df[has_events_mask]['price_cad'].mean()
            without_events_price = df[no_events_mask]['price_cad'].mean()
            
            if not pd.isna(with_events_price) and not pd.isna(without_events_price):
                difference = with_events_price - without_events_price
                pct_diff = (difference / without_events_price) * 100
                
                print(f'Hotels with events: ${with_events_price:.2f} ({has_events_mask.sum():,} hotels)')
                print(f'Hotels without events: ${without_events_price:.2f} ({no_events_mask.sum():,} hotels)')
                print(f'Event premium: ${difference:+.2f} ({pct_diff:+.1f}%)')
            else:
                print('Event data available but insufficient for analysis')
        else:
            print('All hotels have events during stay period')
    else:
        print('No event data available')
    
    # Booking lead time
    print(f'\n⏰ BOOKING LEAD TIME ANALYSIS')
    print('-' * 30)
    if 'booking_lead_time' in df.columns:
        lead_time_valid = df['booking_lead_time'].dropna()
        if len(lead_time_valid) > 0:
            lead_stats = lead_time_valid.describe()
            print(f'Average lead time: {lead_stats["mean"]:.1f} days')
            print(f'Median lead time: {lead_stats["50%"]:.1f} days')
            
            # Lead time categories
            categories = [
                ('Last minute (0-7 days)', 0, 7),
                ('Short term (8-21 days)', 8, 21),
                ('Medium term (22-60 days)', 22, 60),
                ('Long term (60+ days)', 60, float('inf'))
            ]
            
            for name, min_days, max_days in categories:
                if max_days == float('inf'):
                    mask = lead_time_valid >= min_days
                else:
                    mask = (lead_time_valid >= min_days) & (lead_time_valid <= max_days)
                
                count = mask.sum()
                pct = (count / len(lead_time_valid)) * 100
                
                if count > 0:
                    # Get corresponding prices
                    price_mask = df['booking_lead_time'].notna() & mask
                    avg_price = df[price_mask]['price_cad'].mean()
                    if not pd.isna(avg_price):
                        print(f'  {name}: ${avg_price:.2f} ({count:,} bookings, {pct:.1f}%)')
    
    # Missing data
    print(f'\n❓ DATA QUALITY')
    print('-' * 15)
    missing_data = df.isnull().sum()
    missing_data = missing_data[missing_data > 0].sort_values(ascending=False)
    
    if len(missing_data) > 0:
        print('Columns with missing data:')
        for col, count in missing_data.head(10).items():
            pct = (count / len(df)) * 100
            print(f'  {col}: {count:,} ({pct:.1f}%)')
    else:
        print('✅ No missing data!')
    
    # Modeling recommendations
    print(f'\n🤖 MODELING RECOMMENDATIONS')
    print('-' * 30)
    print('1. Target: Use price_cad (log-transform recommended due to skewness)')
    print('2. Key features: District, weekend flag, booking_lead_time, events_count')
    print('3. Feature engineering: Create price categories, seasonal features')
    print('4. Handle outliers: Consider capping at 95th or 99th percentile')
    print('5. Models to try: Linear Regression → Random Forest → XGBoost')
    
    print(f'\n✅ EDA COMPLETE - Ready for model building!')

if __name__ == "__main__":
    main()