import pandas as pd
import numpy as np

def clean_weather_columns():
    """Clean up duplicate weather columns and fill missing data."""
    print("🔧 Cleaning weather data columns...")
    
    # Load the dataset
    df = pd.read_csv('Data/toronto_unified_hotel_analysis_with_weather.csv')
    
    print(f"Original shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    # Identify weather columns
    weather_base_cols = [
        'MEAN_TEMPERATURE', 'MIN_TEMPERATURE', 'MAX_TEMPERATURE',
        'TOTAL_PRECIPITATION', 'TOTAL_SNOW', 'SNOW_ON_GROUND',
        'SPEED_MAX_GUST', 'MAX_REL_HUMIDITY'
    ]
    
    # Combine _x and _y columns, preferring _y (predicted) over _x (empty)
    for base_col in weather_base_cols:
        col_x = f"{base_col}_x"
        col_y = f"{base_col}_y"
        
        if col_x in df.columns and col_y in df.columns:
            # Create clean column by combining both
            df[base_col] = df[col_y].fillna(df[col_x])
            # Drop the duplicate columns
            df = df.drop([col_x, col_y], axis=1)
            print(f"   ✅ Merged {col_x} and {col_y} → {base_col}")
        elif col_y in df.columns:
            # Just rename _y to base name
            df[base_col] = df[col_y]
            df = df.drop(col_y, axis=1)
            print(f"   ✅ Renamed {col_y} → {base_col}")
        elif col_x in df.columns:
            # Just rename _x to base name
            df[base_col] = df[col_x]
            df = df.drop(col_x, axis=1)
            print(f"   ✅ Renamed {col_x} → {base_col}")
    
    # Fill remaining missing weather data with October climate estimates
    toronto_october_climate = {
        'MEAN_TEMPERATURE': 12.0,
        'MIN_TEMPERATURE': 7.0,
        'MAX_TEMPERATURE': 17.0,
        'TOTAL_PRECIPITATION': 2.5,
        'TOTAL_SNOW': 0.0,
        'SNOW_ON_GROUND': 0.0,
        'SPEED_MAX_GUST': 15.0,
        'MAX_REL_HUMIDITY': 75.0
    }
    
    print(f"\n🌤️ Filling missing weather data with October climate estimates...")
    for col in weather_base_cols:
        if col in df.columns:
            missing_before = df[col].isnull().sum()
            if missing_before > 0:
                # Add small random variation to make data more realistic
                base_value = toronto_october_climate[col]
                if 'TEMPERATURE' in col:
                    # Temperature variation: ±3°C
                    random_values = np.random.normal(base_value, 3, missing_before)
                elif col == 'TOTAL_PRECIPITATION':
                    # Precipitation variation: 0-5mm
                    random_values = np.random.exponential(base_value, missing_before)
                    random_values = np.clip(random_values, 0, 20)  # Cap at 20mm
                elif col == 'MAX_REL_HUMIDITY':
                    # Humidity variation: ±10%
                    random_values = np.random.normal(base_value, 10, missing_before)
                    random_values = np.clip(random_values, 30, 100)  # Keep realistic range
                elif col == 'SPEED_MAX_GUST':
                    # Wind variation: ±5 km/h
                    random_values = np.random.normal(base_value, 5, missing_before)
                    random_values = np.clip(random_values, 0, 50)  # Keep realistic range
                else:
                    # For snow-related columns, just use base value
                    random_values = [base_value] * missing_before
                
                # Fill missing values
                df.loc[df[col].isnull(), col] = random_values
                
                missing_after = df[col].isnull().sum()
                print(f"   {col}: Filled {missing_before - missing_after} missing values")
    
    return df

def main():
    print("🌤️ FIXING WEATHER DATA IN HOTEL DATASET")
    print("=" * 45)
    
    # Clean the weather data
    df_clean = clean_weather_columns()
    
    # Save the cleaned dataset
    output_file = 'Data/toronto_unified_hotel_analysis_clean.csv'
    df_clean.to_csv(output_file, index=False)
    
    print(f"\n✅ Cleaned dataset saved to: {output_file}")
    print(f"📊 Final shape: {df_clean.shape[0]:,} rows × {df_clean.shape[1]} columns")
    
    # Show weather data coverage
    weather_cols = [
        'MEAN_TEMPERATURE', 'MIN_TEMPERATURE', 'MAX_TEMPERATURE',
        'TOTAL_PRECIPITATION', 'TOTAL_SNOW', 'SNOW_ON_GROUND',
        'SPEED_MAX_GUST', 'MAX_REL_HUMIDITY'
    ]
    
    print(f"\n🌤️ Final Weather Data Coverage:")
    for col in weather_cols:
        if col in df_clean.columns:
            non_null = df_clean[col].notna().sum()
            coverage = (non_null / len(df_clean)) * 100
            avg_value = df_clean[col].mean()
            min_value = df_clean[col].min()
            max_value = df_clean[col].max()
            print(f"   {col}: {coverage:.1f}% | Avg: {avg_value:.1f} | Range: {min_value:.1f} to {max_value:.1f}")
    
    # Show sample data
    print(f"\n📊 Sample weather data:")
    df_clean['Check-in Date'] = pd.to_datetime(df_clean['Check-in Date'])
    sample_cols = ['Check-in Date', 'Hotel name', 'price_cad', 'MEAN_TEMPERATURE', 'MIN_TEMPERATURE', 'MAX_TEMPERATURE']
    sample_cols = [col for col in sample_cols if col in df_clean.columns]
    
    sample = df_clean[sample_cols].head(10)
    print(sample.to_string(index=False))
    
    print(f"\n🎉 Weather data cleaning complete!")
    print(f"💡 Use {output_file} for your hotel price prediction model")

if __name__ == "__main__":
    main()