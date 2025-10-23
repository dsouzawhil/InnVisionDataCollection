"""
Complete Hotel Data Transformation Pipeline
===========================================

This script consolidates all data transformations from raw scraped data 
to the final analysis-ready dataset with coordinates.

Pipeline Steps:
1. Load raw hotel data
2. Currency conversion (USD → CAD)
3. Data cleaning and duplicate removal
4. Weather data integration
5. Holiday flag detection (all Canadian provinces)
6. District extraction and enhancement
7. Geocoding with coordinates
8. Final data quality checks

Output: Data/toronto_hotels_transformed.csv
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import holidays
import requests
import re
import warnings
warnings.filterwarnings('ignore')

class HotelDataPipeline:
    def __init__(self, input_file='Data/toronto_unified_hotel_analysis.csv'):
        """Initialize the pipeline with input file path."""
        self.input_file = input_file
        self.usd_to_cad_rate = 1.35  # Fixed conversion rate
        
        # Toronto district coordinates for geocoding
        self.district_coords = {
            'Entertainment District': (43.6426, -79.3871),
            'Financial District': (43.6489, -79.3817),
            'Bloor-Yorkville': (43.6706, -79.3951),
            'Fashion District': (43.6553, -79.3906),
            'The Annex': (43.6708, -79.4037),
            'Old Town': (43.6503, -79.3717),
            'Yonge - Dundas': (43.6566, -79.3805),
            'Queen West': (43.6439, -79.4001),
            'King Street West': (43.6435, -79.3929),
            'Church-Wellesley Village': (43.6688, -79.3826),
            'Cabbagetown': (43.6658, -79.3636),
            'Leslieville': (43.6595, -79.3384),
            'The Beaches': (43.6677, -79.2930),
            'Liberty Village': (43.6393, -79.4198),
            'CityPlace': (43.6393, -79.3916),
            'Harbourfront': (43.6389, -79.3817),
            'Distillery District': (43.6503, -79.3589),
            'Regent Park': (43.6581, -79.3624),
            'Moss Park': (43.6551, -79.3724),
            'Garden District': (43.6580, -79.3776),
            'St. Lawrence': (43.6487, -79.3717),
            'Chinatown': (43.6532, -79.3977),
            'Kensington Market': (43.6548, -79.4005),
            'Little Italy': (43.6548, -79.4169),
            'Junction Triangle': (43.6598, -79.4448),
            'Roncesvalles': (43.6458, -79.4489),
            'High Park': (43.6537, -79.4637),
            'Parkdale': (43.6398, -79.4328),
            'Mimico': (43.6055, -79.4969),
            'Airport Area': (43.6777, -79.6248),
            'Etobicoke': (43.6205, -79.5132),
            'North York': (43.7615, -79.4111),
            'Scarborough': (43.7764, -79.2318),
            'East York': (43.6890, -79.3327),
            'York': (43.6896, -79.4879),
            'Toronto': (43.6532, -79.3832)  # Default Toronto center
        }
        
        print("🏗️ HOTEL DATA TRANSFORMATION PIPELINE")
        print("=" * 50)
    
    def load_data(self):
        """Step 1: Load raw hotel data."""
        print("📂 Step 1: Loading raw hotel data...")
        
        try:
            df = pd.read_csv(self.input_file)
            print(f"   ✅ Loaded {len(df)} rows, {len(df.columns)} columns")
            return df
        except Exception as e:
            print(f"   ❌ Error loading data: {e}")
            return None
    
    def currency_conversion(self, df):
        """Step 2: Convert USD prices to CAD."""
        print("💱 Step 2: Converting USD to CAD...")
        
        # Find the price column (could be 'price', 'Price', or 'price_usd')
        price_col = None
        for col in ['price_usd', 'price', 'Price']:
            if col in df.columns:
                price_col = col
                break
        
        if price_col:
            # Clean and convert price column to numeric
            print(f"   🧹 Found price column: {price_col}")
            
            # Remove $ signs, commas, and extract numeric values
            # First filter out obviously non-price values
            df[price_col] = df[price_col].astype(str)
            # Replace common non-price text with NaN
            df[price_col] = df[price_col].replace(['nan', 'None', 'Free airport taxi', '', ' '], np.nan)
            # Only process non-null values
            mask = df[price_col].notna()
            df.loc[mask, price_col] = df.loc[mask, price_col].str.replace('$', '', regex=False)
            df.loc[mask, price_col] = df.loc[mask, price_col].str.replace(',', '', regex=False)
            # Extract only values that look like prices (numbers, possibly with decimal)
            df.loc[mask, price_col] = df.loc[mask, price_col].str.extract(r'^(\d+\.?\d*)$')[0]
            df[price_col] = pd.to_numeric(df[price_col], errors='coerce')
            
            # Rename to price_usd if it's not already
            if price_col != 'price_usd':
                df = df.rename(columns={price_col: 'price_usd'})
                print(f"   🔄 Renamed {price_col} to price_usd")
            
            # Create price_cad column by converting USD to CAD
            # Handle null values explicitly
            df['price_cad'] = np.where(
                df['price_usd'].notna(),
                df['price_usd'] * self.usd_to_cad_rate,
                np.nan
            )
            # Round to 2 decimal places only for non-null values
            df['price_cad'] = df['price_cad'].round(2)
            # Ensure price_cad stays as float type
            df['price_cad'] = df['price_cad'].astype('float64')
            
            converted_count = df['price_usd'].notna().sum()
            print(f"   ✅ Created price_cad column: converted {converted_count} prices from USD to CAD")
            print(f"   💱 Using conversion rate: 1 USD = {self.usd_to_cad_rate} CAD")
        else:
            print("   ⚠️ No price column found (looked for 'price_usd', 'price', 'Price')")
        
        return df
    
    def data_cleaning(self, df):
        """Step 3: Comprehensive data cleaning and type transformations."""
        print("🧹 Step 3: Data cleaning and type transformations...")
        
        original_cols = len(df.columns)
        original_rows = len(df)
        
        # Clean price column - remove $ sign and keep only numeric values
        price_columns = ['Price', 'price', 'price_usd']
        for price_col in price_columns:
            if price_col in df.columns:
                print(f"   💰 Cleaning {price_col} column...")
                # Remove $ sign and any other non-numeric characters except decimal point
                df[price_col] = df[price_col].astype(str).str.replace('$', '', regex=False)
                df[price_col] = df[price_col].str.replace(',', '', regex=False)  # Remove commas
                df[price_col] = df[price_col].str.extract(r'([\d\.]+)')[0]  # Extract numeric part
                df[price_col] = pd.to_numeric(df[price_col], errors='coerce')
                print(f"   ✅ Removed $ sign and cleaned {price_col} column")
                
                # If this is the main price column, ensure it's named price_usd
                if price_col in ['Price', 'price']:
                    df = df.rename(columns={price_col: 'price_usd'})
                    print(f"   🔄 Renamed {price_col} to price_usd")
        
        # Remove duplicate distance columns (case-insensitive)
        distance_cols = [col for col in df.columns if 'distance' in col.lower() and 'attraction' in col.lower()]
        
        if len(distance_cols) > 1:
            print(f"   🔍 Found {len(distance_cols)} distance columns: {distance_cols}")
            
            # Prioritize the column with better coverage
            coverage_stats = {}
            for col in distance_cols:
                non_null_count = df[col].notna().sum()
                coverage_pct = (non_null_count / len(df)) * 100
                coverage_stats[col] = (non_null_count, coverage_pct)
                print(f"     • '{col}': {non_null_count:,} values ({coverage_pct:.1f}% coverage)")
            
            # Keep the column with highest coverage, remove others
            best_col = max(coverage_stats.keys(), key=lambda x: coverage_stats[x][0])
            cols_to_remove = [col for col in distance_cols if col != best_col]
            
            if cols_to_remove:
                df = df.drop(columns=cols_to_remove)
                print(f"   ✅ Kept '{best_col}' (best coverage)")
                print(f"   🗑️ Removed {len(cols_to_remove)} duplicate columns: {cols_to_remove}")
                
                # Rename to standardized name if needed
                if best_col != 'Distance from attraction':
                    df = df.rename(columns={best_col: 'Distance from attraction'})
                    print(f"   🔄 Renamed '{best_col}' → 'Distance from attraction'")
        
        # Clean Distance from attraction column - remove 'miles from downtown' text
        if 'Distance from attraction' in df.columns:
            print("   🧹 Cleaning Distance from attraction column...")
            # Extract numeric values from distance column, remove 'miles from downtown'
            df['Distance from attraction'] = df['Distance from attraction'].astype(str).str.extract(r'([\d\.]+)')[0]
            df['Distance from attraction'] = pd.to_numeric(df['Distance from attraction'], errors='coerce')
            print("   ✅ Removed 'miles from downtown' text from Distance column")
        
        # Convert date columns to datetime
        date_columns = ['Date', 'Check-in Date', 'Check-out Date']
        for col in date_columns:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')
                print(f"   📅 Converted {col} to datetime")
        
        # Fill empty Number of People column with default value of 2
        if 'Number of People' in df.columns:
            df['Number of People'] = df['Number of People'].fillna(2)
            filled_count = df['Number of People'].isna().sum()
            print(f"   👥 Set default value 2 for empty 'Number of People' entries")
        
        # Remove data quality anomalies - identified during EDA
        print("   🚨 Removing data quality anomalies...")
        initial_rows = len(df)
        
        # Remove "scarborough home" anomalies - pricing 6x higher than market rate
        if 'Hotel name' in df.columns:
            anomaly_mask = df['Hotel name'].str.lower() == 'scarborough home'
            anomaly_count = anomaly_mask.sum()
            
            if anomaly_count > 0:
                df = df[~anomaly_mask].copy()
                print(f"   ✅ Removed {anomaly_count} 'scarborough home' pricing anomalies")
                print(f"     • These were outliers: $1,574-$1,580 for basic rooms in Scarborough")
                print(f"     • 6x higher than other Scarborough properties ($74-$334 range)")
        
        # Additional anomaly detection - extreme price outliers
        if 'price_usd' in df.columns:
            # Remove prices beyond reasonable hotel ranges (likely data errors)
            price_data = df['price_usd'].dropna()
            
            if len(price_data) > 0:
                # Define reasonable hotel price bounds for Toronto (conservative)
                min_reasonable_price = 25   # Hostel bed minimum
                max_reasonable_price = 2000  # Luxury suite maximum
                
                price_outliers = ((df['price_usd'] < min_reasonable_price) | 
                                (df['price_usd'] > max_reasonable_price)) & df['price_usd'].notna()
                outlier_count = price_outliers.sum()
                
                if outlier_count > 0:
                    # Log the outliers before removing
                    outlier_data = df.loc[price_outliers, ['Hotel name', 'price_usd', 'Room Type']].head(5)
                    print(f"   ⚠️ Found {outlier_count} extreme price outliers:")
                    for _, row in outlier_data.iterrows():
                        hotel_name = str(row['Hotel name'])[:30] if pd.notna(row['Hotel name']) else 'Unknown'
                        room_type = str(row['Room Type'])[:20] if pd.notna(row['Room Type']) else 'Unknown'
                        print(f"     • {hotel_name}: ${row['price_usd']:.2f} ({room_type})")
                    
                    df = df[~price_outliers].copy()
                    print(f"   ✅ Removed {outlier_count} extreme price outliers")
        
        cleaned_rows = len(df)
        total_removed = initial_rows - cleaned_rows
        if total_removed > 0:
            print(f"   📊 Data quality cleaning: {initial_rows:,} → {cleaned_rows:,} rows ({total_removed} removed)")
        
        # Convert numeric columns
        numeric_columns = {
            'Number of People': 'int64',
            'price_usd': 'float64',
            'price_cad': 'float64',
            'Distance from attraction': 'float64',
            'booking_lead_time': 'int64',
            'length_of_stay': 'int64',
            'week_of_year': 'int64',
            'latitude': 'float64',
            'longitude': 'float64',
            'hotel_latitude': 'float64',
            'hotel_longitude': 'float64',
            'events_count': 'int64',
            'events_total_score': 'float64',
            'events_max_score': 'float64',
            'events_avg_score': 'float64',
            'events_earliest_day': 'int64',
            'events_latest_day': 'int64',
            'events_avg_day': 'float64',
            'events_span_days': 'int64',
            'events_density': 'float64',
            'min_distance_to_event': 'float64',
            'max_distance_to_event': 'float64',
            'avg_distance_to_events': 'float64',
            'median_distance_to_events': 'float64'
        }
        
        converted_count = 0
        for col, dtype in numeric_columns.items():
            if col in df.columns:
                try:
                    if 'int' in dtype:
                        df[col] = pd.to_numeric(df[col], errors='coerce').astype('Int64')  # Nullable integer
                    else:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                    converted_count += 1
                except Exception as e:
                    print(f"   ⚠️ Could not convert {col}: {e}")
        
        if converted_count > 0:
            print(f"   🔢 Converted {converted_count} columns to numeric types")
        
        # Convert categorical columns
        categorical_columns = [
            'Source', 'Hotel name', 'Address', 'Room Type', 'Deal Info',
            'district', 'events_primary_segment', 'room_category'
        ]
        
        cat_converted = 0
        for col in categorical_columns:
            if col in df.columns:
                df[col] = df[col].astype('string')  # Use pandas string type
                cat_converted += 1
        
        if cat_converted > 0:
            print(f"   📝 Converted {cat_converted} columns to string type")
        
        # Convert boolean columns (but skip price columns)
        boolean_columns = [
            'is_holiday', 'is_weekend', 'has_major_event', 'has_multiple_events',
            'has_nearby_event_1km', 'has_nearby_event_5km'
        ]
        
        # Protect price columns from boolean conversion
        price_protection_cols = ['price_usd', 'price_cad']
        
        bool_converted = 0
        for col in boolean_columns:
            if col in df.columns and col not in price_protection_cols:
                df[col] = df[col].astype('boolean')  # Use pandas boolean type
                bool_converted += 1
        
        # Ensure price columns remain as float
        for price_col in price_protection_cols:
            if price_col in df.columns:
                df[price_col] = pd.to_numeric(df[price_col], errors='coerce').astype('float64')
        
        if bool_converted > 0:
            print(f"   ✅ Converted {bool_converted} columns to boolean type")
        print(f"   💰 Protected price columns from type conversion")
        
        # Calculate length of stay if not exists
        if 'Check-in Date' in df.columns and 'Check-out Date' in df.columns:
            if 'length_of_stay' not in df.columns:
                df['length_of_stay'] = (df['Check-out Date'] - df['Check-in Date']).dt.days
                print(f"   🏨 Calculated length_of_stay")
        
        # Round price columns to 2 decimal places
        price_columns = ['price_usd', 'price_cad']
        for col in price_columns:
            if col in df.columns:
                df[col] = df[col].round(2)
        
        # Clean whitespace from string columns
        string_cols = df.select_dtypes(include=['object', 'string']).columns
        for col in string_cols:
            df[col] = df[col].astype(str).str.strip()
        
        final_rows = len(df)
        final_cols = len(df.columns)
        
        print(f"   ✅ Data cleaning complete:")
        print(f"     • Rows: {original_rows:,} → {final_rows:,}")
        print(f"     • Columns: {original_cols} → {final_cols}")
        
        return df
    
    def add_weather_data(self, df):
        """Step 4: Add weather data (check existing data or use seasonal estimates)."""
        print("🌤️ Step 4: Adding weather data...")
        
        # Define weather columns to check
        weather_columns = [
            'MEAN_TEMPERATURE', 'MAX_TEMPERATURE', 'MIN_TEMPERATURE',
            'TOTAL_PRECIPITATION', 'TOTAL_SNOW', 'SNOW_ON_GROUND',
            'SPEED_MAX_GUST', 'MAX_REL_HUMIDITY'
        ]
        
        # Remove duplicate weather columns with "_weather" suffix if they exist
        weather_suffix_cols = [col for col in df.columns if col.endswith('_weather') and any(w in col for w in weather_columns)]
        if weather_suffix_cols:
            df = df.drop(columns=weather_suffix_cols)
            print(f"   🧹 Removed {len(weather_suffix_cols)} duplicate weather columns with '_weather' suffix")
        
        # Check each weather column individually
        october_weather = {
            'MEAN_TEMPERATURE': 10.5,  # °C
            'MAX_TEMPERATURE': 15.2,
            'MIN_TEMPERATURE': 5.8,
            'TOTAL_PRECIPITATION': 66.2,  # mm
            'TOTAL_SNOW': 0.0,
            'SNOW_ON_GROUND': 0.0,
            'SPEED_MAX_GUST': 45.0,  # km/h
            'MAX_REL_HUMIDITY': 85.0
        }
        
        filled_count = 0
        existing_count = 0
        
        for weather_col in weather_columns:
            if weather_col in df.columns:
                # Check if column has data
                non_null_count = df[weather_col].notna().sum()
                if non_null_count == 0:
                    # Column exists but is empty - fill with estimates
                    df[weather_col] = october_weather[weather_col]
                    filled_count += 1
                else:
                    # Column has data - keep existing values
                    existing_count += 1
            else:
                # Column doesn't exist - create with estimates
                df[weather_col] = october_weather[weather_col]
                filled_count += 1
        
        if filled_count > 0:
            print(f"   ✅ Filled {filled_count} empty weather columns with seasonal estimates")
        if existing_count > 0:
            print(f"   ✅ Kept existing data in {existing_count} weather columns")
        
        return df
    
    def add_holiday_flags(self, df):
        """Step 5: Add Canadian holiday flags for all provinces."""
        print("🎉 Step 5: Adding Canadian holiday flags...")
        
        # Get holidays for all Canadian provinces
        all_provinces = ['AB', 'BC', 'MB', 'NB', 'NL', 'NS', 'NT', 'NU', 'ON', 'PE', 'QC', 'SK', 'YT']
        years = [2025, 2026]
        
        # Start with federal holidays
        canada_holidays = holidays.Canada(years=years)
        
        # Add provincial holidays
        for prov in all_provinces:
            try:
                prov_holidays = holidays.Canada(prov=prov, years=years)
                canada_holidays.update(prov_holidays)
            except:
                continue
        
        def check_holiday_period(checkin_date, checkout_date):
            """Check if any day during the stay is a holiday."""
            if pd.isna(checkin_date) or pd.isna(checkout_date):
                return False
            
            # Check each day during the stay
            current_date = checkin_date.date()
            end_date = checkout_date.date()
            
            while current_date <= end_date:
                if current_date in canada_holidays:
                    return True
                current_date += timedelta(days=1)
            return False
        
        # Apply holiday check
        if 'Check-in Date' in df.columns and 'Check-out Date' in df.columns:
            df['is_holiday'] = df.apply(lambda row: check_holiday_period(
                row['Check-in Date'], row['Check-out Date']), axis=1)
            
            holiday_count = df['is_holiday'].sum()
            print(f"   ✅ Found {holiday_count} bookings during holiday periods")
        
        return df
    
    def extract_districts(self, df):
        """Step 6: Extract and enhance district information."""
        print("🗺️ Step 6: Extracting districts...")
        
        def extract_district_from_address(address):
            """Extract district from hotel address."""
            if pd.isna(address):
                return None
            
            address = str(address).lower()
            
            # Toronto district patterns
            district_patterns = {
                'entertainment district': ['entertainment', 'theatre', 'theater'],
                'financial district': ['financial', 'bay street', 'king street'],
                'bloor-yorkville': ['bloor', 'yorkville', 'rosedale'],
                'fashion district': ['fashion', 'garment'],
                'the annex': ['annex', 'university of toronto'],
                'old town': ['old town', 'st lawrence', 'front street'],
                'yonge - dundas': ['yonge', 'dundas', 'eaton'],
                'queen west': ['queen west', 'ossington'],
                'king street west': ['king street west', 'king west'],
                'church-wellesley village': ['church', 'wellesley', 'gay village'],
                'cabbagetown': ['cabbagetown', 'parliament'],
                'leslieville': ['leslieville', 'queen east'],
                'the beaches': ['beaches', 'beach', 'woodbine'],
                'liberty village': ['liberty village', 'liberty'],
                'cityplace': ['cityplace', 'city place', 'fort york'],
                'harbourfront': ['harbourfront', 'harbour', 'waterfront'],
                'distillery district': ['distillery', 'cherry street'],
                'chinatown': ['chinatown', 'spadina'],
                'kensington market': ['kensington'],
                'little italy': ['little italy', 'college street'],
                'parkdale': ['parkdale', 'king street west'],
                'airport area': ['airport', 'pearson', 'dixon'],
                'etobicoke': ['etobicoke', 'islington'],
                'north york': ['north york', 'sheppard', 'finch'],
                'scarborough': ['scarborough'],
                'east york': ['east york'],
                'york': ['york', 'weston']
            }
            
            for district, patterns in district_patterns.items():
                for pattern in patterns:
                    if pattern in address:
                        return district.title()
            
            return None
        
        def extract_district_from_name(hotel_name):
            """Extract district from hotel name."""
            if pd.isna(hotel_name):
                return None
            
            name = str(hotel_name).lower()
            
            # Hotel name patterns
            name_patterns = {
                'Airport Area': ['airport', 'pearson'],
                'Entertainment District': ['theatre', 'theater', 'entertainment'],
                'Financial District': ['financial', 'bay street'],
                'Queen West': ['queen west'],
                'The Beaches': ['beach', 'beaches'],
                'Bloor-Yorkville': ['bloor', 'yorkville'],
                'Harbourfront': ['harbour', 'waterfront'],
                'King Street West': ['king west'],
                'Distillery District': ['distillery'],
                'North York': ['north york'],
                'Etobicoke': ['etobicoke'],
                'Scarborough': ['scarborough']
            }
            
            for district, patterns in name_patterns.items():
                for pattern in patterns:
                    if pattern in name:
                        return district
            
            return None
        
        # Extract districts from address first
        if 'Hotel address' in df.columns:
            df['district_from_address'] = df['Hotel address'].apply(extract_district_from_address)
        
        # Fill missing districts using hotel names
        if 'Hotel name' in df.columns:
            df['district_from_name'] = df['Hotel name'].apply(extract_district_from_name)
        
        # Combine district information
        if 'district' not in df.columns:
            df['district'] = None
        
        # Use address-based district first, then name-based
        if 'district_from_address' in df.columns:
            df['district'] = df['district'].fillna(df['district_from_address'])
        if 'district_from_name' in df.columns:
            df['district'] = df['district'].fillna(df['district_from_name'])
        
        # Clean up temporary columns
        temp_cols = ['district_from_address', 'district_from_name']
        df = df.drop(columns=[col for col in temp_cols if col in df.columns])
        
        district_coverage = (df['district'].notna().sum() / len(df)) * 100
        print(f"   ✅ District coverage: {district_coverage:.1f}%")
        
        return df
    
    def add_coordinates(self, df):
        """Step 7: Add latitude and longitude coordinates."""
        print("📍 Step 7: Adding coordinates...")
        
        def get_coordinates(district):
            """Get coordinates for a district."""
            if pd.isna(district):
                return self.district_coords['Toronto']  # Default to Toronto center
            
            district_clean = str(district).strip()
            
            # Direct match
            if district_clean in self.district_coords:
                return self.district_coords[district_clean]
            
            # Case-insensitive match
            for key, coords in self.district_coords.items():
                if key.lower() == district_clean.lower():
                    return coords
            
            # Partial match
            for key, coords in self.district_coords.items():
                if district_clean.lower() in key.lower() or key.lower() in district_clean.lower():
                    return coords
            
            # Default to Toronto center
            return self.district_coords['Toronto']
        
        # Add coordinates
        coords = df['district'].apply(get_coordinates)
        df['latitude'] = coords.apply(lambda x: x[0])
        df['longitude'] = coords.apply(lambda x: x[1])
        
        print(f"   ✅ Added coordinates for 100% of hotels")
        return df
    
    def fill_districts_by_coordinates(self, df):
        """Step 7.5: Fill empty districts using latitude/longitude coordinates."""
        print("🗺️ Step 7.5: Filling empty districts using coordinates...")
        
        def calculate_distance(lat1, lon1, lat2, lon2):
            """Calculate distance between two points using Haversine formula."""
            import math
            
            # Convert to radians
            lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
            
            # Haversine formula
            dlat = lat2 - lat1
            dlon = lon2 - lon1
            a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
            c = 2 * math.asin(math.sqrt(a))
            r = 6371  # Earth's radius in kilometers
            
            return c * r
        
        def find_closest_district(lat, lon):
            """Find the closest district based on coordinates."""
            if pd.isna(lat) or pd.isna(lon):
                return None
            
            min_distance = float('inf')
            closest_district = None
            
            for district, (dist_lat, dist_lon) in self.district_coords.items():
                if district == 'Toronto':  # Skip the default center
                    continue
                    
                distance = calculate_distance(lat, lon, dist_lat, dist_lon)
                if distance < min_distance:
                    min_distance = distance
                    closest_district = district
            
            return closest_district
        
        # Count empty districts before filling
        empty_districts = df['district'].isna().sum()
        
        if empty_districts > 0:
            print(f"   📍 Found {empty_districts} hotels with missing districts")
            
            # Fill empty districts based on coordinates
            mask = df['district'].isna() & df['latitude'].notna() & df['longitude'].notna()
            
            if mask.sum() > 0:
                print(f"   🔄 Using coordinates to fill {mask.sum()} districts...")
                
                # Apply the closest district function
                df.loc[mask, 'district'] = df.loc[mask].apply(
                    lambda row: find_closest_district(row['latitude'], row['longitude']), 
                    axis=1
                )
                
                # Count how many were filled
                filled_count = empty_districts - df['district'].isna().sum()
                print(f"   ✅ Filled {filled_count} districts using coordinate-based matching")
                
                # Show some examples
                if filled_count > 0:
                    filled_examples = df[mask & df['district'].notna()][['Hotel name', 'latitude', 'longitude', 'district']].head(3)
                    print(f"   📋 Examples of filled districts:")
                    for _, row in filled_examples.iterrows():
                        print(f"     • {row['Hotel name'][:30]:<30} → {row['district']}")
            else:
                print(f"   ⚠️ No coordinates available for hotels with missing districts")
        else:
            print(f"   ✅ All hotels already have district information")
        
        # Final district coverage
        final_coverage = (df['district'].notna().sum() / len(df)) * 100
        print(f"   📊 Final district coverage: {final_coverage:.1f}%")
        
        return df
    
    def add_derived_features(self, df):
        """Step 8: Add derived features for analysis."""
        print("🔧 Step 8: Adding derived features...")
        
        # Weekend flag - check if any day during stay includes weekend
        if 'Check-in Date' in df.columns and 'Check-out Date' in df.columns:
            def has_weekend_during_stay(checkin_date, checkout_date):
                """Check if any day during the stay falls on a weekend (Saturday or Sunday)."""
                if pd.isna(checkin_date) or pd.isna(checkout_date):
                    return False
                
                # Check each day during the stay
                current_date = checkin_date.date()
                end_date = checkout_date.date()
                
                while current_date <= end_date:
                    # weekday(): Monday=0, Tuesday=1, ..., Friday=4, Saturday=5, Sunday=6
                    if current_date.weekday() >= 5:  # Saturday or Sunday
                        return True
                    current_date += timedelta(days=1)
                return False
            
            df['is_weekend'] = df.apply(lambda row: has_weekend_during_stay(
                row['Check-in Date'], row['Check-out Date']), axis=1)
            print(f"   📅 Calculated is_weekend: checks if any day during stay includes weekend")
        
        # Week of year
        if 'Check-in Date' in df.columns:
            df['week_of_year'] = df['Check-in Date'].dt.isocalendar().week
        
        # Booking lead time (Check-in Date - Date when scraped)
        if 'Check-in Date' in df.columns and 'Date' in df.columns:
            df['booking_lead_time'] = (df['Check-in Date'] - df['Date']).dt.days
            df['booking_lead_time'] = df['booking_lead_time'].clip(lower=0)  # No negative lead times
            print(f"   📅 Calculated booking_lead_time: Check-in Date - Scraping Date")
        
        # Categorize room types for ML modeling
        if 'Room Type' in df.columns:
            df['room_category'] = df['Room Type'].apply(self.categorize_room_type)
            print(f"   🏨 Categorized room types: {df['Room Type'].nunique()} unique → {df['room_category'].nunique()} categories")
        
        print(f"   ✅ Added derived features")
        return df
    
    def categorize_room_type(self, room_type):
        """Categorize room types into standard industry categories for ML modeling."""
        if pd.isna(room_type):
            return 'Standard'
        
        room_str = str(room_type).lower()
        
        # Apartment-style accommodations
        if any(keyword in room_str for keyword in ['apartment', 'condo', 'house', 'bungalow']):
            if any(keyword in room_str for keyword in ['three', '3', 'four', '4']):
                return 'Large Apartment'  # 3+ bedrooms
            elif any(keyword in room_str for keyword in ['two', '2']):
                return 'Two-Bedroom Apartment'
            elif any(keyword in room_str for keyword in ['one', '1', 'studio']):
                return 'One-Bedroom Apartment'
            else:
                return 'Apartment'
        
        # Suite categories
        elif any(keyword in room_str for keyword in ['suite', 'junior suite']):
            if any(keyword in room_str for keyword in ['presidential', 'penthouse', 'luxury']):
                return 'Luxury Suite'
            elif any(keyword in room_str for keyword in ['junior']):
                return 'Junior Suite'
            else:
                return 'Suite'
        
        # Premium/Deluxe rooms
        elif any(keyword in room_str for keyword in ['deluxe', 'premium', 'superior', 'executive', 'business']):
            return 'Deluxe Room'
        
        # Studio rooms
        elif 'studio' in room_str:
            return 'Studio'
        
        # Family rooms (multiple beds or explicitly family)
        elif any(keyword in room_str for keyword in ['family', 'triple', 'quad', 'two queen', 'two double', 'two bed']):
            return 'Family Room'
        
        # Accessible rooms
        elif any(keyword in room_str for keyword in ['accessible', 'handicap', 'mobility']):
            return 'Accessible Room'
        
        # Budget/Economy rooms
        elif any(keyword in room_str for keyword in ['budget', 'economy', 'basic', 'shared bathroom', 'hostel']):
            return 'Economy Room'
        
        # Standard rooms by bed type
        elif any(keyword in room_str for keyword in ['king']):
            return 'Standard King'
        elif any(keyword in room_str for keyword in ['queen']):
            return 'Standard Queen'
        elif any(keyword in room_str for keyword in ['double', 'twin']):
            return 'Standard Double'
        elif any(keyword in room_str for keyword in ['single']):
            return 'Standard Single'
        
        # Generic room types
        elif any(keyword in room_str for keyword in ['standard', 'room']):
            return 'Standard Room'
        
        # Default category
        else:
            return 'Standard'
    
    def final_quality_check(self, df):
        """Step 9: Final data quality check and summary."""
        print("✅ Step 9: Final quality check...")
        
        # Basic statistics
        total_rows = len(df)
        total_cols = len(df.columns)
        
        # Check missing data
        missing_data = df.isnull().sum()
        missing_pct = (missing_data / len(df)) * 100
        
        print(f"   📊 Final dataset: {total_rows:,} rows × {total_cols} columns")
        
        # Key feature coverage
        key_features = ['price_cad', 'district', 'latitude', 'longitude', 'is_holiday']
        for feature in key_features:
            if feature in df.columns:
                coverage = ((df[feature].notna().sum() / len(df)) * 100)
                print(f"   • {feature}: {coverage:.1f}% coverage")
        
        # Price statistics
        if 'price_cad' in df.columns:
            price_stats = df['price_cad'].describe()
            print(f"   💰 Price range: ${price_stats['min']:.2f} - ${price_stats['max']:.2f} CAD")
            print(f"   💰 Mean price: ${price_stats['mean']:.2f} CAD")
        
        return df
    
    def remove_empty_columns(self, df):
        """Step 10: Remove columns that are completely empty."""
        print("🗑️ Step 10: Removing empty columns...")
        
        original_cols = len(df.columns)
        
        # Find columns that are completely empty (all null values)
        empty_columns = []
        for col in df.columns:
            if df[col].isnull().all():
                empty_columns.append(col)
        
        if empty_columns:
            # Remove empty columns
            df = df.drop(columns=empty_columns)
            print(f"   🧹 Removed {len(empty_columns)} empty columns:")
            for col in empty_columns:
                print(f"     • {col}")
        else:
            print(f"   ✅ No completely empty columns found")
        
        final_cols = len(df.columns)
        print(f"   📊 Columns: {original_cols} → {final_cols}")
        
        return df
    
    def remove_low_coverage_columns(self, df, coverage_threshold=0.2):
        """Step 11: Remove columns with low data coverage (less than 20% by default)."""
        print(f"🗑️ Step 11: Removing columns with less than {coverage_threshold*100:.0f}% coverage...")
        
        original_cols = len(df.columns)
        low_coverage_columns = []
        
        # Essential columns that should never be removed regardless of coverage
        essential_columns = [
            'Hotel name', 'price_usd', 'price_cad', 'Check-in Date', 'Check-out Date',
            'district', 'latitude', 'longitude', 'room_category', 'is_weekend', 
            'is_holiday', 'booking_lead_time', 'length_of_stay', 'Distance from attraction'
        ]
        
        for col in df.columns:
            if col in essential_columns:
                continue  # Skip essential columns
            
            # Calculate coverage (non-null percentage)
            coverage = df[col].notna().sum() / len(df)
            
            if coverage < coverage_threshold:
                low_coverage_columns.append((col, coverage))
        
        if low_coverage_columns:
            # Sort by coverage for better reporting
            low_coverage_columns.sort(key=lambda x: x[1])
            
            # Remove the columns
            columns_to_drop = [col[0] for col in low_coverage_columns]
            df = df.drop(columns=columns_to_drop)
            
            print(f"   🧹 Removed {len(columns_to_drop)} low coverage columns:")
            for col, coverage in low_coverage_columns:
                print(f"     • {col} ({coverage*100:.1f}% coverage)")
        else:
            print(f"   ✅ No columns with coverage below {coverage_threshold*100:.0f}% found")
        
        final_cols = len(df.columns)
        print(f"   📊 Columns: {original_cols} → {final_cols}")
        
        return df
    
    def save_final_dataset(self, df, output_file='Data/toronto_hotels_transformed.csv'):
        """Save the final processed dataset."""
        print(f"💾 Saving final dataset to {output_file}...")
        
        try:
            df.to_csv(output_file, index=False)
            print(f"   ✅ Saved successfully: {len(df)} rows, {len(df.columns)} columns")
        except Exception as e:
            print(f"   ❌ Error saving: {e}")
    
    def run_pipeline(self, output_file='Data/toronto_hotels_transformed.csv'):
        """Run the complete data transformation pipeline."""
        print(f"🚀 Starting complete data transformation pipeline...")
        print(f"📁 Input: {self.input_file}")
        print(f"📁 Output: {output_file}")
        print()
        
        # Step 1: Load data
        df = self.load_data()
        if df is None:
            return None
        
        # Step 2: Currency conversion
        df = self.currency_conversion(df)
        
        # Step 3: Data cleaning
        df = self.data_cleaning(df)
        
        # Step 4: Weather data
        df = self.add_weather_data(df)
        
        # Step 5: Holiday flags
        df = self.add_holiday_flags(df)
        
        # Step 6: District extraction
        df = self.extract_districts(df)
        
        # Step 7: Coordinates
        df = self.add_coordinates(df)
        
        # Step 7.5: Fill empty districts using coordinates
        df = self.fill_districts_by_coordinates(df)
        
        # Step 8: Derived features
        df = self.add_derived_features(df)
        
        # Step 9: Quality check
        df = self.final_quality_check(df)
        
        # Step 10: Remove empty columns
        df = self.remove_empty_columns(df)
        
        # Step 11: Remove low coverage columns
        df = self.remove_low_coverage_columns(df)
        
        # Save final dataset
        self.save_final_dataset(df, output_file)
        
        print()
        print("🎉 PIPELINE COMPLETE!")
        print("=" * 50)
        print(f"✅ Transformed {self.input_file}")
        print(f"✅ Created {output_file}")
        print("✅ Ready for machine learning and analysis!")
        
        return df


def main():
    """Run the complete hotel data transformation pipeline."""
    
    # Initialize pipeline
    pipeline = HotelDataPipeline('Data/toronto_unified_hotel_analysis.csv')
    
    # Run the complete transformation
    final_df = pipeline.run_pipeline('Data/toronto_hotels_transformed.csv')
    
    if final_df is not None:
        print(f"\n📋 TRANSFORMATION SUMMARY")
        print("-" * 30)
        print(f"Input file: toronto_unified_hotel_analysis.csv")
        print(f"Output file: toronto_hotels_transformed.csv")
        print(f"Transformations applied:")
        print(f"  ✅ USD → CAD conversion")
        print(f"  ✅ Data cleaning & duplicate removal")
        print(f"  ✅ Weather data integration")
        print(f"  ✅ Canadian holiday detection")
        print(f"  ✅ District extraction & enhancement")
        print(f"  ✅ Geocoding with coordinates")
        print(f"  ✅ Derived feature engineering")
        print(f"\n🎯 Ready for hotel price prediction modeling!")


if __name__ == "__main__":
    main()