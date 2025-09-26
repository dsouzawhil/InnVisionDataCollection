import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import glob
import requests

def read_toronto_events():
    """Read the Toronto 2025 events file."""
    events_file = 'Data/toronto_events_with_scores.csv'
    
    if os.path.exists(events_file):
        try:
            events_df = pd.read_csv(events_file)
            print(f"✅ Successfully loaded {len(events_df)} events from {events_file}")
            
            # Display basic info about the events data
            print(f"\n📊 Events Dataset Info (before deduplication):")
            print(f"   Total events: {len(events_df)}")
            print(f"   Columns: {list(events_df.columns)}")
            print(f"   Date range: {events_df['Date'].min()} to {events_df['Date'].max()}")
            
            # Remove duplicate entries
            print(f"\n🔄 Removing duplicate events...")
            initial_count = len(events_df)
            
            # Remove duplicates based on Event Name, Date, and Venue combination
            events_df = events_df.drop_duplicates(subset=['Event Name', 'Date', 'Venue'], keep='first')
            
            final_count = len(events_df)
            duplicates_removed = initial_count - final_count
            
            print(f"   ✅ Removed {duplicates_removed} duplicate events")
            print(f"   📊 Final events count: {final_count}")
            
            # Show first few rows
            print(f"\n📋 First 5 events (after deduplication):")
            print(events_df.head())
            
            return events_df
            
        except Exception as e:
            print(f"❌ Error loading events data: {e}")
            return None
    else:
        print(f"❌ Events file not found: {events_file}")
        print("💡 Please run geteventsdata.py first to generate the events data.")
        return None

def clean_events_data(events_df):
    """Clean the events dataframe by converting date and numeric columns to proper data types."""
    if events_df is None:
        print("❌ No events data provided for cleaning.")
        return None
    
    print("🧹 Starting events data cleaning...")
    
    # Create a copy to avoid modifying original data
    events_cleaned = events_df.copy()
    
    # 1. Convert Date column to datetime
    if 'Date' in events_cleaned.columns:
        print("📅 Converting Date column to datetime...")
        
        # Convert to datetime
        events_cleaned['Date'] = pd.to_datetime(events_cleaned['Date'], errors='coerce')
        
        # Count successful conversions
        valid_dates = events_cleaned['Date'].notna().sum()
        invalid_dates = events_cleaned['Date'].isna().sum()
        print(f"   ✅ Converted {valid_dates} date values to datetime")
        if invalid_dates > 0:
            print(f"   ⚠️ {invalid_dates} invalid dates set to NaN")
    
    # 2. Convert Latitude to numeric (float)
    if 'Latitude' in events_cleaned.columns:
        print("🌍 Converting Latitude to numeric...")
        
        # Replace 'N/A' with NaN first
        events_cleaned['Latitude'] = events_cleaned['Latitude'].replace(['N/A', 'n/a', ''], pd.NA)
        
        # Convert to numeric
        events_cleaned['Latitude'] = pd.to_numeric(events_cleaned['Latitude'], errors='coerce')
        
        # Count valid coordinates
        valid_lat = events_cleaned['Latitude'].notna().sum()
        invalid_lat = events_cleaned['Latitude'].isna().sum()
        print(f"   ✅ Converted {valid_lat} latitude values to float")
        if invalid_lat > 0:
            print(f"   ⚠️ {invalid_lat} invalid latitudes set to NaN")
    
    # 3. Convert Longitude to numeric (float)
    if 'Longitude' in events_cleaned.columns:
        print("🌍 Converting Longitude to numeric...")
        
        # Replace 'N/A' with NaN first
        events_cleaned['Longitude'] = events_cleaned['Longitude'].replace(['N/A', 'n/a', ''], pd.NA)
        
        # Convert to numeric
        events_cleaned['Longitude'] = pd.to_numeric(events_cleaned['Longitude'], errors='coerce')
        
        # Count valid coordinates
        valid_lon = events_cleaned['Longitude'].notna().sum()
        invalid_lon = events_cleaned['Longitude'].isna().sum()
        print(f"   ✅ Converted {valid_lon} longitude values to float")
        if invalid_lon > 0:
            print(f"   ⚠️ {invalid_lon} invalid longitudes set to NaN")
    
    # 4. Convert event_score to numeric (check if column exists)
    if 'event_score' in events_cleaned.columns:
        print("🎯 Converting event_score to numeric...")
        
        # Replace 'N/A' with NaN first
        events_cleaned['event_score'] = events_cleaned['event_score'].replace(['N/A', 'n/a', ''], pd.NA)
        
        # Convert to numeric (int or float)
        events_cleaned['event_score'] = pd.to_numeric(events_cleaned['event_score'], errors='coerce')
        
        # Count valid scores
        valid_scores = events_cleaned['event_score'].notna().sum()
        invalid_scores = events_cleaned['event_score'].isna().sum()
        print(f"   ✅ Converted {valid_scores} event scores to numeric")
        if invalid_scores > 0:
            print(f"   ⚠️ {invalid_scores} invalid event scores set to NaN")
    else:
        print("📝 event_score column not found - may need to run geteventsdata.py first")
    
    # 5. Advanced text cleaning for categorical columns
    print("📝 Applying advanced text cleaning...")
    
    # Define columns that need consistent case treatment
    categorical_columns = ['Segment', 'Genre', 'SubGenre']
    
    for col in categorical_columns:
        if col in events_cleaned.columns:
            print(f"✂️ Cleaning {col}...")
            
            # Basic cleaning: strip whitespace, handle nulls
            events_cleaned[col] = events_cleaned[col].astype(str).str.strip()
            events_cleaned[col] = events_cleaned[col].replace(['', 'N/A', 'n/a', 'nan'], None)
            
            # Convert to title case for consistency
            events_cleaned[col] = events_cleaned[col].str.title()
            
            # Fill missing values with "Unknown" for categorical consistency
            missing_before = events_cleaned[col].isna().sum()
            events_cleaned[col] = events_cleaned[col].fillna('Unknown')
            
            valid_values = events_cleaned[col].notna().sum()
            print(f"   ✅ Cleaned {valid_values} {col} values to title case")
            if missing_before > 0:
                print(f"   🔄 Filled {missing_before} missing {col} values with 'Unknown'")
    
    # Special cleaning for Venue column (common venue name standardization)
    if 'Venue' in events_cleaned.columns:
        print("🏟️ Standardizing venue names...")
        
        # Basic cleaning first
        events_cleaned['Venue'] = events_cleaned['Venue'].astype(str).str.strip()
        events_cleaned['Venue'] = events_cleaned['Venue'].replace(['', 'N/A', 'n/a', 'nan'], None)
        
        # Common venue name corrections
        venue_corrections = {
            'rogers center': 'Rogers Centre',
            'rogers centre': 'Rogers Centre',
            'scotiabank arena': 'Scotiabank Arena',
            'coca-cola coliseum': 'Coca-Cola Coliseum',
            'coca cola coliseum': 'Coca-Cola Coliseum',
            'roy thomson hall': 'Roy Thomson Hall',
            'roy thompson hall': 'Roy Thomson Hall',
            'princess of wales theatre': 'Princess of Wales Theatre',
            'princess of wales theater': 'Princess of Wales Theatre',
            'ed mirvish theatre': 'Ed Mirvish Theatre',
            'ed mirvish theater': 'Ed Mirvish Theatre',
            'royal alexandra theatre': 'Royal Alexandra Theatre',
            'royal alexandra theater': 'Royal Alexandra Theatre',
            'the phoenix concert theatre': 'The Phoenix Concert Theatre',
            'phoenix concert theatre': 'The Phoenix Concert Theatre',
            'the danforth music hall': 'The Danforth Music Hall',
            'danforth music hall': 'The Danforth Music Hall',
            'the opera house': 'The Opera House',
            'opera house': 'The Opera House'
        }
        
        # Apply corrections (case insensitive)
        for incorrect, correct in venue_corrections.items():
            mask = events_cleaned['Venue'].str.lower() == incorrect.lower()
            events_cleaned.loc[mask, 'Venue'] = correct
        
        # Clean up "The" articles consistently
        events_cleaned['Venue'] = events_cleaned['Venue'].str.replace(r'^the\s+(.+)', r'The \1', regex=True, case=False)
        
        valid_venues = events_cleaned['Venue'].notna().sum()
        print(f"   ✅ Standardized {valid_venues} venue names")
    
    # Clean Event Name column (basic cleaning only, preserve original formatting)
    if 'Event Name' in events_cleaned.columns:
        print("📋 Cleaning event names...")
        
        events_cleaned['Event Name'] = events_cleaned['Event Name'].astype(str).str.strip()
        events_cleaned['Event Name'] = events_cleaned['Event Name'].replace(['', 'N/A', 'n/a', 'nan'], None)
        
        valid_events = events_cleaned['Event Name'].notna().sum()
        print(f"   ✅ Cleaned {valid_events} event names")
    
    # Clean Address column
    if 'Address' in events_cleaned.columns:
        print("📍 Cleaning addresses...")
        
        events_cleaned['Address'] = events_cleaned['Address'].astype(str).str.strip()
        events_cleaned['Address'] = events_cleaned['Address'].replace(['', 'N/A', 'n/a', 'nan'], None)
        
        # Standardize common address formats
        events_cleaned['Address'] = events_cleaned['Address'].str.replace(r'\s+', ' ', regex=True)  # Multiple spaces to single
        events_cleaned['Address'] = events_cleaned['Address'].str.replace(r',\s*,', ',', regex=True)  # Double commas
        
        valid_addresses = events_cleaned['Address'].notna().sum()
        print(f"   ✅ Cleaned {valid_addresses} addresses")
    
    print(f"✅ Events data cleaning complete. DataFrame now has {len(events_cleaned.columns)} columns.")
    
    # Show data types after cleaning
    print(f"\n📊 Data Types After Cleaning:")
    for col in events_cleaned.columns:
        dtype = events_cleaned[col].dtype
        non_null = events_cleaned[col].notna().sum()
        print(f"   {col}: {dtype} ({non_null}/{len(events_cleaned)} non-null)")
    
    return events_cleaned

def read_all_hotel_files():
    """Read all hotel CSV files from hotel_listing folder and combine into one DataFrame."""
    hotel_folder = 'Data/hotel_listing'
    
    if not os.path.exists(hotel_folder):
        print(f"❌ Hotel listing folder not found: {hotel_folder}")
        return None
    
    # Find all CSV files in hotel_listing folder
    hotel_files = glob.glob(os.path.join(hotel_folder, 'Toronto_hotels*.csv'))
    
    if not hotel_files:
        print(f"❌ No hotel CSV files found in {hotel_folder}")
        return None
    
    print(f"📂 Found {len(hotel_files)} hotel files:")
    
    all_hotels = []
    total_hotels = 0
    
    for file in hotel_files:
        try:
            df = pd.read_csv(file)
            filename = os.path.basename(file)
            print(f"   ✅ {filename}: {len(df)} hotels")
            all_hotels.append(df)
            total_hotels += len(df)
        except Exception as e:
            print(f"   ❌ Error loading {file}: {e}")
    
    if all_hotels:
        # Combine all DataFrames
        combined_hotels = pd.concat(all_hotels, ignore_index=True)
        print(f"\n🏨 Combined Hotels Dataset:")
        print(f"   Total hotels: {len(combined_hotels)}")
        print(f"   Columns: {list(combined_hotels.columns)}")
        
        # Show data types
        print(f"\n📋 Column Info:")
        for col in combined_hotels.columns:
            non_null = combined_hotels[col].notna().sum()
            print(f"   {col}: {non_null}/{len(combined_hotels)} non-null")
        
        # # Show first few rows
        # print(f"\n📋 First 5 hotel records:")
        # print(combined_hotels.head())
        #
        return combined_hotels
    else:
        print("❌ No hotel data could be loaded.")
        return None

def data_cleaning(hotels_df):
    """Clean the hotels dataframe by removing empty review columns and cleaning distance data."""
    if hotels_df is None:
        print("❌ No hotel data provided for cleaning.")
        return None
    
    print("🧹 Starting data cleaning...")
    
    # Define review columns to remove
    review_columns_to_drop = [
        'Review', 'Review Score', 'Number of Reviews', 
        'Rating', 'Star Rating', 'Location Score', 
        'name', 'score'  # Also removing these sparse columns
    ]
    
    # Find existing review columns in the DataFrame
    existing_review_cols = [col for col in review_columns_to_drop if col in hotels_df.columns]
    
    if existing_review_cols:
        hotels_df_cleaned = hotels_df.drop(columns=existing_review_cols)
        print(f"🗑️ Removed empty review columns: {existing_review_cols}")
    else:
        hotels_df_cleaned = hotels_df.copy()
        print("✅ No review columns found to remove.")
    
    # Clean Distance from attraction column - extract just the number
    distance_cols = ['Distance from attraction', 'Distance from Attraction']
    
    for col in distance_cols:
        if col in hotels_df_cleaned.columns:
            print(f"🔢 Cleaning distance column: {col}")
            
            # Extract numeric value from strings like "0.6 miles from downtown"
            hotels_df_cleaned[col] = hotels_df_cleaned[col].astype(str).str.extract(r'(\d+\.?\d*)')
            hotels_df_cleaned[col] = pd.to_numeric(hotels_df_cleaned[col], errors='coerce')
            
            # Count how many values were successfully converted
            valid_distances = hotels_df_cleaned[col].notna().sum()
            print(f"   ✅ Extracted {valid_distances} numeric distance values")
    
    # Clean price column - remove dollar sign and convert to numeric
    if 'price' in hotels_df_cleaned.columns:
        print(f"💰 Cleaning price column")
        
        # Remove dollar sign, commas, and any other non-numeric characters (except decimal point)
        hotels_df_cleaned['price'] = hotels_df_cleaned['price'].astype(str).str.replace('$', '').str.replace(',', '')
        
        # Extract numeric value (handles cases like "$123.45" or "123")
        hotels_df_cleaned['price'] = hotels_df_cleaned['price'].str.extract(r'(\d+\.?\d*)')
        hotels_df_cleaned['price'] = pd.to_numeric(hotels_df_cleaned['price'], errors='coerce')
        
        # Count how many prices were successfully converted
        valid_prices = hotels_df_cleaned['price'].notna().sum()
        print(f"   ✅ Converted {valid_prices} price values to numeric")
        
        # Convert USD to CAD
        print(f"💱 Converting prices from USD to CAD...")
        
        # Fetch current USD to CAD exchange rate dynamically
        def get_usd_to_cad_rate():
            """Fetch current USD to CAD exchange rate from API."""
            try:
                print("🌐 Fetching current USD to CAD exchange rate...")
                
                # Try multiple API sources for reliability
                apis_to_try = [
                    "https://api.exchangerate-api.com/v4/latest/USD",
                    "https://api.fxratesapi.com/latest?base=USD&symbols=CAD",
                    "https://open.er-api.com/v6/latest/USD"
                ]
                
                for api_url in apis_to_try:
                    try:
                        response = requests.get(api_url, timeout=10)
                        response.raise_for_status()
                        data = response.json()
                        
                        # Extract CAD rate from different API formats
                        if "rates" in data and "CAD" in data["rates"]:
                            rate = float(data["rates"]["CAD"])
                            print(f"   ✅ Current rate fetched: 1 USD = {rate:.4f} CAD")
                            return rate
                            
                    except requests.exceptions.RequestException as e:
                        print(f"   ⚠️ API {api_url} failed: {e}")
                        continue
                
                # If all APIs fail, use fallback
                print("   ⚠️ All exchange rate APIs failed")
                return None
                
            except Exception as e:
                print(f"   ❌ Error fetching exchange rate: {e}")
                return None
        
        # Get exchange rate (dynamic or fallback)
        usd_to_cad_rate = get_usd_to_cad_rate()
        
        if usd_to_cad_rate is None:
            # Fallback to approximate rate
            usd_to_cad_rate = 1.35
            print(f"   🔄 Using fallback rate: 1 USD = {usd_to_cad_rate} CAD")
        
        # Convert prices to CAD
        hotels_df_cleaned['price_cad'] = hotels_df_cleaned['price'] * usd_to_cad_rate
        hotels_df_cleaned['price_cad'] = hotels_df_cleaned['price_cad'].round(2)
        
        # Keep original USD price and rename for clarity
        hotels_df_cleaned.rename(columns={'price': 'price_usd'}, inplace=True)
        
        valid_cad_prices = hotels_df_cleaned['price_cad'].notna().sum()
        print(f"   ✅ Converted {valid_cad_prices} prices to CAD (rate: {usd_to_cad_rate})")
        
        if valid_cad_prices > 0:
            avg_usd = hotels_df_cleaned['price_usd'].mean()
            avg_cad = hotels_df_cleaned['price_cad'].mean()
            print(f"   📊 Average price: ${avg_usd:.2f} USD → ${avg_cad:.2f} CAD")
    
    # Convert date columns to datetime objects
    date_columns = ['Date', 'Check-in Date', 'Check-out Date']
    
    for col in date_columns:
        if col in hotels_df_cleaned.columns:
            print(f"📅 Converting {col} to datetime")
            
            # Convert to datetime
            hotels_df_cleaned[col] = pd.to_datetime(hotels_df_cleaned[col], errors='coerce')
            
            # Count successful conversions
            valid_dates = hotels_df_cleaned[col].notna().sum()
            print(f"   ✅ Converted {valid_dates} date values to datetime")
    
    print(f"✅ Data cleaning complete. DataFrame now has {len(hotels_df_cleaned.columns)} columns.")
    print(f"   Cleaned columns: {list(hotels_df_cleaned.columns)}")
    
    return hotels_df_cleaned

def data_transformation(hotels_df):
    """Transform cleaned hotel data by creating new analytical columns."""
    if hotels_df is None:
        print("❌ No hotel data provided for transformation.")
        return None
    
    print("🔄 Starting data transformation...")
    
    # Create a copy to avoid modifying original data
    hotels_transformed = hotels_df.copy()
    
    # Get current date (scraping date)
    scraping_date = datetime.now().date()
    print(f"📅 Using scraping date: {scraping_date}")
    
    # 1. Booking Lead Time: Check-in Date - Scraping Date
    if 'Check-in Date' in hotels_transformed.columns:
        print("⏰ Calculating booking lead time...")
        
        # Convert check-in dates to date objects for calculation
        checkin_dates = pd.to_datetime(hotels_transformed['Check-in Date']).dt.date
        
        # Calculate lead time in days
        hotels_transformed['booking_lead_time'] = (checkin_dates - scraping_date).apply(
            lambda x: x.days if pd.notna(x) else None
        )
        
        valid_lead_times = hotels_transformed['booking_lead_time'].notna().sum()
        print(f"   ✅ Calculated lead time for {valid_lead_times} bookings")
    
    # 2. Length of Stay: Check-out Date - Check-in Date
    if 'Check-in Date' in hotels_transformed.columns and 'Check-out Date' in hotels_transformed.columns:
        print("🛏️ Calculating length of stay...")
        
        checkin = pd.to_datetime(hotels_transformed['Check-in Date'])
        checkout = pd.to_datetime(hotels_transformed['Check-out Date'])
        
        # Calculate nights stayed
        hotels_transformed['length_of_stay'] = (checkout - checkin).dt.days
        
        valid_stays = hotels_transformed['length_of_stay'].notna().sum()
        print(f"   ✅ Calculated length of stay for {valid_stays} bookings")
    
    # 3. Day of Week for Check-in Date
    if 'Check-in Date' in hotels_transformed.columns:
        print("📆 Extracting day of week...")
        
        hotels_transformed['day_of_week'] = pd.to_datetime(hotels_transformed['Check-in Date']).dt.day_name()
        
        valid_days = hotels_transformed['day_of_week'].notna().sum()
        print(f"   ✅ Extracted day of week for {valid_days} check-ins")
    
    # 4. Month for Check-in Date
    if 'Check-in Date' in hotels_transformed.columns:
        print("📅 Extracting month...")
        
        hotels_transformed['month'] = pd.to_datetime(hotels_transformed['Check-in Date']).dt.month_name()
        
        valid_months = hotels_transformed['month'].notna().sum()
        print(f"   ✅ Extracted month for {valid_months} check-ins")
    
    # 5. Week of Year for Check-in Date
    if 'Check-in Date' in hotels_transformed.columns:
        print("📊 Calculating week of year...")
        
        hotels_transformed['week_of_year'] = pd.to_datetime(hotels_transformed['Check-in Date']).dt.isocalendar().week
        
        valid_weeks = hotels_transformed['week_of_year'].notna().sum()
        print(f"   ✅ Calculated week of year for {valid_weeks} check-ins")
    
    # 6. Is Weekend Flag: Check if Friday, Saturday, or Sunday falls within the stay
    if 'Check-in Date' in hotels_transformed.columns and 'Check-out Date' in hotels_transformed.columns:
        print("🎉 Calculating weekend flag...")
        
        def check_weekend_overlap(checkin, checkout):
            """Check if any Friday, Saturday, or Sunday falls within the stay period."""
            if pd.isna(checkin) or pd.isna(checkout):
                return None
            
            checkin_date = pd.to_datetime(checkin)
            checkout_date = pd.to_datetime(checkout)
            
            # Generate all dates in the stay period (excluding checkout day)
            stay_dates = pd.date_range(start=checkin_date, end=checkout_date - timedelta(days=1))
            
            # Check if any date is Friday (4), Saturday (5), or Sunday (6)
            weekend_days = stay_dates.dayofweek.isin([4, 5, 6])  # Friday, Saturday, Sunday
            
            return weekend_days.any()
        
        # Apply the weekend check function
        hotels_transformed['is_weekend'] = hotels_transformed.apply(
            lambda row: check_weekend_overlap(row['Check-in Date'], row['Check-out Date']), 
            axis=1
        )
        
        valid_weekend_flags = hotels_transformed['is_weekend'].notna().sum()
        weekend_stays = hotels_transformed['is_weekend'].sum() if hotels_transformed['is_weekend'].notna().any() else 0
        print(f"   ✅ Calculated weekend flag for {valid_weekend_flags} bookings")
        print(f"   🎊 {weekend_stays} bookings include weekends")
    
    print(f"✅ Data transformation complete. DataFrame now has {len(hotels_transformed.columns)} columns.")
    
    # Show new columns created
    new_columns = ['booking_lead_time', 'length_of_stay', 'day_of_week', 'month', 'week_of_year', 'is_weekend']
    existing_new_cols = [col for col in new_columns if col in hotels_transformed.columns]
    print(f"   New columns added: {existing_new_cols}")
    
    # 7. Extract District from Address
    if 'Address' in hotels_transformed.columns:
        print("🏙️ Extracting district from address...")
        
        def extract_district(address):
            """Extract district from address using multiple methods."""
            if pd.isna(address) or address == 'N/A':
                return None
            
            address_str = str(address)
            
            # Method 1: Extract from parentheses (most common format)
            # Example: "Entertainment District, Toronto (Entertainment District)"
            import re
            parentheses_match = re.search(r'\(([^)]+)\)', address_str)
            if parentheses_match:
                district = parentheses_match.group(1).strip()
                # Remove "Toronto" if it appears in the district name
                district = re.sub(r',?\s*Toronto$', '', district).strip()
                if district and district != 'Toronto':
                    return district
            
            # Method 2: Extract from comma-separated parts
            # Example: "Financial District, Toronto" or "Etobicoke, Toronto"
            parts = address_str.split(',')
            if len(parts) >= 2:
                potential_district = parts[0].strip()
                # Skip if it's clearly a street address (contains numbers)
                if not re.search(r'\d+', potential_district) and len(potential_district) > 3:
                    return potential_district
            
            # Method 3: Known Toronto districts/neighborhoods
            toronto_districts = [
                'Entertainment District', 'Financial District', 'Yonge - Dundas', 'The Village',
                'Kensington Market', 'Etobicoke', 'Scarborough', 'North York', 'York',
                'Downtown', 'Midtown', 'Uptown', 'Harbourfront', 'Distillery District',
                'King Street West', 'Queen Street West', 'Yorkville', 'The Annex',
                'Liberty Village', 'CityPlace', 'Parkdale', 'Leslieville', 'Riverdale',
                'Corktown', 'St. Lawrence', 'Church-Wellesley', 'Bloor West Village',
                'Little Italy', 'Little Portugal', 'Chinatown', 'Koreatown', 'Greektown'
            ]
            
            # Check if any known district appears in the address
            address_upper = address_str.upper()
            for district in toronto_districts:
                if district.upper() in address_upper:
                    return district
            
            # If no district found, return None
            return None
        
        # Apply district extraction
        hotels_transformed['district'] = hotels_transformed['Address'].apply(extract_district)
        
        valid_districts = hotels_transformed['district'].notna().sum()
        unique_districts = hotels_transformed['district'].nunique()
        print(f"   ✅ Extracted districts for {valid_districts} hotels")
        print(f"   🏘️ Found {unique_districts} unique districts")
        
        # Show district distribution
        if valid_districts > 0:
            district_counts = hotels_transformed['district'].value_counts().head(10)
            print(f"   📊 Top districts:")
            for district, count in district_counts.items():
                print(f"      {district}: {count} hotels")
    
    print(f"✅ Data transformation complete. DataFrame now has {len(hotels_transformed.columns)} columns.")
    
    # Show new columns created
    new_columns = ['booking_lead_time', 'length_of_stay', 'day_of_week', 'month', 'week_of_year', 'is_weekend', 'district']
    existing_new_cols = [col for col in new_columns if col in hotels_transformed.columns]
    print(f"   New columns added: {existing_new_cols}")
    
    return hotels_transformed

def main():

    print("\n1️⃣ Loading Events Data...")
    events_df = read_toronto_events()

    # Clean events data
    if events_df is not None:
        print("\n1️⃣.1 Cleaning Events Data...")
        events_df = clean_events_data(events_df)

    # Read all hotel data
    print("\n2️⃣ Loading Hotel Data...")
    hotels_df = read_all_hotel_files()

    # Clean hotel data
    if hotels_df is not None:
        print("\n3️⃣ Cleaning Hotel Data...")
        hotels_df = data_cleaning(hotels_df)

    # Transform hotel data
    if hotels_df is not None:
        print("\n4️⃣ Transforming Hotel Data...")
        hotels_df = data_transformation(hotels_df)

    # Summary
    print("\n" + "="*45)
    print("📊 DATA LOADING SUMMARY")
    print("="*45)

    if events_df is not None:
        print(f"✅ Events: {len(events_df)} records loaded")
    else:
        print("❌ Events: Failed to load")

    if hotels_df is not None:
        print(f"✅ Hotels: {len(hotels_df)} records loaded")
    else:
        print("❌ Hotels: Failed to load")

    if events_df is not None and hotels_df is not None:
        print("\n🎉 Both datasets loaded successfully!")
        print("💡 Ready for analysis and combination!")
        
        # Save cleaned datasets to CSV
        print("\n5️⃣ Saving Cleaned Data...")
        
        # Save cleaned events data
        if events_df is not None:
            events_output_file = 'Data/toronto_events_cleaned.csv'
            events_df.to_csv(events_output_file, index=False)
            print(f"✅ Cleaned events data saved to: {events_output_file}")
            print(f"   Events records: {len(events_df)}")
        
        # Save cleaned and transformed hotels data  
        if hotels_df is not None:
            hotels_output_file = 'Data/toronto_hotels_cleaned_transformed.csv'
            hotels_df.to_csv(hotels_output_file, index=False)
            print(f"✅ Cleaned & transformed hotel data saved to: {hotels_output_file}")
            print(f"   Hotel records: {len(hotels_df)}")
            print(f"   Columns: {len(hotels_df.columns)}")
    else:
        print("\n⚠️ Some data could not be loaded. Check file paths and run data generation scripts if needed.")

    # Show sample data and info
    if hotels_df is not None:
        print("\n📋 Hotel Data Sample:")
        print(hotels_df.head())
        
        print("\n📊 Data Types and Info:")
        print(hotels_df.dtypes)
        print(f"\nDataFrame Info:")
        hotels_df.info()

    if events_df is not None:
        print("\n📋 Events Data Sample:")
        print(events_df.head())
        
        print("\n📊 Events Data Types:")
        print(events_df.dtypes)

main()

