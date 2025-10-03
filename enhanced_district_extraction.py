"""
Enhanced District Extraction for Toronto Hotels
===============================================

This script improves district extraction from hotel addresses to fill missing values.
"""

import pandas as pd
import re
from collections import Counter

def enhanced_extract_district(address):
    """Enhanced district extraction with more comprehensive patterns."""
    if pd.isna(address) or address == 'N/A' or str(address).strip() == '':
        return None
    
    address_str = str(address).strip()
    
    # Enhanced Toronto districts/neighborhoods with variations
    toronto_districts = {
        # Core districts
        'Entertainment District': ['entertainment district', 'entertainment'],
        'Financial District': ['financial district', 'financial'],
        'Fashion District': ['fashion district', 'fashion'],
        'Distillery District': ['distillery district', 'distillery'],
        'St. Lawrence': ['st. lawrence', 'st lawrence', 'saint lawrence'],
        'Old Town': ['old town', 'oldtown'],
        'Corktown': ['corktown', 'cork town'],
        
        # Yorkville area
        'Bloor-Yorkville': ['bloor-yorkville', 'yorkville', 'bloor yorkville', 'bloor street'],
        'The Annex': ['the annex', 'annex'],
        'Rosedale': ['rosedale'],
        
        # Downtown areas
        'Downtown Toronto': ['downtown toronto', 'downtown', 'dt toronto', 'toronto downtown'],
        'Queen West': ['queen west', 'queen street west', 'west queen west'],
        'King Street West': ['king street west', 'king west'],
        'Church-Wellesley': ['church-wellesley', 'church wellesley', 'church street'],
        'Yonge - Dundas': ['yonge - dundas', 'yonge dundas', 'yonge-dundas', 'dundas square'],
        'The Village': ['the village', 'village'],
        
        # West end
        'Liberty Village': ['liberty village', 'liberty'],
        'King Street West': ['king street west', 'king west'],
        'CityPlace': ['cityplace', 'city place'],
        'Parkdale': ['parkdale'],
        'Little Italy': ['little italy', 'college street'],
        'Little Portugal': ['little portugal', 'portugal'],
        'Kensington Market': ['kensington market', 'kensington'],
        'Chinatown': ['chinatown', 'china town'],
        'Koreatown': ['koreatown', 'korea town'],
        'Greektown': ['greektown', 'greek town', 'danforth'],
        
        # East end
        'Leslieville': ['leslieville', 'leslie'],
        'Riverdale': ['riverdale'],
        'Beaches': ['beaches', 'the beaches', 'beach'],
        'Cabbagetown': ['cabbagetown', 'cabbage town'],
        
        # Harbourfront
        'Harbourfront': ['harbourfront', 'harbour front', 'waterfront', 'harbourfront centre'],
        
        # Outer areas
        'Etobicoke': ['etobicoke', 'etobicoke toronto'],
        'Scarborough': ['scarborough', 'scarborough toronto'],
        'North York': ['north york', 'north toronto'],
        'York': ['york toronto', 'york'],
        'East York': ['east york'],
        
        # Mississauga (nearby)
        'Northeast Mississauga': ['northeast mississauga', 'mississauga', 'mississauga toronto'],
        'Airport': ['airport', 'pearson', 'toronto pearson'],
        
        # Newer/other areas
        'Bloor West Village': ['bloor west village', 'bloor west'],
        'High Park': ['high park'],
        'Junction': ['junction', 'the junction'],
        'Ossington': ['ossington', 'ossington avenue'],
        'Trinity Bellwoods': ['trinity bellwoods', 'trinity'],
        'Regent Park': ['regent park'],
        'Corktown': ['corktown'],
        'Canary District': ['canary district', 'canary'],
        'Port Lands': ['port lands', 'portlands']
    }
    
    # Method 1: Extract from parentheses (most reliable)
    parentheses_match = re.search(r'\(([^)]+)\)', address_str)
    if parentheses_match:
        district = parentheses_match.group(1).strip()
        district = re.sub(r',?\s*Toronto$', '', district, flags=re.IGNORECASE).strip()
        if district and district.lower() != 'toronto':
            # Clean up the extracted district
            district = clean_district_name(district)
            return district
    
    # Method 2: Check against known districts (case insensitive)
    address_lower = address_str.lower()
    for canonical_name, variations in toronto_districts.items():
        for variation in variations:
            if variation in address_lower:
                return canonical_name
    
    # Method 3: Extract from comma-separated parts (enhanced)
    parts = [part.strip() for part in address_str.split(',')]
    for i, part in enumerate(parts):
        part_lower = part.lower()
        
        # Skip if it's clearly a street address (contains numbers)
        if re.search(r'\d+', part) or len(part) <= 3:
            continue
            
        # Check if this part matches any district
        for canonical_name, variations in toronto_districts.items():
            for variation in variations:
                if variation in part_lower:
                    return canonical_name
        
        # If it doesn't contain numbers and is substantial, might be a district
        if len(part) > 8 and not re.search(r'\d+', part):
            # Check if it looks like a district name
            if any(keyword in part_lower for keyword in ['district', 'village', 'town', 'park', 'avenue', 'street']):
                return clean_district_name(part)
    
    # Method 4: Street-based inference
    street_to_district = {
        'king street': 'King Street West',
        'queen street': 'Queen West',
        'yonge street': 'Downtown Toronto',
        'bloor street': 'Bloor-Yorkville',
        'college street': 'Little Italy',
        'dundas street': 'Yonge - Dundas',
        'front street': 'Financial District',
        'bay street': 'Financial District',
        'university avenue': 'Financial District',
        'spadina avenue': 'Chinatown',
        'ossington avenue': 'Ossington',
        'danforth avenue': 'Greektown'
    }
    
    for street, district in street_to_district.items():
        if street in address_lower:
            return district
    
    return None

def clean_district_name(district):
    """Clean and standardize district names."""
    district = district.strip()
    
    # Remove common suffixes/prefixes
    district = re.sub(r',?\s*Toronto$', '', district, flags=re.IGNORECASE)
    district = re.sub(r'^Toronto\s*', '', district, flags=re.IGNORECASE)
    
    # Standardize common variations
    replacements = {
        r'\bDist\b': 'District',
        r'\bSt\b': 'Street',
        r'\bAve\b': 'Avenue',
        r'\bRd\b': 'Road',
        r'\bBlvd\b': 'Boulevard'
    }
    
    for pattern, replacement in replacements.items():
        district = re.sub(pattern, replacement, district, flags=re.IGNORECASE)
    
    # Title case
    district = district.title()
    
    return district

def fill_missing_districts(df):
    """Fill missing districts in the dataframe using enhanced extraction."""
    print("🏙️ ENHANCED DISTRICT EXTRACTION")
    print("=" * 35)
    
    if 'Address' not in df.columns:
        print("❌ No Address column found")
        return df
    
    print(f"📊 Initial district coverage:")
    initial_missing = df['district'].isnull().sum()
    initial_coverage = (len(df) - initial_missing) / len(df) * 100
    print(f"   Missing: {initial_missing:,} ({100-initial_coverage:.1f}%)")
    print(f"   Coverage: {initial_coverage:.1f}%")
    
    # Apply enhanced extraction to missing districts only
    missing_mask = df['district'].isnull()
    
    if missing_mask.sum() > 0:
        print(f"\n🔄 Processing {missing_mask.sum():,} addresses with missing districts...")
        
        # Apply enhanced extraction
        enhanced_districts = df.loc[missing_mask, 'Address'].apply(enhanced_extract_district)
        
        # Fill the missing values
        df.loc[missing_mask, 'district'] = enhanced_districts
        
        # Show improvement
        final_missing = df['district'].isnull().sum()
        final_coverage = (len(df) - final_missing) / len(df) * 100
        filled_count = initial_missing - final_missing
        
        print(f"\n✅ Enhanced extraction results:")
        print(f"   Filled: {filled_count:,} missing districts")
        print(f"   Final missing: {final_missing:,} ({100-final_coverage:.1f}%)")
        print(f"   Final coverage: {final_coverage:.1f}%")
        print(f"   Improvement: +{final_coverage-initial_coverage:.1f} percentage points")
        
        # Show new districts found
        if filled_count > 0:
            new_districts = df.loc[missing_mask & df['district'].notna(), 'district'].value_counts()
            print(f"\n📍 New districts extracted:")
            for district, count in new_districts.head(10).items():
                print(f"   {district}: {count} hotels")
    
    # Show final district distribution
    print(f"\n📊 Final district distribution:")
    district_counts = df['district'].value_counts()
    print(f"   Total unique districts: {len(district_counts)}")
    for district, count in district_counts.head(10).items():
        print(f"   {district}: {count} hotels")
    
    return df

def main():
    """Test the enhanced district extraction."""
    print("🧪 TESTING ENHANCED DISTRICT EXTRACTION")
    print("=" * 45)
    
    # Load the dataset with missing districts
    df = pd.read_csv('Data/toronto_unified_hotel_analysis_clean.csv')
    
    print(f"📂 Loaded dataset: {len(df):,} rows")
    
    # Apply enhanced district extraction
    df_enhanced = fill_missing_districts(df.copy())
    
    # Show some examples of filled districts
    filled_mask = df['district'].isnull() & df_enhanced['district'].notna()
    if filled_mask.sum() > 0:
        print(f"\n📋 Examples of filled districts:")
        examples = df_enhanced[filled_mask][['Address', 'district']].head(10)
        for idx, row in examples.iterrows():
            print(f"   Address: {row['Address'][:60]}...")
            print(f"   District: {row['district']}")
            print()
    
    # Save the enhanced dataset
    output_file = 'Data/toronto_unified_hotel_analysis_enhanced_districts.csv'
    df_enhanced.to_csv(output_file, index=False)
    print(f"✅ Enhanced dataset saved to: {output_file}")
    
    return df_enhanced

if __name__ == "__main__":
    main()