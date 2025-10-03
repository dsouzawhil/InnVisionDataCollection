"""
Extract Districts from Hotel Names
=================================

When addresses only say "Toronto", extract district info from hotel names.
"""

import pandas as pd
import re

def extract_district_from_hotel_name(hotel_name):
    """Extract district information from hotel name."""
    if pd.isna(hotel_name) or str(hotel_name).strip() == '':
        return None
    
    name_lower = str(hotel_name).lower()
    
    # District patterns in hotel names
    name_patterns = {
        # Airport area
        'Airport': ['airport', 'pearson', 'corporate centre'],
        
        # Specific neighborhoods mentioned in names
        'Queen West': ['queen west', 'west queen west'],
        'Trinity Bellwoods': ['trinity bellwoods', 'trinity'],
        'Beaches': ['beaches', 'beach'],
        'Yorkdale': ['yorkdale'],
        'Entertainment District': ['entertainment district'],
        'Financial District': ['financial district', 'financial'],
        'Downtown Toronto': ['downtown toronto', 'downtown'],
        'Harbourfront': ['harbourfront', 'harbour'],
        'Distillery District': ['distillery'],
        'King Street West': ['king west', 'king street west'],
        'Liberty Village': ['liberty village'],
        'The Annex': ['annex'],
        'Bloor-Yorkville': ['yorkville', 'bloor'],
        'Chinatown': ['chinatown'],
        'Little Italy': ['little italy'],
        'Kensington Market': ['kensington'],
        'Parkdale': ['parkdale'],
        'Leslieville': ['leslieville'],
        'Riverdale': ['riverdale'],
        'Corktown': ['corktown'],
        'Old Town': ['old town'],
        'St. Lawrence': ['st. lawrence', 'st lawrence'],
        'Church-Wellesley': ['church wellesley', 'church'],
        'The Village': ['village'],
        'CityPlace': ['cityplace'],
        'High Park': ['high park'],
        'Junction': ['junction'],
        'Ossington': ['ossington'],
        'Danforth': ['danforth'],
        'College': ['college'],
        'Spadina': ['spadina'],
        'Queen East': ['queen east'],
        'King East': ['king east'],
        'Front Street': ['front street'],
        'Bay Street': ['bay street'],
        'Yonge Street': ['yonge street'],
        
        # Outer areas
        'Etobicoke': ['etobicoke'],
        'Scarborough': ['scarborough'],
        'North York': ['north york'],
        'York': ['york'],
        'East York': ['east york'],
        'Mississauga': ['mississauga'],
        
        # Hotels often mention these areas
        'Fashion District': ['fashion district'],
        'Canary District': ['canary district'],
        'Port Lands': ['port lands'],
        'Regent Park': ['regent park']
    }
    
    # Check each pattern
    for district, patterns in name_patterns.items():
        for pattern in patterns:
            if pattern in name_lower:
                return district
    
    # Special case: if hotel name contains "Toronto" + area word
    toronto_area_patterns = {
        'Downtown Toronto': ['toronto downtown', 'downtown toronto'],
        'Airport': ['toronto airport', 'airport toronto'],
        'Harbourfront': ['toronto harbourfront', 'toronto waterfront'],
        'North York': ['toronto north'],
        'Etobicoke': ['toronto etobicoke'],
        'Scarborough': ['toronto scarborough']
    }
    
    for district, patterns in toronto_area_patterns.items():
        for pattern in patterns:
            if pattern in name_lower:
                return district
    
    return None

def fill_districts_from_names(df):
    """Fill missing districts using hotel names."""
    print("🏨 EXTRACTING DISTRICTS FROM HOTEL NAMES")
    print("=" * 42)
    
    # Focus on rows with missing districts
    missing_mask = df['district'].isnull()
    initial_missing = missing_mask.sum()
    
    print(f"📊 Processing {initial_missing:,} hotels with missing districts...")
    
    if initial_missing == 0:
        print("✅ No missing districts to fill!")
        return df
    
    # Extract districts from hotel names for missing entries
    name_districts = df.loc[missing_mask, 'Hotel name'].apply(extract_district_from_hotel_name)
    
    # Fill the missing districts
    df.loc[missing_mask, 'district'] = name_districts
    
    # Show results
    final_missing = df['district'].isnull().sum()
    filled_count = initial_missing - final_missing
    
    print(f"\n✅ Results:")
    print(f"   Filled from names: {filled_count:,} districts")
    print(f"   Still missing: {final_missing:,} districts")
    print(f"   Coverage improvement: +{(filled_count/len(df))*100:.1f} percentage points")
    
    if filled_count > 0:
        # Show which districts were extracted
        newly_filled = df.loc[missing_mask & df['district'].notna(), 'district'].value_counts()
        print(f"\n📍 Districts extracted from hotel names:")
        for district, count in newly_filled.items():
            print(f"   {district}: {count} hotels")
    
    return df

def show_examples(df_before, df_after):
    """Show examples of successful district extractions."""
    missing_before = df_before['district'].isnull()
    filled_after = df_after['district'].notna()
    
    # Find rows that were filled
    filled_mask = missing_before & filled_after
    
    if filled_mask.sum() > 0:
        print(f"\n📋 Examples of successful extractions:")
        examples = df_after[filled_mask][['Hotel name', 'Address', 'district']].head(10)
        
        for idx, row in examples.iterrows():
            print(f"   Hotel: {row['Hotel name']}")
            print(f"   Address: {row['Address']}")
            print(f"   → Extracted: {row['district']}")
            print()

def main():
    """Main function to test district extraction from names."""
    print("🧪 TESTING DISTRICT EXTRACTION FROM HOTEL NAMES")
    print("=" * 50)
    
    # Load dataset
    df = pd.read_csv('Data/toronto_unified_hotel_analysis_clean.csv')
    
    print(f"📂 Loaded dataset: {len(df):,} rows")
    
    # Show initial state
    initial_missing = df['district'].isnull().sum()
    initial_coverage = (len(df) - initial_missing) / len(df) * 100
    print(f"📊 Initial district coverage: {initial_coverage:.1f}% ({initial_missing:,} missing)")
    
    # Create copy for processing
    df_enhanced = df.copy()
    
    # Fill districts from hotel names
    df_enhanced = fill_districts_from_names(df_enhanced)
    
    # Show examples
    show_examples(df, df_enhanced)
    
    # Final statistics
    final_missing = df_enhanced['district'].isnull().sum()
    final_coverage = (len(df_enhanced) - final_missing) / len(df_enhanced) * 100
    
    print(f"📊 Final district coverage: {final_coverage:.1f}% ({final_missing:,} missing)")
    print(f"🎯 Total improvement: +{final_coverage - initial_coverage:.1f} percentage points")
    
    # Save enhanced dataset
    output_file = 'Data/toronto_unified_hotel_analysis_districts_filled.csv'
    df_enhanced.to_csv(output_file, index=False)
    print(f"\n✅ Enhanced dataset saved to: {output_file}")
    
    return df_enhanced

if __name__ == "__main__":
    main()