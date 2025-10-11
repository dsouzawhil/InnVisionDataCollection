# Toronto Hotel Price Prediction Pipeline

A comprehensive data pipeline for scraping, processing, and analyzing hotel pricing data in Toronto, with integration of events and weather data for machine learning price prediction models.

## 🏗️ Project Overview

This project builds a complete hotel price prediction system by:
- Scraping hotel pricing data from multiple sources
- Collecting Toronto events and weather data
- Performing spatial analysis with real coordinates
- Creating ML-ready datasets with comprehensive feature engineering
- Providing exploratory data analysis tools

## 📁 Project Structure

```
├── Data/
│   ├── hotel_listing/          # Daily scraped hotel files
│   ├── toronto_events_*.csv    # Events data files
│   ├── toronto_weather_*.csv   # Weather data files
│   └── *.png                   # Generated analysis plots
├── booking_scraper.py          # Hotel data scraping
├── geteventsdata.py           # Events data collection
├── get_weather.py             # Weather data collection
├── hotel_data_pipeline.py     # Automated pipeline orchestrator
├── data_joining.py            # Spatial data integration
├── data_transformation.py     # ML preprocessing
├── eda_analysis.py            # Exploratory data analysis
└── spatial_analysis.py        # Geographic feature engineering
```

## 🚀 Quick Start

### Prerequisites

1. **Python 3.8+** with required packages:
```bash
pip install -r requirements.txt
```

2. **Required API Keys** (add to environment variables):
```bash
export BOOKING_API_KEY="your_booking_api_key"
export WEATHER_API_KEY="your_weather_api_key"
```

### 🔄 Complete Pipeline Execution

Run the entire pipeline with one command:

```bash
python hotel_data_pipeline.py
```

This automatically executes all steps in the correct order.

## 📋 Manual Step-by-Step Pipeline

For granular control or debugging, run each step manually:

### Step 1: Data Collection

#### 1.1 Scrape Hotel Data
```bash
python booking_scraper.py
```
- Scrapes hotel pricing data for Toronto
- Saves daily files to `Data/hotel_listing/`
- Run daily for continuous data collection

#### 1.2 Collect Events Data
```bash
python geteventsdata.py
```
- Fetches Toronto events from multiple sources
- Creates `Data/toronto_events_2025.csv`
- Includes event scoring and categorization

#### 1.3 Collect Weather Data
```bash
python get_weather.py
```
- Downloads Toronto weather data for 2025
- Creates `Data/toronto_weather_2025.csv`
- Includes temperature, precipitation, and weather conditions

### Step 2: Data Integration

#### 2.1 Combine Hotel Files
```bash
python hotel_data_pipeline.py
```
- Combines daily hotel files into `Data/toronto_hotels_combined.csv`
- Removes duplicates and validates data quality

#### 2.2 Spatial Data Joining
```bash
python data_joining.py
```
- Joins hotels, events, and weather data
- Performs spatial analysis with real coordinates
- Creates `Data/toronto_unified_hotel_analysis.csv`

### Step 3: ML Preprocessing

#### 3.1 Data Transformation
```bash
python data_transformation.py
```
- Converts to ML-ready format
- Adds derived features (districts, holidays, lead times)
- Handles missing values and data quality issues
- Creates `Data/toronto_hotels_transformed.csv`

### Step 4: Analysis

#### 4.1 Exploratory Data Analysis
```bash
python eda_analysis.py
```
- Comprehensive target variable analysis
- Bivariate relationship exploration
- Feature correlation analysis
- Generates visualization plots

## 📊 Generated Datasets

The pipeline creates several key datasets:

| File | Description | Records | Purpose |
|------|-------------|---------|---------|
| `toronto_hotels_combined.csv` | Combined daily hotel data | ~4K | Base hotel dataset |
| `toronto_unified_hotel_analysis.csv` | Spatially enriched data | ~4K | Full feature dataset |
| `toronto_hotels_transformed.csv` | ML-ready dataset | ~4K | Model training |

## 🎯 Key Features

### Hotel Data Features
- **Pricing**: USD/CAD prices with currency conversion
- **Location**: Districts, coordinates, attraction distances
- **Temporal**: Booking lead times, stay duration, seasonality
- **Accommodation**: Room types, guest capacity

### Events Integration
- **Spatial**: Distance-weighted event scores
- **Temporal**: Events during hotel stay periods
- **Impact**: Major event flags and density metrics

### Weather Integration
- **Daily weather**: Temperature, precipitation, conditions
- **Seasonal patterns**: Weather impact on pricing

## 🧹 Data Quality Features

- **Anomaly Detection**: Removes pricing outliers and data errors
- **Duplicate Handling**: Smart deduplication across daily files
- **Missing Value Treatment**: Intelligent imputation strategies
- **Validation**: Comprehensive data quality checks

## 📈 Analysis Outputs

The EDA module generates:
- **Price Distribution Analysis**: Target variable characteristics
- **Feature Correlation Matrix**: Relationship strength visualization
- **Bivariate Analysis**: Feature-price relationship plots
- **Business Insights**: Pricing pattern discoveries

## 🐳 Docker Support

Run the entire pipeline in Docker:

```bash
docker-compose up
```

## 🔄 Scheduling

For automated daily data collection:

```bash
python daily_scheduler.py
```

## 📝 Pipeline Outputs Summary

### ✅ Final ML Dataset Features:
- **Target**: `price_usd` (clean, anomaly-free)
- **Location**: `district`, `latitude`, `longitude`, `Distance from attraction`
- **Temporal**: `is_weekend`, `is_holiday`, `booking_lead_time`, `week_of_year`
- **Events**: `events_total_score`, `has_major_event`, `events_count`
- **Weather**: `MEAN_TEMPERATURE`, `TOTAL_PRECIPITATION`, weather conditions
- **Hotel**: `room_category`, `length_of_stay`, `Number of People`

### 📊 Key Statistics:
- **Records**: ~3,950 (after quality cleaning)
- **Features**: 44 columns
- **Coverage**: 100% for critical features
- **Price Range**: $37 - $1,979 USD
- **Date Range**: 2025 data with historical context

## 🎯 Next Steps

After running the pipeline:

1. **Model Development**: Use `toronto_hotels_transformed.csv` for ML models
2. **Feature Selection**: Based on EDA correlation analysis
3. **Model Training**: Consider log transformation for price target
4. **Validation**: Time-based splits to avoid data leakage

## 🔧 Troubleshooting

### Common Issues:

1. **Missing API Keys**: Ensure environment variables are set
2. **File Permissions**: Check write access to `Data/` directory
3. **Memory Issues**: Large datasets may require chunked processing
4. **Date Parsing**: Ensure consistent date formats across files

### Debug Mode:
Run individual scripts with verbose output to diagnose issues.

## 🤝 Contributing

1. Follow the existing code structure
2. Add comprehensive error handling
3. Update this README for new features
4. Test the complete pipeline after changes

## 📄 License

This project is for educational and research purposes.

---

**🎉 Ready to predict Toronto hotel prices with comprehensive data-driven insights!**