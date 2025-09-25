import asyncio
import schedule
import time
from datetime import datetime
from booking_scraper import scrape_booking, save_hotels_to_csv

async def run_daily_scrape():
    """Run the booking scraper for multiple destinations with different date combinations."""
    print(f"🕘 Starting daily scrape at {datetime.now()}")
    
    # List of destinations to scrape
    destinations = [
        "Toronto",
    ]
    
    # List of check-in and check-out date combinations
    date_combinations = [
        ("2025-10-10", "2025-10-13"),  # 3 nights
        ("2025-10-15", "2025-10-18"),  # 3 nights  
        ("2025-10-20", "2025-10-23"),  # 3 nights
        ("2025-10-25", "2025-10-28"),  # 3 nights
        ("2025-11-01", "2025-11-04"),  # 3 nights
        ("2025-11-05", "2025-11-08"),  # 3 nights
        ("2025-11-10", "2025-11-13"),  # 3 nights
        ("2025-11-15", "2025-11-18"),  # 3 nights
        ("2025-11-20", "2025-11-23"),  # 3 nights
        ("2025-11-25", "2025-11-28"),  # 3 nights
    ]
    
    # Loop through all date combinations
    for combo_index, (checkin_date, checkout_date) in enumerate(date_combinations, 1):
        print(f"📅 Using date combination {combo_index}: {checkin_date} to {checkout_date}")
        
        for destination in destinations:
            try:
                print(f"\n🏨 Scraping {destination}...")
                hotel_data = await scrape_booking(destination, checkin_date, checkout_date)
                save_hotels_to_csv(hotel_data, destination, checkin_date, checkout_date)
                print(f"✅ Completed {destination}: {len(hotel_data)} hotels")
                
                # Wait 2 minutes between destinations to avoid rate limiting
                await asyncio.sleep(120)
                
            except Exception as e:
                print(f"❌ Error scraping {destination}: {e}")
        
        # Wait 3 minutes between date combinations
        if combo_index < len(date_combinations):
            print(f"⏳ Waiting 3 minutes before next date combination...")
            await asyncio.sleep(180)
    
    print(f"🎉 Daily scrape completed at {datetime.now()}")

def job():
    """Wrapper to run async function in scheduler."""
    asyncio.run(run_daily_scrape())

# Set stop date to October 9th, 2025
STOP_DATE = datetime(2025, 10, 9)  # Fixed stop date

def job_with_stop_check():
    """Wrapper to run async function with stop date check."""
    if datetime.now() > STOP_DATE:
        print(f"🛑 Stop date {STOP_DATE.strftime('%Y-%m-%d')} reached. Stopping scheduler.")
        return schedule.CancelJob
    return job()

# Run immediately first, then schedule for daily runs
print("🚀 Running scraper immediately...")
job_with_stop_check()

schedule.every().day.at("09:00").do(job_with_stop_check)  # Run at 9 AM daily

print("📅 Daily booking scraper scheduler started...")
print("⏰ Will run every day at 9:00 AM")
print(f"🛑 Will stop after: {STOP_DATE.strftime('%Y-%m-%d')}")
print("Press Ctrl+C to stop")

# Keep the scheduler running
while True:
    schedule.run_pending()
    time.sleep(60)  # Check every minute