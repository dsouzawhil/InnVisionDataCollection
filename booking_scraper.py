import asyncio
import csv
from datetime import datetime, timedelta
from playwright.async_api import async_playwright
import nest_asyncio

# Apply the patch to allow nested event loops in notebooks
nest_asyncio.apply()


async def main():
    """
    Main function to run the Booking.com scraper.
    """
    destination = "Toronto"

    # today = datetime.now()
    # checkin_date = today + timedelta(days=7)
    # checkout_date = checkin_date + timedelta(days=3)
    # checkin_date = '2025-10-03'
    # checkout_date = '2025-10-06'

    # checkin_str = checkin_date.strftime("%Y-%m-%d")
    # checkout_str = checkout_date.strftime("%Y-%m-%d")
    checkin_str = '2025-10-03'
    checkout_str = '2025-10-06'

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)
        page = await browser.new_page()

        await page.goto("https://www.booking.com", timeout=60000)
        print("✅ Navigated to Booking.com")

        # --- Interacting with the Page ---

        # NEW: Handle cookie consent banner first
        try:
            await page.locator('#onetrust-accept-btn-handler').click(timeout=10000)
            print("🍪 Accepted cookie policy.")
        except Exception:
            print("👍 No cookie banner found or it was already accepted.")

        await page.get_by_placeholder("Where are you going?").fill(destination)
        print(f"🏨 Destination set to: {destination}")

        # UPDATED: The selector for the date picker has changed.
        await page.get_by_test_id("searchbox-dates-container").click()

        await page.locator(f'[data-date="{checkin_str}"]').click()
        await page.locator(f'[data-date="{checkout_str}"]').click()
        print(f"🗓️ Dates selected: {checkin_str} to {checkout_str}")

        await page.get_by_role("button", name="Search").click()
        print("🔍 Search button clicked. Waiting for results...")

        await page.wait_for_selector('[data-testid="property-card"]', timeout=30000)
        print("🏨 Search results loaded.")

        try:
            await page.locator('button[aria-label="Dismiss"]').click(timeout=5000)
            print("👋 Dismissed a pop-up.")
        except Exception:
            print("👍 No pop-up to dismiss.")

        # --- Scraping the Data ---

        hotel_data = []
        listings = await page.locator('[data-testid="property-card"]').all()

        for listing in listings:
            try:
                hotel_name = await listing.locator('[data-testid="title"]').inner_text()
            except Exception:
                hotel_name = "N/A"

            try:
                score_text = await listing.locator('[aria-label*="Scored"]').first.inner_text()
                score = score_text.strip()
            except Exception:
                score = "N/A"

            try:
                price = await listing.locator('[data-testid="price-and-discounted-price"]').first.inner_text()
            except Exception:
                price = "N/A"

            hotel_data.append({
                "name": hotel_name,
                "score": score,
                "price": price.replace('\n', ' ')
            })

        print(f"📊 Scraped data for {len(hotel_data)} hotels.")
        await browser.close()

        # --- Saving the Data ---

        if hotel_data:
            with open(f'{destination}_hotels.csv', 'w', newline='', encoding='utf-8') as file:
                writer = csv.DictWriter(file, fieldnames=["name", "score", "price"])
                writer.writeheader()
                writer.writerows(hotel_data)
            print(f"✅ Data successfully saved to {destination}_hotels.csv")


# Run the async function using asyncio.run()
# This is safe because we applied the nest_asyncio patch at the top
asyncio.run(main())
