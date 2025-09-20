import asyncio
import csv
from datetime import datetime, timedelta, date
from playwright.async_api import async_playwright


async def handle_cookie_consent(page):
    """Handle cookie consent banner if present."""
    try:
        await page.locator('#onetrust-accept-btn-handler').click(timeout=10000)
        print("🍪 Accepted cookie policy.")
    except Exception:
        print("👍 No cookie banner found or it was already accepted.")


async def fill_search_form(page, destination, checkin_date, checkout_date):
    """Fill the search form with destination and dates."""
    # Fill destination
    await page.get_by_placeholder("Where are you going?").fill(destination)
    print(f"🏨 Destination set to: {destination}")
    await page.wait_for_timeout(2000)

    # Select dates
    await page.get_by_test_id("searchbox-dates-container").click()
    await page.locator(f'[data-date="{checkin_date}"]').click()
    await page.locator(f'[data-date="{checkout_date}"]').click()
    print(f"🗓️ Dates selected: {checkin_date} to {checkout_date}")


async def perform_search(page):
    """Perform the search and wait for results."""
    await page.get_by_role("button", name="Search").click()
    print("🔍 Search button clicked. Waiting for results...")
    
    # Wait a moment for potential popup after search click
    await page.wait_for_timeout(3000)
    
    # Dismiss any popup that appeared after search click
    await dismiss_popups(page)

    await page.wait_for_selector('[data-testid="property-card"]', timeout=30000)
    print("🏨 Search results loaded.")

    # Dismiss any additional popup after results load
    await dismiss_popups(page)


async def dismiss_popups(page):
    """Check for and dismiss any popups."""
    popup_selectors = [
        'button[aria-label="Dismiss"]',
        'button[aria-label="Close"]',
        'button[data-testid="modal-close"]',
        'button:has-text("Close")',
        'button:has-text("Dismiss")',
        'button:has-text("×")',
        'button:has-text("Got it")',
        'button:has-text("OK")',
        'button:has-text("Accept")',
        '[role="dialog"] button',
        '.modal button',
        'button[class*="close"]',
        'button[class*="dismiss"]',
        '[data-testid*="close"]',
        '[data-testid*="dismiss"]'
    ]
    
    popup_found = False
    for selector in popup_selectors:
        try:
            popup_button = page.locator(selector).first
            await popup_button.wait_for(state="visible", timeout=2000)
            await popup_button.click()
            print(f"👋 Dismissed popup using selector: {selector}")
            await page.wait_for_timeout(1000)
            popup_found = True
            break
        except:
            continue
    
    if not popup_found:
        print("👍 No popup to dismiss.")
    
    # Try pressing Escape as a fallback
    try:
        await page.keyboard.press("Escape")
        await page.wait_for_timeout(500)
    except:
        pass


async def find_load_more_button(page):
    """Find the load more button using various methods."""
    # Try multiple possible button texts
    possible_texts = ["Load more results", "Load more", "Show more", "More results", "See more"]
    
    for button_text in possible_texts:
        try:
            load_more_button = page.get_by_role("button", name=button_text)
            await load_more_button.wait_for(state="visible", timeout=1000)
            print(f"🔍 Found button with text: '{button_text}'")
            return load_more_button
        except:
            continue
    
    # Try alternative selectors if text-based search fails
    selectors_to_try = [
        "button:has-text('Load')",
        "button:has-text('more')", 
        "[data-testid*='load']",
        "[class*='load-more']"
    ]
    for selector in selectors_to_try:
        try:
            load_more_button = page.locator(selector).first
            await load_more_button.wait_for(state="visible", timeout=1000)
            button_text = await load_more_button.inner_text()
            print(f"🔍 Found button with selector '{selector}': '{button_text}'")
            return load_more_button
        except:
            continue
    
    return None


async def load_more_results(page, max_loads=3):
    """Load more results by clicking load more button multiple times."""
    loads_attempted = 0
    
    while loads_attempted < max_loads:
        try:
            # Scroll down to see more content
            await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
            await page.wait_for_timeout(1000)
            
            # Check for and dismiss any popups after scrolling
            await dismiss_popups(page)
            
            # Find load more button
            load_more_button = await find_load_more_button(page)
            
            if load_more_button:
                # Scroll the button into view before clicking
                await load_more_button.scroll_into_view_if_needed()
                await load_more_button.click()
                print("🔵 Clicked 'Load more' button.")
                await page.wait_for_timeout(2000)
                loads_attempted += 1
            else:
                print("❌ Could not find 'Load more' button with any method.")
                break
        except Exception:
            print("👍 No more 'Load more results' button found or timeout reached.")
            break


async def scrape_hotel_listings(page):
    """Scrape all hotel listings from the current page."""
    hotel_data = []
    listings = await page.locator('[data-testid="property-card"]').all()
    print(f"Listings scraped. Count: {len(listings)}")

    for i, listing in enumerate(listings, 1):
        print(f"📍 Processing listing {i}/{len(listings)}")
        
        # try:
        #     hotel_name = await listing.locator('[data-testid="title"]').inner_text()
        #
        # except Exception:
        #     hotel_name = "N/A"
        #     print(f"   Hotel: {hotel_name} (failed to extract)")
        #
        # try:
        #     score_text = await listing.locator('[aria-label*="Scored"]').first.inner_text()
        #     print(f"   Score: {score}")
        # except Exception:
        #     score = "N/A"
        #     print(f"   Score: {score} (failed to extract)")
        #
        # try:
        #     price = await listing.locator('[data-testid="price-and-discounted-price"]').first.inner_text()
        # except Exception:
        #     price = "N/A"
        #     print(f"   Price: {price} (failed to extract)")
        #
        # hotel_data.append({
        #     "name": hotel_name,
        #     "score": score,
        #     "price": price.replace('\n', ' ')
        # })
        try:
            hotel_name = await listing.locator('[data-testid="title"]').inner_text()
        except Exception:
            try:
                hotel_name = await listing.locator('h3, .sr-hotel__name, .hotel-name').first.inner_text()
            except Exception:
                hotel_name = "N/A"

        try:
            address = await listing.locator('[data-testid="address"]').first.inner_text()
        except Exception:
            try:
                address = await listing.locator('.address, .sr-hotel__address').first.inner_text()
            except Exception:
                address = "N/A"

        try:
            review_score = await listing.locator(
                'div[data-testid="review-score"] div.a3b8729ab1.d86cee9b25').first.inner_text()
        except Exception:
            try:
                review_score = await listing.locator('.bui-review-score__badge, .review-score').first.inner_text()
            except Exception:
                review_score = "N/A"

        try:
            distance_from_attraction = await listing.locator('span[data-testid="distance"]').first.inner_text()
        except Exception:
            try:
                distance_from_attraction = await listing.locator('.distance, .sr-hotel__distance').first.inner_text()
            except Exception:
                distance_from_attraction = "N/A"

        try:
            star_rating_element = await listing.locator('div[data-testid="rating-stars"]').first
            star_rating = await star_rating_element.get_attribute('aria-label')
        except Exception:
            try:
                star_rating_element = await listing.locator('.star-rating, .sr-hotel__stars').first
                star_rating = await star_rating_element.get_attribute(
                    'aria-label') or await star_rating_element.inner_text()
            except Exception:
                star_rating = "N/A"

        try:
            rating_element = await listing.locator('div[data-testid="review-score"] div[aria-label]').first
            rating = await rating_element.get_attribute('aria-label')
        except Exception:
            try:
                rating_element = await listing.locator('.bui-review-score, .review-rating').first
                rating = await rating_element.get_attribute('aria-label') or await rating_element.inner_text()
            except Exception:
                rating = "N/A"

        try:
            room_type = await listing.locator('[data-testid="availability-single"] h4').first.inner_text()
        except Exception:
            try:
                room_type = await listing.locator('.room-type, .sr-hotel__room-info').first.inner_text()
            except Exception:
                room_type = "N/A"

        try:
            location_score = await listing.locator(
                'div[data-testid="review-score"] a[data-testid="secondary-review-score-link"] span').first.inner_text()
        except Exception:
            try:
                location_score = await listing.locator('.location-score').first.inner_text()
            except Exception:
                location_score = "N/A"

        try:
            price = await listing.locator('[data-testid="price-and-discounted-price"]').first.inner_text()
        except Exception:
            try:
                price = await listing.locator('.bui-price-display, .sr-hotel__price, .price').first.inner_text()
            except Exception:
                price = "N/A"

        hotel_data.append({
            "Hotel name": hotel_name,
            "Address": address,
            "Review Score": review_score,
            "Distance from Attraction": distance_from_attraction,
            "Star Rating": star_rating,
            "Rating": rating,
            "Room Type": room_type,
            "Location Score": location_score,
            "price": price.replace('\n', ' ') if price != "N/A" else "N/A"
        })

        # if (i + 1) % 10 == 0:
        #     print(f"    Processed {i + 1}/{len(listings)} properties...")

        print(f"📊 Scraped data for {len(hotel_data)} hotels.")
        
        print(f"   ✅ Added hotel {i} to data")

    return hotel_data


async def scrape_booking(destination, checkin_date, checkout_date):
    """
    Scrape hotel data from Booking.com for a given destination and date range.
    
    Args:
        destination (str): The destination to search for hotels
        checkin_date (str): Check-in date in YYYY-MM-DD format
        checkout_date (str): Check-out date in YYYY-MM-DD format
    
    Returns:
        list: List of dictionaries containing hotel data
    """
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)
        context = await browser.new_context()
        page = await context.new_page()

        await page.goto("https://www.booking.com", timeout=60000)
        print("✅ Navigated to Booking.com")

        # Handle initial setup and search
        await handle_cookie_consent(page)
        await fill_search_form(page, destination, checkin_date, checkout_date)
        await perform_search(page)

        # Load more results to get additional hotels
        # await load_more_results(page)

        # Scrape all hotel data
        hotel_data = await scrape_hotel_listings(page)

        print(f"📊 Scraped data for {len(hotel_data)} hotels.")
        await browser.close()
        
        return hotel_data


def save_hotels_to_csv(hotel_data, destination, checkin_date=None, checkout_date=None):
    """
    Save hotel data to CSV file.

    Args:
        hotel_data (list): List of dictionaries containing hotel data
        destination (str): The destination name for the filename
        checkin_date (str): Check-in date used for the search
        checkout_date (str): Check-out date used for the search
    """
    if hotel_data:
        # Get today's date
        today_date = date.today().strftime("%Y-%m-%d")

        # Add today's date, source, and search dates to each hotel record
        for hotel in hotel_data:
            hotel['Date'] = today_date
            hotel['Source'] = 'booking.com'
            if checkin_date:
                hotel['Check-in Date'] = checkin_date
            if checkout_date:
                hotel['Check-out Date'] = checkout_date

        # Get the fieldnames from the keys of the first hotel dictionary
        # and ensure metadata columns are first
        priority_fields = ['Date', 'Source', 'Check-in Date', 'Check-out Date']
        other_fields = [key for key in hotel_data[0] if key not in priority_fields]
        fieldnames = [field for field in priority_fields if field in hotel_data[0]] + other_fields

        with open(f'{destination}_hotels.csv', 'w', newline='', encoding='utf-8') as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(hotel_data)
        print(f"✅ Data successfully saved to {destination}_hotels.csv")
    else:
        print("❌ No hotel data to save.")


async def main():
    """
    Main function to run the Booking.com scraper.
    """
    # Get user input
    destination = 'Toronto'
    checkin_date ='2025-10-03'
    checkout_date = '2025-10-06'
    
    # Scrape hotel data
    hotel_data = await scrape_booking(destination, checkin_date, checkout_date)
    
    # Save to CSV
    save_hotels_to_csv(hotel_data, destination, checkin_date, checkout_date)


# Run the async function using asyncio.run()
# This is safe because we applied the nest_asyncio patch at the top
asyncio.run(main())
