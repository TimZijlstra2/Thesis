# ===============================
# IMPORTS
# ===============================

# Selenium for browser automation
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# WebDriver Manager for automatically handling ChromeDriver binaries
from webdriver_manager.chrome import ChromeDriverManager

# Time utilities
from datetime import datetime
from dateutil import parser
import time

# For data handling
import pandas as pd
import os

# Progress bar for loops
from tqdm import tqdm

# ===============================
# FUNCTIONS
# ===============================

def scrape_window(handle, scrape_start, scrape_end, match_id=None, save_dir="C:/Master/Master project/"):
    """
    Scrapes tweets posted by a specific Twitter handle within a given time window.
    Uses Nitter (Twitter alternative front-end) to extract public tweets.

    Args:
        handle (str): Twitter handle (e.g., "@AFCAjax").
        scrape_start (str): Start of scraping window (ISO 8601 format).
        scrape_end (str): End of scraping window (ISO 8601 format).
        match_id (str): Optional match identifier for file naming.
        save_dir (str): Directory where the CSV should be saved.
    """

    # Parse start/end timestamps into datetime objects
    scrape_start = datetime.fromisoformat(scrape_start.replace("Z", "+00:00"))
    scrape_end = datetime.fromisoformat(scrape_end.replace("Z", "+00:00"))

    # Format the timestamps for the Nitter URL (just the date portion)
    scrape_start_date_only = scrape_start.date().isoformat()
    scrape_end_date_only = scrape_end.date().isoformat()

    # Clean the handle and define the filename
    handle = handle.lstrip("@")
    filename = f"{handle}_tweets_{match_id}.csv" if match_id else f"{handle}_tweets.csv"
    save_path = os.path.join(save_dir, filename)

    # Skip scraping if file already exists and is not empty
    if os.path.exists(save_path) and os.path.getsize(save_path) > 0:
        print(f"Already exists, skipping: {save_path}")
        return

    # Construct Nitter search URL with the given date window
    url = f"https://nitter.net/search?f=tweets&q=from:{handle} until:{scrape_end_date_only} since:{scrape_start_date_only}"

    # Setup headless browser options (no UI needed)
    options = Options()
    options.add_argument("--disable-gpu")
    options.add_argument("--window-size=1920,1080")

    # Start Chrome driver
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    driver.get(url)
    time.sleep(3)  # wait for page to load

    # Initialize storage
    tweets_collected = []
    tweet_ids = set()

    # Set scrolling behavior
    scroll_pause = 6
    scrolls = 0
    max_scrolls = 30
    stop_scrolling = False

    # Main scroll loop
    while not stop_scrolling and scrolls < max_scrolls:
        scrolls += 1
        print(f"\nScroll #{scrolls}...")

        # Get all tweet containers
        tweets = driver.find_elements(By.CLASS_NAME, "timeline-item")
        print(f"Found {len(tweets)} tweet blocks")

        for tweet in tweets:
            try:
                # Skip boosted (retweeted) content
                boost_elements = tweet.find_elements(By.CLASS_NAME, "boosted")
                if boost_elements:
                    continue

                # Ensure it's a tweet from the correct user
                username_elem = tweet.find_element(By.CSS_SELECTOR, "a.username")
                username_text = username_elem.text.strip()
                if username_text.lower() != f"@{handle.lower()}":
                    continue

                # Skip tweets with no timestamp (invalid structure)
                if not tweet.find_elements(By.CSS_SELECTOR, "span.tweet-date > a"):
                    continue

                # Extract timestamp and convert to datetime
                timestamp_tag = tweet.find_element(By.CSS_SELECTOR, "span.tweet-date > a")
                timestamp_str = timestamp_tag.get_attribute("title").replace("·", "").strip()
                tweet_time = parser.parse(timestamp_str)

                # Stop if tweet is older than desired window
                if tweet_time < scrape_start:
                    stop_scrolling = True
                    print("Found tweet older than scrape start — stopping scrolling.")
                    break

                # Skip if it's too new
                if tweet_time > scrape_end:
                    continue

                # Extract main content and link
                content = tweet.find_element(By.CLASS_NAME, "tweet-content").text.strip()
                tweet_link = timestamp_tag.get_attribute("href")

                # Skip duplicates
                if tweet_link in tweet_ids:
                    continue
                tweet_ids.add(tweet_link)

                # Extract comment count (first stat element)
                stats = tweet.find_elements(By.CLASS_NAME, "tweet-stat")
                comments = stats[0].text.strip() if len(stats) > 0 else "0"

                # Save structured tweet data
                tweets_collected.append({
                    "timestamp": tweet_time,
                    "tweet": content,
                    "link": tweet_link,
                    "comments": int(comments) if comments.isdigit() else 0
                })

            except Exception as e:
                print("Error while parsing a tweet:", e)
                continue

        print(f"Total unique tweets collected so far: {len(tweet_ids)}")

        # Scroll page to load more tweets
        html = driver.find_element(By.TAG_NAME, "html")
        for _ in range(25):
            html.send_keys(Keys.PAGE_DOWN)
            time.sleep(0.3)
        time.sleep(scroll_pause)

        # Try clicking the "Load more" button if available
        try:
            wait = WebDriverWait(driver, 10)
            all_buttons = wait.until(EC.presence_of_all_elements_located((By.CLASS_NAME, "show-more")))
            if all_buttons:
                load_more = all_buttons[-1]
                ActionChains(driver).move_to_element(load_more).perform()
                time.sleep(1)
                load_more.click()
                print("Clicked bottom 'Load more'")
                time.sleep(scroll_pause)
            else:
                print("No 'Load more' button found — ending.")
                break
        except Exception as e:
            print("Could not click 'Load more' —", e)
            break

    # Done scraping — close browser
    driver.quit()

    # Save tweets to CSV
    os.makedirs(save_dir, exist_ok=True)
    if not tweets_collected:
        print("No tweets found — saving empty file with headers.")
        df = pd.DataFrame(columns=["timestamp", "tweet", "link", "comments"])
    else:
        df = pd.DataFrame(tweets_collected)

    df.to_csv(save_path, index=False, encoding="utf-8-sig")
    print(f"\nCollected {len(df)} tweets for {handle} between {scrape_start} and {scrape_end}")
    print(f"Saved to: {save_path}")

def scrape_replies(tweet_url, team_handle, tweet_idx=None, total_tweets=None):
    """
    Scrapes first-level replies under a given tweet URL from Nitter.
    
    Args:
        tweet_url (str): Direct URL of the tweet to scrape replies from.
        team_handle (str): Team Twitter handle (used as fallback for reply-to logic).
        tweet_idx (int): Optional index of the tweet in a batch (for progress display).
        total_tweets (int): Optional total tweet count (for progress display).

    Returns:
        list: List of dictionaries containing reply data.
    """

    # Setup headless browser
    options = Options()
    options.add_argument("--disable-gpu")
    options.add_argument("--window-size=1920,1080")
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    driver.get(tweet_url)
    time.sleep(3)

    # Initialize storage
    replies = []
    seen_replies = set()
    scroll_pause = 3
    scroll_limit = 10
    scrolls = 0

    if tweet_idx is not None and total_tweets is not None:
        print(f"Scraping tweet {tweet_idx}/{total_tweets}: {tweet_url}")

    # Scroll loop to reveal more replies
    while scrolls < scroll_limit:
        scrolls += 1
        previous_reply_count = len(seen_replies)

        # Find all blocks (original + replies)
        reply_blocks = driver.find_elements(By.CLASS_NAME, "timeline-item")
        print(f"Scroll #{scrolls}: Found {len(reply_blocks)} blocks (including original tweet)")

        # Skip first block (it's the original tweet)
        for block in reply_blocks[1:]:
            try:
                # Get the username of the reply author
                author_elem = block.find_elements(By.CSS_SELECTOR, "a.username")
                if not author_elem:
                    continue
                author = author_elem[0].text.strip()

                # Get timestamp of reply
                timestamp_tag = block.find_element(By.CSS_SELECTOR, "span.tweet-date > a")
                timestamp_str = timestamp_tag.get_attribute("title").replace("·", "").strip()
                reply_time = parser.parse(timestamp_str)

                # Unique ID for reply
                reply_url = timestamp_tag.get_attribute("href")
                reply_id = (author, reply_time)
                if reply_id in seen_replies:
                    continue
                seen_replies.add(reply_id)

                # Extract reply text content
                reply_text = block.find_element(By.CLASS_NAME, "tweet-content").text.strip()

                # Determine who the reply is directed to
                try:
                    replying_to_tag = block.find_element(By.CSS_SELECTOR, "div.replying-to > a")
                    in_reply_to_user = replying_to_tag.text.strip()
                except:
                    # If structure not found, default to team
                    in_reply_to_user = team_handle

                # Extract comment count (if available)
                stats = block.find_elements(By.CLASS_NAME, "tweet-stat")
                comments = stats[0].text.strip() if len(stats) > 0 else "0"

                # Store reply information
                replies.append({
                    "author": author,
                    "reply": reply_text,
                    "timestamp": reply_time,
                    "reply_url": reply_url,
                    "comments": int(comments) if comments.isdigit() else 0,
                    "tweet_url": tweet_url,
                    "in_reply_to_user": in_reply_to_user
                })

            except Exception as e:
                print("Error parsing a reply:", e)
                continue

        # Try clicking "Load more" to reveal more replies
        try:
            wait = WebDriverWait(driver, 5)
            load_more_button = wait.until(EC.presence_of_element_located((By.CLASS_NAME, "show-more")))
            if load_more_button:
                ActionChains(driver).move_to_element(load_more_button).perform()
                time.sleep(1)
                load_more_button.click()
                print("Clicked 'Load more' to load more replies")
                time.sleep(scroll_pause)
        except Exception:
            print("No 'Load more' button found — may have reached the end.")

        # If no new replies after scroll, break loop
        if len(seen_replies) == previous_reply_count:
            print("No new replies loaded after scrolling — stopping early.")
            break

        # Scroll to bottom to trigger lazy loading
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(scroll_pause)

    driver.quit()
    print(f"Finished scraping {len(replies)} unique replies for: {tweet_url}\n")
    return replies

def scrape_second_level_replies(first_level_replies_csv, match_id, save_dir="C:/Master/Master project/"):
    """
    Scrapes second-level replies (replies to first-level replies) from tweet reply threads.

    Args:
        first_level_replies_csv (str): Path to CSV containing first-level replies.
        match_id (str): Identifier for the match (used for output file naming).
        save_dir (str): Directory where the result CSV will be saved.
    """

    # Load first-level replies CSV
    first_level_df = pd.read_csv(first_level_replies_csv, encoding="utf-8-sig")

    # Only keep replies that have received at least one comment (i.e., have replies themselves)
    first_level_df = first_level_df[first_level_df["comments"] > 0]

    if first_level_df.empty:
        print(f"No first-level replies with comments for match {match_id} — skipping second-level replies.")
        return

    # Storage for second-level replies
    second_level_replies = []

    # Prepare browser options
    options = Options()
    options.add_argument("--disable-gpu")
    options.add_argument("--window-size=1920,1080")

    # Loop through every reply that may have replies of its own
    for idx, row in tqdm(first_level_df.iterrows(), total=len(first_level_df), desc="🔍 Scraping second-level replies"):
        parent_reply_url = row["reply_url"]
        parent_author = row["author"]
        parent_text = row["reply"]
        parent_timestamp = pd.to_datetime(row["timestamp"])
        team_handle = row["team_handle"]
        tweet_url = row["tweet_url"]

        try:
            # Start browser and open reply URL
            driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
            driver.get(parent_reply_url)
            time.sleep(3)

            scroll_pause = 2
            scroll_limit = 10
            scrolls = 0
            seen_authors_timestamps = set()

            # Scrolling loop to load nested replies
            while scrolls < scroll_limit:
                scrolls += 1
                previous_count = len(seen_authors_timestamps)

                # Get all reply blocks (first block is the parent)
                reply_blocks = driver.find_elements(By.CLASS_NAME, "timeline-item")

                for reply in reply_blocks[1:]:
                    try:
                        # Extract author info
                        reply_author_tag = reply.find_element(By.CSS_SELECTOR, "a.username")
                        reply_author = reply_author_tag.text.strip()

                        # Extract content
                        reply_text = reply.find_element(By.CLASS_NAME, "tweet-content").text.strip()

                        # Extract timestamp
                        reply_timestamp_tag = reply.find_element(By.CSS_SELECTOR, "span.tweet-date > a")
                        reply_timestamp_str = reply_timestamp_tag.get_attribute("title").replace("·", "").strip()
                        reply_timestamp = parser.parse(reply_timestamp_str)

                        # Skip if this is just the parent reply shown again
                        if reply_author == parent_author and pd.Timestamp(reply_timestamp) == parent_timestamp:
                            continue

                        # Check for duplicate
                        reply_id = (reply_author, reply_timestamp)
                        if reply_id in seen_authors_timestamps:
                            continue
                        seen_authors_timestamps.add(reply_id)

                        # Extract comment count
                        stats = reply.find_elements(By.CLASS_NAME, "tweet-stat")
                        comments = stats[0].text.strip() if len(stats) > 0 else "0"

                        # Store the nested reply
                        second_level_replies.append({
                            "author": reply_author,
                            "reply": reply_text,
                            "timestamp": reply_timestamp,
                            "comments": int(comments) if comments.isdigit() else 0,
                            "tweet_url": tweet_url,
                            "parent_reply_url": parent_reply_url,
                            "in_reply_to_author": parent_author,
                            "parent_reply_text": parent_text,
                            "match_id": match_id,
                            "team_handle": team_handle
                        })

                    except Exception:
                        continue

                # Try clicking "Load more" to reveal more second-level replies
                try:
                    wait = WebDriverWait(driver, 5)
                    load_more_button = wait.until(EC.presence_of_element_located((By.CLASS_NAME, "show-more")))
                    if load_more_button:
                        ActionChains(driver).move_to_element(load_more_button).perform()
                        time.sleep(1)
                        load_more_button.click()
                        print("Clicked 'Load more' for second-level replies")
                        time.sleep(scroll_pause)
                except Exception:
                    print("No 'Load more' button found — maybe all second-level replies loaded.")

                # Stop if no new replies appeared after scroll
                if len(seen_authors_timestamps) == previous_count:
                    print("No new second-level replies loaded — stopping early.")
                    break

                # Scroll down to trigger loading more replies
                driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                time.sleep(scroll_pause)

            # Done scraping this thread
            driver.quit()

        except Exception as e:
            print(f"Failed to scrape second-level replies for {parent_reply_url}: {e}")
            continue

    # Save second-level replies to CSV
    second_level_df = pd.DataFrame(second_level_replies)
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"second_level_replies_{match_id}.csv")
    second_level_df.to_csv(save_path, index=False, encoding="utf-8-sig")

    print(f"\nSaved {len(second_level_df)} second-level replies to: {save_path}")

# ===============================
# MAIN SCRIPT
# ===============================

# Define base directory for storing all scraped data
base_dir = "C:/Master/Master project/"

# Define subfolders for organizing the scraped data
folders = ["timelines", "replies", "second_level_replies"]

# Create folders if they don’t exist
for folder in folders:
    folder_path = os.path.join(base_dir, folder)
    os.makedirs(folder_path, exist_ok=True)

# Load match metadata that includes scraping time windows and team handles
matches_df = pd.read_csv(os.path.join(base_dir, "toxic_match_metadata_cleaned.csv"))

# Loop over each match in the dataset
for idx, row in matches_df.iterrows():
    match_id = row["match_id"]
    scrape_start = row["scrape_start_utc"]
    scrape_end = row["scrape_end_utc"]

    # For each match, scrape tweets from both the home and away team's handles
    for handle in [row["home_handle"], row["away_handle"]]:
        print(f"\nScraping tweets for {handle} during match {match_id}...")

        # 1. Scrape timeline tweets during the match window
        scrape_window(
            handle=handle,
            scrape_start=scrape_start,
            scrape_end=scrape_end,
            match_id=f"{match_id}_{handle.lstrip('@')}",
            save_dir=os.path.join(base_dir, "timelines")
        )

        # 2. Load the saved timeline tweets to find which tweets received replies
        timeline_csv_path = os.path.join(
            base_dir, "timelines", f"{handle.lstrip('@')}_tweets_{match_id}_{handle.lstrip('@')}.csv"
        )

        # Skip if the timeline file doesn't exist
        if not os.path.exists(timeline_csv_path):
            print(f"Timeline file not found: {timeline_csv_path} — skipping replies.")
            continue

        timeline_df = pd.read_csv(timeline_csv_path, encoding="utf-8-sig")

        # Skip if there are no tweets with comments
        if timeline_df.empty or (timeline_df["comments"] <= 0).all():
            print(f"No tweets with comments for {handle} during {match_id} — skipping replies.")
            continue

        # 3. Scrape first-level replies for each tweet that received comments
        all_replies = []

        for tweet_idx, tweet_row in timeline_df[timeline_df["comments"] > 0].iterrows():
            tweet_url = tweet_row["link"].split("#")[0]  # Clean the URL (remove anchors)
            team_handle = "@" + tweet_url.split("/")[3]  # Reconstruct handle from URL

            try:
                # Scrape replies under this tweet
                replies = scrape_replies(tweet_url, team_handle=team_handle)

                # Add match metadata to each reply
                for reply in replies:
                    reply["match_id"] = match_id
                    reply["team_handle"] = team_handle

                all_replies.extend(replies)

            except Exception as e:
                print(f"Failed to scrape replies for {tweet_url}: {e}")
                continue

        # 4. Save the collected first-level replies to file
        first_level_save_path = os.path.join(
            base_dir, "replies", f"replies_{handle.lstrip('@')}_{match_id}_{handle.lstrip('@')}.csv"
        )

        if all_replies:
            replies_df = pd.DataFrame(all_replies)
            replies_df.to_csv(first_level_save_path, index=False, encoding="utf-8-sig")
            print(f"Saved {len(replies_df)} first-level replies for {match_id} - {handle}")
        else:
            print(f"No first-level replies found for {match_id} - {handle} — skipping second-level scraping.")
            continue

        # 5. Scrape second-level replies under each first-level reply (if they received comments)
        scrape_second_level_replies(
            first_level_replies_csv=first_level_save_path,
            match_id=f"{match_id}_{handle.lstrip('@')}",
            save_dir=os.path.join(base_dir, "second_level_replies")
        )
