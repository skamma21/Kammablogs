================================================================================
  AlphaLens — AI Equity Research Terminal
  Setup & Usage Guide
================================================================================

TABLE OF CONTENTS
─────────────────
1. What This Is
2. Requirements
3. Setup Instructions
4. How to Use
5. Rate Limits & Usage Costs
6. Embedding on WordPress
7. Troubleshooting


================================================================================
1. WHAT THIS IS
================================================================================

AlphaLens is a single-file AI-powered stock analysis tool. You type in any 
stock ticker and it automatically:

  - Fetches real financial data (income statements, balance sheets, cash flows,
    historical prices, peer companies, and recent news)
  - Runs a Discounted Cash Flow (DCF) model with AI-predicted assumptions
  - Performs comparable company analysis
  - Calculates Graham Number, DDM, and Residual Income valuations
  - Generates a sentiment score with bull/bear cases
  - Displays an interactive price chart with valuation target lines
  - Lets you override any assumption and instantly recalculate

Everything runs in a single HTML file — no server, no build step, no backend.


================================================================================
2. REQUIREMENTS
================================================================================

  - A modern web browser (Chrome, Firefox, Edge, Safari)
  - An Anthropic API key (free tier works)
  - An internet connection

That's it. No other software, accounts, or API keys needed.


================================================================================
3. SETUP INSTRUCTIONS
================================================================================

STEP 1: Get your Anthropic API Key
───────────────────────────────────
  1. Go to https://console.anthropic.com
  2. Create an account or sign in
  3. Navigate to Settings → API Keys (or go directly to 
     https://console.anthropic.com/settings/keys)
  4. Click "Create Key"
  5. Copy the key (it starts with "sk-ant-...")

  IMPORTANT: The free tier gives you $5 of credits which is enough for 
  roughly 50-100 stock analyses. After that you'd need to add credits
  (pay-as-you-go, no subscription required).

STEP 2: Add the key to the HTML file
─────────────────────────────────────
  1. Open stock-analyzer.html in any text editor (Notepad, VS Code, etc.)
  2. Find this line near the top of the <script> section:

     const ANTHROPIC_API_KEY = 'YOUR_ANTHROPIC_KEY_HERE';

  3. Replace YOUR_ANTHROPIC_KEY_HERE with your actual key. For example:

     const ANTHROPIC_API_KEY = 'sk-ant-api03-abc123...your-full-key-here';

  4. Save the file.

STEP 3: Open and use
─────────────────────
  Double-click the HTML file to open it in your browser. Type a ticker 
  symbol (e.g. AAPL, NVDA, TSLA, MSFT) and click "Analyze".


================================================================================
4. HOW TO USE
================================================================================

BASIC USAGE
───────────
  1. Type a stock ticker in the search bar (e.g. AAPL)
  2. Click "Analyze" or press Enter
  3. Wait 20-40 seconds for the AI to fetch data and run analysis
  4. Explore the results across the tabs:

     - DCF Model: AI-predicted assumptions for a 5-year DCF. You can change
       any number in the "Override" column and click "Recalculate" to 
       instantly see updated valuations.

     - Comps Analysis: Peer companies with P/E, EV/Revenue, and margin 
       comparisons. Shows implied price from each multiple.

     - Other Models: Dividend Discount Model (if applicable), Graham Number,
       and Residual Income valuations.

     - Financials: 4-year income statement, balance sheet, and cash flow.

SIDEBAR (right panel)
─────────────────────
  - Sentiment Score: 0-100 scale (0 = max bearish, 100 = max bullish)
  - Bull Case: 3 reasons the stock could go up
  - Bear Case: 3 reasons the stock could go down
  - Recent News: Latest headlines with sentiment labels
  - Key Metrics: P/E, EV/EBITDA, beta, dividend yield, etc.

PRICE CHART
───────────
  - Shows historical prices with timeframe buttons (1M, 3M, 6M, 1Y, 5Y)
  - After analysis completes, dashed lines appear showing:
      Green = DCF target price
      Amber = Comps target price  
      Purple = Blended target price

RECALCULATING DCF
─────────────────
  The DCF tab shows the AI's estimated assumptions. To run your own scenario:
  1. Change any value in the "Override" column
  2. Click the "Recalculate" button
  3. The implied price, upside/downside, and blended target update instantly
  4. The chart target lines also update


================================================================================
5. RATE LIMITS & USAGE COSTS
================================================================================

Each stock analysis makes 2 API calls to Anthropic:

  Call 1 — Data Fetch (uses web search)
    Model: Claude Haiku 4.5
    Input tokens: ~2,000-3,000
    Output tokens: ~3,000-5,000
    Estimated cost: ~$0.005-0.01

  Call 2 — AI Analysis (DCF, sentiment, bull/bear)
    Model: Claude Haiku 4.5
    Input tokens: ~1,500-2,500
    Output tokens: ~1,500-2,500
    Estimated cost: ~$0.003-0.005

  TOTAL PER ANALYSIS: ~$0.01-0.02 (roughly 1-2 cents)

FREE TIER LIMITS (Anthropic)
────────────────────────────
  - $5 free credits on signup
  - That's approximately 250-500 stock analyses
  - Rate limit: 30,000 input tokens per minute
  - If you hit the rate limit, the app automatically waits and retries
  - Wait 60 seconds between analyses if you get rate limit errors

PAID USAGE
──────────
  If you exhaust free credits, you can add funds at:
  https://console.anthropic.com/settings/billing

  There is no monthly subscription — you only pay for what you use.
  At ~$0.02 per analysis, $10 would cover ~500 analyses.

DAILY ANALYSIS CAPACITY
───────────────────────
  - Free tier: ~10-15 analyses per day comfortably (within rate limits)
  - If you space them 1-2 minutes apart, you can do more
  - With paid credits and no rate limit concerns: essentially unlimited


================================================================================
6. EMBEDDING ON WORDPRESS
================================================================================

OPTION A: Custom HTML Block (Easiest)
─────────────────────────────────────
  1. In the WordPress editor, add a "Custom HTML" block
  2. Copy and paste the ENTIRE contents of stock-analyzer.html into it
  3. Publish or preview the page
  
  Note: Some themes may strip <script> tags. If the tool doesn't work, 
  use Option B instead.

OPTION B: Standalone Page
─────────────────────────
  1. Upload stock-analyzer.html to your WordPress media library or via FTP
     to your server (e.g. yoursite.com/tools/stock-analyzer.html)
  2. Link to it from any page or post
  3. Or embed it using an iframe:
     <iframe src="/tools/stock-analyzer.html" width="100%" height="900" 
             frameborder="0" style="border:none;"></iframe>

OPTION C: Page Template (Advanced)
──────────────────────────────────
  1. Create a custom page template in your theme
  2. Paste the HTML contents into the template
  3. Assign the template to a new page

SECURITY NOTE
─────────────
  Your Anthropic API key is visible in the page source code. For a personal
  blog this is generally fine. If you're concerned about key exposure on a 
  public site:
  
  - Set a monthly spending limit on your Anthropic account
    (https://console.anthropic.com/settings/billing)
  - Monitor usage at https://console.anthropic.com/settings/usage
  - You can rotate the key anytime if needed


================================================================================
7. TROUBLESHOOTING
================================================================================

ERROR: "Please add your Anthropic API key"
  → You haven't replaced YOUR_ANTHROPIC_KEY_HERE with your actual key.
    Open the file in a text editor and add your key.

ERROR: "Anthropic API 401: ..."
  → Your API key is invalid or expired. Double-check that you copied the 
    full key correctly, including the "sk-ant-" prefix.

ERROR: "Anthropic API 429: rate_limit_error"
  → You're sending requests too fast. Wait 60 seconds and try again. 
    The app has built-in retry logic that will auto-wait and retry once.

ERROR: "No JSON found in response"
  → The AI returned an unexpected format. Try again — this is rare and 
    usually resolves on a second attempt.

DATA LOOKS WRONG OR OUTDATED
  → The AI searches the web for real data, but web search results vary.
    If data looks off, try analyzing the same ticker again — the second 
    result is often better.

CHART SHOWS NO DATA
  → The historical prices may not have loaded. Check the browser console 
    (F12 → Console tab) for errors. Try a well-known ticker like AAPL first.

PAGE IS BLANK OR UNSTYLED
  → Make sure you're opening the file in a browser, not a text editor.
    If embedding in WordPress, ensure your theme allows <script> and 
    <style> tags in Custom HTML blocks.


================================================================================
  Questions? Issues? Check the browser console (F12) for detailed error 
  messages. Most issues are either API key problems or rate limits.
================================================================================
