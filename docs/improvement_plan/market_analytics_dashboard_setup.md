Global Equity Analytics Dashboard

Hero Section:

- Title: "Global Equity Analytics Dashboard"
- Description: "Comprehensive financial analysis platform for 650 global equities, featuring valuation metrics, earnings
  quality, cash flow analysis, and analyst sentiment to support investment decision-making across multiple sectors and
  regions."
- Logo: "https://dash.plotly.com/assets/images/plotly_logo_dark.png"
- Tags: Data Updated (dynamically populated with timestamp or "2026-02-01"), Created by (Plotly Studio), Data Source (
  Global Equity Analytics Dashboard data)
- Refresh Button: Icon button with refresh icon that triggers data refresh when clicked

Components:

- `filter_component`: 100% width, conditionally rendered if component loads successfully
- `data_cards`: 100% width, conditionally rendered if component loads successfully
- `data_table`: 100% width, conditionally rendered if component loads successfully
- `refresh_trigger`: Hidden data store component that tracks refresh state

Interactions:

- Refresh button click triggers data refresh operation
- Upon successful refresh, the "Data Updated" tag updates with the new timestamp
- Refresh button displays loading state with spinning refresh icon during data refresh operation

Theme Details

Colors:

- Accent: Teal (#0A7EA4)
- Accent Positive: Green (#00A878)
- Accent Negative: Red (#E63946)
- Background Page: Light Blue Gray (#F5F8FA)
- Background Content: Dark (#FFFFFF)
- Body Text: Dark Gray (#2C3E50)
- Border: Light Gray (#D1DBE3)
- Text: Dark Gray (#2C3E50)
- Heading Text: Dark Blue (#1A2332)

Chart Colors - Colorway:

- Teal (#0A7EA4)
- Green (#00A878)
- Purple (#6C63FF)
- Red (#FF6B6B)
- Cyan (#4ECDC4)
- Yellow (#FFD93D)
- Light Green (#95E1D3)
- Pink (#F38181)
- Lavender (#AA96DA)

Chart Colors - Colorscale:

- Light Cyan (#E8F4F8)
- Very Light Cyan (#C8E6F0)
- Light Blue (#A8D8E8)
- Soft Blue (#7FC4DC)
- Medium Blue (#5BB0D0)
- Blue (#379CC4)
- Darker Blue (#1388B8)
- Teal (#0A7EA4)
- Dark Teal (#086A8C)
- Deep Teal (#065674)

Chart Colors - Grid:

- Graph Grid Color: Light Blue Gray (#E8EEF2)

Typography:

- Font Family: Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif
- Font Family Header: Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif
- Font Family Headings: Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif
- Font Size: 15px
- Font Size Smaller Screen: 14px
- Font Size Header: 22px
- Section Title Font Size: 24px

Buttons:

- Background Color: Teal (#0A7EA4)
- Text Color: White (#FFFFFF)
- Border: 0px 4px 2px 0px with Teal (#0A7EA4)
- Border Radius: 4px
- Text Capitalization: none

Cards:

- Background Color: Dark (#FFFFFF)
- Margin: 20px
- Padding: 8px
- Border: 0px solid, 4px radius
- Box Shadow: 4px 4px 0px rgba(10,126,164,0.08)
- Outline: 1px solid Light Blue Gray (#E8EEF2)
- Accent: Teal (#0A7EA4)

Card Header:

- Background Color: Dark (#FFFFFF)
- Margin: 0px 0px 16px 0px
- Padding: 12px
- Border: 0px solid, 0px radius
- Box Shadow: 0px 0px 0px rgba(0,0,0,0)
- Accent: Teal (#0A7EA4)

Card Title:

- Text Color: Dark Blue (#1A2332)
- Font Size: 20px

Card Description:

- Background Color: Dark (#FFFFFF)
- Text Color: Dark Gray (#5A6C7D)
- Font Size: 16px

Card Menu:

- Background Color: Dark (#FFFFFF)
- Text Color: Dark Gray (#2C3E50)

Controls:

- Background Color: Dark (#FFFFFF)
- Text Color: Dark Gray (#2C3E50)
- Border: 1px solid Light Gray (#D1DBE3)
- Border Radius: 4px

Border Style:

- Border Width: 0px 0px 0px 0px
- Border Style: solid
- Border Radius: 0px

Hero:

- Background Color: Dark Blue (#1A2332)
- Title Text: White (#FFFFFF)
- Title Font Size: 48px
- Subtitle Text: Light Gray (#B8C5D0)
- Subtitle Font Size: 16px
- Controls Background Color: rgba(255,255,255,0.9)
- Controls Label Text: Dark Gray (#2C3E50)
- Controls Label Font Size: 14px
- Controls Grid Columns: 4
- Controls Accent: Teal (#0A7EA4)
- Border: 0px solid transparent
- Padding: 24px
- Gap: 24px

Header:

- Background Color: Dark (#FFFFFF)
- Text Color: Dark Blue (#1A2332)
- Content Alignment: spread
- Margin: 0px 0px 32px 0px
- Padding: 0px 16px 0px 0px
- Border: 1px solid Light Blue Gray (#E8EEF2)
- Box Shadow: 4px 4px 0px rgba(0,0,0,0)
- Controls Background Color: Dark (#FFFFFF)

Footer:

- Background Color: Dark Blue (#1A2332)
- Title Text: White (#FFFFFF)
- Title Font Size: 22px

Tags:

- Background Color: Light Cyan (#E8F4F8)
- Text Color: Teal (#0A7EA4)
- Font Size: 14px
- Border: 0px solid, 4px radius

Tooltip:

- Background Color: Dark Blue (#1A2332)
- Text Color: White (#FFFFFF)
- Font Size: 14px

Tables:

- Striped Even: Light Blue Gray (#F8FAFB)
- Striped Odd: Dark (#FFFFFF)
- Border: Light Blue Gray (#E8EEF2)

Top Control Panel:

- Border: 1px solid Light Blue Gray (#E8EEF2)

Layout:

- Section Padding: 24px
- Section Gap: 24px
- Breakpoint Font: 700px
- Breakpoint Stack Blocks: 700px

DBC Colors:

- Primary: Teal (#0A7EA4)
- Secondary: Dark Gray (#5A6C7D)
- Info: Cyan (#4ECDC4)
- Gray: Gray (#95A5A6)
- Success: Green (#00A878)
- Warning: Yellow (#FFD93D)
- Danger: Red (#E63946)

Report:

- Background: Dark (#FFFFFF)
- Background Content: Dark (#FFFFFF)
- Background Page: dark
- Text: black
- Font Family: Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif
- Font Size: 12px
- Border: white

Color Scheme:

- light_with_dark_hero

Filter Component

Sector:

- Multi-select dropdown
- Filters on: `sector`
- Options: All, Information Technology, Industrials, Consumer Discretionary, Health Care, Materials, Communication
  Services, Consumer Staples, Energy, Utilities
- Default: All
- "All" option shows unfiltered data
- When "All" is selected, all sector filters are bypassed

Region:

- Multi-select dropdown
- Filters on: `region`
- Options: All, United States and Canada, Asia / Pacific, Europe, Africa / Middle East, Latin America and Caribbean
- Default: All
- "All" option shows unfiltered data
- When "All" is selected, all region filters are bypassed

Market Cap Range:

- Multi-select dropdown
- Filters on: `market_cap`
- Options: All, Micro Cap <$2B, Small Cap $2B-$10B, Mid Cap $10B-$50B, Large Cap $50B-$200B, Mega Cap >$200B
- Default: All
- "All" option shows unfiltered data
- Micro Cap filters for values less than $2B
- Small Cap filters for values between $2B and $10B
- Mid Cap filters for values between $10B and $50B
- Large Cap filters for values between $50B and $200B
- Mega Cap filters for values greater than $200B
- When "All" is selected, all market cap filters are bypassed

Analyst Sentiment:

- Multi-select dropdown
- Filters on: `analyst_bullish_pct`
- Options: All, Bullish >70%, Neutral 30-70%, Bearish <30%
- Default: All
- "All" option shows unfiltered data
- Bullish filters for analyst bullish percentage greater than 70%
- Neutral filters for analyst bullish percentage between 30% and 70%
- Bearish filters for analyst bullish percentage less than 30%
- When "All" is selected, all analyst sentiment filters are bypassed

Results Display:

- Shows filtered row count and total row count in format "X / Y rows"
- Updates when any filter changes or when data refresh is triggered
- Displays count of rows matching all active filters compared to total dataset rows

Data Cards Component

Card 1: "Total Companies"

- Value: Count of rows in filtered dataset
- Format: Integer with thousands separator (1,234)

Card 2: "Avg Market Cap ($M)"

- Value: Average of `market_cap`
- Format: Integer with thousands separator (1,234)
- Handles null values by displaying "N/A"

Card 3: "Avg P/E Ratio"

- Value: Average of `p_e_ratio`
- Format: Decimal with 1 decimal place (12.5)
- Handles null values by displaying "N/A"

Card 4: "Avg ROE (%)"

- Value: Average of `roe` multiplied by 100
- Format: Percentage with 1 decimal place (12.5%)
- Handles null values by displaying "N/A"

Card 5: "Avg FCF Margin (%)"

- Value: Average of `fcf_margin` multiplied by 100
- Format: Percentage with 1 decimal place (12.5%)
- Handles null values by displaying "N/A"

Data Filters:

- All cards filtered by global filter inputs
- Cards update automatically when filters change or refresh trigger is activated

Layout:

- 5 cards arranged in single row
- Equal width cards (20% width each)
- Cards display placeholder values ("...") until data loads

Error Handling:

- All cards display "Error" if calculation fails
- Exceptions are logged with full traceback
- Empty dataset returns "No Data" for all cards

Data Table Component

The component displays company fundamentals data in an interactive table with filtering, sorting, and pagination
capabilities.

Layout:

Card container with title "Company Fundamentals Data Table"

Ag-Grid table with the following configuration:

- Pagination: Yes (50 rows per page)
- Row selection: Multiple rows selectable
- Row height: 28 pixels
- Header height: 28 pixels
- Default column behavior: Sortable, filterable, resizable with floating filters
- Cell styling: Monospace font
- Table height: 600 pixels
- Maximum rows displayed: 10,000 rows

Card footer with description: "Full data table view with filtering, sorting, and pagination capabilities. Limited to a
maximum of 10000 rows."

Columns:

The table dynamically displays columns based on data availability, prioritizing the following columns in order:

`isin`:

- Header: "Isin"
- Format: Text
- Sortable: Yes
- Filterable: Yes
- Pinned: Left

`ticker`:

- Header: "Ticker"
- Format: Text
- Sortable: Yes
- Filterable: Yes
- Pinned: Left

`name`:

- Header: "Name"
- Format: Text
- Sortable: Yes
- Filterable: Yes
- Pinned: Left

`industry`:

- Header: "Industry"
- Format: Text
- Sortable: Yes
- Filterable: Yes (text filter)

`sector`:

- Header: "Sector"
- Format: Text
- Sortable: Yes
- Filterable: Yes (text filter)

`country`:

- Header: "Country"
- Format: Text
- Sortable: Yes
- Filterable: Yes (text filter)

`region`:

- Header: "Region"
- Format: Text
- Sortable: Yes
- Filterable: Yes (text filter)

`market_cap`:

- Header: "Market Cap"
- Format: Numeric (2 decimal places)
- Sortable: Yes
- Filterable: Yes (numeric filter)

`enterprise_value`:

- Header: "Enterprise Value"
- Format: Numeric (2 decimal places)
- Sortable: Yes
- Filterable: Yes (numeric filter)

`last_price`:

- Header: "Last Price"
- Format: Numeric (2 decimal places)
- Sortable: Yes
- Filterable: Yes (numeric filter)

`price_target`:

- Header: "Price Target"
- Format: Numeric (2 decimal places)
- Sortable: Yes
- Filterable: Yes (numeric filter)

`p_e_ratio`:

- Header: "P E Ratio"
- Format: Numeric (2 decimal places)
- Sortable: Yes
- Filterable: Yes (numeric filter)

`p_b_ratio`:

- Header: "P B Ratio"
- Format: Numeric (2 decimal places)
- Sortable: Yes
- Filterable: Yes (numeric filter)

`ev_ebitda_ratio`:

- Header: "Ev Ebitda Ratio"
- Format: Numeric (2 decimal places)
- Sortable: Yes
- Filterable: Yes (numeric filter)

`ev_sales_ratio`:

- Header: "Ev Sales Ratio"
- Format: Numeric (2 decimal places)
- Sortable: Yes
- Filterable: Yes (numeric filter)

`revenue_ltm`:

- Header: "Revenue Ltm"
- Format: Numeric (2 decimal places)
- Sortable: Yes
- Filterable: Yes (numeric filter)

`ebitda_ltm`:

- Header: "Ebitda Ltm"
- Format: Numeric (2 decimal places)
- Sortable: Yes
- Filterable: Yes (numeric filter)

`net_margin_pct`:

- Header: "Net Margin Pct"
- Format: Numeric (2 decimal places)
- Sortable: Yes
- Filterable: Yes (numeric filter)

`roe`:

- Header: "Roe"
- Format: Numeric (2 decimal places)
- Sortable: Yes
- Filterable: Yes (numeric filter)

`roa`:

- Header: "Roa"
- Format: Numeric (2 decimal places)
- Sortable: Yes
- Filterable: Yes (numeric filter)

`revenue_growth_yoy`:

- Header: "Revenue Growth Yoy"
- Format: Numeric (2 decimal places)
- Sortable: Yes
- Filterable: Yes (numeric filter)

`ebitda_growth_yoy`:

- Header: "Ebitda Growth Yoy"
- Format: Numeric (2 decimal places)
- Sortable: Yes
- Filterable: Yes (numeric filter)

`eps_growth_yoy`:

- Header: "Eps Growth Yoy"
- Format: Numeric (2 decimal places)
- Sortable: Yes
- Filterable: Yes (numeric filter)

`fcf_ltm`:

- Header: "Fcf Ltm"
- Format: Numeric (2 decimal places)
- Sortable: Yes
- Filterable: Yes (numeric filter)

`fcf_margin`:

- Header: "Fcf Margin"
- Format: Numeric (2 decimal places)
- Sortable: Yes
- Filterable: Yes (numeric filter)

`cfo_ltm`:

- Header: "Cfo Ltm"
- Format: Numeric (2 decimal places)
- Sortable: Yes
- Filterable: Yes (numeric filter)

`debt_to_equity`:

- Header: "Debt To Equity"
- Format: Numeric (2 decimal places)
- Sortable: Yes
- Filterable: Yes (numeric filter)

`current_ratio`:

- Header: "Current Ratio"
- Format: Numeric (2 decimal places)
- Sortable: Yes
- Filterable: Yes (numeric filter)

`cash_ratio`:

- Header: "Cash Ratio"
- Format: Numeric (2 decimal places)
- Sortable: Yes
- Filterable: Yes (numeric filter)

`price_momentum_1m`:

- Header: "Price Momentum 1m"
- Format: Numeric (2 decimal places)
- Sortable: Yes
- Filterable: Yes (numeric filter)

`price_momentum_3m`:

- Header: "Price Momentum 3m"
- Format: Numeric (2 decimal places)
- Sortable: Yes
- Filterable: Yes (numeric filter)

`price_momentum_1y`:

- Header: "Price Momentum 1y"
- Format: Numeric (2 decimal places)
- Sortable: Yes
- Filterable: Yes (numeric filter)

`analyst_bullish_pct`:

- Header: "Analyst Bullish Pct"
- Format: Numeric (2 decimal places)
- Sortable: Yes
- Filterable: Yes (numeric filter)

`upside_potential`:

- Header: "Upside Potential"
- Format: Numeric (2 decimal places)
- Sortable: Yes
- Filterable: Yes (numeric filter)

`earnings_quality_score`:

- Header: "Earnings Quality Score"
- Format: Numeric (2 decimal places)
- Sortable: Yes
- Filterable: Yes (numeric filter)

`cash_flow_quality_score`:

- Header: "Cash Flow Quality Score"
- Format: Numeric (2 decimal places)
- Sortable: Yes
- Filterable: Yes (numeric filter)

`accounting_quality_score`:

- Header: "Accounting Quality Score"
- Format: Numeric (2 decimal places)
- Sortable: Yes
- Filterable: Yes (numeric filter)

`next_earnings`:

- Header: "Next Earnings"
- Format: Date as "%Y-%m-%d" (e.g., "2024-01-15")
- Sortable: Yes
- Filterable: Yes

`next_earnings_status`:

- Header: "Next Earnings Status"
- Format: Text
- Sortable: Yes
- Filterable: Yes (text filter)

If priority columns are not available in the data, the first 20 columns from the dataset are displayed instead.

Data:

- Load data from source
- Apply global filters from filter component inputs
- Limit results to maximum of 10,000 rows
- Replace empty strings with null values in numeric columns
- Convert null/NaN values to None for proper display

Callbacks:

Update trigger: Responds to refresh trigger input and filter component inputs

- Updates column definitions and row data based on current filters
- Includes error handling that returns empty columns and rows on failure

Price Momentum Over Time

Chart:

- Type: Line
- X: Date (`last_updated`)
- Y: Price momentum value aggregated by selected method (`price_momentum_1m`, `price_momentum_3m`, `price_momentum_6m`,
  or `price_momentum_1y` depending on selection)

Load data:

- Load from data source

Handle data types:

- String types for `name`, `sector`
- Datetime type for `last_updated`
- Numeric types for momentum columns (`price_momentum_1m`, `price_momentum_3m`, `price_momentum_6m`,
  `price_momentum_1y`)

Filter data:

- Filter by `sector` based on sector filter selection (supports "All Sectors" or specific sector values)
- Apply global filter inputs
- Remove rows where momentum value is null

Aggregate data:

- Group by `last_updated` and aggregate momentum column using selected method (mean, median, or max)
- Sort by `last_updated` in ascending order

Dropdown with label "Momentum Period:":

- Options: 1 Month, 3 Months, 6 Months, 1 Year
- Default: 3 Months

Dropdown with label "Aggregation Method:":

- Options: Average, Median, Max
- Default: Average

Multi-select dropdown with label "Sector Filter:":

- Options: All Sectors, plus all unique values from `sector` column sorted alphabetically
- Default: All Sectors

Labels:

- Card title: "Price Momentum Over Time"
- Card description: "Price momentum trends aggregated by company across different time periods"
- X-axis label: "Date"
- Y-axis label: "[Aggregation Method] Price Momentum (%)" (e.g., "Average Price Momentum (%)", "Median Price
  Momentum (%)", "Max Price Momentum (%)")

Styles:

- Layout: Flexbox row layout for controls with 10px row gap and 15px margins between control groups
- Controls: Minimum width of 200px for period and aggregation dropdowns, 250px for sector filter
- Chart: Minimum height of 550px, responsive height up to viewport minus 600px
- Loading: Circular loading indicator
- Error display: Red text color for error messages

Sector Valuation Comparison

Chart:

- Type: Bar
- X: Company name (`name`)
- Y: Selected valuation metric (`p_e_ratio`, `ev_ebitda_ratio`, or `p_b_ratio`)
- Size: None
- Color: None

Load data:

- Load data from source

Filter data:

- Apply global filter inputs
- Filter by `sector` matching selected sector
- Remove rows where metric value is null
- Select top 15 companies by metric value in descending order

Handle data types:

- Convert metric columns (`p_e_ratio`, `ev_ebitda_ratio`, `p_b_ratio`) to numeric, coercing invalid values to null

Compute additional columns:

- When Y-axis range is set to "Shared", calculate minimum and maximum metric values across both selected sectors' top 15
  companies, then apply 5% padding (multiply min by 0.95 and max by 1.05)

Dropdown with label "Sector 1 (Left)":

- Options: Information Technology, Industrials, Consumer Discretionary, Health Care, Materials, Communication Services,
  Utilities, Consumer Staples, Energy
- Default: Information Technology

Dropdown with label "Sector 2 (Right)":

- Options: Information Technology, Industrials, Consumer Discretionary, Health Care, Materials, Communication Services,
  Utilities, Consumer Staples, Energy
- Default: Industrials

Dropdown with label "Metric":

- Options: P/E Ratio, EV/EBITDA, Price-to-Book
- Default: P/E Ratio

Dropdown with label "Y-Axis Range":

- Options: Shared, Individual
- Default: Shared

Labels:

- Card title: "Sector Valuation Comparison"
- Card description: "Compare valuation multiples of two sectors side by side. Select sectors and metric to analyze top
  15 companies by valuation."
- X-axis label: "Company"
- Y-axis label: Varies by selected metric ("P/E Ratio", "EV/EBITDA", or "Price-to-Book")
- Chart title: "[Sector name] Valuation Multiples"

Styles:

- Layout: Two-column side-by-side layout with equal flex distribution, minimum height of 550px per chart with responsive
  height calculation (calc(100vh - 600px)), 5px margin between columns
- Controls: Horizontal flex layout with 10px row gap, controls aligned to center
- Error display: Red text color for error messages
- Loading: Circular loading spinner for each chart

Stock Lookup Table

Load data:

- Load stock data using data loading function
- Filter data using global filter inputs

Filter data:

- Filter by `sector` column matching selected sector value
- Apply global filters from filter component

Aggregate data:

- Sort by selected columns (`ticker`, `name`, `market_cap`, `p_e_ratio`, `dividend_yield_ltm`, `last_price`) in
  descending order
- Limit results to specified row count (100, 500, 1000, or all rows)

Select columns:

- Display `ticker`, `name`, `sector`, `industry`, `country`, `exchange`, `last_price`, `market_cap`, `p_e_ratio`,
  `dividend_yield_ltm`

Format columns:

- `last_price` and `market_cap` formatted as currency with 2 decimal places and thousands separators
- `p_e_ratio` formatted with 2 decimal places and thousands separators, or "N/A" if missing
- `dividend_yield_ltm` formatted as percentage with 2 decimal places, or "N/A" if missing
- All other columns displayed as strings

Dropdown with label "Sector:":

- Options: Information Technology, Industrials, Consumer Discretionary, Health Care, Materials, Communication Services,
  Utilities, Consumer Staples, Energy
- Default: Information Technology

Multi-select dropdown with label "Sort By:":

- Options: Ticker, Name, Market Cap, P/E Ratio, Dividend Yield, Last Price
- Default: Market Cap

Dropdown with label "Row Limit:":

- Options: 100, 500, 1000, All
- Default: 100

Labels:

- Card title: "Stock Lookup Table"
- Card description: "Look up stocks by sector and industry with key metrics"
- Column headers: "Ticker", "Name", "Sector", "Industry", "Country", "Exchange", "Last Price", "Market Cap", "P/E
  Ratio", "Div Yield"

Styles:

- Table layout: Full width with collapsed borders
- Cell styling: 8px padding, 12px monospace font, 1px solid #ddd bottom border
- Text alignment: Left-aligned for text columns, right-aligned for numeric columns
- Loading indicator: Circular spinner
- Error display: Red text with 10px top and bottom margin
- Container: Horizontal flex layout with 10px row gap, 15px margins between controls
- Scrolling: Horizontal scroll for table overflow, vertical scroll with max height of viewport minus 400px

Valuation Pivot Table

Chart:

- Type: Pivot table with conditional formatting
- Row dimension: `sector`, `industry`, or `trading_country`
- Column dimension: `region`, `sector`, or `trading_country`
- Value metric: `p_e_ratio`, `ev_ebitda_ratio`, or `ev_sales_ratio`

Load data:

- Load data from source

Filter data:

- Apply global filter inputs to dataset

Compute additional columns:

- Calculate row mean values for sorting purposes (temporary, dropped after sorting)

Aggregate data:

- Create pivot table with row dimension, column dimension, and value metric
- Aggregate using selected method: median, mean, min, max, 75th percentile, 95th percentile, or count

Sort data:

- Sort by rows ascending (alphabetical)
- Sort by rows descending (reverse alphabetical)
- Sort by values ascending (by row mean)
- Sort by values descending (by row mean)

Conditional formatting:

- Apply color scale from red (#d73027) to green (#1a9850) with intermediate colors
- Color scaling scope: entire table, by row, or by column
- Calculate contrasting text color (black or dark gray) based on background luminance
- Display missing values as "-" with transparent background

Dropdown with label "Row Dimension":

- Options: Sector, Industry, Trading Country
- Default: Sector

Dropdown with label "Column Dimension":

- Options: Region, Sector, Trading Country
- Default: Region

Dropdown with label "Value Metric":

- Options: P/E Ratio, EV/EBITDA Ratio, EV/Sales Ratio
- Default: P/E Ratio

Dropdown with label "Aggregation":

- Options: Median, Mean, Min, Max, P75, P95, Count
- Default: Median

Dropdown with label "Color By":

- Options: Row, Column, Table
- Default: Table

Dropdown with label "Sort By":

- Options: Rows Ascending, Rows Descending, Values Ascending, Values Descending
- Default: Rows Ascending

Labels:

- Card title: "Valuation Pivot Table"
- Card description: "P/E ratio and EV/EBITDA by sector and region with conditional formatting"
- Row header: Row dimension name (e.g., "Sector")
- Column headers: Column dimension values
- Cell values: Formatted to 2 decimal places, or "-" for missing values
- Legend: "Low" to "High" gradient scale

Styles:

- Colors: Red-to-green diverging color scale (#d73027 to #1a9850) with 9 color stops
- Layout: Flexbox row layout for controls with 10px row gap and 15px right margin between control groups
- Table: Monospace font (12px), 8px padding, 1px solid #eee borders between rows, 2px solid #ddd header border
- Legend: Centered flex layout with 200px gradient bar, 10px font size
- Text: Contrasting text color calculated from background luminance
- Controls: Minimum width of 200px for dropdowns

Quality Score Analysis

Chart:

- Type: Violin plot with box plot overlay
- X: `sector`
- Y: Selected quality score metric (`earnings_quality_score`, `accounting_quality_score`, or `cash_flow_quality_score`)
- Color: `sector`

Load data:

- Load data using data loading function
- Apply global filters from filter component

Handle data types:

- String types for `isin`, `name`, `sector`
- Numeric type for quality score columns

Compute additional columns:

- Convert score metric column to numeric, coercing invalid values to null

Filter data:

- Apply global filters from filter component on all relevant columns
- Filter by score range using minimum and maximum score inputs (0 to 100)
- Remove rows where score metric value is null

Dropdown with label "Quality Metric:":

- Options: Earnings Quality Score, Accounting Quality Score, Cash Flow Quality Score
- Default: Earnings Quality Score

Number input with label "Score Range:" (minimum):

- Range: 0 to 100
- Default: 0

Number input with label "Score Range:" (maximum):

- Range: 0 to 100
- Default: 100

Labels:

- Card title: "Quality Score Analysis"
- Card description: "Distribution of quality scores across companies with filtering capabilities"
- X-axis label: "Sector"
- Y-axis label: Dynamically generated from selected metric (e.g., "Earnings Quality Score")

Styles:

- Y-axis range: Fixed from 0 to 100
- Legend: Hidden
- Points display: Show all individual data points if dataset contains 5000 rows or fewer; hide points for larger
  datasets
- Layout: Minimum height of 550px, responsive height up to viewport minus 600px