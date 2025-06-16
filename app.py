import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# Columns to remove completely
COLUMNS_TO_REMOVE = ['SOURCE', 'FAVORITE', 'INTERESTED', 'LATITUDE', 'LONGITUDE']

st.set_page_config(page_title="Property Filter", layout="wide")
st.title("🏡 Property Filter App")

uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    df.rename(columns={'SQUARE FEET': 'SQFT'}, inplace=True)

    # Drop unwanted columns
    df = df.drop(columns=[col for col in COLUMNS_TO_REMOVE if col in df.columns], errors='ignore')

    # Clean NaNs
    df.fillna('', inplace=True)

    # Coerce types for existing columns
    if 'BEDS' in df.columns:
        df['BEDS'] = pd.to_numeric(df['BEDS'], errors='coerce').fillna(0).astype(int)
    if 'BATHS' in df.columns:
        df['BATHS'] = pd.to_numeric(df['BATHS'], errors='coerce').fillna(0.0)
    if 'PRICE' in df.columns:
        df['PRICE'] = pd.to_numeric(df['PRICE'], errors='coerce').fillna(0.0)
    if 'SQFT' in df.columns:
        df['SQFT'] = pd.to_numeric(df['SQFT'], errors='coerce').fillna(0)

    # Coerce new filter columns if exist
    if 'YEAR BUILT' in df.columns:
        df['YEAR BUILT'] = pd.to_numeric(df['YEAR BUILT'], errors='coerce').fillna(0).astype(int)
    if 'LOT SIZE' in df.columns:
        df['LOT SIZE'] = pd.to_numeric(df['LOT SIZE'], errors='coerce').fillna(0)
    if 'DAYS ON MARKET' in df.columns:
        df['DAYS ON MARKET'] = pd.to_numeric(df['DAYS ON MARKET'], errors='coerce').fillna(0).astype(int)

    # Sidebar filters
    st.sidebar.header("Filter Properties")
    min_beds = st.sidebar.number_input("Minimum Beds", min_value=0, value=0, step=1)
    min_baths = st.sidebar.number_input("Minimum Baths", min_value=0.0, value=0.0, step=0.5)
    price_min = int(df['PRICE'].min()) if 'PRICE' in df.columns else 0
    price_max = int(df['PRICE'].max()) if 'PRICE' in df.columns else 1000000
    price_range = st.sidebar.slider(
        "Price Range",
        min_value=price_min,
        max_value=price_max,
        value=(price_min, price_max),
        step=1000,
        format="$%d"
    )

    # Additional filters if columns exist
    if 'YEAR BUILT' in df.columns:
        year_min = int(df['YEAR BUILT'].min())
        year_max = int(df['YEAR BUILT'].max())
        year_range = st.sidebar.slider(
            "Year Built Range",
            min_value=year_min,
            max_value=year_max,
            value=(year_min, year_max),
            step=1
        )
    else:
        year_range = None

    if 'LOT SIZE' in df.columns:
        lot_min = float(df['LOT SIZE'].min())
        lot_max = float(df['LOT SIZE'].max())
        lot_range = st.sidebar.slider(
            "Lot Size Range",
            min_value=lot_min,
            max_value=lot_max,
            value=(lot_min, lot_max),
            step=0.01,
            format="%.2f"
        )
    else:
        lot_range = None

    if 'DAYS ON MARKET' in df.columns:
        dom_min = int(df['DAYS ON MARKET'].min())
        dom_max = int(df['DAYS ON MARKET'].max())
        dom_range = st.sidebar.slider(
            "Days on Market",
            min_value=dom_min,
            max_value=dom_max,
            value=(dom_min, dom_max),
            step=1
        )
    else:
        dom_range = None

    # Column visibility toggles
    st.sidebar.markdown("---")
    st.sidebar.subheader("Columns to Display")
    all_columns = list(df.columns)
    visible_columns = [
        col for col in all_columns
        if st.sidebar.checkbox(col, value=True, key=f"visible_{col}")
    ]

    # Apply filters
    filter_mask = (
        (df['BEDS'] >= min_beds) &
        (df['BATHS'] >= min_baths) &
        (df['PRICE'] >= price_range[0]) &
        (df['PRICE'] <= price_range[1])
    )

    if year_range:
        filter_mask &= (df['YEAR BUILT'] >= year_range[0]) & (df['YEAR BUILT'] <= year_range[1])
    if lot_range:
        filter_mask &= (df['LOT SIZE'] >= lot_range[0]) & (df['LOT SIZE'] <= lot_range[1])
    if dom_range:
        filter_mask &= (df['DAYS ON MARKET'] >= dom_range[0]) & (df['DAYS ON MARKET'] <= dom_range[1])

    filtered_df = df[filter_mask]

    st.write(f"Showing {len(filtered_df)} properties after filtering.")

    # Calculate price per sqft
    if {'PRICE', 'SQFT'}.issubset(filtered_df.columns):
        filtered_df = filtered_df.copy()  # avoid SettingWithCopyWarning
        filtered_df['PRICE_PER_SQFT'] = filtered_df.apply(
            lambda row: row['PRICE'] / row['SQFT'] if row['SQFT'] > 0 else 0,
            axis=1
        )
    else:
        st.warning("PRICE and SQFT columns required for analysis.")
        st.stop()

    # Filter valid data for regression
    valid_mask = (filtered_df['SQFT'] > 0) & (filtered_df['PRICE_PER_SQFT'] > 0)
    valid_df = filtered_df[valid_mask]

    if len(valid_df) > 1:
        # Linear regression to find expected price per sqft by SQFT
        m, b = np.polyfit(valid_df['SQFT'], valid_df['PRICE_PER_SQFT'], 1)
        expected_price_per_sqft = m * valid_df['SQFT'] + b
        residuals = valid_df['PRICE_PER_SQFT'] - expected_price_per_sqft

        # Threshold for undervalued: bottom 10% residuals
        threshold = residuals.quantile(0.1)
        undervalued_mask = residuals < threshold

        # Mark undervalued in valid_df
        valid_df = valid_df.assign(Undervalued=undervalued_mask)

        # Merge back with filtered_df to include other rows not in valid_df
        filtered_df = filtered_df.merge(
            valid_df[['Undervalued']], left_index=True, right_index=True, how='left'
        )
        filtered_df['Undervalued'] = filtered_df['Undervalued'].fillna(False)
    else:
        filtered_df['Undervalued'] = False

    # Plot with Plotly
    fig = px.scatter(
        filtered_df,
        x='SQFT',
        y='PRICE_PER_SQFT',
        color=filtered_df['Undervalued'].map({True: 'Undervalued', False: 'Normal'}),
        color_discrete_map={'Undervalued': 'red', 'Normal': 'green'},
        hover_data={col: True for col in visible_columns},
        labels={
            'SQFT': 'Square Footage',
            'PRICE_PER_SQFT': 'Price per SQFT ($)',
            'Undervalued': 'Status'
        },
        title="Price per SQFT vs Square Footage"
    )

    # Add regression line
    x_range = np.linspace(filtered_df['SQFT'].min(), filtered_df['SQFT'].max(), 100)
    y_range = m * x_range + b
    fig.add_scatter(x=x_range, y=y_range, mode='lines', name='Regression Line', line=dict(dash='dash', color='blue'))

    st.plotly_chart(fig, use_container_width=True)

    # Show the filtered table
    st.markdown("### Filtered Properties")
    st.dataframe(filtered_df[visible_columns + ['Undervalued']], use_container_width=True)

    # Show only undervalued properties
    st.markdown("### Undervalued Properties")
    undervalued_df = filtered_df[filtered_df['Undervalued']][visible_columns]
    if not undervalued_df.empty:
        st.dataframe(undervalued_df, use_container_width=True)
    else:
        st.info("No undervalued properties found with current filters.")

else:
    st.info("Please upload a CSV file to start.")

st.write("""
## 🏡 Property Filter & Analysis Tool

- Upload your property CSV to filter listings by beds, baths, price, year built, lot size, and days on market.  
- Visualize price per square foot vs. home size with an interactive plot—hover over points for details.  
- Properties priced significantly below the trendline are highlighted as **Undervalued** to help you spot potential deals.  
- Use the tables below to explore filtered and undervalued listings easily.
""")
