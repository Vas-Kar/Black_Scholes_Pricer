
### --------------- Import Libraries --------------- ###

import streamlit as st
import numpy as np
import pandas as pd
from scipy.stats import norm

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm #for custom colormap
import seaborn as sns

### --------------- Main Page Configuration --------------- ###

st.set_page_config(page_title="Black-Scholes Pricing App", layout="wide") #make the page wider

st.title(":blue[Black-Scholes Option Pricing]") #Main page title
st.markdown("""---""")

### --------------- Project Description --------------- ###

with st.expander("ℹ️ Project Description", expanded=False):
    st.markdown("""
    **Black-Scholes Option Pricing App**

    This interactive Streamlit app calculates and visualizes European call and put option prices using the Black-Scholes model.
    Users can input custom parameters—such as stock price, strike price, time to maturity, volatility, and risk-free rate—via a sidebar interface
    The app provides instant pricing outputs and graphical insights, including sensitivity heatmaps that illustrate how option prices respond to changes in key inputs                

    📌 **Features**:
    - Input fields for stock price, strike price, time to maturity, volatility, and risk-free rate.
    - Real-time computation of call and put prices.
    - Heatmaps showing option prices and their across varying parameters.
    - Heatmaps showing color-coded P&Ls based on user purchase price inputs and the calculated option prices.
    - Option buttons to download the heatmap data in csv files
""")

st.markdown("""---""")



### --------------- Author - Linkedin --------------- ###

with st.sidebar:
    st.write(":blue[Created by:]")  # Author heading
    st.markdown(
        """
        <div style='display: flex; align-items: center; gap: 10px;'>
            <img src='https://cdn.jsdelivr.net/gh/devicons/devicon/icons/linkedin/linkedin-original.svg' width='20'/>
            <a href='https://www.linkedin.com/in/vasilis-karantzas' target='_blank'>Vasilis Karantzas</a>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.markdown("---")  # Optional horizontal separator

### --------------- Black-Scholes Option Prices --------------- ###



with st.sidebar:
    st.title("Option Pricing Inputs")
    st.subheader("Black Scholes Inputs")

### --------------- Reset to Default Inputs Button --------------- ###

    if st.button("Reset to Default Black-Scholes Inputs"):
        st.session_state.stock_price = 100.0
        st.session_state.strike_price = 100.0
        st.session_state.time_to_maturity = 1.25
        st.session_state.volatility_input = 0.2
        st.session_state.risk_free_rate = 0.05

### --------------- Stock Price Input --------------- ###

    with st.container(border=True):

        st.caption("Stock Price (e.g. 100)")
        stock_price = st.number_input(
            ":blue[Enter a Stock Price (S)]", 
            format="%0.2f", 
            step=0.1,
            min_value=0.0,
            value=100.0,
            key="stock_price",
            help="Stock Price Input for the Black-Scholes Formula")

### --------------- Strike Price Input --------------- ###

        st.caption("Strike Price (e.g. 100)")
        strike_price = st.number_input(
            ":blue[Enter an Option Strike Price (Κ)]", 
            format="%0.2f", 
            step=0.1,
            min_value=0.0,
            value=100.0, 
            key="strike_price",
            help='Option Strike Price Input for the Black-Scholes Formula')
        
        if strike_price == 0.0:
            st.warning("⚠️ Strike Price cannot be zero. Please enter a positive value.")

### --------------- Time to Maturity Input --------------- ###

        st.caption("Time to Maturity in Years (e.g. 1.25)")
        time_to_maturity = st.number_input(
            ":blue[Enter the Option Time to Maturity (󠁔T) (in years)]", 
            format="%0.2f",
            step=0.25,
            min_value=0.0,
            value=1.25, 
            key="time_to_maturity",
            help='Time to Maturity Input (in years) for the Black-Scholes Formula')
        
        if time_to_maturity == 0.0:
            st.warning("⚠️ Time to maturity cannot be zero. Please enter a positive value.")
    
 ### --------------- Volatility Input --------------- ###

        st.caption("Volatility (e.g. 0.205 = 20.5%)")
        volatility_input = st.number_input(
            ":blue[Enter the Volatility of Stock (σ) (in decimals)]", 
            format="%0.2f",
            step=0.01,
            min_value=0.0,
            max_value=1.0,
            value=0.2, 
            key="volatility_input",
            help='Stock Volatility Input (in decimals) for the Black-Scholes Formula')
        
        if volatility_input == 0.0 :
            st.warning("⚠️ Volatility cannot be 0 in Black-Scholes. Please enter a positive value")

### --------------- Risk-free Rate Input --------------- ###

        st.caption("Risk-free Rate (e.g. 0.05 = 5%)")
        risk_free_rate = st.number_input(
            ":blue[Enter the Risk-free Rate (r) (in decimals)]", 
            format="%0.2f",
            step=0.01,
            min_value=0.0,
            max_value=1.0,
            value=0.05, 
            key="risk_free_rate",
            help='Risk-free Rate Input (in decimals) for the Black-Scholes Formula')
        
        if risk_free_rate == 0.0 :
            st.warning("⚠️ Risk-free rate cannot be 0 in Black-Scholes. Please enter a positive value")

if 0 in (strike_price, time_to_maturity, volatility_input, risk_free_rate):
    st.warning("⚠️ You have entered a 0.0 value for the Strike Price, Time to Maturity, Volatility or Risk-free Rate. Please enter a positive value")
    st.stop()

st.sidebar.markdown("""---""")

### --------------- Define Black Scholes Function --------------- ###

def black_scholes(S, K, T, sigma, r):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2)*T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    call_price = round((S * norm.cdf(d1) - K * np.exp(-r*T) * norm.cdf(d2)), 2)
    put_price = round((K * np.exp(-r*T) * norm.cdf(-d2) - S * norm.cdf(-d1)), 2)
    return round(call_price,2), round(put_price,2)

### --------------- Return Option Prices --------------- ###

call_price, put_price = black_scholes(stock_price, strike_price, time_to_maturity, volatility_input, risk_free_rate)

### --------------- Create Table to Show the Inputs --------------- ###

st.subheader("Black-Scholes Option Pricing Inputs")

inputs = [stock_price, strike_price, time_to_maturity, volatility_input, risk_free_rate]
column_names = ["Stock Price (S)","Strike Price (K)", "Time to Maturity (T)", "Volatility (σ)", "Risk-free Rate (r)"]
df = pd.DataFrame(data=[inputs], columns=column_names)

#Center table cell values and column names using HTML
table_html = df.to_html(index=False, justify='center')
centered_table = f"""
<style>
    table {{
        margin-left: auto;
        margin-right: auto;
        text-align: center;
    }}
    th {{
        text-align: center !important;
    }}
    td {{
        text-align: center !important;
    }}
</style>
{table_html}
"""
st.markdown(centered_table, unsafe_allow_html=True)

st.markdown("""---""")

### --------------- Calculate Option Price Changes using st.session_state() --------------- ###

#initiation new dictionary key in state for the previous call and put values
if "prev_call_price" not in st.session_state:
    st.session_state.prev_call_price = call_price

if "prev_put_price" not in st.session_state:
    st.session_state.prev_put_price = put_price

#calculate deltas based on each session_state values
call_delta = round(call_price - st.session_state.prev_call_price, 2)
put_delta = round(put_price - st.session_state.prev_put_price, 2)

call_delta_pct = (call_delta / st.session_state.prev_call_price) * 100 if st.session_state.prev_call_price != 0 else 0
put_delta_pct = (put_delta / st.session_state.prev_put_price) * 100 if st.session_state.prev_put_price != 0 else 0

### --------------- Show Option Prices and Dynamic Deltas --------------- ###

st.subheader("Black-Scholes Option Prices")

col1, col2 = st.columns(2)

with col1:
    # st.markdown("### 📈 :blue[Call Option Price]")
    st.metric(label=":blue[Call Option Price:]", 
            value=f"€{call_price:.2f}",
            delta=f"{call_delta:.2f} ({call_delta_pct:.2f}%)",
            border=True)
    
with col2:
    # st.markdown("### 📉 :blue[Put Option Price]")
    st.metric(label=":blue[Put Option Price]", 
            value=f"€{put_price:.2f}",
            delta=f"{put_delta:.2f} ({put_delta_pct:.2f}%)",
            border=True)

#Update session_state variable values    
st.session_state.prev_call_price = call_price
st.session_state.prev_put_price = put_price

st.markdown("""---""")

### --------------- Sensitivities --------------- ###

st.subheader("Generated Stock and Volatility Ranges")

### --------------- Sensitivity Widgets --------------- ###

with st.sidebar:
    st.title("Sensitivity Heatmap Inputs")

### --------------- Minimum Stock Price Input --------------- ###

    st.subheader("Stock Price Inputs")

    if st.button("Reset to Default Stock Price Inputs"):
        st.session_state.min_stock_price = 90.0
        st.session_state.max_stock_price = 110.0
        st.session_state.stock_range_method = "Number of Stock Prices"

    with st.container(border=True):

        st.caption("Enter a minimum stock price (e.g. 90)")

        min_stock_price = st.number_input(
            label=":blue[Select a Minimum Stock Price]", 
            format="%0.2f", 
            step=0.1,
            min_value=0.0,
            value=90.0,
            key="min_stock_price",
            help="Minimum Stock Price for the Sensitivity Heatmap")

### --------------- Switch for Stock Price Range Generation Method --------------- ###

        st.caption("Choose how to generate the stock price range: by number of prices or step intervals.")

        #mapping for methods for cleaner code later 
        options_stock ={"Number of Stock Prices" : "count", #user sees Number of Stock Prices but we get count as value
                "Equally Spaced Intervals" : "intervals" #same here with above
        }    
        stock_range_method = st.selectbox(
            label=":green[Select Method of Generating Stock Price Range for Sensitivity Heatmap]",
            options=list(options_stock.keys()),
            key="stock_range_method",
            help="Number of Stock Prices will create equally spaced stock prices based on min and max values  \n "
            "Equally Spaced Intervals will create 10 stock prices with equal space between them, starting from the minimum stock price"
    )

 ### --------------- Create Slider or Cell for Inputs --------------- ###   

    #Create Widgets based on Method chosen and calculate Stock Prices 
        stock_price_range = []
        if options_stock[stock_range_method] == "count": #If the method chosen is Number of Stock Prices
            
            #Maximum Stock Price Input 
            st.caption("Enter a maximum stock price (e.g. 110)")
            
            max_stock_price = st.number_input(
            label=":blue[Select a Maximum Stock Price]", 
            format="%0.2f", 
            step=0.1,
            min_value=0.0,
            value=110.0,
            key="max_stock_price",
            help="Maximum Stock Price for the Sensitivity Heatmap")
            
            #Slider to pick the # of Stock Prices
            number_of_prices = st.slider(
                label=":blue[Choose the Number of Stock Values for the Sensitivity Heatmap]",
                min_value=1,
                max_value=10,
                value=9,
                step=1,
                key="stock_range_slider",
                help="Number of Stock Prices that will be Displayed in the Heatmap")
            
            if max_stock_price == None: #Check that Maximum Stock Price is filled before continuing
                st.stop()       
            else:
                stock_price_range = np.round(np.linspace(min_stock_price, max_stock_price, number_of_prices), 2) #if it is filled calculate the price ranges

        else: #If the method chosen is interval steps
            
            #Input Cell for Interval Step 
            st.caption("Enter an interval step to generate stock prices (e.g. 0.5)")

            stock_interval_step = st.number_input(
                label=":blue[Enter the Step for the Interval of Price Range]",
                format="%.02f",
                step=0.1,
                min_value=0.0,
                value=0.5,
                placeholder="Interval Step for Stock Price Range",
                help="The step will create 10 stock prices starting from the minimum stock price input"
            )

            if stock_interval_step == 0:
                st.warning("⚠️ Please enter a positive interval step")
                st.stop()      

            starting_value = min_stock_price #set starting value of the price range
            last_value = min_stock_price + (11 * stock_interval_step) #ending value of the price range --- We want 10 prices but arange is not inclusive [...). So we calculate 11 steps for the end number
            stock_price_range = np.round(np.arange(start=min_stock_price, stop = (min_stock_price + (11*stock_interval_step)), step=stock_interval_step), 2)

#print the generated stock price range
st.write(f"Generated Stock Price Range:  \n{' | '.join(map(lambda x: f'{x:.2f}', stock_price_range))}") #Map applies the function for 2 decimals in the price range and then we separate them using | 

### --------------- Minimum Volatility Input --------------- ###

with st.sidebar:

    st.subheader("Volatility Inputs")

    if st.button("Reset to Default Volatility Inputs"):
        st.session_state.min_volatility = 0.1
        st.session_state.volatility_range_method = "Number of Volatility Values"
        st.session_state.max_volatility = 0.3

    with st.container(border=True):

        st.caption("Enter a minimum volatility (e.g. 0.1 = 10%)")

        min_volatility = st.number_input(
            label=":blue[Select a Minimum Volatility (in decimals)]", 
            format="%0.2f", 
            step=0.05,
            min_value=0.0,
            max_value=1.0,
            value=0.1, 
            key="min_volatility",
            help="Minimum Volatility for the Sensitivity Heatmap")
        
        if min_volatility == 0.0:
            st.warning("⚠️ Volatility cannot be 0 in Black-Scholes. Please enter a positive value")
            st.stop()


### --------------- Switch for Volatility Range Generation Method --------------- ###

        st.caption("Choose how to generate the volatility range: by number of values or step intervals.")

        options_volatility ={"Number of Volatility Values" : "count", #user sees Number of Volatility Values but we get count as value
                "Equally Spaced Intervals" : "intervals" #same here with above
        }    
        volatility_range_method = st.selectbox(
            label=":green[Select Method of Generating Volatility Range for Sensitivity Heatmap]",
            options=list(options_volatility.keys()),
            key="volatility_range_method",
            help="Number of Volatility Values will create equally spaced volatilities based on min and max values  \n "
            "Equally Spaced Intervals will create 10 volatility values with equal space between them, starting from the minimum volatility"
        )

 ### --------------- Create Slider or Cell for Inputs --------------- ###   

        volatility_range = []
        if options_volatility[volatility_range_method] == "count": #If the method chosen is Number of Volatility Values
            
            #Maximum Volatility Input 

            st.caption("Enter a maximum volatility (e.g. 0.3 = 30%)")

            max_volatility = st.number_input(
            label=":blue[Select a Maximum Volatility (in decimals)]", 
            format="%0.2f", 
            step=0.05,
            min_value=0.0,
            max_value=1.0,
            value=0.3, 
            key="max_volatility",
            help="Maximum Volatility for the Sensitivity Heatmap")
        
            if max_volatility == 0.0:
                st.warning("⚠️ Volatility cannot be 0 in Black-Scholes. Please enter a positive value")
                st.stop()
            
            #Slider to pick the # of Volatility Values
            number_of_volatilities = st.slider(
                label=":blue[Choose the Number of Volatility Values for the Sensitivity Heatmap]",
                min_value=1,
                max_value=10,
                value=9,
                step=1,
                key="volatility_range_slider",
                help="Number of Volatility Values that will be Displayed in the Heatmap")
            
            volatility_range = np.round(np.linspace(min_volatility, max_volatility, number_of_volatilities), 3) #if it is filled calculate the volatility range

        else: #If the method chosen is interval steps
            
            #Input Cell for Interval Step 
            
            st.caption("Enter an interval step to generate volatility values (e.g. 0.0025)")
            
            volatility_interval_step = st.number_input(
                label=":blue[Enter the Step for the Interval of Volatility Range]",
                format="%.03f",
                step=0.0025,
                min_value=0.0,
                max_value=1.0,
                value=0.05,
                placeholder="Interval Step for Volatility Range",
                help="The step will create 10 volatility values starting from the minimum stock price input"
            )
        
            if volatility_interval_step == 0:
                st.warning("⚠️ Please enter a positive interval step")
                st.stop()   

            starting_value = min_volatility #set starting value of the price range
            last_value = min_volatility + (11 * volatility_interval_step) #ending value of the volatility range --- We want 10 values but arange is not inclusive [...). So we calculate 11 steps for the end number
            volatility_range = np.round(np.arange(start=min_volatility, stop = (min_volatility + (11*volatility_interval_step)), step=volatility_interval_step), 3)
    
    st.markdown("""---""")

#print generated volatility range
st.write(f"Generated Volatility Value Range:  \n{' | '.join(map(lambda x: f'{x:.3f}', volatility_range))}") #Map applies the function for 2 decimals in the volatility range and then we separate them using | 

st.markdown("""---""")

### --------------- Initiate Call / Put Option DataFrames  --------------- ###   

call_prices = pd.DataFrame(index=volatility_range, columns=stock_price_range)
put_prices = pd.DataFrame(index=volatility_range, columns=stock_price_range)

# Calculate the option prices using Black-Scholes function
for price in stock_price_range:
    for vol in volatility_range:
        call, put = black_scholes(price, strike_price, time_to_maturity, vol, risk_free_rate)
        call_prices.loc[vol, price] = call
        put_prices.loc[vol, price] = put

#transform into flaots to use with sns.heatmap()
call_prices = call_prices.astype("float")
put_prices = put_prices.astype("float")




### --------------- Plot Heatmaps  --------------- ###   

st.title(":blue[Option Prices & P&Ls Heatmaps]")


tab1, tab2 = st.tabs(["💰 Option Prices Heatmap", "📈 P&L Heatmap"])

with tab1:

    st.subheader("💰 Option Prices Heatmap")
    st.caption("These heatmaps show how call and put prices change with stock price and volatility.")

    show_prices_heatmap = st.checkbox(":blue[📊 Show Option Prices Heatmap]", value=True)

    col1, col2 = st.columns(2)
    with col1:
        st.download_button("📥 Download Call Option Prices as CSV", data=call_prices.to_csv(), file_name="call_option_prices.csv") 
    with col2:
        st.download_button("📥 Download Put Option Prices as CSV", data=put_prices.to_csv(), file_name="put_option_prices.csv")




    
    if show_prices_heatmap:

        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15,6))

        sns.heatmap(call_prices, annot=True, fmt=".2f", cbar=False, cmap="coolwarm", ax=axes[0], annot_kws={"size": 11, "weight": "bold"}, linewidths=0.5, linecolor='lightgrey')
        axes[0].set_title("Call Prices", fontsize=14, weight="bold")
        axes[0].set_ylabel("Volatility (σ)", fontsize=14, weight="bold")
        axes[0].set_xlabel("Stock Price (S)", fontsize=14, weight="bold")
        axes[0].tick_params(axis="x", labelsize=10, rotation=45)
        axes[0].tick_params(axis="y", labelsize=10, rotation=0)
        axes[0].set_xticklabels(labels=stock_price_range, fontweight="bold")
        axes[0].set_yticklabels(labels=volatility_range, fontweight="bold")

        sns.heatmap(put_prices, annot=True, fmt=".2f", cbar=False, cmap="coolwarm", ax=axes[1], annot_kws={"size": 11, "weight": "bold"}, linewidths=0.5, linecolor='lightgrey')
        axes[1].set_title("Put Prices", fontsize=14, weight="bold")
        axes[1].set_xlabel("Stock Price (S)", fontsize=14, weight="bold")
        axes[1].tick_params(axis="x", labelsize=10, rotation=45)
        axes[1].tick_params(axis="y", labelsize=10, rotation=0)
        axes[1].set_xticklabels(labels=stock_price_range, fontweight="bold")
        axes[1].set_yticklabels(labels=volatility_range, fontweight="bold")

        plt.tight_layout()
        st.pyplot(fig)

st.markdown("""---""")

### --------------- Inputs for Option Purchase Prices  --------------- ###

with st.sidebar:
    st.title("P&L Heatmap Inputs")

    if st.button("Reset to Default P&L Heatmap Inputs"):
        st.session_state.call_purchase_price = 11.5
        st.session_state.put_purchase_price = 5.5

    with st.container(border=True):

        col1, col2, = st.columns(2)

        with col1:
            call_purchase_price = st.number_input(
                label=":blue[Enter Call Option Purchase Price (C)]",
                min_value=0.0,
                step=0.5,
                value=11.5,
                format="%.2f",
                key="call_purchase_price",
                help="The purchase price of the Call option on which the P&L will be based on"
            )

            if call_purchase_price == 0.0:
                st.warning("⚠️ Please enter a positive call purchase price to continue")  

        with col2:
            put_purchase_price = st.number_input(
                label=":blue[Enter Put Option Purchase Price (P)]",
                min_value=0.0,
                step=0.5,
                value=5.5,
                format="%.2f",
                key="put_purchase_price",
                help="The purchase price of the Put option on which the P&L will be based on"
        )
            
            if put_purchase_price == 0:
                st.warning("⚠️ Please enter a positive put purchase price to continue") 

if 0 in (call_purchase_price, put_purchase_price):
    st.warning("⚠️ You have entered a 0.0 purchase price for the call or the put option. Please enter a positive value")
    st.stop()

### --------------- Option P&Ls based on Inputs  --------------- ###


call_profits = pd.DataFrame()
put_profits = pd.DataFrame()

call_profits = call_prices - call_purchase_price
put_profits = put_prices - put_purchase_price



### --------------- Create Custom Colormap to give negative values red and positive green and 0s white --------------- ###

colors = ["red", "white", "green"]
custom_colormap = LinearSegmentedColormap.from_list("profit_cmap", colors)

vmin = min(call_profits.min().min(), put_profits.min().min())
vmax = max(call_profits.max().max(), put_profits.max().max())
norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)

### --------------- Plot P&L Heatmaps  --------------- ###

with tab2:

### --------------- P&L Heatmaps  --------------- ###

    st.subheader("📈 P&L Heatmap")
    st.caption("These heatmaps show how the P&L changes based on the calculated option prices and the purchase prices inputs")

    show_profits_heatmap = st.checkbox(":blue[📊 Show Option P&L Heatmap]", value=True)

    col1, col2 = st.columns(2)
    with col1:
        st.download_button("📥 Download Call Option P&L as CSV", data=call_profits.to_csv(), file_name="call_option_P&L.csv") 
    with col2:
        st.download_button("📥 Download Put Option P&L as CSV", data=put_profits.to_csv(), file_name="put_option_P&L.csv")


    if show_profits_heatmap:
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15,6))

        sns.heatmap(call_profits, annot=True, fmt=".2f", cbar=False, cmap=custom_colormap, ax=axes[0], annot_kws={"size": 11, "weight": "bold"}, linewidths=0.5, linecolor='lightgrey')
        axes[0].set_title("Call Option P&L", fontsize=14, weight="bold")
        axes[0].set_ylabel("Volatility (σ)", fontsize=14, weight="bold")
        axes[0].set_xlabel("Stock Price (S)", fontsize=14, weight="bold")
        axes[0].tick_params(axis="x", labelsize=10, rotation=45)
        axes[0].tick_params(axis="y", labelsize=10, rotation=0)
        axes[0].set_xticklabels(labels=stock_price_range, fontweight="bold")
        axes[0].set_yticklabels(labels=volatility_range, fontweight="bold")

        sns.heatmap(put_profits, annot=True, fmt=".2f", cbar=False, cmap=custom_colormap, ax=axes[1], annot_kws={"size": 11, "weight": "bold"}, linewidths=0.5, linecolor='lightgrey')
        axes[1].set_title("Put Option P&L", fontsize=14, weight="bold")
        axes[1].set_xlabel("Stock Price (S)", fontsize=14, weight="bold")
        axes[1].tick_params(axis="x", labelsize=10, rotation=45)
        axes[1].tick_params(axis="y", labelsize=10, rotation=0)
        axes[1].set_xticklabels(labels=stock_price_range, fontweight="bold")
        axes[1].set_yticklabels(labels=volatility_range, fontweight="bold")

        plt.tight_layout()
        st.pyplot(fig)
