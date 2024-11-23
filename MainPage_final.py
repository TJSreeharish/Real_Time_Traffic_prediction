import streamlit as st
import folium
from streamlit_folium import folium_static
import requests
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
import os
import time
from PIL import Image
import pickle
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from datetime import datetime
from prophet import Prophet
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
from datetime import datetime, timedelta, time as dt_time
from geopy.distance import geodesic
import yfinance as yf
from threading import Thread
import polyline
from streamlit_option_menu import option_menu
import json
from streamlit_lottie import st_lottie 

#time1_str = "2024-10-01 15:00:00"
#time2_str = "2024-10-01 15:30:00"

button_style = """
    <style>
    div.stButton > button {
        background-color: #0000FA;
        color: white;
        border: black;
        border-radius: 12px;
        padding: 10px 24px;
        font-size: 16px;
        cursor: pointer;
        transition-duration: 0.4s;
    }

    div.stButton > button:hover {
        background-color: #0000A0;
        color: white;
    }
    </style>
    """

st.set_page_config(
    page_title="Yukthi",
    page_icon="🎉",
    layout="wide"
)

#Lottie handlers
def lottie_load(filepath: str):
    with open(filepath, "r") as f:
        return json.load(f)

# Page Handlers
def handle_login(email, password):
    if email and password:
        st.session_state['logged_in'] = True
        st.session_state['email'] = email
        st.success("Login successful!")

def login_page():
    ls, rs = st.columns((2,1))
    with ls:
        st.markdown("<h1 style='text-align: left;'>Login Page</h1>", unsafe_allow_html=True)

        email = st.text_input("Email", key="login_email")
        password = st.text_input("Password", type="password", key="login_password")

        left, right = st.columns((1,1))
        with left:
            if st.button("Log In", key="login_btn"):
                handle_login(email, password)
        with right:
            if st.button("Sign in through Google", key="google_signin_btn"):
                st.info("Google login is currently under development.")
            if st.button("Sign in through Facebook", key="facebook_signin_btn"):
                st.info("Facebook login is currently under development.")
        
        st.write("New User?")
        if st.button("Create New Account"):
            create_new_account()
    with rs:
        st_lottie(
                lottie_load("Login_animation.json"),
                speed = 0.75,
                loop=True,
            )

def create_new_account():
    st.markdown("<h1 style='text-align: left;'>Create New Account</h1>", unsafe_allow_html=True)

    name = st.text_input("Name", key="create_name")
    email = st.text_input("Email", key="create_email")
    password = st.text_input("Password", type="password", key="create_password")
    workspace = st.text_input("Workspace", key="create_workspace")
    home_address = st.text_area("Home Address", key="create_address")

    if st.button("Create Account", key="create_account_btn"):
        with open("users.txt", "a") as file:
            file.write(f"{name},{email},{password},{workspace},{home_address}\n")
        st.success("Account created successfully!")
        st.info("You can now log in using your credentials.")

def commute_form():
    st.header("Welcome!", anchor=None)
    st.subheader("Please fill in your commute details")

    source = st.text_input("Source", key="source")
    destination = st.text_input("Destination", key="destination")

    junctions = ["Silk Board", "Sarjapur Road", "Bellandur", "Kadubeesanahalli",
                 "Marathahalli", "Mahadevpura", "KR Puram"]
    junction = st.selectbox("Usual Junction of Commute", junctions, key="junction")

    mode_of_commute = st.selectbox("Mode of Commute", ["Car", "Bike", "Bus", "Walk"], key="mode_of_commute")

    if st.button("Submit", key="submit_commute"):
        data = {
            "Email": [st.session_state['email']],
            "Source": [source],
            "Destination": [destination],
            "Junction": [junction],
            "Mode of Commute": [mode_of_commute]
        }
        df = pd.DataFrame(data)
        if not os.path.isfile("commute_route.csv"):
            df.to_csv("commute_route.csv", index=False)
        else:
            df.to_csv("commute_route.csv", mode='a', header=False, index=False)

        st.success("Commute details saved successfully!")

def about():
    ls,rs = st.columns((2,1))
    with ls:
        st.markdown("<h1 style='text-align: left;'> <span style='color: Lime;'>About Us</span></h1>", unsafe_allow_html=True)

        st.markdown("<h4 style='text-align: left;'> <span style='color: white;'>We are Yukthi, a team of six passionate students from the 3rd semester of the AIML department at RV College of Engineering. With a shared interest in artificial intelligence and machine learning, we have been working together on innovative projects that push the boundaries of technology. Over the past two weeks, we have collaborated on a challenging project and developed a cutting-edge solution aimed at solving real-world problems. Our focus on creativity, teamwork, and technical expertise drives us to continuously explore new possibilities in AI and ML.</span></h4>", unsafe_allow_html=True)
    with rs:
        st.write("##")
        st_lottie(
            lottie_load("AboutUs_animation.json"),
            speed = 0.75,
            loop= True
        )
        
    st.markdown("<h2><span style='color:Lime;'>Developed By:</span></h1>", unsafe_allow_html= True)
    l, m, r = st.columns((1,1,1))
    with l:
        st.markdown("<h4>-Sriram A</h4>", unsafe_allow_html= True)
        st.write("")
        st.markdown("<h4>-Niranjan S Kaithota</h4>", unsafe_allow_html= True)
        st.write("")
    with m:
        st.markdown("<h4>-Sreeharish TJ</h4>", unsafe_allow_html= True)
        st.write("")
        st.markdown("<h4>-Pratham M Mallya</h4>", unsafe_allow_html= True)
        st.write("")
    with r:
        st.markdown("<h4>-Samruddhi D</h4>", unsafe_allow_html= True)
        st.markdown("<h4>-Shashank Krishnamani</h4>", unsafe_allow_html= True)
    return
        
def welcome_page():
    #First View
    with st.container():
        ##st.set_page_config(page_title="YUKTHI", layout='wide')
        st.markdown("<h1 style='text-align: left;'>Travel,<br>&nbsp Hassle <span style='color: blue;'> Free</span></h1> <h1 style='text-align: left;'> <span style='color: cyan;'>AI-Powered Traffic Optimization for Seamless Commutes</span></h1>", unsafe_allow_html=True)
        
        # Using columns for better layout
        left_space, image_col, right_space = st.columns([0.2, 0.6, 0.2])
        with image_col:
            image = Image.open("home_page_image2.jpg")
            st.image(image, width=600)
    
        st.markdown("<h3 style='text-align: left;'>Transforming Your Daily Travel with <span style='color: blue;'>Predictive Intelligence</span></h3>", unsafe_allow_html=True)
       
    # Models 
    with st.container():
        st.write("---")
        st.markdown("<h1 style='text-align: left;'> <span style='color: cyan;'>Our Models</span></h1>", unsafe_allow_html=True)
        
        left, right = st.columns(2)
        
        with left:
            st.markdown("""
                        <div style='background-color: #00FF00; padding: 20px; border-radius: 10px;'>
                        <span style='color: black;'>
                        <h2><span style='color: black;'>Time Slot Booking</span></h2>
                        <h6><span style='color: black;'>Find the best time of departure for a hassel free commte.</span></h6>
                        </span>
                        </div>
                        """,unsafe_allow_html=True)
            st.write(" ")
        with right:
            st.markdown("""
                        <div style='background-color: #FF69B4; padding: 20px; border-radius: 10px;'>
                        <span style='color: black;'>
                        <h2><span style='color: black;'>Traffic Prediction</span></h2>
                        <h6><span style='color: black;'>Take a look at the predicted traffic based on previous data.</span></h6>
                        </span>
                        </div>
                        """,unsafe_allow_html=True)
            st.write(" ")
        
        with left:
            st.markdown("""
                        <div style='background-color: #FFFF00; padding: 20px; border-radius: 10px;'>
                        <span style='color: black;'>
                        <h2><span style='color: black;'>Live Traffic data</span></h2>
                        <h6><span style='color: black;'>Look at the live incoming data from our system placed at different junctions across the ORR.</span></h6>
                        </span>
                        </div>
                        """,unsafe_allow_html=True)
        with right:
            st.markdown("""
                        <div style='background-color: #00BFFF; padding: 20px; border-radius: 10px;'>
                        <span style='color: black;'>
                        <h2><span style='color: black;'>Real Time Data Acquisition</span></h2>
                        <h6><span style='color: black;'>The model which does the real time processing of the roads at the junctions and proivde real time data.</span></h6>
                        </span>
                        </div>
                        """,unsafe_allow_html=True)
            st.write("##")
    
    # Methodology     
    with st.container():
        st.write("---")
        #st.markdown("<h1 style='text-align: left;'> <span style='color: cyan;'>Work Flow: </span></h1>", unsafe_allow_html=True)
        #st.write("##")
        
        # Yolo v10
        ls,m, rs = st.columns((1,0.1,2))
        with ls:
            st.write("##")
            st.write("##")
            st.markdown("<h2 style='text-align: centre;'> <span style='color: rgb(0, 221, 6);'>Real-Time Traffic Data Acquisition: </span></h2>",unsafe_allow_html=True)
            st.write("Utilizing YOLO v10 for advanced image processing, we gather and store dynamic traffic data for accurate forecasting and traffic pattern analysis.")
        with rs:
            st.markdown("<h2 style='text-align: left;'> Yolo v10 Image processing and Density calculation </h2>", unsafe_allow_html=True)
            st.video("Yolo_Demo_Video.mp4")
        
        st.write("---")
        
        #Prophet
        ls, rs = st.columns((2,1))
        with ls:
            prophet()
        with rs:
            st.write("##")
            st.write("##")
            st.markdown("<h2 style='text-align: centre;'> <span style='color: rgb(0, 221, 6);'>Predictive Traffic Modeling: </span></h2>",unsafe_allow_html=True)
            st.write("Leveraging historical data with the Prophet model, we generate precise traffic forecasts tailored for each day.")

        st.write("---")

        #time slot
        ls, rs = st.columns((2,1))
        with ls:
            time_slot()
        with rs:
            st.write("##")
            st.write("##")
            st.markdown("<h2 style='text-align: centre;'> <span style='color: rgb(0, 221, 6);'>Time Slot Booking </span></h2>",unsafe_allow_html=True)
            st.write("Receive personalized departure alerts to ensure you arrive on time, minimizing traffic delays by optimizing your travel time through predictive AI models")
        
        st.write("---")
        
        #arima
        ls, rs = st.columns((1,2))
        with ls:
            st.write("##")
            st.write("##")
            st.markdown("<h2 style='text-align: centre;'> <span style='color: rgb(0, 221, 6);'>Live Traffic Monitoring: </span></h2>",unsafe_allow_html=True)
            st.write("Real-time comparison between current traffic patterns and forecasted data. Our model dynamically adjusts based on historical traffic trends and real-time inputs for improved accuracy.")
        with rs:
            pass
            #arima()
                  
    # About Us 
    with st.container():
        
        st.write("---")
        st.markdown("<h1 style='text-align: left;'> <span style='color: Green;'>About Us</span></h1>", unsafe_allow_html=True)
            
        st.markdown("<h5 style='text-align: left;'> <span style='color: white;'>We are Yukthi, a team of six passionate students from the 3rd semester of the AIML department at RV College of Engineering. With a shared interest in artificial intelligence and machine learning, we have been working together on innovative projects that push the boundaries of technology. Over the past two weeks, we have collaborated on a challenging project and developed a cutting-edge solution aimed at solving real-world problems. Our focus on creativity, teamwork, and technical expertise drives us to continuously explore new possibilities in AI and ML.</span></h5>", unsafe_allow_html=True)

##### Models #####     
def time_slot():
    # Function to get coordinates from an address using Nominatim API
    def get_coordinates_from_address(address):
        nominatim_url = "https://nominatim.openstreetmap.org/search"
        params = {
            'q': address,
            'format': 'json',
            'addressdetails': 1,
            'limit': 1
        }
        response = requests.get(nominatim_url, params=params)

        if response.status_code == 200:
            data = response.json()
            if data:
                return float(data[0]['lat']), float(data[0]['lon'])
            else:
                st.error(f"Address '{address}' not found.")
                return None
        else:
            st.error(f"Nominatim API request failed: {response.status_code}")
            return None

    # Function to calculate travel time based on distance and conditions
    def estimate_travel_time(distance_km, departure_time):
        average_speed_kmh = 60  # Assuming average speed of 60 km/h
        estimated_time_hours = distance_km / average_speed_kmh
        estimated_time = timedelta(hours=estimated_time_hours)

        # Adjust estimated travel time based on time of day (rush hour adjustments)
        current_time = departure_time.time()
        if dt_time(8, 0) <= current_time <= dt_time(10, 0) or dt_time(17, 0) <= current_time <= dt_time(20, 0):
            estimated_time += timedelta(minutes=30)

        # Further adjust travel time based on distance ranges
        if 5 < distance_km < 10:
            estimated_time += timedelta(minutes=20)
        elif 10 < distance_km < 20:
            estimated_time += timedelta(minutes=40)
        elif distance_km > 20:
            estimated_time += timedelta(minutes=50)

        return estimated_time

    # Function to fit ARIMA model
    def fit_arima(df):
        model = ARIMA(df['Value'], order=(5, 1, 2))  # Adjusted order for better performance
        arima_model = model.fit()
        return arima_model

    # Function to generate ARIMA forecast
    def generate_arima_forecast(arima_model, steps=10):
        forecast = arima_model.forecast(steps=steps)
        return forecast

    ls, rs = st.columns((2,1))
    with ls:
        # Main Streamlit App
        st.title("Smart Route Optimization:")

        # Taking user input for source and destination addresses
        st.write("Enter the addresses for the source and destination.")
        source_address = st.text_input("Enter Source Address", value="Banashankari, Bangalore")
        dest_address = st.text_input("Enter Destination Address", value="Embassy Techvillage, Bangalore")

        # Time of travel input
        departure_time = st.time_input("Enter the Arrival time:", value=None)

        # Button to generate route
        if st.button("Generate Route"):
            # Get coordinates from addresses
            source_coords = get_coordinates_from_address(source_address)
            dest_coords = get_coordinates_from_address(dest_address)

            if source_coords and dest_coords:
                source = (source_coords[1], source_coords[0])  # Order: (lon, lat)
                dest = (dest_coords[1], dest_coords[0])

                start = "{},{}".format(source[0], source[1])
                end = "{},{}".format(dest[0], dest[1])

                # OSRM API for driving directions
                url = f'http://router.project-osrm.org/route/v1/driving/{start};{end}?alternatives=false&overview=full'
                headers = {'Content-type': 'application/json'}
                r = requests.get(url, headers=headers)

                if r.status_code == 200:
                    routejson = r.json()
                    geometry = routejson['routes'][0]['geometry']
                    coordinates = polyline.decode(geometry)

                    # Creating a dataframe for the coordinates
                    df_out = pd.DataFrame(coordinates, columns=['lat', 'long'])

                    # Plotting the route on a map
                    fig = go.Figure()
                    fig.add_trace(go.Scattermapbox(
                        mode="lines",
                        lat=df_out['lat'],
                        lon=df_out['long'],
                        line=dict(width=4, color='blue'),
                        name="Route"
                    ))

                    fig.add_trace(go.Scattermapbox(
                        mode="markers+text",
                        lat=[source_coords[0], dest_coords[0]],
                        lon=[source_coords[1], dest_coords[1]],
                        marker=dict(size=10, color='red'),
                        text=["Start", "End"],
                        textposition="bottom center"
                    ))

                    fig.update_layout(
                        mapbox=dict(
                            style="open-street-map",
                            zoom=8,
                            center=dict(lat=(source_coords[0] + dest_coords[0]) / 2, lon=(source_coords[1] + dest_coords[1]) / 2)
                        ),
                        margin={"r": 0, "t": 0, "l": 0, "b": 0},
                        height=600,
                        width=900
                    )

                    st.plotly_chart(fig)

                    # Calculate straight-line distance
                    distance_km = geodesic((source_coords[0], source_coords[1]), (dest_coords[0], dest_coords[1])).kilometers
                    st.write(f"**Real-line Distance:** {distance_km:.2f} km")

                    
                    # Estimate travel time
                    arrival_datetime = datetime.combine(datetime.today(), departure_time)
                    estimated_travel_time = estimate_travel_time(distance_km, arrival_datetime)
                    departure_time = arrival_datetime - estimated_travel_time - timedelta(minutes=15)

                    actual_time_range1 = departure_time - timedelta(minutes=5)
                    actual_time_range2 = departure_time + timedelta(minutes=5)
                    st.write(f"**Estimated Travel Time (approx.):** {estimated_travel_time}")
                    st.toast(f"**Optimal Departure Time slot:** {actual_time_range1.strftime('%I:%S')} to {actual_time_range2.strftime('%I:%S')}")
                    st.write(f"**Optimal Departure Time slot:** {actual_time_range1.strftime('%I:%S')} to {actual_time_range2.strftime('%I:%S')}")
                else:
                    distance_km = geodesic((source_coords[0], source_coords[1]), (dest_coords[0], dest_coords[1])).kilometers
                    st.write(f"**Real-line Distance:** {distance_km:.2f} km")
                    arrival_datetime = datetime.combine(datetime.today(), departure_time)
                    estimated_travel_time = estimate_travel_time(distance_km, arrival_datetime)
                    departure_time = arrival_datetime - estimated_travel_time - timedelta(minutes=10)
                    actual_time_range1 = departure_time - timedelta(minutes=5)
                    actual_time_range2 = departure_time + timedelta(minutes=5)
                    st.write(f"**Estimated Travel Time (approx.):** {estimated_travel_time}")
                    st.toast(f"**Optimal Departure Time slot:** {actual_time_range1.strftime('%H:%M')} to {actual_time_range2.strftime('%H:%M')}")
                    st.write(f"**Optimal Departure Time slot:** {actual_time_range1.strftime('%H:%M')} to {actual_time_range2.strftime('%H:%M')}")
                    #st.error("OSRM API request failed.")
    with rs:
        st_lottie(
                lottie_load("Map_animation.json"),
                speed = 0.75,
                loop=True,
            )

def prophet():
    st.markdown("<h3 style='text-align: left;'> Prophet Traffic Prediction </h3>", unsafe_allow_html=True)
    
    class ProphetModel:
        def predict(self, time2):
            # Load the pre-trained Prophet model
            with open('prophet_final.pkl', 'rb') as f:
                loaded_model = pickle.load(f)

            # Define the current time for prediction
            date_string = "2024-10-01 08:00:00"
            global time1
            time1 = datetime.strptime(date_string, "%Y-%m-%d %H:%M:%S")

            # Calculate the difference between time1 and time2 in minutes
            time_difference = time2 - time1
            minutes_difference = time_difference.total_seconds() / 60

            # Calculate the number of periods based on 5-minute intervals
            period = int(minutes_difference / 5)

            # Generate future timestamps for forecasting
            future = loaded_model.make_future_dataframe(periods=period + 1, freq='5min')
            # Predict future values
            forecast = loaded_model.predict(future)
            filtered_forecast = forecast[(forecast['ds'] <= time2) & (forecast['ds'] >= time1)]
            return filtered_forecast

    #if __name__ == "__main__":
    # Load your dataset
    df = pd.read_csv("traffic_synthetic.csv")
    df_prophet = df.reset_index().rename(columns={'time': 'ds', 'pcu': 'y'})

    # Specify the time2 for which you want to predict
    date_string = "2024-10-01 15:30:00"
    time2 = datetime.strptime(date_string, "%Y-%m-%d %H:%M:%S")

    # Initialize the model and predict
    model = ProphetModel()
    forecast = model.predict(time2)
    
    # Convert 'ds' columns to datetime in both actual and forecast datasets
    df_prophet['ds'] = pd.to_datetime(df_prophet['ds'])
    forecast['ds'] = pd.to_datetime(forecast['ds'])

    # Create the figure using Plotly
    fig = go.Figure()

    # Add the actual data trace
    fig.add_trace(go.Scatter(x=df_prophet['ds'], y=df_prophet['y'], mode='lines', name='Actual Data'))

    # Add the forecasted data trace
    fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], mode='lines', name='Forecast Data', line=dict(color='green', width=2, dash='dot')))

    # Customize layout
    fig.update_layout(title="Actual vs Forecast Data", xaxis_title="Time", yaxis_title="PCU", legend_title="Data Type")

    # Create a button to zoom in between time1 and time2
    #if st.button("Back"):
    #    # If button is pressed, zoom between time1 and time2
    #    fig.update_xaxes(range=[time1, time2])

    # Render the Plotly chart with Streamlit
    st.plotly_chart(fig)     
        
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from matplotlib.animation import FuncAnimation
import streamlit as st

# Full ARIMA function inside an if block

def data_create():
    """Function to create synthetic traffic data."""
    start_time = '06:00:00'
    end_time = '22:00:00'
    frequency = '5min'  # 5-minute intervals

    time_index = pd.date_range(start=start_time, end=end_time, freq=frequency)
    
    # Create a sinusoidal base pattern for PCU values
    time_in_hours = (time_index.hour + time_index.minute / 60)
    base_pcu = 200 + 100 * np.sin((time_in_hours - 6) / 16 * 2 * np.pi)  # Peaks around midday
    
    # Define peak traffic hours: 9 AM to 11 AM and 5 PM to 7 PM
    morning_peak = (time_index.hour >= 9) & (time_index.hour < 11)
    evening_peak = (time_index.hour >= 17) & (time_index.hour < 19)
    
    # Adjust PCU for peak hours
    pcu = np.where(morning_peak, base_pcu + 150, base_pcu)
    pcu = np.where(evening_peak, pcu + 150, pcu)
    
    # Add some random noise
    noise = np.random.normal(0, 20, len(time_index))
    pcu += noise
    
    # Create a DataFrame with time and PCU values
    df = pd.DataFrame({'time': time_index, 'pcu': pcu})
    df.set_index('time', inplace=True)
    
    return df

def arima():
    # Data creation function
    def data_create():
        start_time = '06:00:00'
        end_time = '22:00:00'
        frequency = '5min'  # 5-minute intervals
        
        time_index = pd.date_range(start=start_time, end=end_time, freq=frequency)
        
        # Create a sinusoidal base pattern for PCU values
        time_in_hours = (time_index.hour + time_index.minute / 60)
        base_pcu = 200 + 100 * np.sin((time_in_hours - 6) / 16 * 2 * np.pi)  # Peaks around midday
        
        # Define peak traffic hours: 9 AM to 11 AM and 5 PM to 7 PM
        morning_peak = (time_index.hour >= 9) & (time_index.hour < 11)
        evening_peak = (time_index.hour >= 17) & (time_index.hour < 19)
        
        # Adjust PCU for peak hours
        pcu = np.where(morning_peak, base_pcu + 150, base_pcu)
        pcu = np.where(evening_peak, pcu + 150, pcu)
        
        # Add some random noise
        noise = np.random.normal(0, 20, len(time_index))
        pcu += noise
        
        # Create a DataFrame with time and PCU values
        df = pd.DataFrame({'time': time_index, 'pcu': pcu})
        df.set_index('time', inplace=True)
        
        return df

    # Generate data and predictions
    df = data_create()

    # Build ARIMA model
    model = ARIMA(df['pcu'], order=(29, 0, 16), trend='c')
    model_fit = model.fit()

    # Predict the full range of values
    predictions = model_fit.predict(start=0, end=len(df) - 1, typ='levels')

    # Streamlit app
    

    # Custom CSS to set the background color
    st.markdown("""
        <style>
        .stApp {
            background-color: #121212;
        }
        </style>
    """, unsafe_allow_html=True)

    st.title('Dynamic ARIMA Prediction for PCU')

    # Create a plot with dark style
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(12, 6))

    # Set plot limits and labels
    ax.set_xlim(df.index[0], df.index[-1])
    ax.set_ylim(df['pcu'].min(), df['pcu'].max())
    ax.set_title('Dynamic ARIMA Prediction for PCU', color='white', fontsize=16)
    ax.set_xlabel('Time', color='white', fontsize=12)
    ax.set_ylabel('PCU', color='white', fontsize=12)

    # Customize tick labels
    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')

    # Format x-axis to show only time
    ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%H:%M'))

    # Create empty line plots for predicted and actual data
    line_pred, = ax.plot([], [], lw=2, label='Predicted Data', color='#FFA500')  # Bright orange
    line_actual, = ax.plot([], [], lw=2, label='Actual Data', color='#00BFFF')  # Deep sky blue

    # Customize legend
    ax.legend(facecolor='#1C1C1C', edgecolor='#1C1C1C', labelcolor='white')

    # Add a grid for better readability
    ax.grid(True, linestyle='--', alpha=0.3)

    # Streamlit plot
    plot_placeholder = st.empty()

    # Initialize empty lists to store time points and predictions
    x_data_pred, y_data_pred = [], []
    x_data_actual, y_data_actual = [], []

    # Update function
    def update(frame):
        # Plot the predicted data up to the current frame
        x_data_pred.append(df.index[frame])
        y_data_pred.append(predictions.iloc[frame])
        
        # Plot the actual data only if it is 10 steps behind the predicted data
        if frame >= 10:
            x_data_actual.append(df.index[frame - 10])
            y_data_actual.append(df['pcu'].iloc[frame - 10])
        
        # Update the lines
        line_pred.set_data(x_data_pred, y_data_pred)
        line_actual.set_data(x_data_actual, y_data_actual)
        
        # Redraw the plot
        fig.canvas.draw()
        
        # Update the Streamlit plot
        plot_placeholder.pyplot(fig)

    # Animate the plot
    for frame in range(len(df)):
        update(frame)
        time.sleep(0.01)  # Control the speed of the animation

    # Display final data
    st.subheader('Final Data')
    st.dataframe(df.style.background_gradient(cmap='YlOrRd'))


def yolo():
    st.markdown("<h3 style='text-align: left;'> Yolo v10 Image processing and Density calculation </h3>", unsafe_allow_html=True)
    video_file = open('Yolo_Demo_Video.mp4', 'rb')
    st.video(video_file)
    if st.button("Back"):
        return

def main():
    # Custom CSS for styling
    st.markdown("""
        <style>
        body {
            background-color: #ADD8E6;  /* Light Blue Background */
            color: #333;
        }
        .stButton > button {
            background-color: #4CAF50;
            color: white;
        }
        .stButton > button:hover {
            background-color: #45a049;
        }
        .stTextInput > div > input, .stTextArea > div > textarea {
            border: 2px solid #4CAF50;
            border-radius: 5px;
        }
        </style>
    """, unsafe_allow_html=True)

    
    selected = option_menu(
        None,
        options = ["Home", "Login", "Book Your Slot", 'Traffic Prediction', 'Live Traffic Data', 'About Us'], 
        icons = ['house', 'door-open', "geo-alt-fill", 'clipboard-data', 'activity', 'info-circle'], 
        menu_icon="cast",
        default_index=0,
        orientation= "horizontal"
        )
    
    if selected == "Home":
        welcome_page()
    if selected == "Login":
        login_page()
    if selected == "Book Your Slot":
        time_slot()
    if selected == "Traffic Prediction":
        prophet()
    if selected == "Live Traffic Data":
        arima()
    if selected == "About Us":
        about()

    ## Sidebar navigation
    #tab1, tab2, tab3 = st.tabs(["Home Page", "Login", "Create New Account"])
    #with tab1:
    #    welcome_page()
    #with tab2:
    #    login_page()
    #with tab3:
    #    create_new_account()
    
    
    # page = st.sidebar.selectbox("Select Page", ["Home Page", "Login", "Create New Account"])
    # if page == "Login":
        # login_page()
    # elif page == "Home Page":
        # welcome_page()
    # elif page == "Create New Account":
        # create_new_account()
    # else:
        # if 'logged_in' not in st.session_state:
            # st.session_state['logged_in'] = False

        # if st.session_state['logged_in']:
            # commute_form()
        # else:
            # st.info("Please log in to access the commute details.")

if __name__ == "__main__":
    main()