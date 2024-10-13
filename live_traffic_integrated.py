from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Tuple
import pandas as pd
from geopy.distance import great_circle
import gradio as gr
import uvicorn
import threading
import folium
from geopy.geocoders import Nominatim
import requests
from fastapi.responses import Response
import random

app = FastAPI()

# Initialize geocoder
geolocator = Nominatim(user_agent="parking_recommender")

# Load parking data from a JSON file (mock implementation)
parking_spaces = pd.read_json('random_data.json')

# CORS proxy URL
CROS_PROXY_URL = 'https://www.whateverorigin.org/get?url='


# Model for the request body
class UserPreferences(BaseModel):
    lat: float
    lon: float
    max_distance: float
    max_price: float
    min_rating: float


# Model for the parking recommendations
class ParkingRecommendation(BaseModel):
    # address: str
    location: Tuple[float, float]
    price: float
    rating: float
    distance: float
    similarity: float
    traffic_info: str  # Added this line


# Fetch live parking data (mock implementation)
def fetch_live_parking_data() -> pd.DataFrame:
    return parking_spaces


# Function to get traffic data from Google Maps API
# def get_traffic_data(origin: Tuple[float, float], destination: Tuple[float, float]) -> str:
#     api_key = "YOUR_GOOGLE_MAPS_API_KEY"  # Replace with your actual API key
#     url = f"https://maps.googleapis.com/maps/api/directions/json?origin={origin[0]},{origin[1]}&destination={destination[0]},{destination[1]}&key={api_key}&traffic_model=best_guess&departure_time=now"
#
#     response = requests.get(url)
#     if response.status_code == 200:
#         data = response.json()
#         if data['status'] == "OK":
#             duration_in_traffic = data['routes'][0]['legs'][0]['duration_in_traffic']['text']
#             return duration_in_traffic
#     return "N/A"
def get_traffic_duration(origin: Tuple[float, float], destination: Tuple[float, float]) -> int:
    # Mocking traffic duration with a random value
    return random.randint(300, 3600)  # Random duration between 5 and 60 minutes


# Function to recommend parking spaces
def recommend_parking(user_location: Tuple[float, float], max_distance_km: float,
                      max_price: float, min_rating: float) -> List[ParkingRecommendation]:
    recommendations = []
    parking_data = fetch_live_parking_data()

    for index, row in parking_data.iterrows():
        if not row['availability']:
            continue

        distance = great_circle(user_location, row['location']).kilometers

        # Get traffic data
        traffic_duration = get_traffic_duration(user_location, row['location'])
        traffic_info = f"{traffic_duration}" if traffic_duration else "N/A"

        similarity = (1 / (1 + distance)) * (row['rating'] / 5)
        # location_address = geolocator.reverse(row['location']).address

        if distance <= max_distance_km and row['price'] <= max_price and row['rating'] >= min_rating:
            recommendations.append(ParkingRecommendation(
                # address=location_address,
                location=row['location'],
                price=row['price'],
                rating=row['rating'],
                distance=distance,
                similarity=similarity,
                traffic_info=traffic_info  # New field for traffic info
            ))

    recommendations.sort(key=lambda x: x.similarity, reverse=True)

    return recommendations


@app.get("/")
async def read_root():
    return {"message": "Welcome to the Parking Recommendation API!"}


@app.post('/recommend', response_model=List[ParkingRecommendation])
async def recommend(preferences: UserPreferences):
    user_location = (preferences.lat, preferences.lon)
    recommended_parking = recommend_parking(user_location, preferences.max_distance,
                                            preferences.max_price, preferences.min_rating)

    if not recommended_parking:
        raise HTTPException(status_code=404, detail="No parking spaces found matching the criteria.")

    return recommended_parking


# Function to fetch JavaScript through CORS proxy
@app.get('/fetch_js')
async def fetch_js():
    # Construct the Google Maps API URL
    print("In Javascript")
    google_maps_api_url = 'https://maps.googleapis.com/maps/api/js?key=YOUR_GOOGLE_MAPS_API_KEY&callback=initMap'

    # Send the request through CORS proxy
    response = requests.get(CROS_PROXY_URL + google_maps_api_url)

    if response.status_code == 200:
        # Here, we assume the response contains the JavaScript we need.
        javascript_content = response.json().get('contents', '')
        # Modify the JavaScript content as needed
        modified_js = modify_javascript(javascript_content)
        return Response(modified_js, media_type='application/javascript')
    else:
        raise HTTPException(status_code=500, detail="Error fetching JavaScript")


def modify_javascript(js_content):
    # Here you can implement the logic from the JavaScript code you provided.
    # This function will return the modified JS code.

    # Example modifications:
    js_content = js_content.replace("common.js", "modified_common.js")
    # Add more modifications based on your requirements.

    return js_content


# Gradio Interface
def gradio_interface(lat: str, lon: str, max_distance: str, max_price: str, min_rating: str):
    try:
        user_location = (float(lat), float(lon))
        max_distance_km = float(max_distance)
        max_price_value = float(max_price)
        min_rating_value = float(min_rating)

        recommendations = recommend_parking(user_location, max_distance_km, max_price_value, min_rating_value)

        if not recommendations:
            return "<p>No parking spaces found matching the criteria.</p>"

        html_content = "<h2>Parking Recommendations:</h2><ul>"
        for rec in recommendations:
            #html_content += f"<li><strong>Address:</strong> {rec.address}<br>"
            html_content += f"<li><strong>Price:</strong> {rec.price} AZN<br>"
            html_content += f"<strong>Rating:</strong> {rec.rating}<br>"
            html_content += f"<strong>Distance:</strong> {rec.distance:.2f} km<br>"
            html_content += f"<strong>Similarity:</strong> {rec.similarity:.2f}<br>"
            html_content += f"<strong>Traffic Info:</strong> {rec.traffic_info}</li><br>"  # Display traffic info
        html_content += "</ul>"

        # Create a map with Folium
        map_ = folium.Map(location=user_location, zoom_start=14)
        folium.Marker(location=user_location, tooltip="Your Location", icon=folium.Icon(color='blue')).add_to(map_)
#rec.address},
        for rec in recommendations:
            folium.Marker(location=rec.location, tooltip=f"Price: {rec.price} AZN, Rating: {rec.rating}, Traffic: {rec.traffic_info}",
                          icon=folium.Icon(color='green')).add_to(map_)

        # Save map as HTML and embed it directly into the response
        map_html = map_._repr_html_()

        # Include the map HTML directly in the content
        html_content += f"<h2>Map of Parking Locations:</h2>{map_html}"

        return html_content
    except ValueError:
        return "<p>Invalid input. Please enter numeric values.</p>"


# Create Gradio Interface
gr_interface = gr.Interface(
    fn=gradio_interface,
    inputs=[
        gr.Textbox(label="Latitude", placeholder="Enter latitude (e.g., 40.4093)"),
        gr.Textbox(label="Longitude", placeholder="Enter longitude (e.g., 49.8671)"),
        gr.Textbox(label="Max Distance (km)", placeholder="Enter max distance (e.g., 10)"),
        gr.Textbox(label="Max Price", placeholder="Enter max price (e.g., 50)"),
        gr.Textbox(label="Min Rating", placeholder="Enter min rating (e.g., 0)"),
    ],
    outputs="html",
    title="Parking Recommendation System",
    description="Find available parking spaces based on your preferences."
)

# Run both FastAPI and Gradio
if __name__ == '__main__':
    threading.Thread(target=lambda: uvicorn.run(app, host='0.0.0.0', port=8000, log_level="info")).start()
    gr_interface.launch()
