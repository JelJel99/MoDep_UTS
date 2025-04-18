import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load the machine learning model and encode
model = joblib.load('rf_model.pkl')
meal_roomType_encode = joblib.load('meal_roomType_encode.pkl')
market_ohe = joblib.load('market_ohe.pkl')
standard_scaler = joblib.load('standard_scaler.pkl')
robust_scaler = joblib.load('robust_scaler.pkl')


def main():
    st.title('Book Status Model Deployment')

    no_of_adults = st.number_input("Number of adults", 0, 100)
    no_of_children = st.number_input("Number of children", 0, 100)
    no_of_weekend_nights = st.number_input("Number of weekend nights booked (Saturday/Sunday)", 0, 100)
    no_of_week_nights = st.number_input("Number of weeknights booked (Monday–Friday)", 0, 100)
    type_of_meal_plan = st.radio("Choose meal type", ["Not Selected", "Meal Plan 1","Meal Plan 2","Meal Plan 3"])
    required_car_parking_space = st.number_input("Number of required parking space", 0, 10)
    room_type_reserved = st.radio("Choose room type reserved", ["Room_Type 1","Room_Type 2","Room_Type 3", "Room_Type 4", "Room_Type 5", "Room_Type 6", "Room_Type 7"])
    lead_time = st.number_input("Number of days between booking date and arrival date (lead time)", 0, 1000)
    arrival_year = st.number_input("Arrival year", 0, 1000000)
    arrival_month = st.number_input("Arrival month", 0, 12)
    arrival_date = st.number_input("Arrival date", 0, 31)
    market_segment_type = st.radio("Market segment type", ["Online", "Offline","Corporate","Complementary", "Aviation"])
    repeated_guest = st.radio("Repeated guest [0 = No, 1 = Yes]", ["0", "1"])
    no_of_previous_cancellations = st.number_input("Number of previous cancellations", 0, 50)
    no_of_previous_bookings_not_canceled = st.number_input("Number of previous bookings not canceled", 0, 100)
    avg_price_per_room = st.number_input("Average price per room (EUR)", 0.00, 10000.00)
    no_of_special_requests = st.number_input("Number of special requests", 0, 50)
    
    
    data = {
    'no_of_adults': int(no_of_adults),
    'no_of_children': int(no_of_children),
    'no_of_weekend_nights': int(no_of_weekend_nights),
    'no_of_week_nights': int(no_of_week_nights),
    'type_of_meal_plan': type_of_meal_plan,
    'required_car_parking_space': int(required_car_parking_space),  # 0 atau 1
    'room_type_reserved': room_type_reserved,
    'lead_time': int(lead_time),
    'arrival_year': int(arrival_year),
    'arrival_month': int(arrival_month),
    'arrival_date': int(arrival_date),
    'market_segment_type': market_segment_type,
    'repeated_guest': int(repeated_guest),
    'no_of_previous_cancellations': int(no_of_previous_cancellations),
    'no_of_previous_bookings_not_canceled': int(no_of_previous_bookings_not_canceled),
    'avg_price_per_room': float(avg_price_per_room),
    'no_of_special_requests': int(no_of_special_requests)
    }
    
    df=pd.DataFrame([list(data.values())], columns=['no_of_adults','no_of_children', 'no_of_weekend_nights', 'no_of_week_nights','type_of_meal_plan', 
                                                    'required_car_parking_space', 'room_type_reserved','lead_time', 'arrival_year', 'arrival_month', 
                                                    'arrival_date', 'market_segment_type', 'repeated_guest', 'no_of_previous_cancellations',
                                                    'no_of_previous_bookings_not_canceled', 'avg_price_per_room', 'no_of_special_requests'])

    df = df.replace(meal_roomType_encode)

    cat_market = df[['market_segment_type']]
    cat_enc_market=pd.DataFrame(market_ohe.transform(cat_market).toarray(),columns = market_ohe.get_feature_names_out())
    df = pd.concat([df, cat_enc_market], axis=1)
    df = df.drop(['market_segment_type'],axis=1)
    
    if st.button('Make Prediction'):
        features = df      
        result = make_prediction(features)
        result_label = "Canceled" if result == 1 else "Not Canceled"
        st.success(f'The prediction is: {result_label}')

def make_prediction(features):
    input_array = np.array(features).reshape(1, -1)
    prediction = model.predict(input_array)
    return prediction[0]

if __name__ == '__main__':
    main()