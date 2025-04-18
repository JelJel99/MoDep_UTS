import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import RobustScaler, StandardScaler
import statistics as st
import joblib
import pickle as pkl

class BookingStatusPredict:
    def __init__(self):
        self.model = RandomForestClassifier(criterion = 'entropy', max_depth = 40, random_state = 42)
        self.standard_scaler = StandardScaler()
        self.robust_scaler = RobustScaler()
        self.oheEncode = OneHotEncoder()
        self.x_encode = {"type_of_meal_plan": {"Not Selected": 1, "Meal Plan 1": 1, "Meal Plan 2": 2, "Meal Plan 3" : 3}, "room_type_reserved": {"Room_Type 1": 1,"Room_Type 2": 2, "Room_Type 3": 3, "Room_Type 4": 4, "Room_Type 5": 5, "Room_Type 6": 6, "Room_Type 7": 7}}
        self.y_encode = {"Not_Canceled": 0, "Canceled": 1}
    
    def split_data(self):
        self.df = pd.read_csv("C:/Users/Asus/Documents/ANGELA/KULIAH/S4/MoDep/UTS/Dataset_B_hotel.csv")
        input = self.df.drop(['booking_status','Booking_ID'], axis = 1)
        output = self.df['booking_status']
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(input, output, test_size = 0.2, random_state = 42)
    
    def preprocessing(self):
        # Handling Missing Value
        self.x_train['type_of_meal_plan'] = self.x_train['type_of_meal_plan'].fillna(st.mode(self.x_train['type_of_meal_plan']))
        self.x_test['type_of_meal_plan'] = self.x_test['type_of_meal_plan'].fillna(st.mode(self.x_train['type_of_meal_plan']))

        for col in ['required_car_parking_space', 'avg_price_per_room']:
            self.x_train[col] = self.x_train[col].fillna(self.x_train[col].median())
            self.x_test[col] = self.x_test[col].fillna(self.x_test[col].median())
        
        # Encode
        ## Label Encoding
        self.x_train = self.x_train.replace(self.x_encode)
        self.x_test = self.x_test.replace(self.x_encode)

        self.y_train = self.y_train.replace(self.y_encode)
        self.y_test = self.y_test.replace(self.y_encode)


        ## One Hot Encoding
        xtrain_market_enc = pd.DataFrame(self.oheEncode.fit_transform(self.x_train[['market_segment_type']]).toarray(),columns = self.oheEncode.get_feature_names_out())
        self.x_train = self.x_train.reset_index()
        self.x_train = pd.concat([self.x_train, xtrain_market_enc], axis=1)

        xtest_market_enc = pd.DataFrame(self.oheEncode.transform(self.x_test[['market_segment_type']]).toarray(),columns = self.oheEncode.get_feature_names_out())
        self.x_test = self.x_test.reset_index()
        self.x_test = pd.concat([self.x_test, xtest_market_enc], axis=1)

        self.x_train.drop(columns=['market_segment_type'], inplace=True)
        self.x_test.drop(columns=['market_segment_type'], inplace=True)

        # Scaling
        scale_robust = ['no_of_adults', 'no_of_children', 'no_of_weekend_nights', 'no_of_week_nights', 'required_car_parking_space', 'lead_time', 'arrival_year', 'repeated_guest', 'no_of_previous_cancellations', 'no_of_previous_bookings_not_canceled', 'avg_price_per_room']
        scale_standard = ['arrival_month', 'arrival_date', 'type_of_meal_plan', 'room_type_reserved']

        self.x_train[scale_robust] = self.robust_scaler.fit_transform(self.x_train[scale_robust])
        self.x_test[scale_robust] = self.robust_scaler.transform(self.x_test[scale_robust])

        self.x_train[scale_standard] = self.standard_scaler.fit_transform(self.x_train[scale_standard])
        self.x_test[scale_standard] = self.standard_scaler.transform(self.x_test[scale_standard])
    
    def train(self):
        self.model.fit(self.x_train, self.y_train)
    
    def evaluate(self):
        y_pred = self.model.predict(self.x_test)
        print('\nConfusion Matrix\n')
        print(confusion_matrix(self.y_test, y_pred))

        print('\nClassification Report\n')
        print(classification_report(self.y_test, y_pred, target_names = ['0','1']))