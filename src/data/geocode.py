import numpy as np
import pandas as pd
from geopy.geocoders import Nominatim
import time
from tqdm import tqdm

class Get_District:
    def __init__(self, path_read_data : str , path_save_data : str ):
        self.geolocator =  Nominatim(user_agent="agent_geo_decode")
        self.iter_error = 0
        self.path_read_data = path_read_data
        self.path_save_data = path_save_data

    def get_address_by_location(self, latitude, longitude, language="ru"):
        """This function returns an address as raw from a location
        will repeat until success"""
        # build coordinates string to pass to reverse() function
        coordinates = f"{latitude}, {longitude}"
        # sleep for a second to respect Usage Policy
        time.sleep(0.5)
        try:
            result =  self.geolocator.reverse(coordinates, language=language).raw
            self.iter_error = 0
            return result
        except:
            if self.iter_error >= 100:
                self.iter_error = 0
                return {'address' :  {'suburb' : 'NaN' }}
            else:
                self.iter_error += 1
                return self.get_address_by_location(latitude, longitude)

    def get_field_district(self, latitude, longitude):
        # get the address info
        address = self.get_address_by_location(latitude, longitude)
        # print all returned data
        try:
            return address['address']['suburb']
        except KeyError as e:
            try:
                return address['address']['municipality']
            except:
                return 'NaN'
        except:
            return 'NaN'
    def create_csv_file_with_district(self, id_start : int , id_stop : int):
        df = pd.read_csv(self.path_read_data)
        df = df.loc[id_start:id_stop] # вырезаем от id_start до id_stop включительно
        list_district = []
        for i in tqdm(range(id_start,id_stop+1)):
            latitude = df.loc[i]["geo_lat"]
            longitude = df.loc[i]["geo_lon"]
            list_district.append(self.get_field_district(latitude,longitude))
        df['district'] = list_district
        df.to_csv(r'{0}{1}.csv'.format(self.path_save_data, 'id_from_' + str(id_start) +'_to_' + str(id_stop)), index=True)

if __name__ == "__main__":
    print("Start Wolrk programm!")
    obj = Get_District("../../data/interim/msk_geodata.csv","../../data/geocode_msk/")

    # record start time
    start = time.time()
    start_id = 300_001
    stop_id = 381_860
    print('Geodecode from ' + str(start_id) + ' to ' + str(stop_id))
    obj.create_csv_file_with_district(start_id , stop_id)

    # record end time
    end = time.time()
    # print the difference between start
    # and end time in milli. secs
    print("The time of execution of above program is :",
        (end-start) , "s" , "or", (end-start) / 60 , "min")
