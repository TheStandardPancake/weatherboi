from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import pandas as pd
import numpy as np

driver = webdriver.Chrome()

#iterate each year
for year in range(1997,2020):
    #iterate each month
    for month in range(12):
        #iterate each day (Note: some of the days are excluded using this method as some months have more than others, thus I only have 28 days from each month)
        for day in range(28):
            AVG_temperature = np.array([])
            AVG_dewpoint = np.array([])
            AVG_windspeed = np.array([])
            AVG_humidity = np.array([])
            AVG_pressure = np.array([])
            rained = 0
            url = f'https://www.wunderground.com/history/daily/YSSY/date/{year}-{month+1}-{day+1}'
            driver.get(url)
            tables = WebDriverWait(driver,20).until(EC.presence_of_all_elements_located((By.CLASS_NAME, "mat-table.cdk-table.mat-sort.ng-star-inserted")))
            for table in tables:
                newTable = pd.read_html(table.get_attribute('outerHTML'))
                index = newTable.index
                if newTable:
                    for row in range(len(newTable[0])):
                        if newTable[0].fillna('')['Time'][row][-2] == 'A':
                            #gather temp data
                            AVG_temperature = np.append(AVG_temperature, newTable[0].fillna('')['Temperature'][row].rstrip(" F"))
                            #gather dew point data
                            AVG_dewpoint = np.append(AVG_dewpoint, newTable[0].fillna('')['Dew Point'][row].rstrip(" F"))
                            #gather wind speed data
                            AVG_windspeed = np.append(AVG_windspeed, newTable[0].fillna('')['Wind Speed'][row].rstrip(" mph"))
                            #gather humidity data
                            AVG_humidity = np.append(AVG_humidity, newTable[0].fillna('')['Humidity'][row].rstrip(" %"))
                            #gather pressure data
                            AVG_pressure = np.append(AVG_pressure, newTable[0].fillna('')['Pressure'][row].rstrip(" in"))
                        #See if it rained in the afternoon
                        else:
                            if newTable[0].fillna('')['Precip.'][row] != '0.0 in':
                                rained = 1
                    #calculating the mean of the data and storing the data into a file in the appropriate order for the A.I. to Process
                    AVG_temperature = np.sum(AVG_temperature.astype(np.float))/len(AVG_temperature)
                    AVG_dewpoint = np.sum(AVG_dewpoint.astype(np.float))/len(AVG_dewpoint)
                    AVG_windspeed = np.sum(AVG_windspeed.astype(np.float))/len(AVG_windspeed)
                    AVG_humidity = np.sum(AVG_humidity.astype(np.float))/len(AVG_humidity)
                    AVG_pressure = np.sum(AVG_pressure.astype(np.float))/len(AVG_pressure)
                    with open("training_data.csv","w+") as csvfile:
                        csvfile.write(f'{AVG_temperature},{AVG_dewpoint},{AVG_windspeed},{AVG_humidity},{AVG_pressure},{rained}')
driver.quit()
