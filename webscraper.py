from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import pandas as pd

url = 'https://www.wunderground.com/history/daily/YSSY/date/2020-8-26'
driver = webdriver.Chrome(executable_path="C:\\Users\\herob\\Documents\\GitHub\\weatherboi\\chromedriver.exe")
driver.get(url)
tables = WebDriverWait(driver,20).until(EC.presence_of_all_elements_located((By.CLASS_NAME, "mat-table.cdk-table.mat-sort.ng-star-inserted")))
for table in tables:
    newTable = pd.read_html(table.get_attribute('outerHTML'))
    if newTable:
        print(newTable[0].fillna(''))
