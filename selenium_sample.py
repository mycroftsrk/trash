# pip install selenium
# 4.10.0以上
# pip install --upgrade selenium

# %%
from selenium import webdriver
from selenium.webdriver.common.by import By
from time import sleep

START_URL = "https://applications.icao.int/icec/Home/Index"

driver = webdriver.Chrome()
driver.get(START_URL)

elements = driver.find_elements(By.CLASS_NAME, "select2-selection--single")

elements[0].click() # Departure
elm_selectlist = driver.find_element(By.XPATH, f'/html/body/span/span/span[2]')
depr_airport_list = elm_selectlist.find_elements(By.TAG_NAME, 'li')
depr_airport = depr_airport_list[2].text  # 試しに3番目を選択
driver.find_element(By.CLASS_NAME, "select2-search__field").send_keys(depr_airport)
sleep(1)
driver.find_element(By.CLASS_NAME, "select2-results__option--highlighted").click()

elements[1].click() # Destination 
elm_selectlist = driver.find_element(By.XPATH, f'/html/body/span/span/span[2]')
dest_airport_list = elm_selectlist.find_elements(By.TAG_NAME, 'li')
dest_airport = dest_airport_list[1].text  # 試しに2番目を選択（1番目は空白）
driver.find_element(By.CLASS_NAME, "select2-search__field").send_keys(dest_airport)
sleep(1)
driver.find_element(By.CLASS_NAME, "select2-results__option--highlighted").click()

# # add Destination
# driver.find_element(By.CLASS_NAME, "btnAddDestination").click()
# elements = driver.find_elements(By.CLASS_NAME, "select2-selection--single")
# elements[2].click()
# driver.find_element(By.CLASS_NAME, "select2-search__field").send_keys("Liverpool, United Kingdom (Liverpool - LPL)")
# driver.find_element(By.CLASS_NAME, "select2-results__option--highlighted").click()


# Number of passengers
elm_psngr_no = driver.find_element(By.ID, "passengerNo")
elm_psngr_no.clear()
elm_psngr_no.send_keys(5)  # 試しに"5"を入力

# cabin class: Economy, Premium Economy, Business, First
elm_cbn_cls = driver.find_element(By.ID, "selectClass")
elm_cbn_cls.send_keys("First")  # 試しにFirstクラス

# Trip
elm_trip_roundtrip = driver.find_element(By.CLASS_NAME, "active")
elm_trip_roundtrip.click()
elm_trip_oneway = driver.find_element(By.CLASS_NAME, "btnLast")
elm_trip_oneway.click()

# calculate
elm_btn_calc = driver.find_element(By.CLASS_NAME, "btnCalculate")
elm_btn_calc.click()

# HACK
sleep(5)


elm_rslt_tbl = driver.find_element(By.XPATH, f'//*[@id="ChildResultTotalViewHead"]')

airports = elm_rslt_tbl.find_elements(By.CLASS_NAME, 'lblAirportResult')
for elem in airports:
    print(elem.text)

results = elm_rslt_tbl.find_elements(By.CLASS_NAME, 'lblResult')
for elem in results:
    print(elem.text)

result_details = elm_rslt_tbl.find_elements(By.CLASS_NAME, 'lblResultDetail')
for elem in result_details:
    print(elem.text)




