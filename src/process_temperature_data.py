import pandas as pd
def main():
    #Load temperature data from a file
    data = pd.read_csv("data/raw/montreal_temperature_jan2026.csv")
    #Clean the data
    data = data[["Date/Time (LST)", "Temp (°C)"]]
    #save
    data.to_csv("data/processed/processed_montreal_temperature_jan2026.csv", index=False)
if __name__ == "__main__":
    main()