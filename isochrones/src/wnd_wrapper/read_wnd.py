import math
import sys
import pandas as pd


wnd_file_path = 'data/raw/177.wnd'
out_file_path = 'data/pre-processed/test.csv'


def parse_wnd(wnd_file_path):
    longitude = -180
    latitude = 90
    byte_couple_count = 0
    wind_values = []
    wind_value_couple = []
    with open(wnd_file_path, "rb") as f:
        while (byte := f.read(1)):
            if byte_couple_count == 2:
                # Store data
                wind_values.append({'latitude': latitude, 'longitude': longitude,
                                    'u': wind_value_couple[0], 'v': wind_value_couple[1]})

                # Reset local variables
                latitude, longitude = update_lat_lon(latitude, longitude)
                byte_couple_count = 0
                wind_value_couple = []

            raw_value = int.from_bytes(
                byte, byteorder=sys.byteorder, signed=True)
            # Ukmph = sign(Ub) * sqr(Ub / 8)
            wind_value_couple.append(
                math.copysign(1, raw_value) * raw_value**2/8)
            byte_couple_count += 1

        # Store last couple
        wind_values.append({'latitude': latitude, 'longitude': longitude,
                            'u': wind_value_couple[0], 'v': wind_value_couple[1]})
    return wind_values


def update_lat_lon(latitude, longitude):
    # Custom logic linked to the .wnd file structure
    if longitude < 179:
        longitude += 1
    elif longitude == 179:
        longitude = -180
        latitude -= 1
    else:
        print(f'Error with the longitude value: {longitude}')
    return latitude, longitude


def main():
    wind_values = parse_wnd(wnd_file_path)
    df = pd.DataFrame(wind_values)
    df = df.set_index('latitude')
    df.to_csv(out_file_path)


if __name__ == "__main__":
    main()
