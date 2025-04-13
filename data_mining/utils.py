class ParserUtils:
    @staticmethod
    def parse_price(price_str):
        """
        Parses a price string and converts it into a numerical value.
        Handles cases like 'tỷ', 'triệu', '~', and ',' as decimal separator.
        """
        if not price_str:
            return None

        try:
            # Clean the string
            price_str = price_str.strip().lower()
            price_str = price_str.replace("~", "")  # remove tilde
            price_str = price_str.replace(",", ".")  # convert comma to dot for decimal
            price_str = price_str.replace(" ", "")

            # Handle 'tỷ'
            if "tỷ" in price_str:
                price_str = price_str.replace("tỷ", "")
                return float(price_str) * 1_000_000_000

            # Handle 'triệu'
            elif "triệu" in price_str:
                price_str = price_str.replace("triệu", "").replace("/m²", "")
                return float(price_str) * 1_000_000
            
            elif "nghìn" in price_str:
                price_str = price_str.replace("nghìn", "").replace("/m²", "")
                return float(price_str) * 1_000

            # Otherwise just parse as raw float
            return float(price_str)

        except ValueError:
            print(f"[Invalid price format]: {price_str}")
            return None
            
    @staticmethod
    def parse_area(area_str):
        try:
            area = area_str.replace("m²", "").replace(".", "").replace(",", ".").strip()
            return float(area)
        except Exception:
            return None

    @staticmethod
    def parse_coordinates(map_data_src):
        """
        Extracts latitude and longitude from the map_data_src URL.
        Assumes format contains 'q=lat,long' parameters.
        """
        try:
            if not map_data_src or "q=" not in map_data_src:
                return None, None
            coords = map_data_src.split("q=")[1].split("&")[0]
            lat_str, long_str = coords.split(",")
            return float(lat_str), float(long_str)
        except Exception as e:
            print(f"[Invalid coordinates]: {e}")
            return None, None

    @staticmethod
    def parse_property_type(name: str):
        if not name:
            return "land"

        name = name.lower()
        if "căn hộ" in name:
            return "apartment"
        elif "nhà" in name:
            return "house"
        return "land"

    @staticmethod
    def parse_address_components(address: str):
        if not address:
            return "N/A", "N/A", "N/A"

        parts = [part.strip() for part in address.split(",")]
        len_p = len(parts)

        street = parts[len_p - 4] if len_p >= 4 else "N/A"
        ward = parts[len_p - 3] if len_p >= 3 else "N/A"
        district = parts[len_p - 2] if len_p >= 2 else "N/A"

        # Clean up district by removing "Quận" or "Huyện"
        if district != "N/A":
            district = district.replace("Quận", "").replace("Huyện", "").strip()

        # Ensure ward starts with "Phường"
        if ward != "N/A" and not ward.startswith("Phường"):
            ward = f"Phường {ward}"

        return street, ward, district

