from dataclasses import dataclass

@dataclass
class RealEstateProcessed:
    name: str
    url: str
    property_type: str
    street: str
    ward: str
    district: str
    price_total: float
    price_m2: float
    area: float
    long: float
    lat: float
