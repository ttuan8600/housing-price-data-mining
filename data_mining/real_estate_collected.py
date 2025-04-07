from dataclasses import dataclass

@dataclass
class RealStateCollected:
    name: str
    url: str
    address: str
    price_total: str
    price_m2: str
    area: str
    map_data_src: str