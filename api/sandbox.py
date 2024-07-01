import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime


@dataclass
class DataMixin:
    def validate_types(self) -> None:
        for attr_name, type_ in self.__annotations__.items():
            attr_obj = getattr(self, attr_name)
            if not isinstance(attr_obj, type_):
                raise ValueError(
                    f"{self.__class__.__name__}.{attr_name} has incorrect type, "
                    f"expected {type_} recieved {type(attr_obj)}"
                )

    def to_json(self) -> str:
        return json.dumps(asdict(self))


@dataclass
class RowA(DataMixin):
    name: str
    age: int

    def __post_init__(self):
        self.validate_types()


@dataclass
class RowB(DataMixin):
    species: str

    def __post_init__(self):
        self.validate_types()


ATable = list[RowA]
BTable = list[RowB]


@dataclass
class TablesType(DataMixin):
    row_a: ATable
    row_b: BTable


@dataclass
class Dog(DataMixin):
    name: str
    age: str
    id_: str = field(default_factory=lambda: str(uuid.uuid4()))
    tags: list = field(default_factory=list)

    def __post_init__(self):
        self.validate_types()
        self.validate_age_format()

    def validate_age_format(self):
        try:
            datetime.strptime(self.age, "%Y-%m-%d %H:%M:%S")
        except ValueError as e:
            raise ValueError(
                f"{self.__class__.__name__}.age invalid: {self.age}"
            ) from e


d = Dog("moje", "2024-06-30 16:10:28")
d.to_json()

rows_a = [RowA("mole", 60)]
rows_b = [RowB("roley")]


tables = TablesType(row_a=rows_a, row_b=rows_b)

[r.name for r in tables.row_a]
