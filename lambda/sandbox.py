from dataclasses import dataclass


@dataclass
class RowA:
    name: str
    age: int


@dataclass
class RowB:
    species: str


ATable = list[RowA]
BTable = list[RowB]


@dataclass
class TablesType:
    row_a: ATable
    row_b: BTable | None = None


rows_a = [RowA("mole", 60)]
rows_b = [RowB("roley")]


tables = TablesType(row_a=rows_a, row_b=rows_b)


[r.name for r in tables.row_a]
