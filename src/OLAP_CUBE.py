import pandas as pd
import numpy as np
import sys
from functools import lru_cache

np.random.seed(42)

class OLAPCube:
    def __init__(self, data):
        self.data = data
        self.cube = self._create_cube()
        self._cache = {}

    def _create_cube(self):
        return self.data.set_index([
            'Year', 'Month', 'Country', 'Region',
            'ProductCategory', 'SubCategory'
        ]).sort_index()

    def roll_up(self, level='Year'):
        key = f'rollup_{level}'
        if key not in self._cache:
            self._cache[key] = self.cube['StoreSales'].groupby(level=level).sum()
        return self._cache[key]

    def drill_down(self, dimensions):
        result = self.cube['StoreSales'].groupby(level=list(dimensions)).sum()
        return result

    def slice_operation(self, dimension, value):
        try:
            return self.cube.xs(value, level=dimension)
        except KeyError:
            return None

    def pivot_operation(self, index, columns, values='StoreSales', aggfunc='sum'):
        return pd.pivot_table(
            self.data,
            values=values,
            index=index,
            columns=columns,
            aggfunc=aggfunc
        )

def load_data(size=300):
    return pd.DataFrame({
        'Year': np.random.choice([2021, 2022, 2023], size),
        'Month': np.random.choice(range(1, 13), size),
        'Country': np.random.choice(['USA', 'Mexico', 'Canada'], size),
        'Region': np.random.choice(['West', 'East', 'South'], size),
        'ProductCategory': np.random.choice(['Food', 'Drinks', 'Candy'], size),
        'SubCategory': np.random.choice(['Fruit', 'Milk', 'Chocolate'], size),
        'StoreSales': np.random.uniform(10, 500, size),
        'UnitsSold': np.random.randint(1, 50, size)
    })

def display_result(title, data):
    print(f"\n{'='*50}")
    print(f"{title}")
    print('='*50)
    print(data)

def handle_roll_up(cube):
    result = cube.roll_up('Year')
    display_result("ROLL-UP: Sales by Year", result)

def handle_drill_down(cube):
    result = cube.drill_down(['Year', 'Month'])
    display_result("DRILL-DOWN: Year → Month", result)

def handle_slice(cube):
    category = input("Enter product category (Food/Drinks/Candy): ").strip()
    result = cube.slice_operation('ProductCategory', category)

    if result is not None:
        display_result(f"Slice for '{category}'", result.head(20))
    else:
        print("Error: Category not found!")

def handle_pivot(cube):
    result = cube.pivot_operation('Month', 'ProductCategory')
    display_result("PIVOT TABLE: Sales by Month and Category", result)

def main_menu(cube):
    menu_options = {
        '1': ("Roll-up (aggregation)", handle_roll_up),
        '2': ("Drill-down (detalization)", handle_drill_down),
        '3': ("Slice (cross-section)", handle_slice),
        '4': ("Pivot (rotation)", handle_pivot),
    }

    while True:
        print("\n" + "="*50)
        print("OLAP CUBE MENU")
        print("="*50)
        for key, (desc, _) in menu_options.items():
            print(f"{key} — {desc}")
        print("0 — Exit")
        print("="*50)

        choice = input("Select option: ").strip()

        if choice == '0':
            print("Exiting...")
            sys.exit()
        elif choice in menu_options:
            _, handler = menu_options[choice]
            handler(cube)
        else:
            print("Error: Invalid choice!")

def main():
    print("Loading data...")
    data = load_data()
    cube = OLAPCube(data)
    print("Success: OLAP cube created successfully!\n")

    main_menu(cube)

if __name__ == "__main__":
    main()
