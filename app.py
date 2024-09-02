import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import distance_matrix
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
from flask import Flask, render_template, request, redirect, url_for, session
import os

app = Flask(__name__)
app.secret_key = os.urandom(24)

# Dummy user database
users = {
    'user1': 'password1',
    'user2': 'password2'
}

# Adjusted data setup for more realistic scenarios
num_warehouses = 20
num_trucks = 10
num_demands = 40

# Generate warehouse locations
warehouses = pd.DataFrame({
    'WarehouseID': range(1, num_warehouses + 1),
    'X': np.random.randint(0, 500, num_warehouses),
    'Y': np.random.randint(0, 500, num_warehouses)
})

# Generate truck data
trucks = pd.DataFrame({
    'TruckID': range(101, 101 + num_trucks),
    'Capacity': [30] * num_trucks,
    'AC': np.random.choice([True, False], num_trucks),
    'CurrentWarehouseID': np.random.choice(range(1, num_warehouses + 1), num_trucks),
    'Status': np.random.choice(['available', 'in_transit'], num_trucks),
    'CurrentLoad': np.zeros(num_trucks),
    'CurrentPath': [[] for _ in range(num_trucks)]
})

# Generate realistic demands
demands = pd.DataFrame({
    'DemandID': range(1001, 1001 + num_demands),
    'SourceID': np.random.choice(range(1, num_warehouses + 1), num_demands),
    'DestinationID': np.random.choice(range(1, num_warehouses + 1), num_demands),
    'ACRequired': np.random.choice([True, False], num_demands),
    'Priority': np.random.choice(['high', 'low'], num_demands),
    'Quantity': np.random.randint(1, 11, num_demands)
})

# Ensure source and destination are not the same
demands = demands[demands['SourceID'] != demands['DestinationID']].reset_index(drop=True)

# Calculate distance matrix based on warehouse coordinates
coords = warehouses[['X', 'Y']].to_numpy()
dist_matrix = distance_matrix(coords, coords)

# Convert distance to time matrix (assuming speed is 40 km/h)
speed = 40  # km/h
time_matrix = dist_matrix / speed

# Ensure the matrix is symmetric
for i in range(len(time_matrix)):
    for j in range(i + 1, len(time_matrix)):
        time_matrix[j][i] = time_matrix[i][j]

# Update time windows to 6 hours (converted to minutes)
time_windows = [(0, 6 * 60) for _ in range(len(demands))]

# ADMM function for demand assignment, considering AC compatibility, capacity, priority, and cost
def assign_demands_admm(trucks, demands):
    num_trucks = len(trucks)
    num_demands = len(demands)
    z = np.zeros((num_trucks, num_demands))  # Assignment matrix (trucks x demands)
    u = np.zeros((num_trucks, num_demands))  # Dual variables for ADMM
    rho = 1.0  # Penalty parameter
    max_iter = 100  # Maximum iterations
    tolerance = 1e-4  # Tolerance for convergence

    # Define weights for different objectives
    cost_weight = 0.4
    priority_weight = 0.3
    distance_weight = 0.3

    # Normalize priorities, costs, and distances
    demand_priority = demands['Priority'].apply(lambda x: 1 if x == 'high' else 0)
    normalized_priority = demand_priority / demand_priority.sum()
    normalized_cost = demands['Quantity'] / demands['Quantity'].sum()
    distance_matrix_normalized = dist_matrix / dist_matrix.sum()

    for iteration in range(max_iter):
        z_old = z.copy()

        # Update z (assignment variable)
        for i in range(num_trucks):
            truck_capacity = trucks.iloc[i]['Capacity'] - trucks.iloc[i]['CurrentLoad']
            for j in range(num_demands):
                if (demands.iloc[j]['ACRequired'] == trucks.iloc[i]['AC'] and
                        demands.iloc[j]['Quantity'] <= truck_capacity and
                        trucks.iloc[i]['Status'] == 'available'):
                    cost_term = normalized_cost[j] * cost_weight
                    priority_term = normalized_priority[j] * priority_weight
                    distance_term = distance_matrix_normalized[trucks.iloc[i]['CurrentWarehouseID'] - 1, demands.iloc[j]['SourceID'] - 1] * distance_weight
                    z[i, j] = 1 - (cost_term + priority_term + distance_term)
                    truck_capacity -= demands.iloc[j]['Quantity']
                else:
                    z[i, j] = 0

        # Convergence check
        if np.linalg.norm(z - z_old) < tolerance:
            print("Convergence achieved.")
            break

    return z

# Function to calculate and optimize routes using OR-Tools
def solve_vrp(assigned_matrix, trucks, demands):
    num_vehicles = len(trucks)
    num_locations = len(demands)

    # Create the routing index manager
    manager = pywrapcp.RoutingIndexManager(num_locations, num_vehicles, 0)

    # Create Routing Model
    routing = pywrapcp.RoutingModel(manager)

    # Create and register a transit callback
    def time_callback(from_index, to_index):
        # Convert from routing variable Index to distance matrix NodeIndex.
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return time_matrix[demands.iloc[from_node]['SourceID'] - 1][demands.iloc[to_node]['SourceID'] - 1] * 60  # convert hours to minutes

    transit_callback_index = routing.RegisterTransitCallback(time_callback)

    # Define cost of each arc
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    # Add Time Windows constraint
    routing.AddDimension(
        transit_callback_index,
        0,  # no slack
        6 * 60,  # 6 hours in minutes
        True,  # start cumul to zero
        "Time")

    time_dimension = routing.GetDimensionOrDie("Time")
    for i in range(num_locations):
        index = manager.NodeToIndex(i)
        time_dimension.CumulVar(index).SetRange(0, 6 * 60)

    # Add Capacity constraint
    def demand_callback(from_index):
        # Ensure the index is within range
        demand_index = manager.IndexToNode(from_index)
        if demand_index < len(demands):
            return demands.iloc[demand_index]['Quantity']
        return 0

    demand_evaluator_index = routing.RegisterUnaryTransitCallback(demand_callback)
    routing.AddDimensionWithVehicleCapacity(
        demand_evaluator_index,
        0,  # no slack
        [truck['Capacity'] for _, truck in trucks.iterrows()],  # truck capacities
        True,  # start cumul to zero
        "Capacity")

    # Setting first solution heuristic
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    search_parameters.time_limit.seconds = 30

    # Solve the problem
    solution = routing.SolveWithParameters(search_parameters)
    if solution:
        print("Solution on the routes:")
        total_distance = 0
        total_time = 0
        plt.figure(figsize=(12, 8))
        for vehicle_id in range(num_vehicles):
            index = routing.Start(vehicle_id)
            route_load = 0
            route_distance = 0
            route_time = 0
            route_plan = f'Route for vehicle {vehicle_id}:\n'
            route_x, route_y = [], []
            while not routing.IsEnd(index):
                node_index = manager.IndexToNode(index)
                demand_id = demands.iloc[node_index]['DemandID']
                if assigned_matrix[vehicle_id][node_index] > 0:  # Checking if the demand is assigned to this truck
                    route_load += demands.iloc[node_index]['Quantity']
                next_index = solution.Value(routing.NextVar(index))
                if not routing.IsEnd(next_index):
                    next_node_index = manager.IndexToNode(next_index)
                    distance = time_matrix[demands.iloc[node_index]['SourceID'] - 1][demands.iloc[next_node_index]['SourceID'] - 1]
                    route_distance += distance
                    route_time += distance * 60  # converting hours to minutes
                    route_plan += f'{demands.iloc[node_index]["SourceID"]} ({route_load} units) -> '
                    route_x.append(warehouses.iloc[demands.iloc[node_index]['SourceID'] - 1]['X'])
                    route_y.append(warehouses.iloc[demands.iloc[node_index]['SourceID'] - 1]['Y'])
                index = next_index
            route_plan += f'{demands.iloc[node_index]["DestinationID"]} (final)\n'
            route_x.append(warehouses.iloc[demands.iloc[manager.IndexToNode(routing.Start(vehicle_id))]['SourceID'] - 1]['X'])
            route_y.append(warehouses.iloc[demands.iloc[manager.IndexToNode(routing.Start(vehicle_id))]['SourceID'] - 1]['Y'])
            plt.plot(route_x, route_y, label=f'Route for truck {trucks.iloc[vehicle_id]["TruckID"]}')
            print(f'{route_plan}\nDistance of the route: {route_distance:.2f} km\nTime of the route: {route_time:.2f} minutes')
            total_distance += route_distance
            total_time += route_time
        print(f'Total distance of all routes: {total_distance:.2f} km\nTotal time of all routes: {total_time:.2f} minutes')
        plt.scatter(warehouses['X'], warehouses['Y'], c='blue', label='Warehouses')
        for i, row in warehouses.iterrows():
            plt.text(row['X'], row['Y'], f'{row["WarehouseID"]}', fontsize=12)
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        plt.title('Truck Routes')
        plt.legend()
        plt.show()
    else:
        print("No solution found!")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/trucks')
def view_trucks():
    return render_template('trucks.html', trucks=trucks)

@app.route('/warehouses')
def view_warehouses():
    return render_template('warehouses.html', warehouses=warehouses)

@app.route('/demands')
def view_demands():
    return render_template('demands.html', demands=demands)

@app.route('/warehouse/<int:warehouse_id>')
def warehouse_detail(warehouse_id):
    warehouse_trucks = trucks[trucks['CurrentWarehouseID'] == warehouse_id]
    warehouse_demands = demands[demands['SourceID'] == warehouse_id]
    return render_template('warehouse_detail.html', trucks=warehouse_trucks, demands=warehouse_demands)

@app.route('/add_truck', methods=['POST'])
def add_truck():
    new_truck = {
        'TruckID': request.form['TruckID'],
        'Capacity': request.form['Capacity'],
        'AC': request.form['AC'] == 'True',
        'CurrentWarehouseID': request.form['CurrentWarehouseID'],
        'Status': 'available',
        'CurrentLoad': 0,
        'CurrentPath': []
    }
    trucks.loc[len(trucks)] = new_truck
    return redirect(url_for('view_trucks'))

@app.route('/add_warehouse', methods=['POST'])
def add_warehouse():
    new_warehouse = {
        'WarehouseID': request.form['WarehouseID'],
        'X': request.form['X'],
        'Y': request.form['Y']
    }
    warehouses.loc[len(warehouses)] = new_warehouse
    return redirect(url_for('view_warehouses'))

@app.route('/add_demand', methods=['POST'])
def add_demand():
    new_demand = {
        'DemandID': request.form['DemandID'],
        'SourceID': request.form['SourceID'],
        'DestinationID': request.form['DestinationID'],
        'ACRequired': request.form['ACRequired'] == 'True',
        'Priority': request.form['Priority'],
        'Quantity': request.form['Quantity']
    }
    demands.loc[len(demands)] = new_demand
    return redirect(url_for('view_demands'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if username in users and users[username] == password:
            session['username'] = username
            return redirect(url_for('index'))
        else:
            return "Invalid credentials"
    return render_template('login.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if username not in users:
            users[username] = password
            return redirect(url_for('login'))
        else:
            return "User already exists"
    return render_template('signup.html')

@app.route('/logout')
def logout():
    session.pop('username', None)
    return redirect(url_for('login'))

if __name__ == '__main__':
    app.run(debug=True)
