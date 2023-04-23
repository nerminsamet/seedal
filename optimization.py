from ortools.linear_solver import pywraplp
import numpy as np
from ortools.sat.python import cp_model


def apply_optimization(all_pairs, data_stats, pair_scores, reduction_size, area_threshold):

    selected_indexes = lineer_solver(pair_scores, edge_number=reduction_size)
    ls_scans = [all_pairs[ind] for ind in selected_indexes]

    unique_scans  = []
    for scan_t in ls_scans:
        if scan_t[0] not in unique_scans:
            unique_scans.append(scan_t[0])
        if scan_t[1] not in unique_scans:
            unique_scans.append(scan_t[1])

    node_len = len(unique_scans)
    sub_graph_areas = []
    sub_graph_scores = []
    sub_graph_pairs = []
    sub_aff_mat = np.zeros((node_len,node_len))
    for ind, us in enumerate(unique_scans):
        for  ind2 in range(ind+1,len(unique_scans)):
            if (unique_scans[ind], unique_scans[ind2]) in all_pairs:
                ii = all_pairs.index((unique_scans[ind], unique_scans[ind2]))
            else:
                ii = all_pairs.index((unique_scans[ind2], unique_scans[ind]))
            sc = pair_scores[ii]
            sub_graph_scores.append(sc)
            sub_aff_mat[ind, ind2] = sc
            sub_graph_pairs.append(all_pairs[ii])

    for us in unique_scans:
        sub_graph_areas.append(data_stats[us]['area'])

    selected_edge_indexes, selected_node_indexes =\
        lineer_solver_by_node_weight(sub_aff_mat, sub_graph_areas, area_threshold=area_threshold)

    total_area = 0
    final_scenes = []
    for i in selected_node_indexes:
        print(unique_scans[i])
        final_scenes.append(unique_scans[i])
        total_area += data_stats[unique_scans[i]]['area']

    print(final_scenes)
    print(total_area)

    return final_scenes


def create_reduction_data_model(edges, edge_number):
    data = {}
    edges = [np.float64(i) for i in edges]
    num_var = len(edges)
    num_of_xs = [1.] * num_var

    # constraits on sum of pairwise areas
    data['constraint_coeffs'] = [
        num_of_xs
    ]
    data['bounds'] = [edge_number]

    data['obj_coeffs'] = edges # similarity edges

    data['num_vars'] = num_var
    data['num_constraints'] = 1

    return data


def create_linear_data_model(edge_weights, node_weights, area_threshold):
    data = {}
    data['edge_weights'] = edge_weights
    data['node_weights'] =  node_weights
    data['area_threshold'] = area_threshold
    data['len'] = len(data['node_weights'])
    return data


def lineer_solver(edges, edge_number):

    data = create_reduction_data_model(edges, edge_number)
    # Create the mip solver with the SCIP backend.
    solver = pywraplp.Solver.CreateSolver('SCIP')
    if not solver:
        return

    # infinity = solver.infinity()
    x = {}
    for j in range(data['num_vars']):
        x[j] = solver.BoolVar('x[%i]' % j)

    print('Number of variables =', solver.NumVariables())

    for i in range(data['num_constraints']):
        constraint = solver.RowConstraint(0, data['bounds'][i], '')
        for j in range(data['num_vars']):
            constraint.SetCoefficient(x[j], data['constraint_coeffs'][i][j])
    print('Number of constraints =', solver.NumConstraints())

    objective = solver.Objective()
    for j in range(data['num_vars']):
        objective.SetCoefficient(x[j], data['obj_coeffs'][j])
    objective.SetMaximization()

    status = solver.Solve()
    selected_indexes= []
    if status == pywraplp.Solver.OPTIMAL:
        print('Objective value =', solver.Objective().Value())
        for j in range(data['num_vars']):
            if x[j].solution_value() > 0.0:
                print(x[j].name(), ' = ', x[j].solution_value())
                selected_indexes.append(j)
        print()
        print('Problem solved in %f milliseconds' % solver.wall_time())
        print('Problem solved in %d iterations' % solver.iterations())
        print('Problem solved in %d branch-and-bound nodes' % solver.nodes())
    else:
        print('The problem does not have an optimal solution.')

    return selected_indexes


def lineer_solver_by_node_weight(edge_weights, node_weights, area_threshold):

    data = create_linear_data_model(edge_weights,node_weights,area_threshold)
    solver = pywraplp.Solver.CreateSolver('SCIP')

    node_len = data['len']
    if not solver:
        return

    # Variables
    y = {}
    for i in range(node_len):
        for j in range(i+1, node_len):
            y[(i, j)] = solver.BoolVar(f'y_{i}_{j}')

    x = {}
    for j in range(node_len):
        x[j] = solver.BoolVar(f'x[{j}]')

    # Constraints
    for i in range(node_len):
        for j in range(i+1, node_len):
            solver.Add(y[(i, j)] <= x[i])
            solver.Add(y[(i, j)] <= x[j])

    solver.Add(sum(x[i] * data['node_weights'][i] for i in range(node_len)) <= data['area_threshold'])

    solver.Maximize(solver.Sum([y[i,j]*data['edge_weights'][i,j] for i in range(node_len) for j in range(i+1,node_len)]))

    status = solver.Solve()

    selected_node_indexes = []
    selected_edge_indexes = []
    if status == pywraplp.Solver.OPTIMAL or status == cp_model.FEASIBLE:
        print('Objective value =', solver.Objective().Value())
        for j in range(node_len):
            if x[j].solution_value() > 0.0:
                print(x[j].name(), ' = ', x[j].solution_value())
                selected_node_indexes.append(j)

        for i in range(node_len):
            for j in range(i+1, node_len):
                if y[i,j].solution_value() > 0.0:
                    print(y[i,j].name(), ' = ', y[i,j].solution_value())
                    selected_edge_indexes.append((i,j))

        print()
        print('Problem solved in %f milliseconds' % solver.wall_time())
        print('Problem solved in %d iterations' % solver.iterations())
        print('Problem solved in %d branch-and-bound nodes' % solver.nodes())
        print('Number of bins used:')

        print('Time = ', solver.WallTime(), ' milliseconds')
    else:
        print('The problem does not have an optimal solution.')

    return selected_edge_indexes, selected_node_indexes


