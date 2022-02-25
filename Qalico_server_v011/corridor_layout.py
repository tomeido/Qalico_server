"""
corrior_layout.py : "Qalico" corridor layout function
    called from app.py

@version    0.1.1
@author     Naoki TAGAWA <tagawa.naoki@teragroove.com>
@copyright  2022 Naoki TAGAWA
@license    MIT(https://choosealicense.com/licenses/mit/)
@sponserdby "MITOU Target Porject" of IPA(https://www.ipa.go.jp/jinzai/target/index.html)
"""
from os import environ, error

import amplify as ap
import numpy as np
from numpy.lib.index_tricks import diag_indices


def opt(col_loc_list, door_point_nums, adj_matrix):

    print("col_loc_list : \n {}".format(col_loc_list))
    print("door_grid : {}".format(door_point_nums))

    num_locs = len(col_loc_list)
    v_indices = np.arange(num_locs)
    half_num_locs = (num_locs + 1) // 2
    u_indices = np.array(door_point_nums)
    v_diff_u_locations = np.setdiff1d(v_indices, u_indices)
    v_diff_u_indices = np.array([np.where(v_indices == u) for u in v_diff_u_locations])

    print("v_indices : {}".format(v_indices))
    print("u_indices : {}".format(u_indices))
    print("v_diff_u_locations : {}".format(v_diff_u_locations))
    print("v_diff_u_indices : \n {}".format(v_diff_u_indices))

    all_diffs = np.expand_dims(col_loc_list, axis=1) - np.expand_dims(col_loc_list, axis=0)
    uv_edges = distances = np.sqrt(np.sum(all_diffs ** 2, axis=-1))
    edges_max = np.amax(uv_edges)

    gen = ap.SymbolGenerator(ap.BinaryPoly)
    x_v = gen.array(num_locs, half_num_locs) # x_v[v, i] = 1: 頂点 v_locations[v] が根から深さ i の位置に__ある__
    x_uv = gen.array(num_locs, num_locs, half_num_locs) # x_uv[u, v, i] = 1: 辺 v_locations[u] -> v_locations[v] が根から深さ i の位置に__ある__
    for u in range(num_locs):
        for v in range(num_locs):
            if adj_matrix[u, v] != 1.0:
                x_uv[u, v] = 0
                uv_edges[u, v] = 0
    
    cor_constraints = ap.BinaryConstraints()

    # A1: 根 (x_v0 == 1) はただ一つ
    cor_constraints += ap.constraint.one_hot(x_v[:, 0])  # x_v[:, 0].sum() == 1

    # A2: U 内の頂点は必ず木に1回含まれる
    for idx in u_indices:
        # x_v[idx].sum() == 1
        cor_constraints += ap.constraint.one_hot(x_v[idx])

    # A3: U にない頂点は木にただ一回含まれるか含まれない
    for idx in v_diff_u_indices:
        # x_v[idx].sum() <= 1
        cor_constraints += ap.constraint.less_equal(x_v[idx], 1)

    # A4: 根から深さ i + 1 の位置の頂点 v は必ず深さ i の位置の頂点 u と辺で結ばれていなければならない
    for v in range(num_locs):
        for i in range(half_num_locs - 1):
            # x_v[v, i + 1] == x_uv[:, v, i].sum()
            cor_constraints += ap.constraint.equal_to(x_v[v, i + 1] - x_uv[:, v, i].sum(), 0)

    # A5: 
    for u in range(num_locs):
        for v in range(num_locs):
            for i in range(half_num_locs - 1):
                # x_uv[u, v, i] = 0 || (x_uv[u, v, i] = 1 && x_v[v, i] == 1)
                cor_constraints += ap.constraint.penalty(x_uv[u, v, i] * (1 - x_v[u, i]))

    # target function: 木に含まれる辺 u -> v の距離の総和
    cost = ap.einsum("uv,uvi->", uv_edges, x_uv)

    # model
    model =  1 * cost + 1 * edges_max * cor_constraints

    # client settings
    client = ap.client.FixstarsClient()
    client.token = environ['AMPLIFY_API_TOKEN']
    client.parameters.timeout = 20000    # タイムアウト10秒

    # solve
    solver = ap.Solver(client)
    solver.filter_solution = True
    result = solver.solve(model)
    if len(result) == 0:
        raise RuntimeError("Any one of constraints is not satisfied")

    # result
    energy, values, is_feasible = result[0].energy, result[0].values, result[0].is_feasible
    x_v_values = x_v.decode(values)
    x_uv_values = x_uv.decode(values)
    constraint_check = model.check_constraints(result[0].values)
    # print(constraint_check)

    print("-------result-------")
    print(f"is_feasible : {is_feasible}")
    print("energy : {}".format(energy))
    # print(x_v_values)
    # print(x_uv_values)

    print("-------node-------")
    print("#NODE #DEPTH")
    for v, i in zip(*np.where(x_v_values == 1)):
        print(f"{v_indices[v]:>5} {i:>6}")

    print("-------edge-------")
    for u, v, _ in zip(*np.where(x_uv_values == 1)):
        print(f"{v_indices[u]:>3} -> {v_indices[v]:>3}: {uv_edges[u, v]}")
    
    edges = [[u, v] for u, v, _ in zip(*np.where(x_uv_values == 1))]
    
    route = []
    for v in np.where(x_v_values == 1)[0]:
        route.append(int(v_indices[v]))
    print(route)
    
    return route, edges
