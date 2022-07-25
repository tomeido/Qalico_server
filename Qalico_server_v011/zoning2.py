"""
zoning.py : "Qalico" zoning function
    called from app.py

@version    0.1.1
@author     Naoki TAGAWA <tagawa.naoki@teragroove.com>
@copyright  2022 Naoki TAGAWA
@license    MIT(https://choosealicense.com/licenses/mit/)
@sponserdby "MITOU Target Porject" of IPA(https://www.ipa.go.jp/jinzai/target/index.html)
"""
from os import environ
import numpy as np
import time
import itertools
from pyomo.environ import *
from pyrsistent import v
import matplotlib.pyplot as plt

def opt(coe_dict:dict, x_grid:int, y_grid:int, flow_shape:int, flow:list, dire_rooms_dict:dict, area_list:list, area_tolerance:float, grid_shape:np.ndarray, grid_size:float):

    t1 = time.time()    # 処理前時刻 처리 전 시각

    # construct flow matrix
    zones = flow_shape + 1
    flo = np.array(flow)
    flo = np.triu(flo.reshape([flow_shape,flow_shape]), k=1) # np.triu(flo.reshape([5,5], k=1)
    flo_self = np.identity(flow_shape,  int) # 단위행렬
    flo_sum = (flo + flo.T + 100 * flo_self) / 100 # 행렬 + 역행렬 + 100*단위행렬 ???

    print("\n--------Setting Values in main process--------")
    print("zones : {}".format(zones))
    print("flo_sum : \n{}".format(flo_sum))
    print("area_list : {}".format(area_list))
    print("grid_shape : \n{}".format(grid_shape))
    print("x_grid : {}".format(x_grid))
    print("y_grid : {}".format(y_grid))
    print("grid_size : {}".format(grid_size))
    print("n_rooms : {}".format(dire_rooms_dict["N"]))
    print("s_rooms : {}".format(dire_rooms_dict["S"]))
    print("e_rooms : {}".format(dire_rooms_dict["E"]))
    print("w_rooms : {}".format(dire_rooms_dict["W"]))

    # grid settings
    x_nlocations = int(x_grid)
    y_nlocations = int(y_grid)
    nlocations = x_nlocations * y_nlocations
    grid_area = ga = np.square(grid_size) / 1000000

    def gen_locations(x_n, y_n) -> list:
        # 좌표
        grid_list = np.empty((y_n, x_n, 2))
        for i in range(y_n):
            for j in range(x_n):
                grid_list[i][j][0] = i
                grid_list[i][j][1] = j
        grid_list = np.reshape(grid_list, [x_n*y_n, 2])
        print("grid_list : \n{}".format(grid_list))
        # 줄행렬
        all_diffs = np.expand_dims(grid_list, axis=1) - np.expand_dims(grid_list, axis=0)
        distances_ori = np.sqrt(np.sum(all_diffs ** 2, axis=-1))
        distances = distances_ori / np.amax(distances_ori)

        return grid_list, distances, distances_ori
                    
    grid_list, distances, distances_ori = gen_locations(x_nlocations, y_nlocations)


    model = ConcreteModel()

    # 변수 배열 생성
    model.zones_param = Param(mutable=True, default=zones)
    model.nlocations_param = Param(mutable=True, default=nlocations)
    model.zones_range = RangeSet(1, model.zones_param)
    model.nlocations_range = RangeSet(1, model.nlocations_param)
    model.q = Var(model.zones_range, model.nlocations_range, domain=Binary, initialize=1)

    
    # 비활성 그리드 배제


    # 클러스터화·존 배치
    flo_dis = np.einsum("ik,jl->ijkl",flo_sum,distances)
    cost_place_matrix = flo_dis.reshape((zones-1)*nlocations, (zones-1)*nlocations)
    cost_place_matrix = np.array(np.triu(cost_place_matrix + np.triu(cost_place_matrix.T, k=1))).tolist()

    model.I_param = Param(mutable=True, default=len(cost_place_matrix))
    model.J_param = Param(mutable=True, default=len(cost_place_matrix[0]))
    model.I_range = RangeSet(1, model.I_param)
    model.J_range = RangeSet(1, model.J_param)
    model.I_set = Set(initialize=model.I_range)
    model.J_set = Set(initialize=model.J_range)
  
    model.zn_param = Param(mutable=True, default=(zones-1)*nlocations)
    model.zn_range = RangeSet(1, model.zn_param)
    model.cost_place_var = Var(model.zn_range, model.zn_range, domain=Binary, initialize=1)
    
    model.cost_place = quicksum(cost_place_matrix(i,j) for i in range(len(cost_place_matrix)) for j in range(len(cost_place_matrix[0])) * model.cost_place_var[i,j].value for i in model.I_range for j in model.J_range) / (((zones-1)*nlocations) ** 2)


    # 직사각형화
    grid_n = np.arange(nlocations)
    grid_loc = grid_n.reshape([y_nlocations, x_nlocations])
    grid_loc_edge_x = [grid_loc[0][i] for i in range(x_nlocations-1)] + [grid_loc[-1][i] for i in range(x_nlocations-1)]
    grid_loc_edge_y = [grid_loc[i][0] for i in range(1, y_nlocations-1)] + [grid_loc[i][-1] for i in range(y_nlocations)]
    grid_loc_edge = grid_loc_edge_x + grid_loc_edge_y
    grid_n_t = np.reshape(grid_loc.T, [nlocations])

    k_range = range(1, nlocations-y_nlocations)
    i_range = range(1, zones)

    model.cost_rec_row = Param(mutable=True)
    model.cost_rec_column = Param(mutable=True)
    model.cost_rec_row = quicksum(((model.q[k,grid_n_t[i]].value + (-model.q[k,grid_n_t[i+y_nlocations]].value)) ** 2) for k in i_range for i in k_range)
    model.cost_rec_column = quicksum((((model.q[k,i].value + (-model.q[k,i+x_nlocations].value)) ** 2) for k in i_range for i in k_range))

    zones1_range = range(1, zones-1)
    grid_loc_edge_range = range(1, len(grid_loc_edge))
    model.cost_rec_edge = Param(mutable=True)
    model.cost_rec_edge = quicksum(model.q[k,i].value for k in zones1_range for i in grid_loc_edge_range)
    model.cost_rec = (model.cost_rec_row + model.cost_rec_column + model.cost_rec_edge) / (nlocations*2 + x_nlocations + y_nlocations)
    

    # 방향 배치
    def cardinal(model):
        cost_dire_sum = 0
        dire_room_n = 0
        for direction, dire_rooms_list in dire_rooms_dict.items():
            if dire_rooms_list is None:
                cost_dire_sum += 0
            elif direction == "N": # Face north
                cost_dire_sum += quicksum(((y_nlocations-1)-grid_list[j][0] * model.q[dire_rooms_list[1],j].value) / (area_list[dire_rooms_list[i]]/ga) for j in range(1,nlocations) for i in range(1,len(dire_rooms_list)))
                dire_room_n += len(dire_rooms_list)
            elif direction == "E": # Face east
                cost_dire_sum += quicksum(((y_nlocations-1)-grid_list[j][1] * model.q[dire_rooms_list[i],1].value) / (area_list[dire_rooms_list[i]]/ga) for j in range(1,nlocations) for i in range(1,len(dire_rooms_list)))
                dire_room_n += len(dire_rooms_list)
            elif direction == "S": # Face south
                cost_dire_sum += quicksum((grid_list[j][0] * model.q[dire_rooms_list[i],j].value) / (area_list[dire_rooms_list[i]]/ga) for j in range(1,nlocations) for i in range(1,len(dire_rooms_list)))
                dire_room_n += len(dire_rooms_list)
            elif direction == "W": # Face west
                cost_dire_sum += quicksum((grid_list[j][1] * model.q[dire_rooms_list[i],j].value) / (area_list[dire_rooms_list[i]]/ga) for j in range(1,nlocations) for i in range(1,len(dire_rooms_list)))
                dire_room_n += len(dire_rooms_list)
            else:
                cost_dire_sum = 0
                dire_room_n = 1
        if all([i == None for i in dire_rooms_dict.values()]):
            dire_room_n = 1
        cost_dire_sum /= dire_room_n
        return cost_dire_sum
    
    cost_dire_sum = cardinal(model)
    model.cost_dire_sum = Param(mutable=True)
    model.cost_dire_sum = cost_dire_sum
    

    # 면적 제약
    model.area_constraints = ConstraintList()
    for n in range(len(area_list)):
        model.area_constraints.add(((area_list[n]/ga - sum([model.q[n+1,i+1] for i in range(nlocations)])) ** 2 ) <= (area_list[n]/ga * area_tolerance/100) ** 2)


    # 중복 방지 제약
    model.zones_constraints = ConstraintList()
    for n in range(nlocations):
        model.zones_constraints.add(sum(model.q[:,n+1]) == 1)   
    # model.zones_constraints.pprint()
    

    # # penalty coefficients scale adjustment
    # coe_dict["cost_place"] *= 1
    # coe_dict["cost_rec"] *= 10
    # coe_dict["cost_dire_sum"] /= 10
    # coe_dict["area_constraints"] /= 10
    # coe_dict["zones_constraints"] /= 0.1
    

    # # constraints containerization
    # model.constraints = coe_dict["zones_constraints"] * model.zones_constraints + coe_dict["area_constraints"] * model.area_constraints


    # model
    model.obj = Objective(expr=model.cost_place + model.cost_rec + model.cost_dire_sum, sense=minimize)
    instance = model.create_instance()
    opt = SolverFactory('ipopt')
    
    result = opt.solve(instance,tee=True)
    print('OF= ',value(instance.obj))
    print('cost_place : ',value(model.cost_place))
    print('cost_rec : ',value(model.cost_rec))
    print('cost_dire_sum : ',value(model.cost_dire_sum))
    
    print(value(model.q[1,1]))

    result_q = np.empty((zones, nlocations))
    for i in range(1,zones):
        for j in range(1,nlocations):
            result_q[i][j] = value(model.q[i,j])
    
    print(result_q)
    result_q = np.array(result_q).transpose((1,0))

    print("\n--------result--------")
    ans_grid = result_q[:, np.newaxis, :].reshape([y_nlocations, x_nlocations, zones]).transpose(2,0,1)
    print("answer grid : \n{}".format(ans_grid))
    print("answer grid shape : {}".format(ans_grid.shape))


    # f, ax = plt.subplots(1, 1)
    # print(result)
    # for i in instance.zones_set:
    #     for j in instance.nlocations_set:
    #         X=value(instance.q[i,j])
    #         if X==1:
    #             plt.scatter(i,j,s=55,color='black')
    #         else:
    #             plt.scatter(i,j,s=10,color='red')
    # plt.axis('off')
    # plt.show()
    # f.savefig('NQueen.png', format='png', dpi=1200)
    
    # # result
    # energy, values, is_feasible = result[0].energy, result[0].values, result[0].is_feasible
    # q_values = ap.decode_solution(q, values, 1)
    # q_values = np.array(q_values).transpose((1,0))

    # print("\n--------result--------")
    # ans_grid = q_values[:, np.newaxis, :].reshape([y_nlocations, x_nlocations, zones]).transpose(2,0,1)
    # print("answer grid : \n{}".format(ans_grid))
    # print("answer grid shape : {}".format(ans_grid.shape))   # 各ゾーンごとのグリッド表現 각 구역별 그리드 표현

    # route = np.where(np.array(q_values) == 1)[1]
    # routes_list = [np.where(np.array(route) == i)[0] for i in range(zones-1)]

    # # show result
    # for i in range(zones-1):
    #     print("\n")
    #     print("zone" + str(i))
    #     print(routes_list[i])
    #     print(len(routes_list[i]) * ga)
    # print("\n")
    # print(f"is_feasible : {is_feasible}")
    # # print(f"check_constraints : {model.check_constraints(values)}")
    # print(f"energy : {energy}")

    # t2 = time.time()    # 処理後時刻 처리 후 시각
    # elapsed_time = t2 -t1
    # print(f"経過時間 : {elapsed_time}")

    # ans_list = [[int(n) for n in routes_list[i]] for i in range(len(routes_list))]
    # answer_list = [[str(i) + '-' + str(s) for s in ans_list[i]] for i in range(len(ans_list))]
    # a = list(itertools.chain.from_iterable(answer_list))


    # return a
