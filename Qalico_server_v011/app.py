"""
appy.py : "Qalico" server

"Qalico server" application for bridging quantum annealing machine(QAM) and "Qalico" client applications.
This application uses "Fixtars Amplify" as QAM.
So, You need to get an access token to connect Amplify.
https://amplify.fixstars.com/en/#gettingstarted
Befor starting this server, set your access token to an environment variable "AMPLIFY_API_TOKEN".

usage
    "flask run" to start up this "Qalico" server

@version    0.1.1
@author     Naoki TAGAWA <tagawa.naoki@teragroove.com>
@copyright  2022 Naoki TAGAWA
@license    MIT(https://choosealicense.com/licenses/mit/)
@sponserdby "MITOU Target Porject" of IPA(https://www.ipa.go.jp/jinzai/target/index.html)
"""
from flask import Flask
import ghhops_server as hs
import rhino3dm
import numpy as np

# register hops app as middleware
qalico = Flask(__name__)
hops = hs.Hops(qalico)

# ゾーニング最適化
@hops.component(
    "/zoning_opt",
    name="zoning_OPT",
    description="zoning_OPT\nOptimize Zoning",
    icon="icon/zoning_opt_0131.png",
    inputs=[
        hs.HopsNumber("Cost Rectangle", "cost_rec", "coefficient of rectangle objective function"),
        hs.HopsNumber("Cost Room Direction", "cost_dire", "coefficient of direction objective function"),
        hs.HopsCurve("Grid Curve", "GridCrv", "GridCrv", hs.HopsParamAccess.LIST),
        hs.HopsSurface("Site Grid Surface", "SiteGridSrf", "Grid Surfaces", hs.HopsParamAccess.LIST),
        hs.HopsInteger("X Grid", "X", "Number of x-axis grid"),
        hs.HopsInteger("Y_Grid", "Y", "Number of y-axis grid"),
        hs.HopsInteger("Zone", "Zone", "Zones(Rooms)"),
        hs.HopsNumber("Area", "Area", "Room areas", hs.HopsParamAccess.LIST),
        hs.HopsNumber("Area Tolerance(%)", "Tolerance"),
        hs.HopsNumber("Flow Matrix", "Flow", "Flow matrix", hs.HopsParamAccess.LIST),
        hs.HopsInteger("N Rooms", "N_Rooms", "Rooms faceing North", hs.HopsParamAccess.LIST, optional=True, default=-1),
        hs.HopsInteger("S Rooms", "S_Rooms", "Rooms faceing South", hs.HopsParamAccess.LIST, optional=True, default=-1),
        hs.HopsInteger("E Rooms", "E_Rooms", "Rooms faceing East", hs.HopsParamAccess.LIST, optional=True, default=-1),
        hs.HopsInteger("W Rooms", "W_Rooms", "Rooms faceing West", hs.HopsParamAccess.LIST, optional=True, default=-1),
        hs.HopsBoolean("Run", "Run", "Run Boolean(Toggle)"),
    ],
    outputs=[
        hs.HopsInteger("Zone", "Zone", "Zones(Rooms)"),
        hs.HopsCurve("Grid Curve", "GridCrv", "GridCrv"),
        hs.HopsString("Result", "Result", "Result(text)",),
    ],
)
def zoning_opt(cost_rec, cost_dire, gc, site_gs, x_grid:int, y_grid:int, flow_shape:int, area_list:list, area_tolerance:float, flow:list, n_rooms:list, s_rooms:list, e_rooms:list, w_rooms:list, run:bool):
    if run:
        
        #ERRORS
        if x_grid * y_grid * flow_shape > 3000:
            raise RuntimeError("Unable to process because there are too many Qbits")

        """
        # for debug of coefficient
        def remap(value, source_min, source_max, target_min, target_max):
            value_ratio = (value - source_min)/(source_max - source_min)
            return (value_ratio * (target_max - target_min) + target_min)

        remap_cost_rec = remap(cost_rec, 0, 2, 1, 5)
        remap_cost_dire = remap(cost_dire, 0, 2, 0, 2)
        coe_dict = {"cost_place": 1, "area_constraints": 1, "cost_rec": remap_cost_rec, "cost_dire_sum": remap_cost_dire, "zones_constraints": 2}
        print("coe_dict : {}".format(coe_dict))
        """

        # dire_input
        dire_rooms_dict = {"N": n_rooms, "S": s_rooms, "E": e_rooms, "W": w_rooms}
        for dire, rooms_list in dire_rooms_dict.items():
            if rooms_list[0] == -1:
                dire_rooms_dict[dire] = None

        # Organizing geometry info
        grid_size = gc[0].TryGetPolyline().Length /4
        startPt = [crv.PointAt(0) for crv in gc]
        cornerPt = [srf.PointAt(0,0) for srf in site_gs]
        grid_shape = np.zeros((y_grid*x_grid))
        for i in range(len(startPt)):
            for j in range(len(cornerPt)):
                if startPt[i].X == cornerPt[j].X and startPt[i].Y == cornerPt[j].Y:
                    grid_shape[i] = 1
        grid_shape = np.reshape(grid_shape, [y_grid, x_grid])

        # zoning_optimizer
        import zoning
        rslt = zoning.opt(coe_dict, x_grid, y_grid, flow_shape, flow, dire_rooms_dict, area_list, area_tolerance, grid_shape, grid_size)
        return flow_shape, gc, rslt


# 通路配置最適化
@hops.component(
    "/corridor_layout_opt",
    name="Corridor_Layout_OPT",
    description="Corridor_Layout_OPT\nOptimize corridor layout",
    icon="icon/corridor_layout_opt_0201.png",
    inputs=[
        hs.HopsPoint("Corridor Points", "CP", "cor_points",  hs.HopsParamAccess.LIST),
        hs.HopsPoint("Door Points", "DP", "door_points",  hs.HopsParamAccess.LIST),
        hs.HopsBoolean("Run", "Run", "Run Boolean"),
    ],
    outputs=[
        hs.HopsCurve("Result Lines", "RL", "Result lines",),
    ],
)
def corridor_layout_opt(col_points, door_points, run:bool):
    if run:
        # Rhinoジオメトリーの変換
        x = [p.X for p in col_points]
        y = [p.Y for p in col_points]

        def interval(x:list):
            x_min = min(x)
            x_min_sec = min(x for x in list(set(x)) if x != x_min)
            return x_min_sec - x_min

        gird_size = max([interval(x), interval(y)])
        x_loc = list(map(lambda a: int((a - min(x)) / gird_size), x))
        y_loc = list(map(lambda a: int((a - min(y)) / gird_size), y))
        col_loc_list = np.stack([np.array(x_loc), np.array(y_loc)], -1)
        print("col_loc_list : \n {}".format(col_loc_list))

        door_points_num = list(set([rhino3dm.PointCloud(col_points).ClosestPoint(p) for p in door_points]))
        print("door_points_num : {}".format(door_points_num))

        # 隣接行列の作成

        # 角の点の抽出
        all_diffs = np.expand_dims(col_loc_list, axis=1) - np.expand_dims(col_loc_list, axis=0)
        all_diffs_square = np.square(all_diffs)

        def x_adj_points(diffs): # x軸上の隣接する点の数
            y_ad = diffs[:,1] == 0
            x_ad = diffs[:,0] == 1
            ad = y_ad * x_ad
            return len([i for i, x in enumerate(ad) if x == True])

        def y_adj_points(diffs): # y軸上の隣接する点の数
            y_ad = diffs[:,1] == 1
            x_ad = diffs[:,0] == 0
            ad = y_ad * x_ad
            return len([i for i, x in enumerate(ad) if x == True])

        valid_value_index = []
        count = 0
        for diffs in all_diffs_square:
            if (x_adj_points(diffs) == 1 or x_adj_points(diffs) == 2) and (y_adj_points(diffs) == 1 or y_adj_points(diffs) == 2):
                if count not in door_points_num:
                    valid_value_index.append(count)
            count += 1
        print("valid_value_index : {}".format(valid_value_index))

        for i in door_points_num:
            valid_value_index.append(i)
        
        # 隣接判定
        col_tuple_list = [tuple(i) for i in col_loc_list]
        valid_tuple_list = [tuple(col_loc_list[i]) for i in valid_value_index]
        
        adj_matrix = np.zeros((len(valid_value_index), len(valid_value_index)))
        dire_list = [np.array([0,1]), np.array([0,-1]),
         np.array([1,0]), np.array([-1,0])]

        for i in valid_value_index:
            for j in dire_list:
                if tuple(col_loc_list[i] + j) in col_tuple_list:
                    col = col_loc_list[i] + j
                    while_count = 0
                    while tuple(col) not in valid_tuple_list:
                        col += j
                        while_count += 1
                        if while_count >= max([len(x_loc), len(y_loc)]):
                            break
                    else:
                        adj_matrix[valid_value_index.index(i), valid_tuple_list.index(tuple(col))] = 1
        print("adj_matrix : \n {}".format(adj_matrix))


        import corridor_layout
        rslt_route, rslt_edges = corridor_layout.opt(np.array(valid_tuple_list), list(range(len(valid_tuple_list)-len(door_points_num), len(valid_tuple_list))), adj_matrix)

        rslt_lines = [rhino3dm.LineCurve(col_points[valid_value_index[rslt_edges[i][0]]], col_points[valid_value_index[rslt_edges[i][1]]]) for i in range(len(rslt_edges))]
        return rslt_lines


if __name__ == "__main__":
    qalico.run('0.0.0.0', 5000, debug=True, threaded=True)