#
#         batlow
#                   www.fabiocrameri.ch/colourmaps
from matplotlib.colors import LinearSegmentedColormap

cm_data = [
    [0.0051932, 0.098238, 0.34984],
    [0.0090652, 0.10449, 0.35093],
    [0.012963, 0.11078, 0.35199],
    [0.01653, 0.11691, 0.35307],
    [0.019936, 0.12298, 0.35412],
    [0.023189, 0.12904, 0.35518],
    [0.026291, 0.13504, 0.35621],
    [0.029245, 0.14096, 0.35724],
    [0.032053, 0.14677, 0.35824],
    [0.034853, 0.15256, 0.35923],
    [0.037449, 0.15831, 0.36022],
    [0.039845, 0.16398, 0.36119],
    [0.042104, 0.16956, 0.36215],
    [0.044069, 0.17505, 0.36308],
    [0.045905, 0.18046, 0.36401],
    [0.047665, 0.18584, 0.36491],
    [0.049378, 0.19108, 0.36581],
    [0.050795, 0.19627, 0.36668],
    [0.052164, 0.20132, 0.36752],
    [0.053471, 0.20636, 0.36837],
    [0.054721, 0.21123, 0.36918],
    [0.055928, 0.21605, 0.36997],
    [0.057033, 0.22075, 0.37075],
    [0.058032, 0.22534, 0.37151],
    [0.059164, 0.22984, 0.37225],
    [0.060167, 0.2343, 0.37298],
    [0.061052, 0.23862, 0.37369],
    [0.06206, 0.24289, 0.37439],
    [0.063071, 0.24709, 0.37505],
    [0.063982, 0.25121, 0.37571],
    [0.064936, 0.25526, 0.37636],
    [0.065903, 0.25926, 0.37699],
    [0.066899, 0.26319, 0.37759],
    [0.067921, 0.26706, 0.37819],
    [0.069002, 0.27092, 0.37877],
    [0.070001, 0.27471, 0.37934],
    [0.071115, 0.2785, 0.37989],
    [0.072192, 0.28225, 0.38043],
    [0.07344, 0.28594, 0.38096],
    [0.074595, 0.28965, 0.38145],
    [0.075833, 0.29332, 0.38192],
    [0.077136, 0.297, 0.38238],
    [0.078517, 0.30062, 0.38281],
    [0.079984, 0.30425, 0.38322],
    [0.081553, 0.30786, 0.3836],
    [0.083082, 0.31146, 0.38394],
    [0.084778, 0.31504, 0.38424],
    [0.086503, 0.31862, 0.38451],
    [0.088353, 0.32217, 0.38473],
    [0.090281, 0.32569, 0.38491],
    [0.092304, 0.32922, 0.38504],
    [0.094462, 0.33271, 0.38512],
    [0.096618, 0.33616, 0.38513],
    [0.099015, 0.33962, 0.38509],
    [0.10148, 0.34304, 0.38498],
    [0.10408, 0.34641, 0.3848],
    [0.10684, 0.34977, 0.38455],
    [0.1097, 0.3531, 0.38422],
    [0.11265, 0.35639, 0.38381],
    [0.11575, 0.35964, 0.38331],
    [0.11899, 0.36285, 0.38271],
    [0.12232, 0.36603, 0.38203],
    [0.12589, 0.36916, 0.38126],
    [0.12952, 0.37224, 0.38038],
    [0.1333, 0.37528, 0.3794],
    [0.13721, 0.37828, 0.37831],
    [0.14126, 0.38124, 0.37713],
    [0.14543, 0.38413, 0.37584],
    [0.14971, 0.38698, 0.37445],
    [0.15407, 0.38978, 0.37293],
    [0.15862, 0.39253, 0.37132],
    [0.16325, 0.39524, 0.36961],
    [0.16795, 0.39789, 0.36778],
    [0.17279, 0.4005, 0.36587],
    [0.17775, 0.40304, 0.36383],
    [0.18273, 0.40555, 0.36171],
    [0.18789, 0.408, 0.35948],
    [0.19305, 0.41043, 0.35718],
    [0.19831, 0.4128, 0.35477],
    [0.20368, 0.41512, 0.35225],
    [0.20908, 0.41741, 0.34968],
    [0.21455, 0.41966, 0.34702],
    [0.22011, 0.42186, 0.34426],
    [0.22571, 0.42405, 0.34146],
    [0.23136, 0.4262, 0.33857],
    [0.23707, 0.42832, 0.33563],
    [0.24279, 0.43042, 0.33263],
    [0.24862, 0.43249, 0.32957],
    [0.25445, 0.43453, 0.32643],
    [0.26032, 0.43656, 0.32329],
    [0.26624, 0.43856, 0.32009],
    [0.27217, 0.44054, 0.31683],
    [0.27817, 0.44252, 0.31355],
    [0.28417, 0.44448, 0.31024],
    [0.29021, 0.44642, 0.30689],
    [0.29629, 0.44836, 0.30351],
    [0.30238, 0.45028, 0.30012],
    [0.30852, 0.4522, 0.29672],
    [0.31465, 0.45411, 0.29328],
    [0.32083, 0.45601, 0.28984],
    [0.32701, 0.4579, 0.28638],
    [0.33323, 0.45979, 0.28294],
    [0.33947, 0.46168, 0.27947],
    [0.3457, 0.46356, 0.276],
    [0.35198, 0.46544, 0.27249],
    [0.35828, 0.46733, 0.26904],
    [0.36459, 0.46921, 0.26554],
    [0.37092, 0.47109, 0.26206],
    [0.37729, 0.47295, 0.25859],
    [0.38368, 0.47484, 0.25513],
    [0.39007, 0.47671, 0.25166],
    [0.3965, 0.47859, 0.24821],
    [0.40297, 0.48047, 0.24473],
    [0.40945, 0.48235, 0.24131],
    [0.41597, 0.48423, 0.23789],
    [0.42251, 0.48611, 0.23449],
    [0.42909, 0.48801, 0.2311],
    [0.43571, 0.48989, 0.22773],
    [0.44237, 0.4918, 0.22435],
    [0.44905, 0.49368, 0.22107],
    [0.45577, 0.49558, 0.21777],
    [0.46254, 0.4975, 0.21452],
    [0.46937, 0.49939, 0.21132],
    [0.47622, 0.50131, 0.20815],
    [0.48312, 0.50322, 0.20504],
    [0.49008, 0.50514, 0.20198],
    [0.49709, 0.50706, 0.19899],
    [0.50415, 0.50898, 0.19612],
    [0.51125, 0.5109, 0.1933],
    [0.51842, 0.51282, 0.19057],
    [0.52564, 0.51475, 0.18799],
    [0.53291, 0.51666, 0.1855],
    [0.54023, 0.51858, 0.1831],
    [0.5476, 0.52049, 0.18088],
    [0.55502, 0.52239, 0.17885],
    [0.56251, 0.52429, 0.17696],
    [0.57002, 0.52619, 0.17527],
    [0.57758, 0.52806, 0.17377],
    [0.5852, 0.52993, 0.17249],
    [0.59285, 0.53178, 0.17145],
    [0.60052, 0.5336, 0.17065],
    [0.60824, 0.53542, 0.1701],
    [0.61597, 0.53723, 0.16983],
    [0.62374, 0.539, 0.16981],
    [0.63151, 0.54075, 0.17007],
    [0.6393, 0.54248, 0.17062],
    [0.6471, 0.54418, 0.17146],
    [0.65489, 0.54586, 0.1726],
    [0.66269, 0.5475, 0.17404],
    [0.67048, 0.54913, 0.17575],
    [0.67824, 0.55071, 0.1778],
    [0.686, 0.55227, 0.18006],
    [0.69372, 0.5538, 0.18261],
    [0.70142, 0.55529, 0.18548],
    [0.7091, 0.55677, 0.18855],
    [0.71673, 0.5582, 0.19185],
    [0.72432, 0.55963, 0.19541],
    [0.73188, 0.56101, 0.19917],
    [0.73939, 0.56239, 0.20318],
    [0.74685, 0.56373, 0.20737],
    [0.75427, 0.56503, 0.21176],
    [0.76163, 0.56634, 0.21632],
    [0.76894, 0.56763, 0.22105],
    [0.77621, 0.5689, 0.22593],
    [0.78342, 0.57016, 0.23096],
    [0.79057, 0.57142, 0.23616],
    [0.79767, 0.57268, 0.24149],
    [0.80471, 0.57393, 0.24696],
    [0.81169, 0.57519, 0.25257],
    [0.81861, 0.57646, 0.2583],
    [0.82547, 0.57773, 0.2642],
    [0.83227, 0.57903, 0.27021],
    [0.839, 0.58034, 0.27635],
    [0.84566, 0.58167, 0.28263],
    [0.85225, 0.58304, 0.28904],
    [0.85875, 0.58444, 0.29557],
    [0.86517, 0.58588, 0.30225],
    [0.87151, 0.58735, 0.30911],
    [0.87774, 0.58887, 0.31608],
    [0.88388, 0.59045, 0.32319],
    [0.8899, 0.59209, 0.33045],
    [0.89581, 0.59377, 0.33787],
    [0.90159, 0.59551, 0.34543],
    [0.90724, 0.59732, 0.35314],
    [0.91275, 0.59919, 0.36099],
    [0.9181, 0.60113, 0.369],
    [0.9233, 0.60314, 0.37714],
    [0.92832, 0.60521, 0.3854],
    [0.93318, 0.60737, 0.39382],
    [0.93785, 0.60958, 0.40235],
    [0.94233, 0.61187, 0.41101],
    [0.94661, 0.61422, 0.41977],
    [0.9507, 0.61665, 0.42862],
    [0.95457, 0.61914, 0.43758],
    [0.95824, 0.62167, 0.4466],
    [0.9617, 0.62428, 0.4557],
    [0.96494, 0.62693, 0.46486],
    [0.96798, 0.62964, 0.47406],
    [0.9708, 0.63239, 0.48329],
    [0.97342, 0.63518, 0.49255],
    [0.97584, 0.63801, 0.50183],
    [0.97805, 0.64087, 0.51109],
    [0.98008, 0.64375, 0.52035],
    [0.98192, 0.64666, 0.5296],
    [0.98357, 0.64959, 0.53882],
    [0.98507, 0.65252, 0.548],
    [0.98639, 0.65547, 0.55714],
    [0.98757, 0.65842, 0.56623],
    [0.9886, 0.66138, 0.57526],
    [0.9895, 0.66433, 0.58425],
    [0.99027, 0.66728, 0.59317],
    [0.99093, 0.67023, 0.60203],
    [0.99148, 0.67316, 0.61084],
    [0.99194, 0.67609, 0.61958],
    [0.9923, 0.67901, 0.62825],
    [0.99259, 0.68191, 0.63687],
    [0.99281, 0.68482, 0.64542],
    [0.99297, 0.68771, 0.65393],
    [0.99306, 0.69058, 0.6624],
    [0.99311, 0.69345, 0.67081],
    [0.99311, 0.69631, 0.67918],
    [0.99307, 0.69916, 0.68752],
    [0.993, 0.70201, 0.69583],
    [0.9929, 0.70485, 0.70411],
    [0.99277, 0.70769, 0.71238],
    [0.99262, 0.71053, 0.72064],
    [0.99245, 0.71337, 0.72889],
    [0.99226, 0.71621, 0.73715],
    [0.99205, 0.71905, 0.7454],
    [0.99184, 0.72189, 0.75367],
    [0.99161, 0.72475, 0.76196],
    [0.99137, 0.72761, 0.77027],
    [0.99112, 0.73049, 0.77861],
    [0.99086, 0.73337, 0.78698],
    [0.99059, 0.73626, 0.79537],
    [0.99031, 0.73918, 0.80381],
    [0.99002, 0.7421, 0.81229],
    [0.98972, 0.74504, 0.8208],
    [0.98941, 0.748, 0.82937],
    [0.98909, 0.75097, 0.83798],
    [0.98875, 0.75395, 0.84663],
    [0.98841, 0.75695, 0.85533],
    [0.98805, 0.75996, 0.86408],
    [0.98767, 0.763, 0.87286],
    [0.98728, 0.76605, 0.8817],
    [0.98687, 0.7691, 0.89057],
    [0.98643, 0.77218, 0.89949],
    [0.98598, 0.77527, 0.90845],
    [0.9855, 0.77838, 0.91744],
    [0.985, 0.7815, 0.92647],
    [0.98447, 0.78462, 0.93553],
    [0.98391, 0.78776, 0.94463],
    [0.98332, 0.79091, 0.95375],
    [0.9827, 0.79407, 0.9629],
    [0.98205, 0.79723, 0.97207],
    [0.98135, 0.80041, 0.98127],
]

batlow_map = LinearSegmentedColormap.from_list("batlow", cm_data)
# For use of "viscm view"
test_cm = batlow_map

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np

    try:
        from viscm import viscm

        viscm(batlow_map)
    except ImportError:
        print("viscm not found, falling back on simple display")
        plt.imshow(np.linspace(0, 100, 256)[None, :], aspect="auto", cmap=batlow_map)
    plt.show()
