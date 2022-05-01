import json
import numpy as np

def swap_sample(dir2):
    #open
    with open(dir2+'sample_class_stats_dict_copy.json') as f:
        data = json.load(f)
    # process
    for kx,vx in data.items():
        new_val = {}
        for ky,vy in vx.items():
            new_id = 18-int(ky)
            new_val[str(new_id)] = vy
        data[kx] =  new_val
    # save
    json_string = json.dumps(data, indent=2)
    with open(dir2+'sample_class_stats_dict.json', 'w') as f:
        f.write(json_string)

    # open
    with open(dir2+'sample_class_stats_copy.json') as f:
        data = json.load(f)
    # process
    my_list = []
    for vx in data:
        new_val = {}
        for ky,vy in vx.items():
            if ky.isdigit():
                new_id = 18-int(ky)
                new_val[str(new_id)] = vy
            else:
                new_val[ky] = vy
        my_list.append(new_val)
    # save
    json_string = json.dumps(my_list, indent=2)
    with open(dir2+'sample_class_stats.json', 'w') as f:
        f.write(json_string)

    # open
    with open(dir2+'samples_with_class_copy.json') as f:
        data = json.load(f)
    # process
    new_data = {}
    for kx,vx in data.items():
        new_id = 18-int(kx)
        new_data[str(new_id)] = vx
    # save
    json_string = json.dumps(new_data, indent=2)
    with open(dir2+'samples_with_class.json', 'w') as f:
        f.write(json_string)

        



'''
# Main file Cityscape #################################################
file1 = 'data/cityscapes/gtFine/cityscapes_panoptic_val.json'
file2 = 'data/cityscapes/gtFine/cityscapes_panoptic_dapanformer_val.json'
file3 = 'data/cityscapes/gtFine/cityscapes_panoptic_train.json'
file4 = 'data/cityscapes/gtFine/cityscapes_panoptic_dapanformer_train.json'
city_idx = np.array([
    [7,  0 ],
    [8,  1 ],
    [11, 2 ],
    [12, 3 ],
    [13, 4 ],
    [17, 5 ],
    [19, 6 ], 
    [20, 7 ],
    [21, 8 ],
    [22, 9 ], 
    [23, 10],
    [24, 11],
    [25, 12],
    [26, 13],
    [27, 14], 
    [28, 15],
    [31, 16], 
    [32, 17],
    [33, 18]
])

with open(file1) as f:
    data = json.load(f)

for x in data['annotations']:
    for y in x['segments_info']:
        temp = y['category_id']
        if temp in city_idx[:,0]:
            temp = 18 - city_idx[temp==city_idx[:,0],1].item()
        else:
            temp = 255
        y['category_id'] = temp

for x in data['categories']:
    temp = x['id'] 
    if temp in city_idx[:,0]:
        temp = 18 - city_idx[temp==city_idx[:,0],1].item()
    else:
        temp = 255
    x['id'] = temp

json_string = json.dumps(data, indent=2)
with open(file2, 'w') as f:
    f.write(json_string)
    
with open(file3) as f:
    data = json.load(f)

for x in data['annotations']:
    for y in x['segments_info']:
        temp = y['category_id']
        if temp in city_idx[:,0]:
            temp = 18 - city_idx[temp==city_idx[:,0],1].item()
        else:
            temp = 255
        y['category_id'] = temp

for x in data['categories']:
    temp = x['id'] 
    if temp in city_idx[:,0]:
        temp = 18 - city_idx[temp==city_idx[:,0],1].item()
    else:
        temp = 255
    x['id'] = temp

json_string = json.dumps(data, indent=2)
with open(file4, 'w') as f:
    f.write(json_string)



'''
# Main file Synthia #################################################
file5 = 'data/synthia/GT/panoptic-labels-crowdth-0/synthia_panoptic.json'
file6 = 'data/synthia/GT/panoptic-labels-crowdth-0/synthia_panoptic_dapanformer.json'
with open(file5) as f:
    data = json.load(f)

synthia_idx = np.array([
    [3,  0 ],
    [4,  1 ],
    [2,  2 ],
    [21, 3 ],
    [5,  4 ],
    [7,  5 ],
    [15, 6 ],
    [9,  7 ],
    [6,  8 ],
    [16, 9 ],
    [1,  10],
    [10, 11],
    [17, 12],
    [8,  13],
    [18, 14],
    [19, 15],
    [20, 16],
    [12, 17],
    [11, 18]
])

for x in data['annotations']:
    for y in x['segments_info']:
        temp = y['category_id']
        if temp<19:
            temp = 18 - temp
        else:
            temp = 255
        y['category_id'] = temp

#for x in data['images']:
#    x['file_name'] = x['file_name'].replace('.png', '_panoptic.png')

json_string = json.dumps(data, indent=2)
with open(file6, 'w') as f:
    f.write(json_string)



# Sample file #################################################
swap_sample('data/cityscapes/')
swap_sample('data/synthia/')
