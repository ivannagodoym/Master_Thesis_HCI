import pandas as pd

#read the data
data_path = "/Users/ivannagodoymunoz/Desktop/Master Thesis/Testing"
thesis_data = pd.read_csv(f"{data_path}/thesis_data.csv", sep=",",on_bad_lines='skip')

length_data = len(thesis_data)
print(length_data)


# Divide conversations into three parts
part1 = thesis_data[:17473]
part2 = thesis_data[17473:30246]
print(len(part2))
part3 = thesis_data[30246:]
print(len(part3))

values_to_remove = ['hit:7971_conv:15943', 'hit:8075_conv:16150', 'hit:11677_conv:23355',
                    'hit:2257_conv:4515', 'hit:1559_conv:3118', 'hit:11151_conv:22303',
                    'hit:4597_conv:9195', 'hit:1243_conv:2487', 'hit:4672_conv:9345',
                    'hit:11626_conv:23253', 'hit:11439_conv:22879', 'hit:1068_conv:2136',
                    'hit:5678_conv:11356', 'hit:832_conv:1665', 'hit:8754_conv:17508', 'hit:5289_conv:10578',
                    'hit:3196_conv:6393', 'hit:3617_conv:7235', 'hit:826_conv:1653', 'hit:10015_conv:20030',
                    'hit:3484_conv:6969', 'hit:7288_conv:14577', 'hit:2816_conv:5632', 'hit:5996_conv:11992',
                    'hit:558_conv:1117', 'hit:8514_conv:17028', 'hit:11100_conv:22200', 'hit:9417_conv:18835',
                    'hit:818_conv:1637', 'hit:5493_conv:10987']
thesis_data = thesis_data[~thesis_data['conv_id'].isin(values_to_remove)]
thesis_data.to_csv("thesis_data.csv", index=False)

#values_to_remove_1 = ['hit:7971_conv:15943', 'hit:8075_conv:16150', 'hit:11677_conv:23355', 'hit:2257_conv:4515', 'hit:1559_conv:3118', 'hit:11151_conv:22303', 'hit:4597_conv:9195']
#part1 = part1[~part1['conv_id'].isin(values_to_remove_1)]
#part1.to_csv("thesis_data1.csv", index=False)

#values_to_remove_2 = ['hit:1243_conv:2487', 'hit:4672_conv:9345', 'hit:11626_conv:23253', 'hit:11439_conv:22879', 'hit:1068_conv:2136', 'hit:5678_conv:11356', 'hit:832_conv:1665', 'hit:8754_conv:17508', 'hit:5289_conv:10578', 'hit:3196_conv:6393', 'hit:3617_conv:7235', 'hit:826_conv:1653']
#part2 = part2[~part2['conv_id'].isin(values_to_remove_2)]
#part2.to_csv("thesis_data2.csv", index=False)

#values_to_remove_3 = ['hit:10015_conv:20030', 'hit:3484_conv:6969', 'hit:7288_conv:14577', 'hit:2816_conv:5632', 'hit:5996_conv:11992', 'hit:558_conv:1117', 'hit:8514_conv:17028', 'hit:11100_conv:22200', 'hit:9417_conv:18835', 'hit:818_conv:1637', 'hit:5493_conv:10987']
#part3 = part3[~part3['conv_id'].isin(values_to_remove_3)]
#part3.to_csv("thesis_data3.csv", index=False)

