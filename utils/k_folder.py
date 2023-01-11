import csv

k = 5
for i in range(k):
    result = []
    slide_file = open('/home/zhw/Siam_SSA/table/slide_dir.csv', 'r')
    slide_reader = csv.reader(slide_file)
    for idx,(slide_id, folder, label, patch_num, ratio) in enumerate(slide_reader,0):
        if idx%k==i:
            result.append([slide_id, folder, label, patch_num, ratio])
    slide_file.close()
    with open("../table/folder_"+str(i)+".csv","w",encoding="utf-8",newline='') as f:
        writer=csv.writer(f)
        writer.writerows(result)
    f.close()