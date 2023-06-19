import os

# 指定目标文件夹路径
output_dir = "D:/Python"  # 设置新建文件夹的位置，例如在 D:/Python/Output 下创建新文件夹

# 指定新文件夹名称
folder_name = "NewFolder"

# 使用 os.path.join() 创建完整路径
folder_path = os.path.join(output_dir, folder_name)

# 创建新文件夹
try:
    os.mkdir(folder_path)
except:
    print("文件夹已经存在")

file_name = "output.txt"

save_path = os.path.join(folder_path,file_name)

with open(save_path, "w") as file:
    file.write("This is a new file.")

print("Folder created at:", folder_path)

