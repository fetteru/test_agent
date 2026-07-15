import os
import shutil
import kagglehub

# 项目根目录
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# 目标目录
target_dir = os.path.join(PROJECT_ROOT, "myprojectdata")
os.makedirs(target_dir, exist_ok=True)

MAX_RETRIES = 5

for attempt in range(1, MAX_RETRIES + 1):
    try:
        print(f"第 {attempt}/{MAX_RETRIES} 次尝试下载...")
        path = kagglehub.dataset_download("bratjay/ua-detrac-orig")
        print("下载完成，缓存路径：", path)
        break
    except Exception as e:
        print(f"下载中断：{e}")
        if attempt < MAX_RETRIES:
            print("等待 5 秒后重试...\n")
        else:
            print("已达最大重试次数，下载失败")
            exit(1)

# 复制到项目目录
print(f"正在复制到 {target_dir} ...")
for item in os.listdir(path):
    src = os.path.join(path, item)
    dst = os.path.join(target_dir, item)
    if os.path.isdir(src):
        shutil.copytree(src, dst, dirs_exist_ok=True)
    else:
        shutil.copy2(src, dst)

print(f"完成！数据集位置：{target_dir}")
