import time

start_time = time.time()  # 获取当前时间

# 放置你的代码
for i in range(1000000):
    pass

end_time = time.time()  # 再次获取当前时间

elapsed_time = end_time - start_time  # 计算两次时间的差值
print(f"程序运行时间：{elapsed_time}秒")