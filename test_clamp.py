# import torch
# import torch.nn.functional as F

# # ----------------------------------------------------
# # 方法一：您要验证的函数
# # ----------------------------------------------------
# def max_fn(x):
#     """使用 PyTorch 内置高级函数实现"""
#     return F.normalize(F.relu(x), p=1, dim=1)


# # ----------------------------------------------------
# # 方法二：手动实现数学步骤
# # ----------------------------------------------------
# def manual_verification_fn(x):
#     """手动实现 L1 归一化 max(x, 0) 的每个步骤"""
#     # 步骤 1: max(x, 0)
#     x_relu = torch.clamp(x, min=0) # F.relu(x) 的另一种等价写法

#     # 步骤 2: 计算 L1 范数 (因为x_relu非负，所以就是求和)
#     # keepdim=True 确保维度匹配，以便进行广播除法
#     l1_norm = torch.sum(x_relu, dim=1, keepdim=True)
    
#     # 步骤 3: 处理和为0的行，防止除以0得到NaN
#     # 添加一个极小的数 (epsilon) 来保证数值稳定性
#     l1_norm = l1_norm + 1e-8 

#     # 步骤 4: 元素相除
#     return x_relu / l1_norm


# # ----------------------------------------------------
# # 执行验证
# # ----------------------------------------------------
# # 创建一个包含正数、负数、零和全为负数的行的测试张量
# test_tensor = torch.tensor([
#     [1.0, -2.0, 7.0],      # 混合行
#     [5.0,  5.0, 10.0],     # 全正数行
#     [-1.0, -3.0, -5.0]     # 全负数行（关键测试用例）
# ], dtype=torch.float32)


# # 运行两个函数
# output_max_fn = max_fn(test_tensor)
# output_manual = manual_verification_fn(test_tensor)

# print("--- 测试张量 ---\n", test_tensor)
# print("\n--- 方法一: F.normalize(F.relu(x)) 的结果 ---")
# print(output_max_fn)
# # 预期第一行: [1/(1+7), 0, 7/(1+7)] = [0.125, 0, 0.875]
# # 预期第三行: [0, 0, 0] (F.normalize 会优雅地处理和为0的情况)

# print("\n--- 方法二: 手动计算的结果 ---")
# print(output_manual)
# # 我们手动添加了epsilon，所以结果会非常接近，但不是100%比特相同

# # ----------------------------------------------------
# # 最终验证：使用 torch.allclose 来比较浮点数张量
# # ----------------------------------------------------
# # 注意：因为我们手动加了epsilon，而F.normalize有自己的实现，
# # 所以使用 allclose 是最合适的比较方式
# are_they_equivalent = torch.allclose(output_max_fn, output_manual, atol=1e-7)

# print(f"\n两个函数的输出是否等价? -> {are_they_equivalent}")


import torch

# 两个效果等价的输入
probs_sum_one = torch.tensor([0.1, 0.2, 0.7])
weights_sum_ten = torch.tensor([1.0, 2.0, 7.0])

# 进行大量抽样
num_draws = 100000
samples_from_probs = torch.multinomial(probs_sum_one, num_samples=num_draws, replacement=True)
samples_from_weights = torch.multinomial(weights_sum_ten, num_samples=num_draws, replacement=True)

# 统计每个索引出现的次数
counts_from_probs = torch.bincount(samples_from_probs)
counts_from_weights = torch.bincount(samples_from_weights)

# 计算频率
freq_from_probs = counts_from_probs.float() / num_draws
freq_from_weights = counts_from_weights.float() / num_draws

print(f"理论概率: {[0.1, 0.2, 0.7]}\n")

print(f"从【和为1的概率】张量抽样的频率: {freq_from_probs}")
print(f"从【和不为1的权重】张量抽样的频率: {freq_from_weights}")
