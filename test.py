import sglang as sl

# 请确保 SGLang server 正在运行，并且可以从你的客户端访问
# 如果你的 server 运行在本地的默认端口，则不需要修改此行
sl.set_default_backend(sl.RuntimeEndpoint("http://127.0.0.1:30000"))

# 定义一个简单的 SGLang 程序
@sl.function
def gen_with_logprobs(s, character, num):
    s += "Hello, " + character
    s += sl.gen("answer", max_tokens=10, return_logprob = True, top_logprobs_num = num)
# 调用函数。注意：logprobs 参数在这里被传递给了 run 方法，而不是 gen_with_logprobs 函数本身。
# run() 方法是实际执行 @sl.function 的入口。
state = gen_with_logprobs.run(character="world", num=5)

# 打印结果
print(state.text())
print(state.get_meta_info("answer")["output_token_logprobs"])
print(state.get_meta_info("answer")["output_top_logprobs"])