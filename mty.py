# 读取用户输入的两个整数
try:
    num1 = int(input("请输入第一个整数："))
    num2 = int(input("请输入第二个整数："))
    
    # 计算和、差、积
    sum_result = num1 + num2
    sub_result = num1 - num2
    mul_result = num1 * num2
    
    # 处理除法（避免除数为0）
    if num2 == 0:
        div_result = "除数不能为零"
    else:
        div_result = num1 / num2  # 保留浮点数结果，符合示例中的0.5输出
    
    # 输出结果
    print(f"和是 {sum_result}")
    print(f"差是 {sub_result}")
    print(f"积是 {mul_result}")
    print(f"商是 {div_result}")

# 处理非整数输入的异常（可选，提升程序健壮性）
except ValueError:
    print("输入错误！请输入有效的整数。")