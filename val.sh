#!/bin/bash

# 定义起始值和结束值
start=10
end=200
step=10

# 初始化一个空数组来收集输出
test_accuracies=()

# 循环从 start 到 end，步长为 step
for (( id_classes=start; id_classes<=end; id_classes+=step )); do
    # 执行 Python 脚本并收集输出
    echo "Evaluating with ID classes: $id_classes"
    output=$(python eval.py --ID_classes "$id_classes")
    
    # 提取 Test Accuracy
    accuracy=$(echo "$output" | grep 'Test Accuracy' | awk '{print $3}')  # 获取第三个字段
    test_accuracies+=("$accuracy")  # 将准确率添加到数组中
done


# 写入收集到的 Test Accuracy 到 val.txt
echo "Collected Test Accuracies:" > val.txt  # 清空并写入标题
for accuracy in "${test_accuracies[@]}"; do
    echo "$accuracy" >> val.txt  # 追加每个准确率
done
