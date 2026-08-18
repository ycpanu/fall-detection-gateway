from rknn.api import RKNN
import os

if __name__ == "__main__":
    # 1. 初始化 RKNN 对象
    rknn = RKNN(verbose=True)

    # 2. 配置目标平台
    # mean_values 和 std_values 保持默认，因为 C++ 里传进去的是标准的 RGB 图像
    rknn.config(mean_values=[[0, 0, 0]], std_values=[[255, 255, 255]], target_platform='rk3566')

    # 3. 加载 ONNX 模型
    print("正在加载 ONNX 模型...")
    ret = rknn.load_onnx(model='./yolov8n-pose.onnx')
    if ret != 0:
        print("ONNX 模型加载失败！")
        exit(ret)

    # 4. 构建 RKNN 模型
    print("正在编译构建 RKNN 模型...")
    # do_quantization=False 表示暂不使用 INI8 极致量化，避免精度掉点，FP16 速度也极快
    ret = rknn.build(do_quantization=False)
    if ret != 0:
        print("RKNN 模型构建失败！")
        exit(ret)

    # 5. 导出为 .rknn 文件
    print("正在导出 .rknn 模型文件...")
    ret = rknn.export_rknn('./yolov8n-pose.rknn')
    if ret != 0:
        print("RKNN 模型导出失败！")
        exit(ret)

    print("RKNN 模型转换完成！")
    rknn.release()
