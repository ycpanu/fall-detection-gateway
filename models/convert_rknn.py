"""YOLOv8-Pose 混合量化：默认卷积保持 INT8，仅合并坐标与置信度的输出使用 FP16。"""

import argparse
import io
import os
import re
import sys
import time
from pathlib import Path


MODEL_DIR = Path(__file__).resolve().parent
OUTPUT_ALIASES = {"Reshape", "Transpose", "Identity", "exDataConvert"}


def check_result(result, operation):
    if result != 0:
        raise RuntimeError(f"{operation}失败，错误码：{result}")


def pose_head_prefix(model_path):
    """识别当前导出图中的检测头。"""
    import onnx

    model = onnx.load(str(model_path))
    if len(model.graph.output) != 1:
        raise ValueError("当前 C++ 后处理要求单输出 [1,56,N] 的 Pose 模型")
    output = model.graph.output[0]
    shape = [dim.dim_value for dim in output.type.tensor_type.shape.dim]
    if len(shape) != 3 or shape[0] != 1 or shape[1] != 56 or shape[2] <= 0:
        raise ValueError(f"不支持的 Pose 输出尺寸：{shape}，要求 [1,56,N]")
    producers = {tensor: node for node in model.graph.node for tensor in node.output}
    last_node = producers[output.name]
    while last_node.op_type in OUTPUT_ALIASES:
        last_node = producers[last_node.input[0]]
    if last_node.op_type != "Concat" or "/" not in last_node.name.strip("/"):
        raise ValueError("无法识别 Pose 检测头，请检查 ONNX 导出结构")
    return last_node.name.rsplit("/", 1)[0] + "/"


def configure_mixed_precision(model_path, config_path, scope):
    """保护坐标与概率的混合输出，并允许复用旧版全检测头 FP16 的标定文件。"""
    import onnx
    from ruamel.yaml import YAML

    model = onnx.load(str(model_path))
    head_prefix = pose_head_prefix(model_path)
    head_tensors = {tensor for node in model.graph.node
                    if node.name.startswith(head_prefix) and node.op_type not in ("Constant", "Shape")
                    for tensor in node.output}
    output_tensors = {output.name for output in model.graph.output}
    managed = head_tensors | output_tensors
    if len(head_tensors) < 2:
        raise ValueError("临时模型中未找到检测头，无法设置混合量化")

    if scope == "head":
        tensors = managed
    else:
        producers = {tensor: node for node in model.graph.node for tensor in node.output}

        def from_sigmoid(tensor):
            node = producers.get(tensor)
            while node is not None and node.op_type in OUTPUT_ALIASES:
                node = producers.get(node.input[0])
            return node is not None and node.op_type == "Sigmoid"

        # 必须同时保护关键点的 XY/conf 合并和最终 box/conf/keypoints 合并。
        # 只将最终输出设为 FP16，无法恢复在关键点合并处丢失的置信度。
        merges = [node for node in model.graph.node
                  if node.name.startswith(head_prefix) and node.op_type == "Concat"
                  and any(from_sigmoid(tensor) for tensor in node.input)]
        if len(merges) != 2:
            raise ValueError(f"预期找到 2 个坐标/置信度合并节点，实际 {len(merges)}；请检查导出结构")
        tensors = {tensor for node in merges for tensor in node.output}
        while True:
            aliases = {tensor for node in model.graph.node
                       if node.op_type in OUTPUT_ALIASES and node.input[0] in tensors
                       for tensor in node.output}
            if aliases <= tensors:
                break
            tensors.update(aliases)
        if not output_tensors <= tensors:
            raise ValueError("未能沿合并节点追踪到最终输出，停止转换以避免置信度再次量化为零")

    # Toolkit 可能已在检测头之外生成 FP16 建议；也要清除，否则卷积仍会使用 FP16。
    # 只替换覆盖列表。标定浮点数必须逐字保留，YAML 重写的小数舍入可能触发
    # Toolkit 的 ori_min/ori_max 不可修改校验。
    original = config_path.read_text()
    section = re.search(r"(?m)^custom_quantize_layers:[^\n]*(?:\n|$)", original)
    if section is None:
        raise ValueError("量化配置缺少 custom_quantize_layers 字段")
    next_section = re.search(r"(?m)^[^\s#][^\n]*:", original[section.end():])
    end = section.end() + next_section.start() if next_section else len(original)
    replacement = io.StringIO()
    YAML().dump({"custom_quantize_layers": {tensor: "float16" for tensor in sorted(tensors)}}, replacement)
    config_path.write_text(original[:section.start()] + replacement.getvalue() + original[end:])
    print(f"FP16 范围：{scope}，共 {len(tensors)} 个张量；配置：{config_path}", flush=True)


def calibration_file(dataset_path, work_dir, limit):
    """将路径锚定到原标定列表目录；limit=0 使用完整列表。"""
    dataset_path = dataset_path.resolve()
    print(f"读取标定列表：{dataset_path}", flush=True)
    entries = [line.strip() for line in dataset_path.read_text().splitlines() if line.strip()]
    if not entries:
        raise ValueError("标定列表为空")
    if limit and len(entries) > limit:
        entries = [entries[i * len(entries) // limit] for i in range(limit)]
    total = len(entries)
    print(f"开始检查标定图片路径：0/{total}（挂载盘上的文件检查可能较慢）", flush=True)
    started = last_report = time.monotonic()
    paths = []
    for index, entry in enumerate(entries, start=1):
        path = Path(entry)
        if not path.is_absolute():
            path = dataset_path.parent / path
        if not path.is_file():
            raise FileNotFoundError(f"标定图片不存在：{path}")
        # path 已是绝对路径，无需为每张图片重复解析整条路径上的符号链接。
        paths.append(str(path))
        now = time.monotonic()
        if index == total or now - last_report >= 2:
            print(f"标定路径检查：{index}/{total}，已用时 {now - started:.1f} 秒", flush=True)
            last_report = now
    normalized = work_dir / "calibration.txt"
    normalized.write_text("\n".join(paths) + "\n")
    print(f"使用 {len(paths)} 张标定图片", flush=True)
    return normalized


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx", type=Path, default=MODEL_DIR / "yolov8n-pose.onnx")
    parser.add_argument("--dataset", type=Path, default=MODEL_DIR / "dataset.txt")
    parser.add_argument("--output", type=Path, default=MODEL_DIR / "yolov8n-pose-hybrid.rknn")
    parser.add_argument("--work-dir", type=Path, default=MODEL_DIR.parent / "build/rknn_pose_hybrid")
    parser.add_argument("--step", choices=("1", "2", "all"), default="all")
    parser.add_argument("--fp16-scope", choices=("output", "head"), default="output",
                        help="output：仅保护合并输出（默认）；head：整个检测头 FP16，速度较慢")
    parser.add_argument("--calibration-limit", type=int, default=0,
                        help="仅用于快速验证的标定图片上限；默认 0 使用全部图片")
    args = parser.parse_args()
    if args.calibration_limit < 0:
        parser.error("--calibration-limit 不能小于 0")
    print(f"开始转换，Python：{sys.executable}", flush=True)
    print("正在加载 RKNN Toolkit…", flush=True)
    from rknn.api import RKNN

    model_path = args.onnx.resolve()
    dataset_path = args.dataset.resolve()
    output_path = args.output.resolve()
    work_dir = args.work_dir.resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"工作目录：{work_dir}\n输出模型：{output_path}", flush=True)
    # Toolkit 将 .model/.data/.quantization.cfg 写入当前目录。
    os.chdir(work_dir)

    if args.step in ("1", "all"):
        print(f"[1/2] 检查 ONNX 模型：{model_path}", flush=True)
        pose_head_prefix(model_path)
        dataset = calibration_file(dataset_path, work_dir, args.calibration_limit)
        rknn = RKNN(verbose=True)
        try:
            # C++ 传入 RGB uint8；归一化由 RKNN 完成，不要在 C++ 再除以 255。
            rknn.config(mean_values=[[0, 0, 0]], std_values=[[255, 255, 255]],
                        target_platform="rk3566")
            print("[1/2] 加载 ONNX 到 RKNN…", flush=True)
            check_result(rknn.load_onnx(model=str(model_path)), "加载 ONNX")
            print("[1/2] 开始标定和量化，完整数据集可能耗时较长…", flush=True)
            check_result(rknn.hybrid_quantization_step1(
                dataset=str(dataset), proposal=False), "混合量化第一步")
        finally:
            rknn.release()
        if args.step == "1":
            configure_mixed_precision(work_dir / f"{model_path.stem}.model",
                                      work_dir / f"{model_path.stem}.quantization.cfg", args.fp16_scope)

    if args.step in ("2", "all"):
        print("[2/2] 开始构建混合量化模型…", flush=True)
        artifacts = [work_dir / f"{model_path.stem}{suffix}"
                     for suffix in (".model", ".data", ".quantization.cfg")]
        for artifact in artifacts:
            if not artifact.is_file():
                raise FileNotFoundError(f"缺少 {artifact}，请先执行 --step 1")
        configure_mixed_precision(artifacts[0], artifacts[2], args.fp16_scope)
        rknn = RKNN(verbose=True)
        try:
            check_result(rknn.hybrid_quantization_step2(
                model_input=str(artifacts[0]), data_input=str(artifacts[1]),
                model_quantization_cfg=str(artifacts[2])), "混合量化第二步")
            check_result(rknn.export_rknn(str(output_path)), "导出 RKNN")
            print(f"混合量化模型已导出：{output_path}", flush=True)
        finally:
            rknn.release()


if __name__ == "__main__":
    main()
