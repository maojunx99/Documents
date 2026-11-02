GitHub 目录说明

完整复现请先下载并解压本目录的 matmulv2_static_kshift_repro.zip，然后进入解压后的 matmulv2_static_kshift_repro 目录运行。
固定输入 fixture.pt 和完整验证日志随 ZIP 提供；本目录其余文件供直接浏览。
ZIP SHA256: 737cee224d3cd8e0cb4f8d20e6349a40754cf10fe14d0114809ff99430b0d739

MatMulV2 Static Kernel / 默认 K-shift 普通 Kernel 数值差异复现

1. 复现对象

  调用链：torch.matmul -> aclnnMatmulWeightNz -> MatMulCommon -> MatMulV2。
  A：BF16、ND，逻辑 shape [1, 2048]。
  B：BF16、FRACTAL_NZ，逻辑 shape [2048, 2048]。
  输出：BF16、ND，shape [1, 2048]。
  两个 transpose 属性均为 false。

  fixture.pt 是已经固定的真实模型投影输入与权重切片，包含 CPU Tensor a、b。
  来源是 Qwen3.5-35B-A3B 的一次 decode 投影；运行脚本不需要模型或模型目录。
  权重在测量前显式转为 FRACTAL_NZ，并检查实际格式值为 29。

2. 环境

  Linux / Ascend A3；CANN Toolkit 与对应 ops 包，以及匹配的 torch、torch_npu。
  默认 SoC：Ascend910_9382。本描述文件固定为该平台的 24 Cube / 48 Vector 配置；
  其他 SoC 未验证，不能只修改 --soc 就假定等价。
  已知复现环境：CANN 9.1.0 / torch_npu 2.10.0.post4+git5dd8ef3；
                CANN 9.2.0 V100R001C25B114 / torch_npu 2.10.0.post7。
  repro.py 会用当前原包 op_compiler 重新编译一个静态算子，安装到结果目录。
  不修改 CANN 源文件，不向 SDK 全局目录安装，不运行完整模型。

3. 一键运行

  解压后进入本目录。先选择一个实际空闲 die；下面的编号 9 只是示例。

  source /usr/local/Ascend/cann-9.1.0/set_env.sh
  export ASCEND_RT_VISIBLE_DEVICES=9
  python3 repro.py --output ./result_cann91

  CANN 9.2 示例：

  source /usr/local/Ascend/cann-9.2.0/set_env.sh
  export ASCEND_RT_VISIBLE_DEVICES=9
  python3 repro.py --output ./result_cann92

  快速只复现真实输入这一组：

  python3 repro.py --cases 1 --output ./result_one_case

  --output 目录必须尚不存在。脚本会自动执行：
    a. 校验固定 fixture 的 SHA256。
    b. 使用原包 op_compiler -f true 编译 MatMulV2.json。
    c. 在新进程中取消 CLOSE_MATMUL_K_SHIFT，比较普通与 Static，包含 graph replay。
    d. 在另一新进程中设置 CLOSE_MATMUL_K_SHIFT=1，进行同输入对照。
    e. 另外采集一组原生日志，检查普通与 Static 实际命中的 kernel 名称。

  确定性和强一致性运行时进程选项均显式保持默认值 0。
  CLOSE_MATMUL_K_SHIFT 的两个设置只作用于各自的子进程。
  请使用未注入第三方 Python hook / LD_PRELOAD 的干净 CANN 环境。

4. 已知预期结果

  默认 64 组输入：第一组是固定真实输入，另外 63 组为固定种子的随机激活，B 相同。
  共同包含 131072 个输出元素。

  普通 vs Static，默认 K-shift：22 个输出元素不同。
  普通 vs Static，CLOSE_MATMUL_K_SHIFT=1：0 个输出元素不同。
  设置 CLOSE_MATMUL_K_SHIFT=1 后，普通输出变化，Static 输出保持不变。
  各自 eager 与各自 graph replay 一致。

  真实输入（--cases 1 也可复现）：
    输出坐标：1053
    普通：   -0.8671875
    Static： -0.86328125
    CPU FP64 参考：-0.8652345320442691
    两个 BF16 结果相差 1 ULP。

  普通 kernel 的已知名称：
    MatMulV2_NDNZ_ND_FP16_FP16_false_false_all_229443
  其中名字中的 FP16 是该通用二进制的命名，实际输入 dtype 仍是 BF16。
  Static kernel 名称为本次编译的 static_kernel_MatMulV2_*__kernel0。
  已知两条路径均使用 22 个 block；Static 的 opMode=static、opParaSize=0。

  不同 CANN / 驱动 / SoC 可能改变选择和数值结果。
  如果没有复现差异，summary.json 的 numerical_mismatch_reproduced 为 false；
  不会为了符合预期而伪造或强制制造差异。
  如果实际 kernel 命中不符合预期，脚本会报错，不能把结果当作本问题的复现。

5. 输出材料

  summary.json                 汇总、差异数量、真实输入示例、kernel 名称。
  default/report.json          默认 K-shift 逐输入差异。
  close_shift/report.json      关闭 K-shift 后的逐输入对照。
  default/outputs.pt           普通 / Static / graph 输出，供进一步检查。
  close_shift/outputs.pt       关闭 K-shift 后的输出。
  kernel_selection.log         过滤后的实际 kernel 选择与发射日志。
  proof.log                    完整的原生日志。
  compile_command.json         编译命令、环境和编译器入口文件校验值。
  compile.log / install.log    原包编译和本地安装记录。
  install/                    私有 Static Kernel 包，不修改 SDK 全局安装。

6. 如何表述问题

  这个复现证明特定输入上，静态 / 普通 MatMulV2 的数值结果不同，
  且该差异与 K-shift 控制变量相关。
  它本身不等价于证明算子超出了官方误差标准，也不是完整模型精度评测。

  输入文件 SHA256：
  e2cdc5c2489c768dd58d768e106cfa7523b2db2b0145715ff6ec59cef38b34f1

7. 本包实测记录

  2026-11-02：98 机器现有 codex-static-kernel-cann91-20261031 容器，
  CANN 9.1.0，Ascend910_9382，空闲 die 9。
  本包 repro.py 已完整执行；默认 22 个差异，关闭 K-shift 后 0 个差异，
  普通和 Static 原生 kernel 命中检查通过，进程已退出并释放该 die。
  verified_cann91/ 保存本次汇总、逐输入报告、输出 Tensor、编译日志和原生选择日志。
  verified_cann91/proof.log 是完整的 DEBUG 日志，体积较大；优先查看 kernel_selection.log。
  CANN 9.2 的已知结论来自此前相同 fixture 的独立测试；本次交付脚本新验证环境为 9.1。
