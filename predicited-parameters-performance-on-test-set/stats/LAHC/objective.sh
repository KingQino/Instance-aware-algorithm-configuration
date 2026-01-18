#!/bin/bash
set -euo pipefail

output_file="a.txt"
> "$output_file"   # clear/create

# 按 ins 的顺序写死所有实例（目录名 = 去掉 .evrp 后缀）
instances=(
  "E-n22-k4"
  "E-n112-k8-s11"
  "M-n212-k16-s12"
  "F-n140-k5-s5"
  "X-n221-k11-s7"
  "X-n469-k26-s10"
  "X-n698-k75-s13"
  "X-n1006-k43-s5"
)

# 把每个 stats 文件的最后 3 行按顺序拼到 a.txt
for dir in "${instances[@]}"; do
  stats_file="${dir}/stats.${dir}.evrp"
  if [[ -f "$stats_file" ]]; then
    tail -n 3 "$stats_file" >> "$output_file"
  else
    echo "File not found: $stats_file" >&2
  fi
done

# 保持你原来的输出格式：每个实例输出 3 行（min / mean / std）
# 兼容 "Min:" 或 "Min" 开头，mean/std 从 Mean 行取
awk '
  /^Mean/ {
    mean = $2
    std  = $NF
  }
  /^Min/ {
    # Min 行最后一个数值字段当作 min（兼容 Min: 123 或 Min: xxx 123）
    min=""
    for (i = NF; i >= 1; --i) {
      if ($i ~ /^-?[0-9]+(\.[0-9]+)?([eE][-+]?[0-9]+)?$/) { min=$i; break }
    }
    if (min=="" || mean=="" || std=="") {
      # 如果解析不到，仍然输出占位，避免打乱顺序
      print "NA"
      print "NA"
      print "NA"
    } else {
      print min
      print mean
      print std
    }
    # reset for next block
    min=mean=std=""
  }
' "$output_file"

rm -f "$output_file"

