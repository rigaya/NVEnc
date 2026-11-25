# meson_embed/meson.build - バイナリリソース埋め込み

embed_script = files('embed_resource.sh')

# nnedi3_weights.bin
# シンボル: _binary_resource_nnedi3_weights_bin_start/end
nnedi3_weights = '../resource/nnedi3_weight/nnedi3_weights.bin'
if nvenc_probe
  # 診断で使わない重みは、既存のシンボルを解決できるように1バイトだけ埋め込む。
  nnedi3_weights = custom_target('nnedi3_probe_weights',
    output: 'nnedi3_probe_weights.bin',
    command: ['sh', '-c', 'head -c 1 /dev/zero > "$1"', 'sh', '@OUTPUT@'],
  )
endif
resource_nnedi3 = custom_target('embed_nnedi3_weights',
  input: nnedi3_weights,
  output: 'nnedi3_weights.o',
  command: ['bash', embed_script, '@INPUT@', '@OUTPUT@', 'resource_nnedi3_weights_bin'],
)

# NVEncFilterColorspaceFunc.h
# シンボル: _binary_NVEncCore_NVEncFilterColorspaceFunc_h_start/end
resource_colorspace = custom_target('embed_colorspace_func',
  input: '../NVEncCore/NVEncFilterColorspaceFunc.h',
  output: 'NVEncFilterColorspaceFunc.o',
  command: ['bash', embed_script, '@INPUT@', '@OUTPUT@', 'NVEncCore_NVEncFilterColorspaceFunc_h'],
)

# perf_monitor.pyw
# シンボル: _binary_PerfMonitor_perf_monitor_pyw_start/end
resource_perfmon = custom_target('embed_perf_monitor',
  input: '../PerfMonitor/perf_monitor.pyw',
  output: 'perf_monitor.o',
  command: ['bash', embed_script, '@INPUT@', '@OUTPUT@', 'PerfMonitor_perf_monitor_pyw'],
)
