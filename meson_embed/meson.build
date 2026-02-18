# meson_embed/meson.build - バイナリリソース埋め込み

embed_script = files('embed_resource.sh')

# nnedi3_weights.bin
# シンボル: _binary_resource_nnedi3_weights_bin_start/end
resource_nnedi3 = custom_target('embed_nnedi3_weights',
  input: '../resource/nnedi3_weights.bin',
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
