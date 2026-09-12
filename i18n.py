"""GUI translations. Stable mode IDs and driver diagnostics are not translated."""
import re

LANGUAGES={'English':'en','Português (Brasil)':'pt-BR','简体中文':'zh-CN'}
DEFAULT_LANGUAGE='en'
# Portuguese source, English, Simplified Chinese. Placeholders preserve GPU names
# and numeric values when switching languages without restarting a benchmark.
MESSAGES=[
('Benchmark GPU','GPU Benchmark','GPU 基准测试'),
('AMD · NVIDIA · Intel Arc   |   Medições nativas de cálculo','AMD · NVIDIA · Intel Arc   |   Native compute benchmarks','AMD · NVIDIA · Intel Arc   |   原生计算基准测试'),
('Idioma','Language','语言'),
('Placa de vídeo','Graphics card','显卡'),
('Amostras por configuração','Samples per configuration','每种配置的采样次数'),
('Atualizar GPUs','Refresh GPUs','刷新 GPU'),
('Testes disponíveis','Available tests','可用测试'),
('Separar em Vetor e Matriz','Separate Vector and Matrix','按向量和矩阵分类'),
('Selecionar todos','Select all','全选'),
('Limpar seleção','Clear selection','清除选择'),
('Compatibilidade','Compatibility','兼容性'),
('Iniciar benchmark','Start benchmark','开始测试'),
('Cancelar','Cancel','取消'),
('Abrir resultados','Open results','打开结果'),
('Detectando GPUs…','Detecting GPUs…','正在检测 GPU…'),
('Resultados da execução','Benchmark results','测试结果'),
('Resultados','Results','结果'),
('Registro da execução','Technical log (original language)','技术日志（原始语言）'),
('Estatísticas da configuração com melhor mediana. Esparsos: taxa equivalente da matriz expandida.','Statistics for the configuration with the best median. Sparse: dense-equivalent throughput.','显示中位数最高配置的统计数据。稀疏测试：按等效稠密矩阵计算吞吐量。'),
('Teste','Test','测试'),('Unidade','Unit','单位'),('Mínimo','Minimum','最小值'),
('Máximo','Maximum','最大值'),('Média','Mean','平均值'),('Mediana','Median','中位数'),
('{gpu} · ainda sem resultados','{gpu} · no results yet','{gpu} · 暂无结果'),
('Vetor ({count})','Vector ({count})','向量 ({count})'),
('Matriz ({count})','Matrix ({count})','矩阵 ({count})'),
('Todos ({count})','All ({count})','全部 ({count})'),
('Nenhum teste desta categoria disponível.','No tests available in this category.','此类别暂无可用测试。'),
('Verificando os testes compatíveis…','Checking compatible tests…','正在检查兼容测试…'),
('Todos os testes estão disponíveis.','All tests are available.','所有测试均可用。'),
('Amostras','Samples','采样次数'),
('Informe um número inteiro entre 5 e 200.','Enter a whole number from 5 to 200.','请输入 5 到 200 之间的整数。'),
('Resultados · {gpu} · {count} amostras por configuração','Results · {gpu} · {count} samples per configuration','结果 · {gpu} · 每种配置 {count} 次采样'),
('Iniciando benchmark…','Starting benchmark…','正在启动测试…'),
('{count} de {total} testes concluídos','{count} of {total} tests completed','已完成 {count}/{total} 项测试'),
('Cancelamento','Cancellation','取消测试'),
('Cancelando após o teste atual. Os resultados concluídos serão preservados.','Cancelling after the current test. Completed results will be preserved.','将在当前测试结束后取消。已完成的结果将被保留。'),
('Nenhuma GPU detectada. Consulte o registro e os drivers.','No GPU detected. Check the technical log and drivers.','未检测到 GPU。请检查技术日志和驱动程序。'),
('Pronto · {count} testes disponíveis','Ready · {count} tests available','就绪 · {count} 项可用测试'),
('Falha ao detectar capacidades. Consulte o registro.','Could not detect capabilities. Check the technical log.','无法检测设备功能。请查看技术日志。'),
('Concluído. Resultados salvos.','Completed. Results saved.','测试完成，结果已保存。'),
('Cancelado. Resultados parciais salvos.','Cancelled. Partial results saved.','已取消，部分结果已保存。'),
('Execução com falha. Consulte o registro.','Benchmark failed. Check the technical log.','测试失败。请查看技术日志。'),
('unsupported','Unsupported','不支持'),('failed','Failed','失败'),
('Requisitos não atendidos para {mode}. Detalhes técnicos originais:','Requirements not met for {mode}. Original technical details:','不满足 {mode} 的要求。原始技术详情：'),
('FP8 E4M3 - matriz densa','FP8 E4M3 - dense matrix','FP8 E4M3 - 稠密矩阵'),
('INT8 - matriz densa / DOT4','INT8 - dense matrix / DOT4','INT8 - 稠密矩阵 / DOT4'),
('INT4 - matriz densa / DOT8','INT4 - dense matrix / DOT8','INT4 - 稠密矩阵 / DOT8'),
('FP32 - vetor','FP32 - vector','FP32 - 向量'),
('FP16 - vetor packed','FP16 - packed vector','FP16 - 打包向量'),
('FP16 - matriz densa (acumulacao FP32)','FP16 - dense matrix (FP32 accumulation)','FP16 - 稠密矩阵（FP32 累加）'),
('FP16 - matriz esparsa 2:4 (acumulacao FP32)','FP16 - sparse 2:4 (FP32 accumulation)','FP16 - 2:4 稀疏矩阵（FP32 累加）'),
('FP8 E4M3 - matriz esparsa 2:4','FP8 E4M3 - sparse 2:4 matrix','FP8 E4M3 - 2:4 稀疏矩阵'),
('FP8 E5M2 - matriz esparsa 2:4','FP8 E5M2 - sparse 2:4 matrix','FP8 E5M2 - 2:4 稀疏矩阵'),
('INT8 - matriz esparsa 2:4','INT8 - sparse 2:4 matrix','INT8 - 2:4 稀疏矩阵'),
('INT4 - matriz esparsa 2:4','INT4 - sparse 2:4 matrix','INT4 - 2:4 稀疏矩阵'),
('INT16 — adição packed de 16 bits','INT16 — packed 16-bit addition','INT16 — 打包 16 位加法'),
('INT32 — inteiros de 32 bits','INT32 — 32-bit integers','INT32 — 32 位整数'),
('INT4 — produto escalar packed INT4 (DOT8)','INT4 — packed dot product (DOT8)','INT4 — 打包点积（DOT8）'),
('INT1 — binário AND + popcount','INT1 — binary AND + popcount','INT1 — 二进制 AND + 位计数'),
('MXFP6 — E3M2 + escala E8M0','MXFP6 — E3M2 + E8M0 scale','MXFP6 — E3M2 + E8M0 缩放'),
('MXFP8 — E4M3 + escala E8M0','MXFP8 — E4M3 + E8M0 scale','MXFP8 — E4M3 + E8M0 缩放'),
('MXFP4 — E2M1 + escala E8M0','MXFP4 — E2M1 + E8M0 scale','MXFP4 — E2M1 + E8M0 缩放'),
('NVFP4 — E2M1 + escala E4M3','NVFP4 — E2M1 + E4M3 scale','NVFP4 — E2M1 + E4M3 缩放'),
('{format} — cálculo nativo','{format} — native compute','{format} — 原生计算'),
]

_INDEX={'pt-BR':0,'en':1,'zh-CN':2}
_EXACT={text:row for row in MESSAGES for text in row if '{' not in text}
_PATTERNS=[]
for row in MESSAGES:
    if '{' not in row[0]: continue
    for template in row:
        parts=re.split(r'(\{\w+\})',template)
        expression=''.join('(?P<'+p[1:-1]+'>.+?)' if p.startswith('{') else re.escape(p) for p in parts)
        _PATTERNS.append((re.compile(expression,re.DOTALL),row))

def translate(text,language=DEFAULT_LANGUAGE):
    text=str(text); index=_INDEX[language]
    if text in _EXACT: return _EXACT[text][index]
    for pattern,row in _PATTERNS:
        match=pattern.fullmatch(text)
        if match: return row[index].format(**match.groupdict())
    return text
