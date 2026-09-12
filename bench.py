"""Native dense RDNA4 WMMA microbenchmark. Python 3.10+, AMD Windows driver."""
import argparse
import ctypes as C
import csv
import datetime
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parent
P = C.c_void_p
I = C.c_int
U = C.c_uint
S = C.c_size_t


class Handle(C.Structure):
    _fields_ = [('handle', C.c_uint64)]


def bind(dll, name, args, result=I):
    f = getattr(dll, name)
    f.argtypes, f.restype = args, result
    return f


def amd_library(names):
    errors = []
    for name in names:
        try:
            return C.CDLL(str(Path(os.environ['SystemRoot']) / 'System32' / name))
        except OSError as exc:
            errors.append(str(exc))
    raise RuntimeError('Biblioteca AMD nao encontrada. Requer driver HIP/COMGR compativel: ' + '; '.join(names))


class Compiler:
    def __init__(self):
        self.dll = amd_library(['amd_comgr_3.dll', 'amd_comgr_2.dll'])
        specs = {
            'create_data': [I, C.POINTER(Handle)], 'set_data': [Handle, S, P],
            'set_data_name': [Handle, C.c_char_p], 'release_data': [Handle],
            'create_data_set': [C.POINTER(Handle)], 'destroy_data_set': [Handle],
            'data_set_add': [Handle, Handle], 'create_action_info': [C.POINTER(Handle)],
            'destroy_action_info': [Handle], 'action_info_set_language': [Handle, I],
            'action_info_set_isa_name': [Handle, C.c_char_p],
            'action_info_set_option_list': [Handle, C.POINTER(C.c_char_p), S],
            'action_info_set_logging': [Handle, C.c_bool],
            'do_action': [I, Handle, Handle, Handle],
            'action_data_count': [Handle, I, C.POINTER(S)],
            'action_data_get_data': [Handle, I, S, C.POINTER(Handle)],
            'get_data': [Handle, C.POINTER(S), P],
        }
        for n, a in specs.items():
            setattr(self, n, bind(self.dll, 'amd_comgr_' + n, a))

    @staticmethod
    def ok(code):
        if code:
            raise RuntimeError('AMD COMGR error ' + str(code))

    def extract(self, dataset, kind):
        count = S()
        self.ok(self.action_data_count(dataset, kind, C.byref(count)))
        blobs = []
        for idx in range(count.value):
            h, size = Handle(), S()
            self.ok(self.action_data_get_data(dataset, kind, idx, C.byref(h)))
            try:
                self.ok(self.get_data(h, C.byref(size), None))
                buf = C.create_string_buffer(size.value)
                self.ok(self.get_data(h, C.byref(size), buf))
                blobs.append(buf.raw)
            finally:
                self.release_data(h)
        return blobs

    def compile(self, source, arch='gfx1201'):
        info, data, inp = Handle(), Handle(), Handle()
        sets = []
        self.ok(self.create_action_info(C.byref(info)))
        self.ok(self.create_data(1, C.byref(data)))
        self.ok(self.create_data_set(C.byref(inp)))
        sets.append(inp)
        try:
            self.ok(self.set_data(data, len(source), source))
            self.ok(self.set_data_name(data, b'bench.hip'))
            self.ok(self.data_set_add(inp, data))
            self.ok(self.action_info_set_language(info, 3))
            self.ok(self.action_info_set_isa_name(info, ('amdgcn-amd-amdhsa--' + arch).encode()))
            self.ok(self.action_info_set_logging(info, True))

            def action(kind, src, options):
                dst = Handle()
                self.ok(self.create_data_set(C.byref(dst)))
                sets.append(dst)
                opts = (C.c_char_p * len(options))(*(x.encode() for x in options))
                self.ok(self.action_info_set_option_list(info, opts, len(options)))
                code = self.do_action(kind, info, src, dst)
                if code:
                    log = b'\n'.join(self.extract(dst, 5)).decode(errors='replace')
                    raise RuntimeError(f'Compilation action {kind}: {code}\n{log}')
                return dst

            bc = action(2, inp, ['-O3', '-std=c++17', '-nogpuinc', '-nogpulib'])
            asm = action(5, bc, ['-O3'])
            assembly = b'\n'.join(self.extract(asm, 1))
            obj = action(4, bc, ['-O3'])
            exe = action(7, obj, [])
            blobs = self.extract(exe, 8)
            if len(blobs) != 1:
                raise RuntimeError('Expected one GPU code object')
            return blobs[0], assembly
        finally:
            for h in reversed(sets):
                self.destroy_data_set(h)
            self.release_data(data)
            self.destroy_action_info(info)


def kernel_source(mode):
    fp = mode == 'FP8'
    typ = 'float' if fp else 'int'
    operand = 'int' if mode == 'INT4' else 'v2i'
    builtin = {'FP8': '__builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12',
               'INT8': '__builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12',
               'INT4': '__builtin_amdgcn_wmma_i32_16x16x16_iu4_w32_gfx12'}[mode]
    call = lambda j: f'{builtin}(a{j}, b{j}, c{j})' if fp else f'{builtin}(true, a{j}, true, b{j}, c{j}, false)'
    decl = '\n'.join(f'{operand} a{j} = in[{2*j}], b{j} = in[{2*j+1}]; acc c{j} = {{}}; c{j} += {j};' for j in range(4))
    ops = '\n'.join(f'c{j} = {call(j)};' for j in range(4))
    stores = '\n'.join(f'out[tid*4+{j}] = c{j};' for j in range(4))
    return f'''
typedef int v2i __attribute__((ext_vector_type(2)));
typedef {typ} acc __attribute__((ext_vector_type(8)));
extern "C" __attribute__((global)) void bench(const {operand}* in, acc* out, int iterations) {{
    unsigned tid = __builtin_amdgcn_workgroup_id_x()*128 + __builtin_amdgcn_workitem_id_x();
    {decl}
    #pragma clang loop unroll(disable)
    for (int i=0; i<iterations; ++i) {{ {ops} }}
    {stores}
}}
'''.encode()


class GPU:
    def __init__(self, requested, allow_other=False):
        self.dll = amd_library(['amdhip64_7.dll', 'amdhip64_6.dll'])
        specs = {
            'hipInit': [U], 'hipGetDeviceCount': [C.POINTER(I)], 'hipSetDevice': [I],
            'hipDeviceGetName': [P, I, I], 'hipDeviceGetAttribute': [C.POINTER(I), I, I],
            'hipRuntimeGetVersion': [C.POINTER(I)], 'hipDriverGetVersion': [C.POINTER(I)],
            'hipMalloc': [C.POINTER(P), S], 'hipFree': [P], 'hipMemcpy': [P, P, S, I],
            'hipModuleLoadData': [C.POINTER(P), P], 'hipModuleGetFunction': [C.POINTER(P), P, C.c_char_p],
            'hipModuleUnload': [P],
            'hipModuleLaunchKernel': [P, U, U, U, U, U, U, U, P, C.POINTER(P), C.POINTER(P)],
            'hipDeviceSynchronize': [], 'hipEventCreate': [C.POINTER(P)],
            'hipEventRecord': [P, P], 'hipEventSynchronize': [P],
            'hipEventElapsedTime': [C.POINTER(C.c_float), P, P], 'hipEventDestroy': [P],
        }
        for n, a in specs.items():
            setattr(self, n, bind(self.dll, n, a))
        self.error_string = bind(self.dll, 'hipGetErrorString', [I], C.c_char_p)
        self.ok(self.hipInit(0))
        count = I()
        self.ok(self.hipGetDeviceCount(C.byref(count)))
        self.devices = []
        for i in range(count.value):
            name = C.create_string_buffer(256)
            self.ok(self.hipDeviceGetName(name, 256, i))
            self.devices.append(name.value.decode())
        matches = [i for i, n in enumerate(self.devices) if '9070' in n and 'XT' in n]
        if requested is None:
            if len(matches) != 1:
                raise RuntimeError(f'Select --device from {self.devices}')
            requested = matches[0]
        if requested not in range(len(self.devices)):
            raise RuntimeError('Invalid GPU index')
        if requested not in matches and not allow_other:
            raise RuntimeError('This benchmark is restricted to RX 9070 XT / gfx1201')
        self.device = requested
        self.ok(self.hipSetDevice(requested))

    def ok(self, status):
        if status:
            raise RuntimeError(self.error_string(status).decode())

    def alloc(self, size):
        p = P()
        self.ok(self.hipMalloc(C.byref(p), size))
        return p


def run_mode(gpu, compiler, mode, folder, samples):
    print(f'Compilando {mode} nativo...', flush=True)
    source = kernel_source(mode)
    binary, assembly = compiler.compile(source)
    (folder / f'{mode}.hip').write_bytes(source)
    (folder / f'{mode}.s').write_bytes(assembly)
    mnemonic = {'FP8': 'v_wmma_f32_16x16x16_fp8_fp8', 'INT8': 'v_wmma_i32_16x16x16_iu8', 'INT4': 'v_wmma_i32_16x16x16_iu4'}[mode]
    asm_text = assembly.decode(errors='replace')
    native_count = sum(1 for line in asm_text.splitlines() if line.strip().startswith(mnemonic))
    if native_count != 4:
        raise RuntimeError(f'{mode}: expected exactly 4 native WMMA instructions, found {native_count}')
    if '.amdhsa_wavefront_size32 1' not in asm_text:
        raise RuntimeError('Wave32 not confirmed in assembly')
    module, fn, start, end = P(), P(), P(), P()
    codebuf = C.create_string_buffer(binary)
    gpu.ok(gpu.hipModuleLoadData(C.byref(module), codebuf))
    gpu.ok(gpu.hipModuleGetFunction(C.byref(fn), module, b'bench'))
    gpu.ok(gpu.hipEventCreate(C.byref(start)))
    gpu.ok(gpu.hipEventCreate(C.byref(end)))
    din, dout = gpu.alloc(64), gpu.alloc(1024*128*4*8*4)
    scalar = C.c_float if mode == 'FP8' else I
    try:
        def inputs(negative=False):
            # A=1 (or -1), B=1, all dense lanes; no sparsity metadata.
            a = (0xb8 if negative else 0x38) if mode == 'FP8' else ((0xf if mode == 'INT4' else 0xff) if negative else 1)
            b = 0x38 if mode == 'FP8' else 1
            if mode == 'INT4':
                packed = bytes([a | a << 4])*4 + bytes([b | b << 4])*4
            else:
                packed = bytes([a])*8 + bytes([b])*8
            host = C.create_string_buffer(packed*4)
            gpu.ok(gpu.hipMemcpy(din, host, len(packed)*4, 1))

        def launch(blocks, iterations):
            it = I(iterations)
            args = (P * 3)(C.cast(C.byref(din), P), C.cast(C.byref(dout), P), C.cast(C.byref(it), P))
            gpu.ok(gpu.hipModuleLaunchKernel(fn, blocks, 1, 1, 128, 1, 1, 0, None, args, None))

        def timed(blocks, iterations):
            gpu.ok(gpu.hipEventRecord(start, None))
            launch(blocks, iterations)
            gpu.ok(gpu.hipEventRecord(end, None))
            gpu.ok(gpu.hipEventSynchronize(end))
            ms = C.c_float()
            gpu.ok(gpu.hipEventElapsedTime(C.byref(ms), start, end))
            if ms.value <= 0:
                raise RuntimeError('Invalid GPU timer')
            return ms.value

        def validate(blocks, iterations, sign):
            host = (scalar * (blocks*128*4*8))()
            gpu.ok(gpu.hipMemcpy(host, dout, C.sizeof(host), 2))
            for idx, value in enumerate(host):
                expected = sign*16*iterations + (idx//8) % 4
                if value != expected:
                    raise RuntimeError(f'{mode} validation failed at {idx}: {value} != {expected}')

        for negative in (False, True):
            inputs(negative)
            launch(2, 3)
            gpu.ok(gpu.hipDeviceSynchronize())
            validate(2, 3, -1 if negative else 1)
        inputs()
        rows = []
        for blocks in (256, 512, 1024):
            iterations = 4096
            probe = timed(blocks, iterations)
            iterations = max(4096, min(65536, int(iterations*8/max(probe, 0.01))))
            # Warm up for at least 0.3 seconds, bounded individual launches.
            until = time.perf_counter() + 0.3
            while time.perf_counter() < until:
                timed(blocks, iterations)
            times = [timed(blocks, iterations) for _ in range(samples)]
            validate(blocks, iterations, 1)
            ops = blocks * 4 * iterations * 4 * 2 * 16 * 16 * 16
            median = statistics.median(times)
            row = {'mode': mode, 'blocks': blocks, 'threads_per_block': 128,
                   'iterations': iterations, 'operations_per_launch': ops,
                   'median_ms': median, 'min_ms': min(times), 'max_ms': max(times),
                   'median_tops': ops/(median*1e9), 'best_tops': ops/(min(times)*1e9),
                   'samples_ms': times, 'validated': True}
            rows.append(row)
            print(f'  {blocks:4} blocos: {row["median_tops"]:.2f} {"TFLOPS" if mode == "FP8" else "TOPS"} (mediana)', flush=True)
        return {'mode': mode, 'native_instruction': mnemonic, 'native_instructions_in_loop': native_count,
                'source_sha256': hashlib.sha256(source).hexdigest(), 'configurations': rows,
                'best_configuration': max(rows, key=lambda r: r['median_tops'])}
    finally:
        gpu.hipFree(din)
        gpu.hipFree(dout)
        gpu.hipEventDestroy(start)
        gpu.hipEventDestroy(end)
        gpu.hipModuleUnload(module)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--device', type=int)
    ap.add_argument('--samples', type=int, default=21)
    ap.add_argument('--modes', nargs='+', choices=['FP8', 'INT8', 'INT4'], default=['FP8', 'INT8', 'INT4'])
    args = ap.parse_args()
    if not 5 <= args.samples <= 200:
        ap.error('--samples must be 5..200')
    gpu = GPU(args.device)
    print('GPU:', gpu.devices[gpu.device], '| indice HIP:', gpu.device, flush=True)
    folder = ROOT / 'results' / datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    folder.mkdir(parents=True, exist_ok=False)
    report = {'timestamp': datetime.datetime.now().astimezone().isoformat(),
              'gpu': gpu.devices[gpu.device], 'device_index': gpu.device, 'devices': gpu.devices,
              'benchmark': 'dense native WMMA register throughput, not end-to-end GEMM/inference',
              'operation_convention': 'multiply + add = 2 operations; no sparse multiplier',
              'python': sys.version, 'results': [], 'errors': []}
    for key, call in [('hip_runtime', gpu.hipRuntimeGetVersion), ('hip_driver', gpu.hipDriverGetVersion)]:
        ver = I()
        gpu.ok(call(C.byref(ver)))
        report[key] = ver.value
    compiler = Compiler()
    for mode in args.modes:
        try:
            report['results'].append(run_mode(gpu, compiler, mode, folder, args.samples))
        except Exception as exc:
            report['errors'].append({'mode': mode, 'error': str(exc)})
            print(f'FALHA {mode}: {exc}', flush=True)
        (folder / 'results.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print('Resultados:', folder, flush=True)
    with (folder / 'results.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['precision', 'unit', 'blocks', 'iterations', 'median_ms', 'median_throughput', 'best_throughput', 'validated'])
        for result in report['results']:
            for row in result['configurations']:
                writer.writerow([row['mode'], 'TFLOPS' if row['mode'] == 'FP8' else 'TOPS', row['blocks'], row['iterations'], row['median_ms'], row['median_tops'], row['best_tops'], row['validated']])
    lines = ['# Benchmark RX 9070 XT', '', report['timestamp'], '',
             'Throughput denso de instruções WMMA, com operandos reutilizados em registradores.',
             'Melhor configuração pela mediana de cada conjunto de amostras. Multiplicação + soma = 2 operações.', '',
             '| Precisão | Mediana da melhor configuração | Blocos |', '|---|---:|---:|']
    for result in report['results']:
        row = result['best_configuration']
        unit = 'TFLOPS' if row['mode'] == 'FP8' else 'TOPS'
        lines.append(f'| {row["mode"]} | {row["median_tops"]:.2f} {unit} | {row["blocks"]} |')
    lines += ['', 'FP8 E4M3 acumula em FP32; INT8 e INT4 assinados acumulam em INT32.',
              'Instruções nativas e Wave32 verificadas no assembly. Saídas verificadas com entradas uniformes +1 e -1, e após cada configuração cronometrada.',
              'Estes números não medem inferência, GEMM completo, largura de banda, desempenho esparso ou o limite absoluto da placa.',
              'Clocks, temperatura, consumo e outras cargas não são controlados nem medidos.', '',
              f'Amostras por configuração: {args.samples}. Tempos brutos em results.json; comparação em results.csv.']
    if report['errors']:
        lines += ['', 'Falhas: ' + json.dumps(report['errors'], ensure_ascii=False)]
    (folder / 'RESUMO.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return 1 if report['errors'] else 0


if __name__ == '__main__':
    try:
        from gpu_bench import main as universal_main
        sys.exit(universal_main())
    except Exception as exc:
        print('ERRO:', exc, file=sys.stderr)
        sys.exit(1)
