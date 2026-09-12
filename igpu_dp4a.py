"""INT8 signed native dot4 benchmark for gfx1036; no WMMA or emulation."""
import ctypes as C
import datetime
import json
import re
import statistics
import time
from bench import GPU, Compiler, bind, P, I, ROOT


def main():
    gpu = GPU(1, allow_other=True)
    props = C.create_string_buffer(16384)
    getprops = bind(gpu.dll, 'hipGetDevicePropertiesR0600', [P, I])
    gpu.ok(getprops(props, gpu.device))
    arches = re.findall(rb'gfx[0-9a-f]+', props.raw)
    if arches != [b'gfx1036']:
        raise RuntimeError(f'Expected gfx1036 iGPU; found {arches}')
    print('GPU:', gpu.devices[gpu.device], 'gfx1036', flush=True)
    chains = 8
    decl = '\n'.join(f'int a{j}=input[{j*2}], b{j}=input[{j*2+1}], c{j}={j};' for j in range(chains))
    ops = '\n'.join(f'c{j}=__builtin_amdgcn_sdot4(a{j},b{j},c{j},false);' for j in range(chains))
    stores = '\n'.join(f'output[tid*{chains}+{j}]=c{j};' for j in range(chains))
    source = f'''extern "C" __attribute__((global)) void bench(const int* input, int* output, int iterations) {{
    unsigned tid=__builtin_amdgcn_workgroup_id_x()*128+__builtin_amdgcn_workitem_id_x();
    {decl}
    #pragma clang loop unroll(disable)
    for(int i=0;i<iterations;++i) {{ {ops} }}
    {stores}
    }}'''.encode()
    folder = ROOT / 'results' / ('igpu-dp4a-' + datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
    folder.mkdir(parents=True)
    binary, assembly = Compiler().compile(source, 'gfx1036')
    (folder / 'INT8-DP4A.hip').write_bytes(source)
    (folder / 'INT8-DP4A.s').write_bytes(assembly)
    native = [line.strip() for line in assembly.decode().splitlines() if line.strip().startswith('v_dot4c_i32_i8')]
    if len(native) != chains:
        raise RuntimeError(f'Expected {chains} native dot4 instructions; got {len(native)}')
    module, fn, start, end = P(), P(), P(), P()
    codebuf = C.create_string_buffer(binary)
    gpu.ok(gpu.hipModuleLoadData(C.byref(module), codebuf))
    gpu.ok(gpu.hipModuleGetFunction(C.byref(fn), module, b'bench'))
    gpu.ok(gpu.hipEventCreate(C.byref(start)))
    gpu.ok(gpu.hipEventCreate(C.byref(end)))
    din, dout = gpu.alloc(chains*8), gpu.alloc(128*128*chains*4)
    rows = []
    try:
        def inputs(negative=False):
            # Nonuniform bytes, signed test, distinct operands per chain.
            values, dots = [], []
            for j in range(chains):
                a = [1, -2 if negative else 2, 3, j+1]
                b = [2, 3, -1 if negative else 1, 2]
                dots.append(sum(x*y for x,y in zip(a,b)))
                for vec in (a,b):
                    values.append(int.from_bytes(bytes(x & 255 for x in vec), 'little', signed=True))
            host = (I*len(values))(*values)
            gpu.ok(gpu.hipMemcpy(din,host,C.sizeof(host),1))
            return dots

        def launch(blocks, iterations):
            it = I(iterations)
            args = (P*3)(C.cast(C.byref(din),P),C.cast(C.byref(dout),P),C.cast(C.byref(it),P))
            gpu.ok(gpu.hipModuleLaunchKernel(fn,blocks,1,1,128,1,1,0,None,args,None))

        def timed(blocks, iterations):
            gpu.ok(gpu.hipEventRecord(start,None))
            launch(blocks,iterations)
            gpu.ok(gpu.hipEventRecord(end,None))
            gpu.ok(gpu.hipEventSynchronize(end))
            ms=C.c_float()
            gpu.ok(gpu.hipEventElapsedTime(C.byref(ms),start,end))
            if ms.value<=0:
                raise RuntimeError('Invalid GPU timer')
            return ms.value

        def validate(blocks, iterations, dots):
            host=(I*(blocks*128*chains))()
            gpu.ok(gpu.hipMemcpy(host,dout,C.sizeof(host),2))
            for k,value in enumerate(host):
                j=k%chains
                if value != j+iterations*dots[j]:
                    raise RuntimeError(f'Validation failed at {k}: {value}')

        for negative in (False,True):
            dots=inputs(negative)
            launch(2,7)
            gpu.ok(gpu.hipDeviceSynchronize())
            validate(2,7,dots)
        dots=inputs()
        for blocks in (16,32,64,128):
            probe=timed(blocks,1024)
            iterations=max(1024,min(65536,int(1024*8/probe)))
            until=time.perf_counter()+0.5
            while time.perf_counter()<until:
                timed(blocks,iterations)
            times=[timed(blocks,iterations) for _ in range(21)]
            validate(blocks,iterations,dots)
            count=blocks*128*iterations*chains*8
            row={'blocks':blocks,'threads_per_block':128,'iterations':iterations,
                 'operations':count,'samples_ms':times,'median_ms':statistics.median(times),
                 'median_tops':count/(statistics.median(times)*1e9),'validated':True}
            rows.append(row)
            print(f'{blocks} blocos: {row["median_tops"]:.4f} TOPS INT8 DP4A',flush=True)
        report={'gpu':gpu.devices[gpu.device],'arch':'gfx1036','device_index':gpu.device,
                'instruction':'v_dot4c_i32_i8','precision':'signed INT8 x INT8, INT32 accumulator',
                'method':'Register throughput. 4 multiplies + 4 adds = 8 ops per dot4 per thread. No WMMA.',
                'chains':chains,'samples_per_configuration':21,'configurations':rows,
                'best_configuration':max(rows,key=lambda r:r['median_tops']),
                'validation':'All outputs, mixed signed bytes before timing; positive bytes after each timed configuration.',
                'limitations':'Not GEMM/inference. Clocks, temperature and power not measured or controlled.'}
        (folder/'results.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
        print('Resultados:',folder,flush=True)
    finally:
        gpu.hipFree(din)
        gpu.hipFree(dout)
        gpu.hipEventDestroy(start)
        gpu.hipEventDestroy(end)
        gpu.hipModuleUnload(module)


if __name__=='__main__':
    main()

