"""Windows GPU benchmark: AMD HIP, NVIDIA CUDA and Intel OpenCL/XMX.
No conversion fallback. Unsupported modes are reported individually.
"""
import argparse
import ctypes as C
import csv
import datetime
import json
import math
import os
from pathlib import Path
import re
import statistics
import sys
import time
from functools import lru_cache
from bench import GPU, Compiler, bind, P, I, U, S, ROOT, kernel_source, amd_library
from extended_kernels import EXTRA_MODES, describe, amd_extended, nvidia_extended, payload
from intel_opencl import OpenCL, intel_kernel, run_intel
from fp32_kernels import VARIANTS, named_candidate
from more_formats import NEW_MODES,is_float,category,amd_more,nvidia_more

MODES = ('FP8', 'INT8', 'INT4') + EXTRA_MODES + NEW_MODES

def unit_for(mode): return 'TFLOPS' if is_float(mode) else 'TOPS'

def sample_statistics(operations, times, sparse=False):
    if not times or any(not math.isfinite(t) or t <= 0 for t in times):
        raise ValueError('Amostras devem conter tempos positivos e finitos.')
    rates = [operations / (t * 1e9) for t in times]
    row = {'samples_ms': times, 'samples_tops': rates}
    for name, fn in [('min', min), ('max', max), ('mean', statistics.mean), ('median', statistics.median)]:
        row[name + '_ms'] = fn(times)
        row[name + '_tops'] = fn(rates)
        if sparse: row['nonzero_' + name + '_tops'] = fn(rates) / 2
    if sparse: row['nonzero_tops'] = row['nonzero_median_tops']
    return row

class Unsupported(RuntimeError): pass


def parse_modes(values):
    tokens = [x.upper() for v in values for x in re.split(r'[,;\s]+', v.strip()) if x]
    if tokens in (['ALL'],['TODOS']): return list(MODES)
    aliases = {str(i+1):m for i,m in enumerate(MODES)}
    tokens = [aliases.get(x, x) for x in tokens]
    tokens = ['MXFP4_MATRIX' if x in ('MXPF4','MXPF4_MATRIX','MXFP4') else x for x in tokens]
    if not tokens or any(x not in MODES for x in tokens):
        raise ValueError('Escolha nomes ou numeros do menu, separados por virgula/espaco; ALL seleciona todos.')
    return list(dict.fromkeys(tokens))


@lru_cache(maxsize=32)
def compile_amd(source,arch):
    return Compiler().compile(source,arch)


def audit_amd(mode,spec,asm):
    if spec.get('audit_regex'):
        count=sum(bool(re.match(spec['audit_regex'],line.strip())) for line in asm.decode().splitlines())
    elif mode=='FP32_VECTOR' and spec['mnemonic']=='v_fma':
        count=sum(len(re.findall(r'\bv_(?:dual_)?fma(?:c|ak|mk)?_f32(?:_e32|_e64)?\b',line)) for line in asm.decode().splitlines())
    else: count=sum(line.strip().startswith(spec['mnemonic']) for line in asm.decode().splitlines())
    expected=spec.get('native_count',4)
    if count!=expected: raise RuntimeError(f'Instruction audit: expected {expected} {spec["mnemonic"]}, got {count}')
    return expected


def available_modes(info,int8_path='auto',int4_k=32,probe_amd=False):
    available=[]; unavailable={}
    for mode in MODES:
        try:
            if info['vendor']=='Intel': intel_kernel(info,mode,Unsupported)
            elif info['vendor']=='NVIDIA': nvidia_kernel(info,mode,int8_path)
            else:
                source,spec=amd_kernel(info,mode,int4_k)
                if probe_amd:
                    _,asm=compile_amd(source,info['arch']); audit_amd(mode,spec,asm)
                elif mode in ('INT8','INT4') and not info['arch'].startswith(('gfx10','gfx11','gfx12')):
                    raise Unsupported('DOT requer verificacao pelo compilador nesta arquitetura.')
            available.append(mode)
        except (Unsupported,RuntimeError) as exc: unavailable[mode]=str(exc)
    return available,unavailable


def select_modes(values,available,unavailable):
    tokens=[x.upper() for v in values for x in re.split(r'[,;\s]+',v.strip()) if x]
    selected=list(available) if tokens in (['ALL'],['TODOS']) else parse_modes(values)
    rejected=[m for m in selected if m not in available]
    if rejected:
        raise ValueError('Indisponivel para esta GPU/backend: '+'; '.join(m+': '+unavailable.get(m,'sem suporte') for m in rejected))
    if not selected: raise ValueError('Nenhum teste disponivel para esta GPU/backend.')
    return selected


class CUDA:
    def __init__(self, index=None):
        self.dll = C.CDLL(str(Path(os.environ['SystemRoot'])/'System32/nvcuda.dll'))
        self.ctx = P()
        specs = {
            'cuInit': [U], 'cuDeviceGetCount': [C.POINTER(I)],
            'cuDeviceGet': [C.POINTER(I), I], 'cuDeviceGetName': [P,I,I],
            'cuDeviceComputeCapability': [C.POINTER(I),C.POINTER(I),I],
            'cuDeviceGetAttribute': [C.POINTER(I),I,I],
            'cuCtxCreate_v2': [C.POINTER(P),U,I], 'cuCtxDestroy_v2': [P],
            'cuGetErrorString': [I,C.POINTER(C.c_char_p)],
            'cuDriverGetVersion': [C.POINTER(I)],
            'cuMemAlloc_v2': [C.POINTER(P),S], 'cuMemFree_v2': [P],
            'cuMemcpyHtoD_v2': [P,P,S], 'cuMemcpyDtoH_v2': [P,P,S],
            'cuModuleLoadDataEx': [C.POINTER(P),P,U,C.POINTER(I),C.POINTER(P)],
            'cuModuleGetFunction': [C.POINTER(P),P,C.c_char_p], 'cuModuleUnload': [P],
            'cuLaunchKernel': [P,U,U,U,U,U,U,U,P,C.POINTER(P),C.POINTER(P)],
            'cuEventCreate': [C.POINTER(P),U], 'cuEventRecord': [P,P],
            'cuEventSynchronize': [P], 'cuEventElapsedTime': [C.POINTER(C.c_float),P,P],
            'cuEventDestroy_v2': [P], 'cuCtxSynchronize': []}
        for n,a in specs.items(): setattr(self,n,bind(self.dll,n,a))
        self.ok(self.cuInit(0))
        count=I(); self.ok(self.cuDeviceGetCount(C.byref(count)))
        self.devices=[]
        for i in range(count.value):
            dev=I(); self.ok(self.cuDeviceGet(C.byref(dev),i))
            name=C.create_string_buffer(256); major,minor=I(),I()
            self.ok(self.cuDeviceGetName(name,256,dev))
            self.ok(self.cuDeviceComputeCapability(C.byref(major),C.byref(minor),dev))
            self.devices.append({'id':f'nvidia:{i}','vendor':'NVIDIA','index':i,'handle':dev.value,
                                 'name':name.value.decode(),'arch':f'sm_{major.value}{minor.value}',
                                 'cc':major.value*10+minor.value})
        if index is not None:
            self.info=self.devices[index]
            self.ok(self.cuCtxCreate_v2(C.byref(self.ctx),0,self.info['handle']))
        for hip,cu in {'hipFree':'cuMemFree_v2','hipModuleGetFunction':'cuModuleGetFunction',
                       'hipModuleUnload':'cuModuleUnload','hipModuleLaunchKernel':'cuLaunchKernel',
                       'hipDeviceSynchronize':'cuCtxSynchronize','hipEventRecord':'cuEventRecord',
                       'hipEventSynchronize':'cuEventSynchronize','hipEventElapsedTime':'cuEventElapsedTime',
                       'hipEventDestroy':'cuEventDestroy_v2'}.items(): setattr(self,hip,getattr(self,cu))
    def ok(self,status):
        if status:
            msg=C.c_char_p(); self.cuGetErrorString(status,C.byref(msg))
            raise RuntimeError(f'CUDA {status}: '+(msg.value or b'unknown').decode())
    def alloc(self,size):
        p=P(); self.ok(self.cuMemAlloc_v2(C.byref(p),size)); return p
    def hipMemcpy(self,dst,src,size,kind):
        return self.cuMemcpyHtoD_v2(dst,src,size) if kind==1 else self.cuMemcpyDtoH_v2(dst,src,size)
    def hipEventCreate(self,out): return self.cuEventCreate(out,0)
    def load(self,module,code):
        log=C.create_string_buffer(16384)
        options=(I*2)(5,6)  # CU_JIT_ERROR_LOG_BUFFER / BUFFER_SIZE_BYTES
        values=(P*2)(C.cast(log,P),P(len(log)))
        status=self.cuModuleLoadDataEx(C.byref(module),code,2,options,values)
        if status:
            raise RuntimeError(f'CUDA JIT {status}: {log.value.decode(errors="replace")}')
    def close(self):
        if self.ctx: self.cuCtxDestroy_v2(self.ctx); self.ctx=P()


def inventory():
    devices=[]; diagnostics=[]
    try:
        dll=amd_library(['amdhip64_7.dll','amdhip64_6.dll'])
        init=bind(dll,'hipInit',[U]); countfn=bind(dll,'hipGetDeviceCount',[C.POINTER(I)])
        props=bind(dll,'hipGetDevicePropertiesR0600',[P,I])
        count=I()
        if init(0) or countfn(C.byref(count)): raise RuntimeError('HIP nao inicializou')
        for index in range(count.value):
            data=C.create_string_buffer(16384)
            if props(data,index): continue
            arch=re.search(rb'gfx[0-9a-f]+',data.raw)
            if arch:
                devices.append({'id':f'amd:{index}','vendor':'AMD','index':index,
                                'name':data.value.decode(),'arch':arch.group().decode()})
    except (OSError,AttributeError,RuntimeError) as e: diagnostics.append('AMD: '+str(e))
    try:
        cuda=CUDA(); devices.extend(cuda.devices)
    except (OSError,AttributeError,RuntimeError) as e: diagnostics.append('NVIDIA: '+str(e))
    try:
        devices.extend(OpenCL().devices())
    except (OSError,AttributeError,RuntimeError) as e: diagnostics.append('Intel OpenCL: '+str(e))
    return devices,diagnostics


def amd_kernel(info,mode,int4_k=32):
    if mode in NEW_MODES: return amd_more(info,mode,Unsupported)
    if mode in EXTRA_MODES: return amd_extended(info,mode,Unsupported)
    arch=info['arch']; matrix=arch.startswith(('gfx11','gfx12'))
    if mode=='FP8' and not arch.startswith('gfx12'):
        raise Unsupported('FP8 nativo ainda nao implementado para esta arquitetura; nenhuma conversao aplicada.')
    if matrix:
        source=kernel_source(mode).decode()
        if arch.startswith('gfx11'):
            source=source.replace('_gfx12','')
            if mode=='INT8': source=source.replace('ext_vector_type(2)','ext_vector_type(4)')
            else:
                source=source.replace('const int* in','const v2i* in')
                source=re.sub(r'\bint ([ab][0-3]) =',r'v2i \1 =',source)
        mnemonic={'FP8':'v_wmma_f32_16x16x16_fp8_fp8','INT8':'v_wmma_i32_16x16x16_iu8','INT4':'v_wmma_i32_16x16x16_iu4'}[mode]
        # Uniform A/B allow each lane to load the same packed vector.
        source=re.sub(r'in\[[0246]\]', 'in[0]',source)
        source=re.sub(r'in\[[1357]\]', 'in[1]',source)
        stride=(16 if arch.startswith('gfx11') else 8) if mode!='INT4' else (8 if arch.startswith('gfx11') else 4)
        if mode=='INT4' and arch.startswith('gfx12') and int4_k==32:
            source=source.replace('16x16x16_iu4','16x16x32_iu4')
            source=source.replace('const int* in','const v2i* in')
            source=re.sub(r'\bint (a[0-3]) =',r'v2i \1 =',source)
            return source.encode(),{'path':'WMMA 16x16x32','lanes':8,'k':32,
                'ops_per_thread':512,'stride':8,'mnemonic':'v_wmma_i32_16x16x32_iu4'}
        return source.encode(),{'path':'WMMA','lanes':8,'k':16,'ops_per_thread':256,'stride':stride,'mnemonic':mnemonic}
    builtin='sdot4' if mode=='INT8' else 'sdot8'
    k=4 if mode=='INT8' else 8
    decl='\n'.join(f'int a{j}=in[0], b{j}=in[1], c{j}={j};' for j in range(4))
    ops='\n'.join(f'c{j}=__builtin_amdgcn_{builtin}(a{j},b{j},c{j},false);' for j in range(4))
    stores='\n'.join(f'out[tid*4+{j}]=c{j};' for j in range(4))
    source=f'''extern "C" __attribute__((global)) void bench(const int* in,int* out,int iterations) {{
    unsigned tid=__builtin_amdgcn_workgroup_id_x()*128+__builtin_amdgcn_workitem_id_x();
    {decl}
    #pragma clang loop unroll(disable)
    for(int i=0;i<iterations;++i) {{ {ops} }}
    {stores}
    }}'''
    return source.encode(),{'path':'DOT4' if k==4 else 'DOT8','lanes':1,'k':k,'ops_per_thread':2*k,'stride':4,'mnemonic':f'v_dot{k}'}


def nvidia_kernel(info,mode,int8_path='auto'):
    if mode in NEW_MODES: return nvidia_more(info,mode,Unsupported)
    if mode in EXTRA_MODES: return nvidia_extended(info,mode,Unsupported)
    cc=info['cc']
    if cc<61: raise Unsupported('Requer compute capability >= 6.1 (DP4A).')
    if mode=='FP8' and cc<89: raise Unsupported('FP8 MMA requer compute capability >= 8.9.')
    if mode=='INT4' and cc<75: raise Unsupported('INT4 MMA requer compute capability >= 7.5.')
    if mode=='INT8' and int8_path=='tensor' and cc<75: raise Unsupported('INT8 Tensor MMA requer compute capability >= 7.5.')
    matrix=mode!='INT8' or (cc>=75 and int8_path!='dp4a')
    fp=mode=='FP8'; lanes=4 if fp else (2 if matrix else 1)
    k=32 if mode in ('FP8','INT4') else (16 if matrix else 4)
    if fp: mnemonic='mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32'
    elif matrix: mnemonic=f'mma.sync.aligned.m8n8k{k}.row.col.s32.{"s4.s4" if mode=="INT4" else "s8.s8"}.s32'
    else: mnemonic='dp4a.s32.s32'
    version='8.4' if fp else ('6.5' if matrix else '6.0')
    target=89 if fp else (75 if matrix else 61)
    n=4*lanes; typ='f32' if fp else 'b32'
    init='\n'.join(f'mov.{typ} %c{i}, '+(f'0f{__import__("struct").pack(">f",float(i//lanes)).hex()}' if fp else str(i//lanes))+';' for i in range(n))
    operations=[]
    for j in range(4):
        acc=', '.join(f'%c{j*lanes+x}' for x in range(lanes))
        if matrix:
            a='{'+','.join([f'%a{j}']*(4 if fp else 1))+'}'
            b='{'+','.join([f'%b{j}']*(2 if fp else 1))+'}'
            operations.append(f'{mnemonic} {{{acc}}}, {a}, {b}, {{{acc}}};')
        else: operations.append(f'{mnemonic} %c{j}, %a{j}, %b{j}, %c{j};')
    stores='\n'.join(f'st.global.{typ} [%out+{i*4}], %c{i};' for i in range(n))
    source=f'''.version {version}
.target sm_{target}
.address_size 64
.visible .entry bench(.param .u64 input, .param .u64 output, .param .u32 iterations) {{
.reg .b64 %in,%out,%offset;
.reg .b32 %a<4>,%b<4>,%idx,%block,%count;
.reg .pred %again;
.reg .{typ} %c<{n}>;
ld.param.u64 %in,[input]; ld.param.u64 %out,[output]; ld.param.u32 %count,[iterations];
{chr(10).join(f'ld.global.b32 %a{j},[%in+{8*j}]; ld.global.b32 %b{j},[%in+{8*j+4}];' for j in range(4))}
mov.u32 %idx,%tid.x; mov.u32 %block,%ctaid.x;
mad.lo.u32 %idx,%block,128,%idx;
mul.wide.u32 %offset,%idx,{n*4}; add.u64 %out,%out,%offset;
{init}
LOOP:
{chr(10).join(operations)}
sub.u32 %count,%count,1; setp.ne.u32 %again,%count,0; @%again bra LOOP;
{stores}
ret;
}}
'''
    return source.encode(),{'path':'MMA Tensor' if matrix else 'DP4A','lanes':lanes,'k':k,
        'ops_per_thread':(8192 if fp else 2*8*8*k)//32 if matrix else 8,'stride':4,'mnemonic':mnemonic}


def run(gpu,info,mode,folder,samples,int8_path,int4_k=32,kernel_override=None,blocks_override=None):
    amd=info['vendor']=='AMD'
    source,spec=kernel_override if kernel_override is not None else (amd_kernel(info,mode,int4_k) if amd else nvidia_kernel(info,mode,int8_path))
    (folder/(mode+('.hip' if amd else '.ptx'))).write_bytes(source)
    if amd:
        try:
            binary,asm=compile_amd(source,info['arch'])
        except RuntimeError as exc:
            if 'needs target feature' in str(exc):
                raise Unsupported('A arquitetura nao oferece as instrucoes deste kernel. '+str(exc)) from exc
            raise
        (folder/(mode+'.s')).write_bytes(asm)
        expected_count=audit_amd(mode,spec,asm)
        spec['verification']=f'AMDGPU assembly checked; exactly {expected_count} native operations in loop'
        if mode=='FP32_VECTOR':
            assembly=asm.decode()
            spec['assembly_metrics']={
                'dual_fma_pairs':sum(line.strip().startswith('v_dual_fmac_f32') for line in assembly.splitlines()),
                'delay_instructions':sum(line.strip().startswith('s_delay_alu') for line in assembly.splitlines())}
            for key in ('vgpr_count','private_segment_fixed_size'):
                match=re.search(r'\.'+key+r':\s*(\d+)',assembly)
                if match: spec['assembly_metrics'][key]=int(match.group(1))
            if spec.get('positive_fma') and spec['assembly_metrics'].get('private_segment_fixed_size',0):
                raise RuntimeError('FP32 candidate spills to private memory; excluded from peak tuning.')
    else:
        binary=source+b'\0'
        spec['verification']='Explicit PTX instructions; CUDA driver JIT and numeric validation (SASS not inspected)'
    module,fn,start,end=P(),P(),P(),P(); din=dout=None
    try:
        code=C.create_string_buffer(binary)
        if amd: gpu.ok(gpu.hipModuleLoadData(C.byref(module),code))
        else: gpu.load(module,code)
        gpu.ok(gpu.hipModuleGetFunction(C.byref(fn),module,b'bench'))
        gpu.ok(gpu.hipEventCreate(C.byref(start))); gpu.ok(gpu.hipEventCreate(C.byref(end)))
        # Smaller grids for scalar dot kernels and small GPUs; matrix RX9070 keeps old grid sweep.
        blocks_list=(256,512,1024) if info['arch'].startswith('gfx12') else (16,64,256)
        if blocks_override is not None: blocks_list=blocks_override
        lanes=spec['lanes']; stride=spec['stride']; scalar=C.c_double if spec.get('output_type')=='FP64' else (C.c_float if is_float(mode) else I)
        chains=spec.get('chains',4)
        din=gpu.alloc(8*stride); dout=gpu.alloc(max(blocks_list)*128*chains*lanes*C.sizeof(scalar))
        def inputs(negative=False):
            if spec.get('extended'):
                blob=payload(spec,(0 if spec.get('unsigned_binary') else -1) if negative else 1)
                data=C.create_string_buffer(blob); gpu.ok(gpu.hipMemcpy(din,data,len(blob),1)); return
            a=(0xb8 if negative else 0x38) if mode=='FP8' else (255 if negative else (0x11 if mode=='INT4' else 1))
            b=0x38 if mode=='FP8' else (0x11 if mode=='INT4' else 1)
            data=C.create_string_buffer((bytes([a])*stride+bytes([b])*stride)*4)
            gpu.ok(gpu.hipMemcpy(din,data,8*stride,1))
        def launch(blocks,it):
            iterations=I(it)
            params=(P*3)(C.cast(C.byref(din),P),C.cast(C.byref(dout),P),C.cast(C.byref(iterations),P))
            gpu.ok(gpu.hipModuleLaunchKernel(fn,blocks,1,1,128,1,1,0,None,params,None))
        def timed(blocks,it):
            gpu.ok(gpu.hipEventRecord(start,None)); launch(blocks,it)
            gpu.ok(gpu.hipEventRecord(end,None)); gpu.ok(gpu.hipEventSynchronize(end))
            ms=C.c_float(); gpu.ok(gpu.hipEventElapsedTime(C.byref(ms),start,end))
            if ms.value<=0: raise RuntimeError('GPU timer returned zero')
            if ms.value>250: raise RuntimeError('Kernel excedeu 250 ms; interrompido para evitar cargas longas.')
            return ms.value
        def validate(blocks,it,sign,dot=None):
            out=(scalar*(blocks*128*chains*lanes))(); gpu.ok(gpu.hipMemcpy(out,dout,C.sizeof(out),2))
            for idx,v in enumerate(out):
                expected=sign*(spec.get('dot',spec['k']) if dot is None else dot)*it+(idx//lanes)%chains
                if spec.get('wrap_bits'):
                    bits=spec['wrap_bits']; expected=((expected+(1<<(bits-1)))%(1<<bits))-(1<<(bits-1))
                if v!=expected: raise RuntimeError(f'Validation: output[{idx}]={v}, expected {expected}')
        for neg in (False,True):
            inputs(neg); launch(2,3); gpu.ok(gpu.hipDeviceSynchronize()); validate(2,3,(0 if spec.get('unsigned_binary') else -1) if neg else 1)
        if spec.get('block_scale') or spec.get('unsigned_binary'):
            for sign in ((0,1) if spec.get('unsigned_binary') else (-1,1)):
                blob=payload(spec,sign,pattern=spec.get('unsigned_binary',False),scales=(2,4))
                data=C.create_string_buffer(blob); gpu.ok(gpu.hipMemcpy(din,data,len(blob),1))
                launch(2,3); gpu.ok(gpu.hipDeviceSynchronize())
                validate(2,3,sign,dot=spec['k']//2 if spec.get('unsigned_binary') else spec['k']*8)
            spec['validation']='All outputs; signed/zero A; non-unit block scales 2x4 or alternating binary B; timed outputs checked'
        if spec.get('vector'):
            for sign in (1,-1):
                blob=payload(spec,sign,pattern=True)
                data=C.create_string_buffer(blob); gpu.ok(gpu.hipMemcpy(din,data,len(blob),1))
                launch(2,3); gpu.ok(gpu.hipDeviceSynchronize()); validate(2,3,sign,dot=spec.get('pattern_dot',-8))
            spec['validation']='Balanced FMA pairs; asymmetric B+/B- test with signed A; all outputs verified'
            if spec.get('positive_fma'): spec['validation']='Exact FP32 sums: signed A, B=1/2, all outputs; timed sums <= 2^20+31'
        if spec.get('sparse'):
            for second in (False,True):
                for sign in (1,-1):
                    blob=payload(spec,sign,pattern=True,second=second)
                    data=C.create_string_buffer(blob); gpu.ok(gpu.hipMemcpy(din,data,len(blob),1))
                    launch(2,3); gpu.ok(gpu.hipDeviceSynchronize())
                    validate(2,3,sign,dot=(7 if second else 3)*spec['k']//4)
            spec['validation']='All outputs: +/- A; nonuniform B; metadata selects [0,1] and [2,3]; timed outputs checked'
        if amd and mode=='INT4' and spec['k']==32:
            # Each dot has 16 values of 1 and 16 of 2: both packed words matter.
            for sign in (1,-1):
                packed_a=bytes([0x11 if sign==1 else 0xff])*4+bytes([0x22 if sign==1 else 0xee])*4
                data=C.create_string_buffer((packed_a+bytes([0x11])*8)*4)
                gpu.ok(gpu.hipMemcpy(din,data,64,1))
                launch(2,3); gpu.ok(gpu.hipDeviceSynchronize()); validate(2,3,sign,dot=48)
            spec['validation']='Signed +/-1 and mixed K words (+/-1,+/-2); all outputs checked; timed outputs checked'
        inputs(); rows=[]
        for blocks in blocks_list:
            # Short gfx12 probes can fall below the Windows HIP event timer resolution.
            it=4096 if info['arch'].startswith('gfx12') else 256
            probe=timed(blocks,it)
            it=max(256,min(65536,int(it*8/probe)))
            until=time.perf_counter()+0.3
            while time.perf_counter()<until: timed(blocks,it)
            times=[timed(blocks,it) for _ in range(samples)]
            validate(blocks,it,1)
            ops=blocks*128*chains*it*spec['ops_per_thread']
            row={'blocks':blocks,'iterations':it,'operations':ops,'validated':True,
                 **sample_statistics(ops,times,spec.get('sparse',False))}
            rows.append(row)
            if spec.get('sparse'): row['nonzero_tops']=row['median_tops']/2
            print(f'  {mode} / {spec["path"]}: min {row["min_tops"]:.3f}, max {row["max_tops"]:.3f}, media {row["mean_tops"]:.3f}, mediana {row["median_tops"]:.3f} {unit_for(mode)} ({blocks} blocos)',flush=True)
        return {'mode':mode,'status':'ok','kernel':spec,'configurations':rows,
                'best_configuration':max(rows,key=lambda x:x['median_tops'])}
    finally:
        if din: gpu.hipFree(din)
        if dout: gpu.hipFree(dout)
        if start: gpu.hipEventDestroy(start)
        if end: gpu.hipEventDestroy(end)
        if module: gpu.hipModuleUnload(module)


def run_fp32(gpu,info,folder,samples,int8_path,int4_k,variant='auto'):
    if info['vendor']!='AMD' or not info['arch'].startswith('gfx12'):
        return run(gpu,info,'FP32_VECTOR',folder,samples,int8_path,int4_k)
    reference=amd_kernel(info,'FP32_VECTOR',int4_k)
    reference[1]['variant']='reference'
    kernels={name:reference if name=='reference' else named_candidate(name) for name in VARIANTS}
    if variant!='auto':
        return run(gpu,info,'FP32_VECTOR',folder,samples,int8_path,int4_k,kernel_override=kernels[variant])
    tuning_folder=folder/'FP32_tuning'; tuning_folder.mkdir()
    trials=[]
    print('FP32: selecao curta de variantes (5 amostras/configuracao); medicao final independente.',flush=True)
    for name,kernel in kernels.items():
        sub=tuning_folder/name; sub.mkdir()
        try:
            trial=run(gpu,info,'FP32_VECTOR',sub,5,int8_path,int4_k,kernel_override=kernel)
            trial['variant']=name
        except Exception as exc:
            trial={'variant':name,'status':'failed','reason':str(exc)}
            print('FP32 candidato '+name+': '+str(exc),flush=True)
        trials.append(trial)
        (tuning_folder/'results.json').write_text(json.dumps(trials,indent=2),encoding='utf-8')
    valid=[trial for trial in trials if trial['status']=='ok']
    if not valid: raise RuntimeError('Nenhum candidato FP32 passou na validacao.')
    winner=max(valid,key=lambda trial:trial['best_configuration']['median_tops'])
    name=winner['variant']; blocks=winner['best_configuration']['blocks']
    print(f'FP32: escolhido {name}, {blocks} blocos. Iniciando {samples} novas amostras.',flush=True)
    result=run(gpu,info,'FP32_VECTOR',folder,samples,int8_path,int4_k,kernel_override=kernels[name],blocks_override=(blocks,))
    result['selection']={'method':'Best tuning median; new final samples at fixed selected variant/grid',
        'samples_per_configuration':5,'variant':name,'blocks':blocks,'trials':trials}
    return result


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--list',action='store_true',help='Listar GPUs detectadas')
    ap.add_argument('--interactive',action='store_true',help='Menu de GPU e formatos')
    ap.add_argument('--device',help='Identificador da GPU: amd:0, amd:1, nvidia:0, intel:0')
    ap.add_argument('--modes',nargs='+',help='Formatos do menu, por nome/numero; ALL para todos')
    ap.add_argument('--samples',type=int,default=21)
    ap.add_argument('--cancel-file',help=argparse.SUPPRESS)
    ap.add_argument('--fp32-kernel',choices=('auto',)+VARIANTS,default='auto',help='RDNA4 FP32: auto compara variantes e mede a vencedora; reference preserva o kernel anterior')
    ap.add_argument('--int8-path',choices=['auto','dp4a','tensor'],default='auto',help='NVIDIA: Tensor quando disponivel, ou DP4A')
    ap.add_argument('--amd-int4-k',type=int,choices=[16,32],default=32,help='RDNA4 INT4: 32 (padrao, maior throughput) ou 16 (comparacao com versao antiga)')
    args=ap.parse_args()
    if not 5<=args.samples<=200: ap.error('--samples deve ser 5..200')
    try: modes=parse_modes(args.modes) if args.modes else list(MODES)
    except ValueError as e: ap.error(str(e))
    devices,diagnostics=inventory()
    for i,d in enumerate(devices): print(f'[{i+1}] {d["id"]}: {d["name"]} ({d["arch"]})')
    if args.list or not devices:
        for d in diagnostics: print(d)
        return 0 if devices else 1
    info=next((d for d in devices if d['id']==args.device),None)
    if args.device and info is None: ap.error('GPU nao encontrada; use --list')
    if args.interactive:
        if info is None:
            while True:
                choice=input('Escolha a GPU [1]: ').strip() or '1'
                if choice.isdigit() and 1<=int(choice)<=len(devices): info=devices[int(choice)-1]; break
                print('Opcao invalida.')
    if info is None:
        if len(devices)>1: ap.error('Ha varias GPUs. Use --device ou --interactive para escolher.')
        info=devices[0]
    print('Verificando testes disponiveis para '+info['name']+'...',flush=True)
    available,unavailable=available_modes(info,args.int8_path,args.amd_int4_k,probe_amd=True)
    if not available: ap.error('Nenhum teste disponivel. '+ '; '.join(m+': '+r for m,r in unavailable.items()))
    if args.interactive and not args.modes:
        for m in available: print(f'{MODES.index(m)+1:2}: {m} - {describe(m)}')
        print('Numeros fixos; ALL seleciona somente os testes disponiveis. ? mostra os motivos das opcoes ocultas.')
        while True:
            choice=input('Escolha formatos por numero/nome ou ALL [ALL]: ').strip() or 'ALL'
            if choice=='?':
                for m,reason in unavailable.items(): print(m+': '+reason)
                continue
            try: modes=select_modes([choice],available,unavailable); break
            except ValueError as e: print(e)
    else:
        try: modes=select_modes(args.modes or ['ALL'],available,unavailable)
        except ValueError as e: ap.error(str(e))
    if args.fp32_kernel not in ('auto','reference') and (info['vendor']!='AMD' or not info['arch'].startswith('gfx12')):
        ap.error('Variantes FP32 explicitas requerem AMD gfx12.')
    print(f'Testando {info["name"]}: {", ".join(modes)}',flush=True)
    folder=ROOT/'results'/('gpu-'+datetime.datetime.now().strftime('%Y%m%d-%H%M%S-%f')); folder.mkdir(parents=True)
    print('Pasta de resultados:',folder,flush=True)
    report={'gpu':info,'requested_modes':modes,'samples':args.samples,'amd_int4_k':args.amd_int4_k,'results':[],
            'available_modes':available,'unavailable_modes':unavailable,
            'fp32_kernel':args.fp32_kernel,
            'method':'Register throughput; multiply+add=2. Sparse kernels report dense-equivalent throughput AND nonzero arithmetic throughput; no emulation.',
            'limitations':'Not GEMM/inference or guaranteed peak. Different instruction paths must be compared explicitly.'}
    gpu=GPU(info['index'],allow_other=True) if info['vendor']=='AMD' else (OpenCL(info) if info['vendor']=='Intel' else CUDA(info['index']))
    try:
        for mode in modes:
            if args.cancel_file and Path(args.cancel_file).exists():
                report['cancelled']=True
                break
            try:
                if info['vendor']=='Intel': result=run_intel(gpu,info,mode,folder,args.samples,Unsupported,sample_statistics)
                elif mode=='FP32_VECTOR': result=run_fp32(gpu,info,folder,args.samples,args.int8_path,args.amd_int4_k,args.fp32_kernel)
                else: result=run(gpu,info,mode,folder,args.samples,args.int8_path,args.amd_int4_k)
            except Unsupported as e: result={'mode':mode,'status':'unsupported','reason':str(e)}
            except Exception as e: result={'mode':mode,'status':'failed','reason':str(e)}
            report['results'].append(result)
            if result['status']!='ok': print(f'{mode}: {result["status"]} - {result["reason"]}',flush=True)
            (folder/'results.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
            print('Formato concluido:',mode,flush=True)
    finally:
        if info['vendor'] in ('NVIDIA','Intel'): gpu.close()
    lines=['# Benchmark GPU', '',info['name']+' / '+info['arch'],'', '| Formato | Caminho | Unidade | Minimo | Maximo | Media | Mediana |','|---|---|---|---:|---:|---:|---:|']
    reasons=[]
    with (folder/'results.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.writer(f); writer.writerow(['mode','status','path','unit','median_throughput','nonzero_throughput','reason','min_throughput','max_throughput','mean_throughput','blocks','nonzero_min_throughput','nonzero_max_throughput','nonzero_mean_throughput'])
        for r in report['results']:
            if r['status']=='ok':
                value=r['best_configuration']['median_tops']; unit=unit_for(r['mode'])
                row=r['best_configuration']
                lines.append(f'| {r["mode"]} | {r["kernel"]["path"]} | {unit} | '+ ' | '.join(f'{row[k+"_tops"]:.3f}' for k in ('min','max','mean','median'))+' |')
                writer.writerow([r['mode'],'ok',r['kernel']['path'],unit,value,row.get('nonzero_tops',value),'',row['min_tops'],row['max_tops'],row['mean_tops'],row['blocks'],*[row.get('nonzero_'+k+'_tops',row[k+'_tops']) for k in ('min','max','mean')]])
            else:
                lines.append(f'| {r["mode"]} | {r["status"]} | — | — | — | — | — |'); reasons.append(r['mode']+': '+r['reason'].replace('\n',' '))
                writer.writerow([r['mode'],r['status'],'','','','',r['reason'],*(['']*7)])
    lines+=['',f'Melhor mediana entre configuracoes; {args.samples} amostras por configuracao.','Esparsos: taxa equivalente da matriz expandida; taxa das operacoes nao nulas tambem no JSON. Nenhum fallback de precisao.']+['\n'+r for r in reasons]
    lines+=['','Minimo, maximo, media e mediana calculados sobre as taxas individuais da mesma configuracao escolhida pela melhor mediana. Todas as configuracoes e tempos estao no JSON.', 'Maximo observado = amostra mais rapida; nao e o pico teorico do fabricante. A media das taxas nao e a taxa calculada pelo tempo medio.']
    for result in report['results']:
        if result.get('selection'):
            selection=result['selection']
            lines+=['',f'FP32: variante {selection["variant"]}, {selection["blocks"]} blocos, escolhidos em rodada de 5 amostras por configuracao. A tabela usa somente a rodada final independente. Detalhes em FP32_tuning/results.json.']
    (folder/'RESUMO.md').write_text('\n'.join(lines),encoding='utf-8')
    if report.get('cancelled'):
        (folder/'results.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
        with (folder/'RESUMO.md').open('a',encoding='utf-8') as f: f.write('\n\nExecucao cancelada; resultados parciais.\n')
    print('Resultados:',folder)
    if report.get('cancelled'): return 130
    return 1 if any(r['status']=='failed' for r in report['results']) or not any(r['status']=='ok' for r in report['results']) else 0

if __name__=='__main__':
    try: sys.exit(main())
    except (EOFError,KeyboardInterrupt): print('\nCancelado.'); sys.exit(130)
    except Exception as e: print('ERRO:',e,file=sys.stderr); sys.exit(1)
