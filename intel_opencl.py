"""Intel OpenCL driver backend. No SDK, precision conversion, or matrix emulation.
Matrix API: cl_intel_subgroup_matrix_multiply_accumulate, minimum subgroup 8/16.
"""
import ctypes as C
import os
from pathlib import Path
import time
from extended_kernels import payload
from more_formats import NEW_MODES,is_float

P=C.c_void_p; U=C.c_uint; I=C.c_int; S=C.c_size_t; Q=C.c_ulonglong
MATRIX_EXT='cl_intel_subgroup_matrix_multiply_accumulate'

class OpenCL:
    def __init__(self, info=None):
        self.dll=C.WinDLL(str(Path(os.environ['SystemRoot'])/'System32/OpenCL.dll'))
        self.context=self.queue=None
        signatures={
            'clGetPlatformIDs':(I,[U,P,P]), 'clGetDeviceIDs':(I,[P,Q,U,P,P]),
            'clGetDeviceInfo':(I,[P,U,S,P,P]),
            'clCreateContext':(P,[P,U,P,P,P,P]),
            'clCreateCommandQueue':(P,[P,P,Q,P]),
            'clCreateBuffer':(P,[P,Q,S,P,P]),
            'clCreateProgramWithSource':(P,[P,U,P,P,P]),
            'clBuildProgram':(I,[P,U,P,C.c_char_p,P,P]),
            'clGetProgramBuildInfo':(I,[P,P,U,S,P,P]),
            'clCreateKernel':(P,[P,C.c_char_p,P]),
            'clSetKernelArg':(I,[P,U,S,P]),
            'clGetKernelWorkGroupInfo':(I,[P,P,U,S,P,P]),
            'clEnqueueWriteBuffer':(I,[P,P,U,S,S,P,U,P,P]),
            'clEnqueueReadBuffer':(I,[P,P,U,S,S,P,U,P,P]),
            'clEnqueueNDRangeKernel':(I,[P,P,U,P,P,P,U,P,P]),
            'clWaitForEvents':(I,[U,P]), 'clGetEventProfilingInfo':(I,[P,U,S,P,P]),
            'clFinish':(I,[P]),
        }
        for name in ('Context','CommandQueue','MemObject','Program','Kernel','Event'):
            signatures['clRelease'+name]=(I,[P])
        for name,(ret,args) in signatures.items():
            fn=getattr(self.dll,name); fn.restype=ret; fn.argtypes=args; setattr(self,name,fn)
        if info is not None:
            self.device=P(info['handle']); error=I()
            self.context=self.clCreateContext(None,1,C.byref(self.device),None,None,C.byref(error))
            self.ok(error.value)
            try:
                self.queue=self.clCreateCommandQueue(self.context,self.device,2,C.byref(error))
                self.ok(error.value)
            except Exception:
                self.close(); raise

    @staticmethod
    def ok(status):
        if status: raise RuntimeError(f'OpenCL error {status}')

    def device_bytes(self,dev,key,optional=False):
        size=S(); status=self.clGetDeviceInfo(dev,key,0,None,C.byref(size))
        if status and optional: return b''
        self.ok(status)
        data=C.create_string_buffer(size.value)
        self.ok(self.clGetDeviceInfo(dev,key,len(data),data,None)); return data.raw

    def devices(self, intel_only=True):
        count=U(); status=self.clGetPlatformIDs(0,None,C.byref(count))
        if status==-1001: return []
        self.ok(status); platforms=(P*count.value)()
        self.ok(self.clGetPlatformIDs(count,platforms,None)); devices=[]
        for platform in platforms:
            n=U(); status=self.clGetDeviceIDs(platform,4,0,None,C.byref(n))
            if status==-1: continue
            self.ok(status); handles=(P*n.value)()
            self.ok(self.clGetDeviceIDs(platform,4,n,handles,None))
            for handle in handles:
                vendor=int.from_bytes(self.device_bytes(handle,0x1001),'little')
                if intel_only and vendor!=0x8086: continue
                getstr=lambda key:self.device_bytes(handle,key).rstrip(b'\0').decode(errors='replace')
                name=getstr(0x102b); ext=getstr(0x1030).split()
                raw=self.device_bytes(handle,0x4108,True) if 'cl_intel_required_subgroup_size' in ext else b''
                sizes=[int.from_bytes(raw[i:i+C.sizeof(S)],'little') for i in range(0,len(raw),C.sizeof(S))]
                devices.append({'id':f'{"intel" if vendor==0x8086 else "opencl"}:{len(devices)}','index':len(devices),'vendor':{0x8086:'Intel',0x1002:'AMD',0x10de:'NVIDIA'}.get(vendor,'Unknown'),
                    'handle':handle,'name':name,'arch':'OpenCL / subgroup '+str(min(sizes) if sizes else '?'),
                    'extensions':ext,'subgroup_sizes':sizes,'driver':getstr(0x102d),
                    'single_fp_config':int.from_bytes(self.device_bytes(handle,0x101b),'little'),
                    'half_fp_config':int.from_bytes(self.device_bytes(handle,0x1033,True),'little'),
                    'double_fp_config':int.from_bytes(self.device_bytes(handle,0x1032,True),'little'),
                    'max_work_group_size':int.from_bytes(self.device_bytes(handle,0x1004),'little')})
        return devices

    def build(self,source):
        error=I(); src=C.c_char_p(source); size=S(len(source))
        program=self.clCreateProgramWithSource(self.context,1,C.byref(src),C.byref(size),C.byref(error))
        self.ok(error.value)
        try:
            status=self.clBuildProgram(program,1,C.byref(self.device),b'-cl-std=CL2.0',None,None)
            if status:
                length=S(); self.clGetProgramBuildInfo(program,self.device,0x1183,0,None,C.byref(length))
                log=C.create_string_buffer(max(1,length.value))
                self.clGetProgramBuildInfo(program,self.device,0x1183,len(log),log,None)
                raise RuntimeError(f'OpenCL build {status}: '+log.value.decode(errors='replace'))
            kernel=self.clCreateKernel(program,b'bench',C.byref(error)); self.ok(error.value)
            return program,kernel
        except Exception:
            self.clReleaseProgram(program); raise

    def close(self):
        if self.queue: self.clReleaseCommandQueue(self.queue); self.queue=None
        if self.context: self.clReleaseContext(self.context); self.context=None


def intel_kernel(info,mode,Unsupported):
    if mode in NEW_MODES: return intel_more(info,mode,Unsupported)
    ext=set(info.get('extensions',[])); vector=mode.endswith('_VECTOR')
    if mode not in ('FP32_VECTOR','FP16_VECTOR','FP16_MATRIX','INT8','INT4'):
        raise Unsupported('Formato sem caminho nativo implementado no backend Intel OpenCL.')
    if info.get('max_work_group_size',0)<128:
        raise Unsupported('Este kernel requer grupos de 128 work-items.')
    fp16=mode.startswith('FP16'); fp=mode.startswith('FP')
    if vector:
        if fp16 and 'cl_khr_fp16' not in ext: raise Unsupported('Driver nao anuncia cl_khr_fp16.')
        if not info.get('half_fp_config' if fp16 else 'single_fp_config',0)&32:
            raise Unsupported('Driver nao anuncia FMA nativo para esta precisao.')
    else:
        if MATRIX_EXT not in ext or 'cl_intel_required_subgroup_size' not in ext:
            raise Unsupported('Driver nao anuncia extensoes XMX de matriz/subgrupo exigidas.')
        sizes=info.get('subgroup_sizes',[])
        if not sizes or min(sizes) not in (8,16): raise Unsupported('XMX requer subgrupo minimo 8 ou 16.')
    subgroup=min(info.get('subgroup_sizes') or [8])
    fmt='FP16' if fp16 else ('FP32' if fp else mode)
    k=1 if vector else (16 if fp16 else (32 if mode=='INT8' else 64))
    lanes=2 if vector else 8; chains=4
    scalar='float' if fp else 'int'; acc=('half2' if fp16 else 'float2') if vector else scalar+'8'
    atype=acc if vector else ('int8' if subgroup==8 else 'short8')
    btype=acc if vector else 'int8'
    decl=[]; operations=[]; stores=[]
    builtin=f'intel_sub_group_{"f16_f16" if fp16 else ("i8_i8" if mode=="INT8" else "i4_i4")}_matrix_mad_k{k}'
    for j in range(chains):
        decl.append(f'{atype} a{j}=*(__global const {atype}*)(input+{j*256}); {btype} b{j}=*(__global const {btype}*)(input+{j*256+64}); {acc} c{j}=({acc})({j});')
        if vector:
            decl.append(f'{acc} n{j}=*(__global const {acc}*)(input+{j*256+96});')
        else: operations.append(f'c{j}={builtin}(a{j},b{j},c{j});')
        for lane in range(lanes):
            stores.append(f'output[tid*{chains*lanes}+{j*lanes+lane}]=({scalar})c{j}.s{lane};')
    if vector:
        operations=[f'c{j}=fma({"-" if rep%2 else ""}a{j},{"n" if rep%2 else "b"}{j},c{j});' for rep in range(16) for j in range(chains)]
    pragma=('#pragma OPENCL EXTENSION cl_khr_fp16 : enable\n' if fp16 and vector else '')
    if not vector: pragma+=f'#pragma OPENCL EXTENSION {MATRIX_EXT} : enable\n#pragma OPENCL EXTENSION cl_intel_required_subgroup_size : enable\n'
    attribute='' if vector else f'__attribute__((intel_reqd_sub_group_size({subgroup})))'
    source=f'''{pragma}
{attribute} __kernel void bench(__global const char* input,__global {scalar}* output,int iterations) {{
size_t tid=get_global_id(0);
{chr(10).join(decl)}
#pragma unroll 1
for(int i=0;i<iterations;++i) {{ {chr(10).join(operations)} }}
{chr(10).join(stores)}
}}
'''
    # M=8, N=subgroup: divide 2*M*N*K by subgroup exactly once.
    bits=16 if fp16 else (8 if mode=='INT8' else 4)
    ae=2 if vector else 8*k//subgroup
    return source.encode(),{'path':'OpenCL vector FMA' if vector else f'XMX {8}x{subgroup}x{k}',
        'input_format':fmt,'a_elements':ae,'b_elements':2 if vector else 256//bits,
        'chains':chains,'lanes':lanes,'k':k,'dot':0 if vector else k,
        'ops_per_thread':64 if vector else 16*k,'vector':vector,'sparse':False,
        'subgroup_size':None if vector else subgroup,
        'verification':'OpenCL native FMA capability / explicit Intel matrix builtin; driver JIT; numeric validation. ISA not inspected.'}


def intel_more(info,mode,Unsupported):
    ext=set(info.get('extensions',[]))
    if mode in ('BF16_MATRIX','TF32_MATRIX'):
        source,spec=intel_kernel(info,'FP16_MATRIX',Unsupported)
        source=source.decode()
        if mode=='BF16_MATRIX':
            source=source.replace('f16_f16','bf16_bf16'); spec.update(input_format='BF16',path='XMX BF16')
        else:
            extension=MATRIX_EXT+'_tf32'
            if extension not in ext or min(info.get('subgroup_sizes') or [0])!=16:
                raise Unsupported('TF32 XMX requer extensao TF32 e subgrupo minimo 16.')
            source=source.replace('f16_f16_matrix_mad_k16','tf32_tf32_matrix_mad_k8')
            source=source.replace('short8 a','float4 a').replace('const short8*','const float4*').replace('int8 b','float8 b').replace('const int8*','const float8*')
            source=f'#pragma OPENCL EXTENSION {extension} : enable\n'+source
            spec.update(input_format='TF32',path='XMX TF32',a_elements=4,b_elements=8,k=8,dot=8,ops_per_thread=128)
        spec['output_type']='FP32'; return source.encode(),spec
    if mode=='FP64_VECTOR':
        if 'cl_khr_fp64' not in ext or not info.get('double_fp_config',0)&32:
            raise Unsupported('Driver nao anuncia FP64 FMA nativo.')
        source,spec=intel_kernel(info,'FP32_VECTOR',Unsupported)
        source=b'#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n'+source.replace(b'float',b'double')
        spec.update(input_format='FP64',output_type='FP64',path='OpenCL FP64 FMA'); return source,spec
    if mode=='INT32_VECTOR':
        import re
        source,spec=intel_kernel(info,'FP32_VECTOR',Unsupported)
        source=source.decode().replace('float','int')
        # A runtime multiplier on the accumulator prevents integer cancellation
        # from deleting balanced +A/-A pairs. Timed B=1; A=+/-1 or +/-2.
        source=re.sub(r'fma\(-?a(\d+),[bn]\d+,c\d+\)',lambda m:f'(c{m[1]}*b{m[1]}+a{m[1]})',source)
        spec.update(input_format='INT32',output_type='INT32',path='OpenCL INT32 MUL+ADD',dot=16,integer_recurrence=True)
        return source.encode(),spec
    raise Unsupported('Formato sem caminho nativo implementado no backend Intel OpenCL.')


def run_intel(gpu,info,mode,folder,samples,Unsupported,statistics_fn):
    source,spec=intel_kernel(info,mode,Unsupported)
    (folder/(mode+'.cl')).write_bytes(source)
    program=kernel=None; buffers=[]
    try:
        program,kernel=gpu.build(source)
        maxgroup=S(); gpu.ok(gpu.clGetKernelWorkGroupInfo(kernel,gpu.device,0x11b0,C.sizeof(maxgroup),C.byref(maxgroup),None))
        if maxgroup.value<128: raise Unsupported('Kernel compilado nao permite grupo de 128 work-items.')
        blocks_list=(256,512,1024); chains=spec['chains']; lanes=spec['lanes']
        scalar=C.c_double if spec.get('output_type')=='FP64' else (C.c_float if is_float(mode) else I)
        for size in (256*chains,max(blocks_list)*128*chains*lanes*C.sizeof(scalar)):
            err=I(); handle=gpu.clCreateBuffer(gpu.context,1,size,None,C.byref(err)); gpu.ok(err.value); buffers.append(P(handle))
        for idx,buf in enumerate(buffers): gpu.ok(gpu.clSetKernelArg(kernel,idx,C.sizeof(P),C.byref(buf)))
        def inputs(sign=1,pattern=False):
            if spec.get('integer_recurrence') and pattern: sign*=2; pattern=False
            blob=payload(spec,sign,pattern); host=C.create_string_buffer(blob)
            gpu.ok(gpu.clEnqueueWriteBuffer(gpu.queue,buffers[0],1,0,len(blob),host,0,None,None))
        def timed(blocks,it):
            iterations=I(it); gpu.ok(gpu.clSetKernelArg(kernel,2,C.sizeof(I),C.byref(iterations)))
            global_size=S(blocks*128); local_size=S(128); event=P()
            gpu.ok(gpu.clEnqueueNDRangeKernel(gpu.queue,kernel,1,None,C.byref(global_size),C.byref(local_size),0,None,C.byref(event)))
            try:
                gpu.ok(gpu.clWaitForEvents(1,C.byref(event))); start=Q(); end=Q()
                for key,value in ((0x1282,start),(0x1283,end)):
                    gpu.ok(gpu.clGetEventProfilingInfo(event,key,C.sizeof(Q),C.byref(value),None))
                ms=(end.value-start.value)/1e6
                if not 0<ms<=250: raise RuntimeError(f'OpenCL kernel time out of range: {ms} ms')
                return ms
            finally: gpu.clReleaseEvent(event)
        def validate(blocks,it,sign,dot):
            out=(scalar*(blocks*128*chains*lanes))()
            gpu.ok(gpu.clEnqueueReadBuffer(gpu.queue,buffers[1],1,0,C.sizeof(out),out,0,None,None))
            for idx,value in enumerate(out):
                expected=sign*dot*it+(idx//lanes)%chains
                if value!=expected: raise RuntimeError(f'Validation output[{idx}]={value}; expected {expected}')
        for pattern in (False,True):
            dot=(32 if spec.get('integer_recurrence') else (-8 if spec['vector'] else 5*spec['k']//2)) if pattern else spec['dot']
            for sign in (1,-1):
                inputs(sign,pattern); timed(2,3); validate(2,3,sign,dot)
        inputs(); rows=[]
        for blocks in blocks_list:
            it=4096; probe=timed(blocks,it); it=max(256,min(65536,int(it*8/probe)))
            until=time.perf_counter()+0.3
            while time.perf_counter()<until: timed(blocks,it)
            times=[timed(blocks,it) for _ in range(samples)]; validate(blocks,it,1,spec['dot'])
            ops=blocks*128*chains*it*spec['ops_per_thread']
            row={'blocks':blocks,'iterations':it,'operations':ops,'validated':True,**statistics_fn(ops,times)}; rows.append(row)
            unit='TFLOPS' if is_float(mode) else 'TOPS'
            print(f'  {mode} / {spec["path"]}: '+', '.join(f'{key} {row[key+"_tops"]:.3f}' for key in ('min','max','mean','median'))+f' {unit} ({blocks} blocos)',flush=True)
        spec['validation']='All outputs: signed A, nonuniform B / asymmetric vector operands; timed outputs checked'
        return {'mode':mode,'status':'ok','kernel':spec,'configurations':rows,'best_configuration':max(rows,key=lambda row:row['median_tops'])}
    finally:
        for buf in buffers: gpu.clReleaseMemObject(buf)
        if kernel: gpu.clReleaseKernel(kernel)
        if program: gpu.clReleaseProgram(program)
