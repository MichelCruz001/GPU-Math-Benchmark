"""Additional native formats. Sparse throughput counts the expanded matrix.
Input block per accumulator: A at 0, B at 64, metadata at 128; 256 bytes.
"""
import struct
import re
from more_formats import NEW_MODES,label

EXTRA_MODES = (
    'FP32_VECTOR', 'FP16_VECTOR', 'FP16_MATRIX', 'FP16_SPARSE',
    'FP8_E4M3_SPARSE', 'FP8_E5M2_SPARSE', 'INT8_SPARSE', 'INT4_SPARSE',
)

def describe(mode):
    if mode in NEW_MODES: return label(mode)
    return {
        'FP8':'FP8 E4M3 - matriz densa', 'INT8':'INT8 - matriz densa / DOT4',
        'INT4':'INT4 - matriz densa / DOT8',
        'FP32_VECTOR':'FP32 - vetor', 'FP16_VECTOR':'FP16 - vetor packed',
        'FP16_MATRIX':'FP16 - matriz densa (acumulacao FP32)',
        'FP16_SPARSE':'FP16 - matriz esparsa 2:4 (acumulacao FP32)',
        'FP8_E4M3_SPARSE':'FP8 E4M3 - matriz esparsa 2:4',
        'FP8_E5M2_SPARSE':'FP8 E5M2 - matriz esparsa 2:4',
        'INT8_SPARSE':'INT8 - matriz esparsa 2:4',
        'INT4_SPARSE':'INT4 - matriz esparsa 2:4',
    }[mode]

def pack_values(fmt,values):
    if fmt=='FP64': return struct.pack('<'+'d'*len(values),*values)
    if fmt in ('INT32','INT16'): return struct.pack('<'+('i' if fmt=='INT32' else 'h')*len(values),*values)
    if fmt=='BF16': return b''.join(struct.pack('<f',v)[2:] for v in values)
    if fmt=='TF32': return struct.pack('<'+'f'*len(values),*values)
    if fmt=='INT1': return bytes(sum((values[i+j]&1)<<j for j in range(8)) for i in range(0,len(values),8))
    if fmt in ('NVFP4','MXFP4'):
        codes={0:0,1:2,2:4,3:5,4:6}
        n=[codes[abs(v)]|(8 if v<0 else 0) for v in values]
        return bytes(n[i]|(n[i+1]<<4) for i in range(0,len(n),2))
    if fmt=='MXFP6': return bytes({0:0,1:12,2:16,3:18,4:20}[abs(v)]|(32 if v<0 else 0) for v in values)
    if fmt=='MXFP8': fmt='FP8_E4M3'
    if fmt in ('FP32','FP16'):
        return struct.pack('<'+('f' if fmt=='FP32' else 'e')*len(values),*values)
    if fmt.startswith('FP8'):
        positive=[0,0x38,0x40,0x44,0x48] if fmt=='FP8_E4M3' else [0,0x3c,0x40,0x42,0x44]
        return bytes(positive[abs(v)] | (0x80 if v<0 else 0) for v in values)
    if fmt=='INT8': return bytes(v&255 for v in values)
    return bytes((values[i]&15)|((values[i+1]&15)<<4) for i in range(0,len(values),2))

def payload(spec,sign=1,pattern=False,second=False,scales=(1,1)):
    a=pack_values(spec['input_format'],[sign]*spec['a_elements'])
    b=pack_values(spec['input_format'],([1,2,3,4]*((spec['b_elements']+3)//4))[:spec['b_elements']] if pattern and not spec.get('vector') else [1]*spec['b_elements'])
    if spec.get('positive_fma'): b=pack_values(spec['input_format'],[2 if pattern else 1]*spec['b_elements'])
    if spec.get('unsigned_binary') and pattern: b=pack_values('INT1',[0,1]*(spec['b_elements']//2))
    block=bytearray(256); block[:len(a)]=a; block[64:64+len(b)]=b
    if spec.get('vector'):
        block[96:96+len(b)]=pack_values(spec['input_format'],[2 if pattern else 1]*spec['b_elements'])
    # Two 2-bit indices per nibble: 0,1 => 0x4; 2,3 => 0xe.
    block[128:132]=struct.pack('<I',0xeeeeeeee if second else 0x44444444)
    if spec.get('block_scale'):
        enc={1:0x38,2:0x40,4:0x48} if spec['input_format']=='NVFP4' else {1:127,2:128,4:129}
        block[128:136]=bytes([enc[scales[0]]])*4+bytes([enc[scales[1]]])*4
    return bytes(block)*spec.get('chains',4)

def amd_extended(info,mode,Unsupported):
    arch=info['arch']; vector=mode.endswith('_VECTOR'); sparse=mode.endswith('_SPARSE')
    if sparse and not arch.startswith('gfx12'):
        raise Unsupported('SWMMAC esparso 2:4 requer RDNA4 / gfx12 neste backend.')
    if not vector and not sparse and not arch.startswith(('gfx11','gfx12')):
        raise Unsupported('FP16 WMMA requer gfx11/gfx12 neste backend.')
    fmt='FP32' if mode=='FP32_VECTOR' else ('FP16' if mode.startswith('FP16') else mode.removesuffix('_SPARSE'))
    chains=8 if mode=='FP16_VECTOR' else 4
    integer=fmt.startswith('INT'); k=64 if fmt=='INT4' else (32 if sparse else 16)
    adecl=bdecl=''
    if vector:
        lanes=2 if fmt=='FP16' or arch.startswith('gfx12') else 1
        atype='h2' if fmt=='FP16' else ('f2' if lanes==2 else 'float'); btype=atype; ctype=atype
        ae=be=lanes; mnemonic=('v_pk_fma_f16' if fmt=='FP16' else 'v_dual_fmac_f32') if lanes==2 else 'v_fma'
        expr=lambda j: f'__builtin_elementwise_fma(a{j},b{j},c{j})'
        k=1; opt=2*lanes*16
    else:
        lanes=8; ctype='i8' if integer else 'f8'
        if fmt=='FP16':
            ae=8 if arch.startswith('gfx12') else 16; be=16 if sparse else ae
            atype=f'h{ae}'; btype=f'h{be}'; suffix='f16'
        else:
            ae=16 if fmt=='INT4' else 8; be=ae*2
            atype='i2'; btype='i4'; suffix={'INT4':'iu4','INT8':'iu8','FP8_E4M3':'fp8_fp8','FP8_E5M2':'bf8_bf8'}[fmt]
        prefix='swmmac' if sparse else 'wmma'; accum='i32' if integer else 'f32'
        mnemonic=f'v_{prefix}_{accum}_16x16x{k}_{suffix}'
        builtin='__builtin_amdgcn_'+mnemonic[2:]+'_w32'+('' if sparse or arch.startswith('gfx11') else '_gfx12')
        if integer: expr=lambda j: f'{builtin}(true,a{j},true,b{j},c{j},meta,false)'
        else: expr=lambda j: f'{builtin}(a{j},b{j},c{j}'+(',meta)' if sparse else ')')
        opt=2*16*16*k//32
    decl='\n'.join(f'{atype} a{j}=*(const {atype}*)(in+{j*256}); {btype} b{j}=*(const {btype}*)(in+{j*256+64}); {ctype} c{j}={{}}; c{j}+={j};' for j in range(chains))
    ops='\n'.join(f'c{j}={expr(j)};' for j in range(chains))
    if vector:
        decl+='\n'+'\n'.join(f'{btype} n{j}=*(const {btype}*)(in+{j*256+96}); {atype} neg{j}=-a{j}; asm volatile("" : "+v"(a{j}), "+v"(neg{j}), "+v"(b{j}), "+v"(n{j}));' for j in range(chains))
        ops='\n'.join(f'c{j}=__builtin_elementwise_fma({"neg" if rep%2 else "a"}{j},{"n" if rep%2 else "b"}{j},c{j});' for rep in range(16) for j in range(chains))
    if vector and lanes==2:
        stores='\n'.join(f'out[tid*{chains*2}+{2*j}]=(float)c{j}[0]; out[tid*{chains*2}+{2*j+1}]=(float)c{j}[1];' for j in range(chains))
    elif vector: stores='\n'.join(f'out[tid*{chains}+{j}]=c{j};' for j in range(chains))
    else: stores='\n'.join(f'(({ctype}*)out)[tid*{chains}+{j}]=c{j};' for j in range(chains))
    source=f'''
typedef _Float16 h2 __attribute__((ext_vector_type(2)));
typedef _Float16 h8 __attribute__((ext_vector_type(8)));
typedef _Float16 h16 __attribute__((ext_vector_type(16)));
typedef float f2 __attribute__((ext_vector_type(2)));
typedef float f8 __attribute__((ext_vector_type(8)));
typedef int i8 __attribute__((ext_vector_type(8)));
typedef int i2 __attribute__((ext_vector_type(2)));
typedef int i4 __attribute__((ext_vector_type(4)));
extern "C" __attribute__((global)) void bench(const char* in, {'int' if integer else 'float'}* out,int iterations) {{
unsigned tid=__builtin_amdgcn_workgroup_id_x()*128+__builtin_amdgcn_workitem_id_x();
int meta=*(const int*)(in+128);
{decl}
#pragma clang loop unroll(disable)
for(int i=0;i<iterations;++i) {{ {ops} }}
{stores}
}}
'''
    return source.encode(),{'path':('vector dual FMA' if fmt=='FP32' else 'vector packed FMA') if vector and lanes==2 else ('vector FMA' if vector else ('SWMMAC 2:4' if sparse else 'WMMA FP16')),
        'lanes':lanes,'k':k,'ops_per_thread':opt,'stride':256 if vector else 128,'chains':chains,'mnemonic':mnemonic,
        'input_format':fmt,'a_elements':ae,'b_elements':be,'extended':True,'sparse':sparse,'vector':vector,
        'dot':0 if vector else (k//2 if sparse else k),'unit':'TOPS' if integer else 'TFLOPS',
        'native_count':chains*16 if vector else chains,
        'counting':'dense-equivalent (expanded 2:4 matrix)' if sparse else 'executed arithmetic',
        'nonzero_ops_per_thread':opt//2 if sparse else opt}


def nvidia_extended(info,mode,Unsupported):
    cc=info['cc']; vector=mode.endswith('_VECTOR'); sparse=mode.endswith('_SPARSE')
    fmt='FP32' if mode=='FP32_VECTOR' else ('FP16' if mode.startswith('FP16') else mode.removesuffix('_SPARSE'))
    fp=fmt.startswith('FP'); halfvector=mode=='FP16_VECTOR'
    if cc<61: raise Unsupported('Este backend requer CC 6.1+.')
    if not vector and cc<75: raise Unsupported('FP16 MMA requer CC 7.5+.')
    if sparse and cc<80: raise Unsupported('MMA esparso requer CC 8.0+.')
    if fmt.startswith('FP8') and cc<89: raise Unsupported('FP8 MMA esparso requer CC 8.9+.')
    k=1 if vector else (128 if fmt=='INT4' else (64 if fmt in ('INT8','FP8_E4M3','FP8_E5M2') else (32 if sparse else (16 if cc>=80 else 8))))
    if vector:
        lanes=2 if halfvector else 1; ae=be=lanes; na=nb=1
        mnemonic='fma.rn.f16x2' if halfvector else 'fma.rn.f32'; typ='b32' if halfvector else 'f32'
        opt=2*lanes*16; target=61; version='6.0'
    else:
        lanes=4; typ='f32' if fp else 'b32'
        na=4 if sparse or k==16 else 2; nb=4 if sparse else na//2
        elements_per_word=2 if fmt=='FP16' else (8 if fmt=='INT4' else 4)
        ae=na*elements_per_word; be=nb*elements_per_word
        suffix={'FP16':'f16.f16','FP8_E4M3':'e4m3.e4m3','FP8_E5M2':'e5m2.e5m2','INT8':'s8.s8','INT4':'s4.s4'}[fmt]
        accumulator='f32' if fp else 's32'
        mnemonic=f'mma{".sp" if sparse else ""}.sync.aligned.m16n8k{k}.row.col.{accumulator}.{suffix}.{accumulator}'
        opt=2*16*8*k//32; target=89 if fmt.startswith('FP8') else (80 if sparse or k==16 else 75)
        version='8.4' if fmt.startswith('FP8') else ('7.1' if target>=80 else '6.5')
    regs=[]; loads=[]; init=[]; operations=[]; stores=[]
    for j in range(4):
        regs.append(f'.reg .{typ} %c{j}<{lanes if not halfvector else 1}>;')
        regs.append(f'.reg .b32 %a{j}<{na}>,%b{j}<{nb}>;')
        for i in range(na): loads.append(f'ld.global.b32 %a{j}{i},[%in+{256*j+4*i}];')
        for i in range(nb): loads.append(f'ld.global.b32 %b{j}{i},[%in+{256*j+64+4*i}];')
        for i in range(lanes if not halfvector else 1):
            value=(struct.unpack('<I',struct.pack('<ee',j,j))[0] if halfvector else (f'0f{struct.pack(">f",float(j)).hex()}' if fp else j))
            init.append(f'mov.{typ} %c{j}{i},{value};')
        if vector:
            regs.append(f'.reg .b32 %n{j},%neg{j};')
            loads.append(f'ld.global.b32 %n{j},[%in+{256*j+96}];')
            # Flip sign bits once, outside the timed arithmetic loop.
            loads.append(f'xor.b32 %neg{j},%a{j}0,{"0x80008000" if halfvector else "0x80000000"};')
        else:
            acc='{'+','.join(f'%c{j}{i}' for i in range(lanes))+'}'
            a='{'+','.join(f'%a{j}{i}' for i in range(na))+'}'
            b='{'+','.join(f'%b{j}{i}' for i in range(nb))+'}'
            operations.append(f'{mnemonic} {acc},{a},{b},{acc}'+(',%meta,0;' if sparse else ';'))
        if halfvector:
            regs.append(f'.reg .b16 %lo{j},%hi{j}; .reg .f32 %fl{j},%fh{j};')
            stores.extend([f'mov.b32 {{%lo{j},%hi{j}}},%c{j}0;',f'cvt.f32.f16 %fl{j},%lo{j};',f'cvt.f32.f16 %fh{j},%hi{j};',
                           f'st.global.f32 [%out+{j*8}],%fl{j};',f'st.global.f32 [%out+{j*8+4}],%fh{j};'])
        else:
            stores.extend(f'st.global.{typ} [%out+{(j*lanes+i)*4}],%c{j}{i};' for i in range(lanes))
    if vector:
        operations=[f'{mnemonic} %c{j}0,{"%neg"+str(j) if rep%2 else "%a"+str(j)+"0"},{"%n"+str(j) if rep%2 else "%b"+str(j)+"0"},%c{j}0;' for rep in range(16) for j in range(4)]
    source=f'''.version {version}
.target sm_{target}
.address_size 64
.visible .entry bench(.param .u64 input,.param .u64 output,.param .u32 iterations) {{
.reg .b64 %in,%out,%offset;
.reg .b32 %idx,%block,%count,%meta; .reg .pred %again;
{chr(10).join(regs)}
ld.param.u64 %in,[input]; ld.param.u64 %out,[output]; ld.param.u32 %count,[iterations];
ld.global.b32 %meta,[%in+128];
{chr(10).join(loads)}
mov.u32 %idx,%tid.x; mov.u32 %block,%ctaid.x;
mad.lo.u32 %idx,%block,128,%idx;
mul.wide.u32 %offset,%idx,{4*lanes*4}; add.u64 %out,%out,%offset;
{chr(10).join(init)}
LOOP:
{chr(10).join(operations)}
sub.u32 %count,%count,1; setp.ne.u32 %again,%count,0; @%again bra LOOP;
{chr(10).join(stores)}
ret;
}}
'''
    source=re.sub(r'%([abc])([0-3])<',r'%\1\2_<',source)
    source=re.sub(r'%([abc])([0-3])([0-3])\b',r'%\1\2_\3',source)
    return source.encode(),{'path':('vector FMA packed' if halfvector else 'vector FMA') if vector else ('MMA sparse 2:4' if sparse else 'MMA FP16'),
        'lanes':lanes,'k':k,'ops_per_thread':opt,'stride':128,'mnemonic':mnemonic,'extended':True,
        'input_format':fmt,'a_elements':ae,'b_elements':be,'sparse':sparse,'vector':vector,
        'dot':0 if vector else (k//2 if sparse else k),'unit':'TFLOPS' if fp else 'TOPS',
        'counting':'dense-equivalent (expanded 2:4 matrix)' if sparse else 'executed arithmetic',
        'nonzero_ops_per_thread':opt//2 if sparse else opt}
