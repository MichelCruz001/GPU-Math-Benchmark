"""Native format catalog and kernels. Unsupported paths are never emulated."""
import struct
NEW_MODES=('FP64_VECTOR','BF16_VECTOR','INT32_VECTOR','INT16_VECTOR','INT4_VECTOR',
 'INT1_MATRIX','INT2_MATRIX','NVFP4_MATRIX','MXFP4_MATRIX','MXFP6_MATRIX','MXFP8_MATRIX',
 'BF16_MATRIX','TF32_MATRIX','FP32_MATRIX','FP64_MATRIX')

def is_float(mode): return not mode.startswith('INT')

def category(mode,info=None):
    if mode.endswith('_VECTOR'): return 'Vetor'
    if mode in ('INT8','INT4') and info and info['vendor']=='AMD' and not info['arch'].startswith(('gfx11','gfx12')): return 'Vetor'
    if mode=='INT8' and info and info['vendor']=='NVIDIA' and (info['cc']<75 or info.get('int8_path')=='dp4a'): return 'Vetor'
    return 'Matriz'

def label(mode):
    fmt=mode.split('_')[0]
    suffix={'INT16_VECTOR':'adição packed de 16 bits','INT32_VECTOR':'inteiros de 32 bits',
            'INT4_VECTOR':'produto escalar packed INT4 (DOT8)','INT1_MATRIX':'binário AND + popcount',
            'MXFP6_MATRIX':'E3M2 + escala E8M0','MXFP8_MATRIX':'E4M3 + escala E8M0',
            'MXFP4_MATRIX':'E2M1 + escala E8M0','NVFP4_MATRIX':'E2M1 + escala E4M3'}.get(mode,'cálculo nativo')
    return f'{fmt} — {suffix}'

def base(fmt,lanes,k,ops,ae,be,**extra):
    return dict(path=fmt,lanes=lanes,k=k,dot=k,ops_per_thread=ops,stride=128,chains=4,
        input_format=fmt,a_elements=ae,b_elements=be,extended=True,vector=False,sparse=False,**extra)

def amd_more(info,mode,Unsupported):
    from extended_kernels import amd_extended
    arch=info['arch']
    if mode=='BF16_VECTOR':
        if not arch.startswith(('gfx11','gfx12')): raise Unsupported('DOT2 BF16 deste backend requer RDNA3/4.')
        spec=base('BF16',1,2,4,2,2,output_type='FP32')
        spec.update(path='vector BF16 DOT2, acumulador FP32',mnemonic='v_dot2_f32_bf16',native_count=4)
        decl='\n'.join(f's2 a{j}=*(const s2*)(in+{256*j}), b{j}=*(const s2*)(in+{256*j+64}); float c{j}={j};' for j in range(4))
        ops='\n'.join(f'c{j}=__builtin_amdgcn_fdot2_f32_bf16(a{j},b{j},c{j},false);' for j in range(4))
        stores='\n'.join(f'out[tid*4+{j}]=c{j};' for j in range(4))
        return ('typedef short s2 __attribute__((ext_vector_type(2)));\n'+amd_source('float',decl,ops,stores)).encode(),spec
    if (mode=='BF16_MATRIX' and arch in ('gfx90a','gfx940','gfx941','gfx942','gfx950')) or (mode=='TF32_MATRIX' and arch in ('gfx940','gfx941','gfx942')):
        bf16=mode=='BF16_MATRIX'; fmt='BF16' if bf16 else 'TF32'
        typ='short' if bf16 else 'float'; width=4 if bf16 else 2; k=16 if bf16 else 8
        builtin='__builtin_amdgcn_mfma_f32_16x16x16bf16_1k' if bf16 else '__builtin_amdgcn_mfma_f32_16x16x8_xf32'
        spec=base(fmt,4,k,8*k,width,width,output_type='FP32')
        spec.update(path='MFMA '+fmt,mnemonic='v_mfma_f32_16x16x16bf16' if bf16 else 'v_mfma_f32_16x16x8_xf32')
        spec['audit_regex']=r'v_mfma_f32_16x16x'+('16_?bf16(?:_1k)?' if bf16 else '8_?xf32')+r'\b'
        decl='\n'.join(f'input a{j}=*(const input*)(in+{256*j}), b{j}=*(const input*)(in+{256*j+64}); acc c{j}={{}}; c{j}+={j};' for j in range(4))
        ops='\n'.join(f'c{j}={builtin}(a{j},b{j},c{j},0,0,0);' for j in range(4))
        stores='\n'.join(f'((acc*)out)[tid*4+{j}]=c{j};' for j in range(4))
        return (f'typedef {typ} input __attribute__((ext_vector_type({width}))); typedef float acc __attribute__((ext_vector_type(4)));\n'+amd_source('float',decl,ops,stores)).encode(),spec
    if mode=='BF16_MATRIX':
        src,spec=amd_extended(info,'FP16_MATRIX',Unsupported)
        src=src.replace(b'_Float16',b'short').replace(b'_f16_',b'_bf16_')
        spec.update(path='WMMA BF16',input_format='BF16',mnemonic=spec['mnemonic'].replace('_f16','_bf16'))
        return src,spec
    if mode in ('FP32_MATRIX','FP64_MATRIX'):
        fp64=mode=='FP64_MATRIX'
        supported=('gfx90a','gfx940','gfx941','gfx942','gfx950') if fp64 else ('gfx908','gfx90a','gfx940','gfx941','gfx942','gfx950')
        if arch not in supported: raise Unsupported('MFMA FP32/FP64 requer uma arquitetura CDNA implementada; TF32 nao substitui FP32.')
        typ='double' if fp64 else 'float'; fmt='FP64' if fp64 else 'FP32'
        mnemonic='v_mfma_'+('f64' if fp64 else 'f32')+'_16x16x4'+('f64' if fp64 else 'f32')
        builtin='__builtin_amdgcn_'+mnemonic[2:]
        spec=base(fmt,4,4,32,1,1)
        spec.update(path='MFMA '+fmt,mnemonic=mnemonic,output_type=fmt)
        spec['audit_regex']=r'v_mfma_'+('f64' if fp64 else 'f32')+r'_16x16x4_?'+('f64' if fp64 else 'f32')+r'\b'
        decl='\n'.join(f'{typ} a{j}=*(const {typ}*)(in+{256*j}), b{j}=*(const {typ}*)(in+{256*j+64}); acc c{j}={{}}; c{j}+={j};' for j in range(4))
        ops='\n'.join(f'c{j}={builtin}(a{j},b{j},c{j},0,0,0);' for j in range(4))
        stores='\n'.join(f'((acc*)out)[tid*4+{j}]=c{j};' for j in range(4))
        return (f'typedef {typ} acc __attribute__((ext_vector_type(4)));\n'+amd_source(typ,decl,ops,stores)).encode(),spec
    if mode not in ('FP64_VECTOR','INT32_VECTOR','INT16_VECTOR','INT4_VECTOR'):
        raise Unsupported('Sem instrucao nativa implementada para este formato nesta arquitetura AMD.')
    fmt=mode.split('_')[0]; typ='double' if fmt=='FP64' else 'unsigned int'
    lanes=2 if fmt=='INT16' else 1; k=8 if fmt=='INT4' else 16
    spec=base(fmt,lanes,k,16 if fmt=='INT4' else (32 if fmt in ('FP64','INT16') else 16),
              8 if fmt=='INT4' else lanes,8 if fmt=='INT4' else lanes)
    spec.update(output_type='FP64' if fmt=='FP64' else 'INT32',native_count=4 if fmt=='INT4' else 64)
    mnemonic={'FP64':'v_fma_f64','INT32':'v_add_nc_u32','INT16':'v_pk_add_u16','INT4':'v_dot8_i32_i4'}[fmt]
    if fmt=='INT32' and not arch.startswith(('gfx10','gfx11','gfx12')): mnemonic='v_add_u32'
    spec.update(mnemonic=mnemonic,path=fmt+' vector '+('FMA' if fmt=='FP64' else ('DOT8' if fmt=='INT4' else 'ADD')))
    if fmt=='FP64': spec['audit_regex']=r'v_fmac?_f64(?:_e32|_e64)?\b'
    if fmt=='INT4': spec['audit_regex']=r'v_dot8c?_i32_i?u?4(?:_e32|_e64)?\b'
    if fmt=='INT16': spec['wrap_bits']=16
    decl='\n'.join(f'{typ} a{j}=*(const {typ}*)(in+{j*256}), b{j}=*(const {typ}*)(in+{j*256+64}), c{j}={j*65537 if fmt=="INT16" else j}; asm volatile("" : "+v"(a{j}), "+v"(b{j}));' for j in range(4))
    if fmt=='FP64': expr=lambda j:f'c{j}=__builtin_fma(a{j},b{j},c{j});'
    elif fmt=='INT4':
        expr=(lambda j:f'c{j}=__builtin_amdgcn_sudot8(true,a{j},true,b{j},c{j},false);') if arch.startswith(('gfx11','gfx12')) else (lambda j:f'c{j}=__builtin_amdgcn_sdot8(a{j},b{j},c{j},false);')
    else: expr=lambda j:f'asm volatile("{mnemonic} %0, %1, %2" : "=v"(c{j}) : "v"(a{j}), "v"(c{j}));'
    ops='\n'.join(expr(j) for _ in range(1 if fmt=='INT4' else 16) for j in range(4))
    stores='\n'.join((f'out[tid*8+{j*2}]=(short)(c{j}&65535); out[tid*8+{j*2+1}]=(short)(c{j}>>16);' if fmt=='INT16' else f'out[tid*4+{j}]=c{j};') for j in range(4))
    return amd_source('double' if fmt=='FP64' else 'int',decl,ops,stores).encode(),spec

def amd_source(out,decl,ops,stores):
    return f'''extern "C" __attribute__((global)) void bench(const char* in,{out}* out,int iterations) {{
unsigned tid=__builtin_amdgcn_workgroup_id_x()*128+__builtin_amdgcn_workitem_id_x();
{decl}
#pragma clang loop unroll(disable)
for(int i=0;i<iterations;++i) {{ {ops} }}
{stores}
}}'''

def nvidia_more(info,mode,Unsupported):
    from extended_kernels import nvidia_extended,pack_values
    cc=info['cc']
    if mode=='BF16_VECTOR':
        if cc<80: raise Unsupported('BF16 FMA requer CC 8.0+.')
        src,spec=nvidia_extended(info,'FP16_VECTOR',Unsupported)
        src=src.decode().replace('.version 6.0','.version 7.1').replace('.target sm_61','.target sm_80').replace('f16','bf16')
        for j in range(4):
            old=struct.unpack('<I',struct.pack('<ee',j,j))[0]
            new=int.from_bytes(pack_values('BF16',[j,j]),'little')
            src=src.replace(f'mov.b32 %c{j}_0,{old};',f'mov.b32 %c{j}_0,{new};')
        spec.update(path='vector BF16 FMA',input_format='BF16',output_type='FP32')
        return src.encode(),spec
    vector=mode in ('FP64_VECTOR','INT32_VECTOR')
    block=mode in ('NVFP4_MATRIX','MXFP4_MATRIX','MXFP6_MATRIX','MXFP8_MATRIX')
    binary=mode=='INT1_MATRIX'; fp64=mode.startswith('FP64')
    if mode in ('INT16_VECTOR','INT4_VECTOR','INT2_MATRIX','FP32_MATRIX'):
        raise Unsupported('Sem caminho nativo desta precisao implementado; nenhuma promocao ou decomposicao aplicada.')
    if block and cc!=120: raise Unsupported('Este backend block-scale requer SM120a (Blackwell CC12.0); TCGEN05 de SM100/103 ainda nao implementado.')
    if mode=='FP64_MATRIX' and cc not in (80,90): raise Unsupported('FP64 Tensor habilitado apenas em SM80/SM90 neste backend.')
    if mode in ('BF16_MATRIX','TF32_MATRIX','INT1_MATRIX') and cc<80: raise Unsupported('Este caminho de matriz requer CC8.0+.')
    if binary and cc>=90: raise Unsupported('MMA binario deste backend limitado a SM80/86/89.')
    if not vector and mode not in ('FP64_MATRIX','BF16_MATRIX','TF32_MATRIX','INT1_MATRIX') and not block: raise Unsupported('Formato nao implementado para NVIDIA.')
    fmt=mode.split('_')[0]; outtype='f64' if fp64 else ('s32' if mode.startswith('INT') else 'f32')
    k=16 if vector else (4 if fp64 else (16 if fmt=='BF16' else (8 if fmt=='TF32' else (128 if binary else (64 if fmt in ('NVFP4','MXFP4') else 32)))))
    lanes=1 if vector else (2 if fp64 or binary else 4)
    na=nb=1
    if not vector and not fp64 and not binary: na,nb=4,2
    target='120a' if block else ('80' if not vector else '61')
    version='8.7' if block else ('7.1' if not vector else '6.0')
    itype='f64' if fp64 else 'b32'; bytesreg=8 if fp64 else 4
    if vector: mnemonic='fma.rn.f64' if fp64 else 'mad.lo.s32'
    else:
        shape=f'm{8 if fp64 or binary else 16}n8k{k}'
        suffix={'BF16':'bf16.bf16','TF32':'tf32.tf32','FP64':'f64.f64','INT1':'b1.b1'}.get(fmt)
        if block:
            kind='mxf4nvf4.block_scale.scale_vec::4X' if fmt=='NVFP4' else ('mxf4.block_scale' if fmt=='MXFP4' else 'mxf8f6f4.block_scale.scale_vec::1X')
            elem='e2m1' if k==64 else ('e3m2' if fmt=='MXFP6' else 'e4m3')
            mnemonic=f'mma.sync.aligned.{shape}.row.col.kind::{kind}.f32.{elem}.{elem}.f32.'+('ue4m3' if fmt=='NVFP4' else 'ue8m0')
        else: mnemonic=f'mma.sync.aligned.{shape}.row.col.{outtype}.{suffix}.{outtype}'+('.and.popc' if binary else '')
    decl=[]; loads=[]; init=[]; operations=[]; stores=[]
    for j in range(4):
        decl += [f'.reg .{itype} %a{j}_<{na}>,%b{j}_<{nb}>;',f'.reg .{outtype} %c{j}_<{lanes}>;']
        loads += [f'ld.global.{itype} %a{j}_{i},[%in+{256*j+bytesreg*i}];' for i in range(na)]
        loads += [f'ld.global.{itype} %b{j}_{i},[%in+{256*j+64+bytesreg*i}];' for i in range(nb)]
        literal=str(j) if outtype=='s32' else ('0d'+struct.pack('>d',j).hex() if fp64 else '0f'+struct.pack('>f',j).hex())
        init += [f'mov.{outtype} %c{j}_{i},{literal};' for i in range(lanes)]
        c='{'+','.join(f'%c{j}_{i}' for i in range(lanes))+'}'
        a='{'+','.join(f'%a{j}_{i}' for i in range(na))+'}'; b='{'+','.join(f'%b{j}_{i}' for i in range(nb))+'}'
        operations.append(f'{mnemonic} %c{j}_0,%a{j}_0,%b{j}_0,%c{j}_0;' if vector else f'{mnemonic} {c},{a},{b},{c}'+(',%scaleA,{0,0},%scaleB,{0,0};' if block else ';'))
        stores += [f'st.global.{outtype} [%out+{(j*lanes+i)*bytesreg}],%c{j}_{i};' for i in range(lanes)]
    scale_load='ld.global.b32 %scaleA,[%in+128]; ld.global.b32 %scaleB,[%in+132];' if block else ''
    src=f'''.version {version}
.target sm_{target}
.address_size 64
.visible .entry bench(.param .u64 input,.param .u64 output,.param .u32 iterations) {{
.reg .b64 %in,%out,%offset;
.reg .b32 %idx,%block,%count,%scaleA,%scaleB; .reg .pred %again;
{chr(10).join(decl)}
ld.param.u64 %in,[input]; ld.param.u64 %out,[output]; ld.param.u32 %count,[iterations];
mov.u32 %idx,%tid.x; mov.u32 %block,%ctaid.x; mad.lo.u32 %idx,%block,128,%idx;
mul.wide.u32 %offset,%idx,{4*lanes*bytesreg}; add.u64 %out,%out,%offset;
{chr(10).join(loads)}
{scale_load}
{chr(10).join(init)}
LOOP:
{chr(10).join(operations*(16 if vector else 1))}
sub.u32 %count,%count,1; setp.ne.u32 %again,%count,0; @%again bra LOOP;
{chr(10).join(stores)}
ret;
}}'''
    epw=2 if fmt=='BF16' else (32 if binary else (8 if k==64 and block else (4 if block else 1)))
    spec=base(fmt,lanes,k,32 if vector else 2*(8 if fp64 or binary else 16)*8*k//32,na*epw,nb*epw,
              output_type='FP64' if fp64 else ('INT32' if outtype=='s32' else 'FP32'),unsigned_binary=binary,block_scale=block)
    spec.update(path=('vector FMA' if fp64 else 'vector MAD') if vector else ('MMA AND+POPC' if binary else ('MMA block-scale '+fmt if block else 'MMA '+fmt)),mnemonic=mnemonic)
    return src.encode(),spec
