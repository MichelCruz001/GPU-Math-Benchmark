"""FP32 candidates: exact bounded positive accumulation, no fast-math."""
VARIANTS=('reference','c8w1','c16w1','c32w1','c8w2','c16w2','c32w2')

def named_candidate(name):
    if name not in VARIANTS[1:]: raise ValueError('Unknown FP32 candidate: '+name)
    chains,width=name[1:].split('w')
    return candidate(int(chains),int(width))

def candidate(chains=16,width=1):
    if chains not in (8,16,32) or width not in (1,2): raise ValueError('Invalid FP32 variant')
    typ='float' if width==1 else 'f2'
    decl=f'{typ} a=*(const {typ}*)in, b=*(const {typ}*)(in+64); asm volatile("" : "+v"(a), "+v"(b));\n'
    decl+='\n'.join(f'{typ} c{j}={{}}; c{j}+={j};' for j in range(chains))
    ops='\n'.join(f'c{j}=__builtin_elementwise_fma(a,b,c{j});' for _ in range(16) for j in range(chains))
    stores='\n'.join(f'out[tid*{chains*width}+{j*width+l}]='+ (f'c{j}' if width==1 else f'c{j}[{l}]')+';' for j in range(chains) for l in range(width))
    src=f'''typedef float f2 __attribute__((ext_vector_type(2)));
extern "C" __attribute__((global)) void bench(const char* in,float* out,int iterations) {{
unsigned tid=__builtin_amdgcn_workgroup_id_x()*128+__builtin_amdgcn_workitem_id_x();
{decl}
#pragma clang loop unroll(disable)
for(int i=0;i<iterations;++i) {{ {ops} }}
{stores}
}}'''
    return src.encode(),{'path':f'FP32 FMA {chains}x{width} independent','variant':f'c{chains}w{width}',
        'lanes':width,'k':1,'ops_per_thread':32*width,'stride':32*chains,'chains':chains,
        'mnemonic':'v_fma','input_format':'FP32','a_elements':width,'b_elements':width,
        'extended':True,'sparse':False,'vector':True,'positive_fma':True,
        'dot':16,'pattern_dot':32,'native_count':16*chains*width,'counting':'executed arithmetic'}
