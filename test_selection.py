"""Host-only regression tests. Run: python -m unittest test_selection.py"""
import unittest
from gpu_bench import parse_modes,nvidia_kernel,amd_kernel,Unsupported

class SelectionTests(unittest.TestCase):
    def test_fp32_candidate_count_and_precision(self):
        from fp32_kernels import VARIANTS,named_candidate
        from extended_kernels import payload
        import struct
        for name in VARIANTS[1:]:
            source,spec=named_candidate(name)
            self.assertEqual(spec['ops_per_thread']*spec['chains'],2*spec['native_count'])
            self.assertEqual(source.count(b'__builtin_elementwise_fma'),spec['chains']*16)
            self.assertEqual(len(payload(spec)),8*spec['stride'])
            data=payload(spec,-1,True)
            self.assertEqual(struct.unpack_from('<f',data,0)[0],-1)
            self.assertEqual(struct.unpack_from('<f',data,64)[0],2)
            self.assertLess(spec['dot']*65536+spec['chains'],2**24)
            self.assertEqual(spec['pattern_dot'],32)

    def intel_info(self,subgroup):
        from intel_opencl import MATRIX_EXT
        return {'vendor':'Intel','subgroup_sizes':[subgroup,32],
                'extensions':[MATRIX_EXT,'cl_intel_required_subgroup_size','cl_khr_fp16'],
                'single_fp_config':32,'half_fp_config':32,'max_work_group_size':1024}

    def test_intel_subgroup_layouts(self):
        from intel_opencl import intel_kernel
        from extended_kernels import payload
        for subgroup in (8,16):
            for mode,k,bits in [('INT8',32,8),('INT4',64,4),('FP16_MATRIX',16,16)]:
                src,spec=intel_kernel(self.intel_info(subgroup),mode,Unsupported)
                self.assertIn(f'intel_reqd_sub_group_size({subgroup})'.encode(),src)
                self.assertIn(('short8 a0' if subgroup==16 else 'int8 a0').encode(),src)
                self.assertEqual(spec['a_elements']*subgroup,8*k)
                self.assertEqual(spec['b_elements'],k)
                self.assertEqual(spec['ops_per_thread']*subgroup,2*8*subgroup*k)
                self.assertEqual(len(payload(spec)),1024)
                self.assertEqual(spec['lanes'],8)

    def test_intel_capabilities(self):
        from gpu_bench import available_modes
        for subgroup in (8,16):
            available,hidden=available_modes(self.intel_info(subgroup))
            self.assertEqual(available,['INT8','INT4','FP32_VECTOR','FP16_VECTOR','FP16_MATRIX','INT32_VECTOR','BF16_MATRIX'])
            self.assertIn('FP8',hidden)
            self.assertIn('INT4_SPARSE',hidden)
        info=self.intel_info(8); info['extensions']=[]
        self.assertEqual(available_modes(info)[0],['FP32_VECTOR','INT32_VECTOR'])
        info=self.intel_info(32)
        self.assertEqual(available_modes(info)[0],['FP32_VECTOR','FP16_VECTOR','INT32_VECTOR'])

    def test_filtered_selection(self):
        from gpu_bench import available_modes,select_modes
        for cc,expected in [(61,['INT8','FP32_VECTOR','FP16_VECTOR']),
                            (75,['INT8','INT4','FP32_VECTOR','FP16_VECTOR','FP16_MATRIX'])]:
            expected+=['FP64_VECTOR','INT32_VECTOR']
            available,hidden=available_modes({'vendor':'NVIDIA','cc':cc})
            self.assertEqual(available,expected)
            self.assertEqual(select_modes(['ALL'],available,hidden),expected)
            self.assertEqual(select_modes(['2,4'],available,hidden),['INT8','FP32_VECTOR'])
            with self.assertRaises(ValueError): select_modes(['1'],available,hidden)
        available,_=available_modes({'vendor':'AMD','arch':'gfx1036'})
        self.assertEqual(available,['INT8','INT4','FP32_VECTOR','FP16_VECTOR','FP64_VECTOR','INT32_VECTOR','INT16_VECTOR','INT4_VECTOR'])
        from gpu_bench import MODES
        self.assertEqual(available_modes({'vendor':'AMD','arch':'gfx1201'})[0],list(MODES[:16])+['BF16_MATRIX'])

    def test_new_formats_native_gates_and_units(self):
        from gpu_bench import unit_for
        from more_formats import category
        for mode in ('BF16_VECTOR','TF32_MATRIX','MXFP4_MATRIX','NVFP4_MATRIX'):
            self.assertEqual(unit_for(mode),'TFLOPS')
        self.assertEqual(parse_modes(['MXPF4']),['MXFP4_MATRIX'])
        for info in ({'cc':80},{'cc':120}):
            for mode in ('INT2_MATRIX','FP32_MATRIX'):
                with self.assertRaises(Unsupported): nvidia_kernel(info,mode)
        with self.assertRaises(Unsupported): nvidia_kernel({'cc':100},'NVFP4_MATRIX')
        self.assertEqual(category('INT8',{'vendor':'NVIDIA','cc':89,'int8_path':'dp4a'}),'Vetor')
        self.assertEqual(category('BF16_MATRIX'),'Matriz')
        for mode in ('FP32_MATRIX','FP64_MATRIX'):
            _,spec=amd_kernel({'arch':'gfx90a'},mode)
            self.assertEqual(spec['ops_per_thread']*64,2*16*16*4)

    def test_format_payload_and_scale_encoding(self):
        from extended_kernels import pack_values,payload
        self.assertEqual(pack_values('BF16',[1,-1]),bytes.fromhex('803f80bf'))
        self.assertEqual(pack_values('INT1',[0,1]*4),bytes.fromhex('aa'))
        self.assertEqual(pack_values('MXFP4',[1,2,-1,-2]),bytes.fromhex('42ca'))
        for mode,scale in [('MXFP4_MATRIX','80808181'),('NVFP4_MATRIX','40404848')]:
            _,spec=nvidia_kernel({'cc':120},mode)
            data=payload(spec,scales=(2,4))
            self.assertEqual(data[128:136],bytes.fromhex(scale[:4]*2+scale[4:]*2))
            self.assertEqual(len(data),1024)
        _,spec=amd_kernel({'arch':'gfx1201'},'INT16_VECTOR')
        self.assertEqual(spec['wrap_bits'],16)
        self.assertEqual(spec['ops_per_thread']*spec['chains'],128)

    def test_intel_new_precision_gates(self):
        from intel_opencl import intel_kernel,MATRIX_EXT
        info=self.intel_info(16)
        with self.assertRaises(Unsupported): intel_kernel(info,'FP64_VECTOR',Unsupported)
        with self.assertRaises(Unsupported): intel_kernel(info,'TF32_MATRIX',Unsupported)
        info['extensions'] += ['cl_khr_fp64',MATRIX_EXT+'_tf32']; info['double_fp_config']=32
        source,spec=intel_kernel(info,'TF32_MATRIX',Unsupported)
        self.assertIn(b'float4 a0',source); self.assertIn(b'float8 b0',source)
        self.assertEqual(spec['ops_per_thread']*16,2*8*16*8)
        source,spec=intel_kernel(info,'FP64_VECTOR',Unsupported)
        self.assertIn(b'__global double* output',source)
        source,spec=intel_kernel(info,'INT32_VECTOR',Unsupported)
        self.assertNotIn(b'fma(',source)
        self.assertNotIn(b'-a0',source)
        self.assertEqual(source.count(b'c0*b0+a0'),16)
        self.assertEqual(spec['dot'],16)

    def test_rate_statistics(self):
        from gpu_bench import sample_statistics
        row=sample_statistics(12_000_000_000,[1,2,3,6],True)
        self.assertEqual([row[k+'_tops'] for k in ('min','max','mean','median')],[2,12,6,5])
        self.assertEqual(row['median_ms'],2.5)
        self.assertEqual(row['nonzero_mean_tops'],3)
        for times in ([],[0],[-1],[float('nan')],[float('inf')]):
            with self.assertRaises(ValueError): sample_statistics(100,times)

    def test_one_two_three(self):
        self.assertEqual(parse_modes(['2']),['INT8'])
        self.assertEqual(parse_modes(['1,3']),['FP8','INT4'])
        self.assertEqual(parse_modes(['FP8','int8','INT4']),['FP8','INT8','INT4'])
    def test_duplicates(self):
        self.assertEqual(parse_modes(['2,2','INT8']),['INT8'])
    def test_bad_inputs(self):
        for value in ([],['0'],['1,99'],['FP16']):
            with self.assertRaises(ValueError): parse_modes(value)
    def test_pascal(self):
        _,spec=nvidia_kernel({'cc':61},'INT8')
        self.assertEqual((spec['path'],spec['ops_per_thread'] ),('DP4A',8))
        with self.assertRaises(Unsupported): nvidia_kernel({'cc':61},'INT4')
        with self.assertRaises(Unsupported): nvidia_kernel({'cc':60},'INT8')
    def test_tensor_and_explicit_dp4a(self):
        _,spec=nvidia_kernel({'cc':80},'INT8')
        self.assertEqual(spec['ops_per_thread'],2*8*8*16//32)
        _,spec=nvidia_kernel({'cc':80},'INT8','dp4a')
        self.assertEqual(spec['path'],'DP4A')
        _,spec=nvidia_kernel({'cc':75},'INT4')
        self.assertEqual(spec['ops_per_thread'],128)
    def test_fp8_gate(self):
        with self.assertRaises(Unsupported): nvidia_kernel({'cc':86},'FP8')
        ptx,spec=nvidia_kernel({'cc':89},'FP8')
        self.assertIn(b'.version 8.4',ptx)
        self.assertEqual(spec['ops_per_thread'],256)
    def test_amd_routes(self):
        _,spec=amd_kernel({'arch':'gfx1201'},'FP8')
        self.assertEqual(spec['path'],'WMMA')
        _,spec=amd_kernel({'arch':'gfx1036'},'INT8')
        self.assertEqual(spec['path'],'DOT4')
        _,spec=amd_kernel({'arch':'gfx1036'},'INT4')
        self.assertEqual(spec['path'],'DOT8')
        with self.assertRaises(Unsupported): amd_kernel({'arch':'gfx1036'},'FP8')

    def test_rdna4_int4_full_k(self):
        src,spec=amd_kernel({'arch':'gfx1201'},'INT4')
        self.assertIn(b'wmma_i32_16x16x32_iu4_w32_gfx12',src)
        self.assertIn(b'const v2i* in',src)
        self.assertEqual((spec['k'],spec['stride'],spec['ops_per_thread']),(32,8,512))
        _,old=amd_kernel({'arch':'gfx1201'},'INT4',16)
        self.assertEqual((old['k'],old['stride'],old['ops_per_thread']),(16,4,256))
        _,rdna3=amd_kernel({'arch':'gfx1100'},'INT4')
        self.assertEqual(rdna3['k'],16)

    def test_expanded_menu(self):
        from gpu_bench import MODES
        self.assertEqual(parse_modes(['ALL']),list(MODES))
        self.assertEqual(parse_modes(['4,5,11']),['FP32_VECTOR','FP16_VECTOR','INT4_SPARSE'])

    def test_sparse_payload_and_count(self):
        from extended_kernels import EXTRA_MODES,payload
        for mode in EXTRA_MODES:
            for vendor,info in [('AMD',{'arch':'gfx1201'}),('NVIDIA',{'cc':89})]:
                _,spec=amd_kernel(info,mode) if vendor=='AMD' else nvidia_kernel(info,mode)
                for pattern in (False,True):
                    self.assertEqual(len(payload(spec,pattern=pattern)),256*spec.get('chains',4))
                if spec['sparse']:
                    self.assertEqual(spec['nonzero_ops_per_thread']*2,spec['ops_per_thread'])
                    self.assertEqual(spec['dot']*2,spec['k'])
                    self.assertEqual(payload(spec)[128:132],bytes.fromhex('44444444'))
                    self.assertEqual(payload(spec,second=True)[128:132],bytes.fromhex('eeeeeeee'))

    def test_extended_gates(self):
        with self.assertRaises(Unsupported): amd_kernel({'arch':'gfx1036'},'FP16_SPARSE')
        with self.assertRaises(Unsupported): nvidia_kernel({'cc':75},'FP16_SPARSE')
        with self.assertRaises(Unsupported): nvidia_kernel({'cc':80},'FP8_E4M3_SPARSE')

if __name__=='__main__': unittest.main()
