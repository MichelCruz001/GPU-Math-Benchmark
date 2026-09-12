# Benchmark GPU — AMD, NVIDIA e Intel Arc (Windows x64)

O atalho **GPU Benchmark.lnk** usa o novo ícone de chip, também aplicado à janela. O ícone está em `assets/chip.ico`, com tamanhos de 16 a 256 pixels. O atalho aponta para a localização atual desta pasta; recrie-o se mover a ferramenta. **INICIAR.cmd** continua funcionando.

Abra **INICIAR.cmd** para usar a interface gráfica. Escolha a GPU, marque os testes e clique em **Iniciar benchmark**. A janela exibe apenas testes compatíveis, permite ajustar de 5 a 200 amostras (50 por padrão na GUI), acompanha o progresso por formato e mostra mínimo, máximo, média e mediana. A aba de registro mostra a saída completa e os diagnósticos. **Abrir resultados** abre a pasta com os relatórios.

**Idiomas da GUI:** a interface inicia em **English**. No seletor **Language**, no canto superior direito, escolha **Português (Brasil)** ou **简体中文** (chinês simplificado). A troca é imediata, inclusive durante um teste, e preserva a GPU selecionada, os formatos, o progresso e as abas de resultados. Botões, descrições dos testes, categorias, cabeçalhos, mensagens de andamento e diálogos da interface são traduzidos. A preferência vale para a sessão; uma nova abertura inicia em inglês. O console, os relatórios em disco e os detalhes técnicos de compatibilidade/driver permanecem no idioma original; a aba de registro está identificada como log técnico no idioma original. IDs de formatos, nomes de GPUs e unidades não são traduzidos.

O botão **Cancelar** aguarda o formato atual terminar; os resultados concluídos são preservados. Fechar a janela durante a execução solicita esse mesmo cancelamento e aguarda o processo terminar. A descoberta, compilação e execução não bloqueiam a interface. Intel/NVIDIA mantêm as limitações de validação em hardware descritas abaixo.

A GUI usa Tkinter, incluído na instalação padrão do Python para Windows; não exige pacotes pip. Também pode ser aberta com `python gui.py` ou `./run.ps1 -Gui`. A linha de comando continua disponível em **CONSOLE.cmd** e `run.ps1` sem `-Gui`.

Marque **Separar em Vetor e Matriz** para organizar os testes em duas abas, ou desmarque para uma lista única. A seleção é preservada ao alternar essa visualização. A lista tem rolagem. Na parte inferior, cada GPU possui sua própria aba de resultados; executar outra GPU não apaga a anterior. **Abrir resultados** acompanha a aba selecionada. As abas mantêm a última execução de cada GPU durante a sessão; todos os relatórios continuam salvos em disco.

No menu de console, escolha a GPU e qualquer combinação de formatos:

```text
1 = FP8   2 = INT8   3 = INT4
2       -> somente INT8
1,3     -> FP8 e INT4
1,2,3   -> os tres formatos
ALL     -> todos os testes disponiveis para a GPU escolhida
```

Também aceita nomes como `FP8 INT8`, vírgulas e letras minúsculas. Repetições são removidas. Uma opção inválida faz o menu perguntar novamente. Ctrl+C cancela.

Depois da escolha da GPU, o menu mostra apenas os caminhos implementados e compatíveis. Os números permanecem fixos: INT8 sempre é 2, mesmo quando FP8 está oculto. Digite `?` no menu de formatos para consultar os motivos das opções ocultas. Na AMD, os kernels são previamente compilados para a arquitetura selecionada e reutilizados durante a execução; NVIDIA consulta a compute capability e Intel consulta extensões, capacidades FMA e tamanhos de subgrupo do driver. A compilação/validação durante a execução ainda pode identificar limitações adicionais do driver. Uma seleção explícita incompatível na linha de comando é rejeitada antes das medições.

## Menu completo

| Número | Nome para linha de comando | Operação |
|---:|---|---|
| 1 | FP8 | Matriz densa FP8 E4M3 |
| 2 | INT8 | Matriz densa INT8 / DOT4 conforme a GPU |
| 3 | INT4 | Matriz densa INT4 / DOT8 conforme a GPU |
| 4 | FP32_VECTOR | Vetor FP32, FMA; dual FMA na RDNA4 |
| 5 | FP16_VECTOR | Vetor FP16 packed, FMA |
| 6 | FP16_MATRIX | Matriz FP16 densa, acumulação FP32 |
| 7 | FP16_SPARSE | Matriz FP16 esparsa 2:4, acumulação FP32 |
| 8 | FP8_E4M3_SPARSE | Matriz FP8 E4M3 esparsa 2:4 |
| 9 | FP8_E5M2_SPARSE | Matriz FP8 E5M2 esparsa 2:4 |
| 10 | INT8_SPARSE | Matriz INT8 esparsa 2:4 |
| 11 | INT4_SPARSE | Matriz INT4 esparsa 2:4 |
| 12 | FP64_VECTOR | FMA FP64 |
| 13 | BF16_VECTOR | AMD DOT2 com acumulação FP32; NVIDIA FMA BF16 packed |
| 14 | INT32_VECTOR | AMD ADD; NVIDIA MAD; Intel MUL+ADD |
| 15 | INT16_VECTOR | AMD adição packed de dois inteiros de 16 bits |
| 16 | INT4_VECTOR | AMD DOT8 de inteiros signed de 4 bits, acumulação INT32 |
| 17 | INT1_MATRIX | NVIDIA matriz binária AND + popcount |
| 18 | INT2_MATRIX | Reservado; nenhum backend nativo implementado |
| 19 | NVFP4_MATRIX | E2M1 com escala E4M3 |
| 20 | MXFP4_MATRIX | E2M1 com escala E8M0 |
| 21 | MXFP6_MATRIX | E3M2 com escala E8M0 |
| 22 | MXFP8_MATRIX | E4M3 com escala E8M0 |
| 23 | BF16_MATRIX | Matriz BF16, acumulação FP32 |
| 24 | TF32_MATRIX | Matriz TF32, acumulação FP32 |
| 25 | FP32_MATRIX | AMD CDNA MFMA FP32 nativo |
| 26 | FP64_MATRIX | AMD CDNA MFMA / NVIDIA MMA FP64 |

`MXPF4` e `MXPF4_MATRIX` são aceitos como aliases de `MXFP4_MATRIX`. Os formatos MX/NV desta lista são densos, não as variantes esparsas.

### Cobertura dos novos caminhos

| Backend | Caminhos implementados e condições |
|---|---|
| AMD | FP64 FMA, INT32 ADD, INT16 packed ADD e INT4 DOT8 sujeitos à compilação e auditoria da arquitetura; BF16 DOT2 em gfx11/12; BF16 WMMA em gfx11/12; BF16 MFMA em gfx90a/940/941/942/950; TF32 via MFMA XF32 em gfx940/941/942; FP32 MFMA em gfx908/90a/940/941/942/950; FP64 MFMA em gfx90a/940/941/942/950. |
| NVIDIA | FP64 e INT32 vetoriais no backend CC6.1+; BF16 vetorial/matricial e TF32 matricial CC8.0+; INT1 AND+POPC em CC8.0/8.6/8.9; FP64 matricial habilitado em SM80/SM90; NVFP4 e MXFP4/6/8 via MMA block-scale em SM120a (CC12.0). |
| Intel OpenCL | INT32 MUL+ADD; FP64 apenas se o driver anunciar `cl_khr_fp64` e FMA FP64; BF16 matricial com extensão XMX e subgrupo mínimo 8/16; TF32 requer também a extensão específica TF32 e subgrupo mínimo 16. |

**Limites da ampliação:** INT2 foi registrado no catálogo, mas permanece oculto porque não há kernel nativo implementado. Os backends não implementam INT16/INT4 vetoriais na NVIDIA, BF16/INT16/INT4 vetoriais na Intel, nem FP32 matricial na NVIDIA/Intel. MX/NV não estão implementados para AMD, Intel ou NVIDIA SM100/103 (que exigem outro caminho, TCGEN05). Portanto, a existência de suporte no hardware não garante cobertura nesta ferramenta. O botão Compatibilidade informa essas restrições; não há substituição silenciosa por uma precisão maior, decomposição em operações menores ou TF32 no lugar de FP32.

As operações têm contagens distintas: ADD conta uma operação por componente; MUL+ADD/FMA contam duas; DOT8 conta 16 operações; INT1 AND+POPC usa a convenção equivalente de duas operações por contribuição binária. INT1 representa entradas 0/1, não inteiros signed ±1. BF16 DOT2 acumula em FP32, enquanto o FMA BF16 da NVIDIA arredonda o acumulador BF16. Compare o caminho e a operação, além do nome do formato. INT16 valida o resultado com wrap de 16 bits. Os testes MX/NV verificam escalas 1 e 2×4 fora da medição; o intervalo medido usa escalas 1. NVFP4 usa escala global do tensor igual a 1. Essas validações dirigidas não cobrem todos os valores especiais ou padrões de escala.

Exemplo com todas as categorias e 50 amostras por configuração:

```powershell
.\run.ps1 -Device amd:0 -Modes ALL -Samples 50 -NonInteractive
```

Os nomes descrevem a categoria de cálculo das especificações do fabricante. O resultado é o throughput **medido por este kernel**, não uma certificação de que ele atingiu o pico teórico. Em particular, mesmo o kernel FP32 com instruções duais não esgota necessariamente o teto de execução vetorial da placa.

## Linha de comando

No PowerShell, dentro desta pasta:

```powershell
# Listar placas e seus identificadores
.\run.ps1 -List

# Escolher GPU e dois formatos, sem perguntas
.\run.ps1 -Device amd:0 -Modes FP8,INT4 -NonInteractive

# Somente INT8 na NVIDIA
.\run.ps1 -Device nvidia:0 -Modes INT8 -NonInteractive

# Intel Arc: todos os formatos disponiveis ou uma selecao
.\run.ps1 -Device intel:0 -Modes ALL -NonInteractive
.\run.ps1 -Device intel:0 -Modes INT8,FP16_MATRIX -NonInteractive

# Forçar DP4A para comparar com outra GPU que usa dot4
.\run.ps1 -Device nvidia:0 -Modes INT8 -Int8Path dp4a -NonInteractive

# Mais amostras
.\run.ps1 -Device amd:0 -Modes INT8 -Samples 51 -NonInteractive
```

Com Python 3.10+ de 64 bits:

```text
python bench.py --interactive
python bench.py --list
python bench.py --device amd:0 --modes FP8 INT4
python bench.py --device nvidia:0 --modes INT8 --int8-path dp4a
```

`bench.py` e `gpu_bench.py` abrem a mesma interface. Sem `--modes`, a execução não interativa testa todas as categorias disponíveis para a GPU. Quando há várias GPUs é obrigatório selecionar uma ou usar o menu. O lançador procura primeiro o Python do Codex e depois Python no PATH.

## Caminhos implementados

| GPU / arquitetura | INT8 | INT4 | FP8 E4M3 |
|---|---|---|---|
| AMD RDNA 4, gfx12 | WMMA → INT32 | WMMA 16×16×32 → INT32 | WMMA → FP32 |
| AMD RDNA 3, gfx11 | WMMA → INT32 | WMMA → INT32 | Não implementado |
| Outras AMD expostas pelo HIP | DOT4, se o compilador confirmar suporte | DOT8, se o compilador confirmar suporte | Não implementado |
| NVIDIA CC 6.1–7.4 | DP4A → INT32 | Não suportado neste caminho | Não suportado |
| NVIDIA CC 7.5–8.8 | MMA Tensor → INT32 ou DP4A | MMA Tensor → INT32 | Não suportado |
| NVIDIA CC 8.9+ | MMA Tensor → INT32 ou DP4A | MMA Tensor → INT32 | MMA Tensor → FP32 |
| Intel Arc A/B com extensão de matriz | XMX → INT32 | XMX → INT32 | Não implementado |

`--int8-path auto` escolhe Tensor MMA na NVIDIA a partir de CC 7.5, senão DP4A. `dp4a` força DP4A; `tensor` exige Tensor MMA. Esta opção é específica do backend NVIDIA. Na AMD o caminho é definido pela arquitetura.

Novas categorias: AMD usa FMA vetorial, WMMA FP16 em gfx11/gfx12 e SWMMAC esparso em gfx12. NVIDIA usa FMA vetorial, MMA FP16 em CC 7.5+, MMA esparso em CC 8.0+ e MMA FP8 esparso em CC 8.9+. Cada kernel continua sujeito ao suporte do compilador/driver; não há fallback de precisão.

### Intel Arc A e B

Backend OpenCL pelo driver Intel, sem exigir instalação do SDK oneAPI ou bibliotecas Python adicionais. Disponibiliza FP32 vetorial, FP16 vetorial, FP16 matricial, INT8 matricial e INT4 matricial conforme as capacidades anunciadas. FP16/INT8/INT4 matriciais usam diretamente `cl_intel_subgroup_matrix_multiply_accumulate` (XMX), sem expandir INT4 para INT8. FP16 matricial acumula em FP32; inteiros acumulam em INT32.

A implementação consulta `CL_DEVICE_SUB_GROUP_SIZES_INTEL` e exige o menor subgrupo anunciado: 8 ou 16. Usa o operando A `int8` para subgrupo 8 e `short8` para 16, B `int8` e acumuladores de oito componentes. Esses nomes são tipos de vetor OpenCL: `int8` significa oito inteiros de 32 bits, cujos bits contêm os valores compactados. As dimensões são M=8, N=subgrupo e K=16/32/64 para FP16/INT8/INT4. A contagem é `2*M*N*K/subgrupo` por work-item.

FP8 e todos os modos esparsos ficam ocultos na Intel: este backend não possui um caminho nativo para eles. Isso descreve a cobertura do programa, não uma inferência de suporte de hardware baseada no nome da GPU. Outros dispositivos Intel expostos pelo OpenCL também podem aparecer, com os testes filtrados pelas capacidades reais do driver.

Tempos Intel usam os timestamps START/END de eventos OpenCL com profiling habilitado. Compilação e cópias de validação ficam fora do intervalo. São verificados sinais positivos/negativos, B não uniforme e todas as saídas após cada configuração. A ISA Intel não é inspecionada e não se garante atingir o pico XMX.

O filtro pelo nome RX 9070 XT foi removido: qualquer GPU AMD detectada pelo HIP pode ser selecionada. **Isso não significa que toda placa AMD suporte os três formatos**, nem que todos os aceleradores de todas as famílias estejam implementados. Por exemplo, este programa ainda não implementa os kernels MFMA FP8 de GPUs AMD Instinct. Em placas antigas sem instruções DOT, o compilador rejeita o teste correspondente. Nenhum desses casos é convertido silenciosamente em FP16/FP32 ou CPU.

Status `unsupported` significa ausência de caminho implementado/compatível. `failed` identifica falha de compilação, driver, validação ou medição; o diagnóstico é salvo. Outros formatos selecionados continuam sendo avaliados. Nenhuma taxa é inventada para um formato que falhou.

## Dependências

- Intel: driver de gráficos com runtime OpenCL de 64 bits e `OpenCL.dll`. Os modos XMX exigem as extensões de matriz/subgrupo citadas acima. Não requer SDK oneAPI.

- Windows de 64 bits e Python 3.10+ de 64 bits, usando apenas a biblioteca padrão.
- AMD: driver/runtime HIP (`amdhip64_7.dll` ou `_6.dll`) e COMGR (`amd_comgr_3.dll` ou `_2.dll`) em System32. O driver precisa expor a GPU ao HIP; instalar apenas um driver gráfico não garante suporte computacional para toda placa.
- NVIDIA: driver CUDA com `nvcuda.dll`. PTX é compilado pelo JIT do próprio driver; **não precisa de CuPy, PyTorch, NVCC ou CUDA Toolkit separado**. O driver precisa entender o PTX usado: 6.0 para DP4A, 6.5 para INT8/INT4 MMA e 8.4 para FP8 MMA. Driver antigo pode precisar ser atualizado, mesmo em placa compatível.
- Não baixa nem instala dependências ao executar. HIP/COMGR e CUDA só são usados no backend correspondente.

## Metodologia

**Correção INT4 RDNA 4:** o padrão agora usa `v_wmma_i32_16x16x32_iu4`, com K=32, dois registradores de entrada e 16.384 operações por wave (512 por thread). A versão anterior usava K=16, válido mas insuficiente para explorar a taxa INT4 máxima. Não basta dobrar o número publicado: a instrução, o tamanho dos operandos e a validação também mudaram. Há um teste dirigido com valores diferentes nas duas metades de K, positivos e negativos, para conferir que ambas participam da conta. RDNA 3 permanece em K=16.

Para comparar com o kernel antigo: `python bench.py --device amd:0 --modes INT4 --amd-int4-k 16`. O padrão é `--amd-int4-k 32`. No PowerShell, use `-AmdInt4K 16` ou `32`. Não se trata de esparsidade. Resultados antigos medem apenas o kernel K=16 e não devem ser tratados como limite INT4 da RX 9070 XT.

São kernels que reutilizam operandos em registradores. Matrizes usam quatro cadeias; FP16 vetorial AMD usa oito. Os kernels vetoriais contêm 16 repetições de FMA por cadeia e por iteração; essa quantidade está incluída em `ops_per_thread`. INT8 e INT4 são assinados × assinados com acumulação INT32; matrizes FP16/FP8 acumulam em FP32. FP16 vetorial calcula em FP16, convertendo apenas a saída para validação em FP32, fora do laço. Não mede GEMM completo, tokens/s, VRAM/RAM ou o máximo absoluto garantido da GPU.

Multiplicação e soma contam separadamente:

```text
operacoes = blocos × 128 threads × iteracoes × cadeias × operacoes_por_thread
taxa de cada amostra = operacoes / (tempo_ms × 10^9)
resultado = minimo, maximo, media e mediana das taxas individuais
```

DOT4/DP4A conta 8 operações por thread por instrução; DOT8 conta 16. WMMA/MMA divide a contagem de operações da matriz pela wave/warp de 32 threads, evitando multiplicar a operação matricial inteira por cada thread. O JSON registra a contagem e o caminho escolhidos.

### Esparsidade estruturada real

A matriz A é comprimida com duas posições não nulas por grupo de quatro; B é densa. O kernel executa SWMMAC (AMD) ou `mma.sp` (NVIDIA), recebendo os metadados de seleção. A taxa principal usa **2×M×N×K da matriz expandida**, a convenção de throughput equivalente usada para os picos esparsos. O JSON e CSV também incluem a taxa das operações com valores não nulos, que é a metade. Não se mede um kernel denso para depois simplesmente dobrar seu resultado.

Na RDNA4, FP16/FP8/INT8 usam SWMMAC 16×16×32 e INT4 usa 16×16×64. Os testes validam metadados `0x44444444` (posições 0/1) e `0xeeeeeeee` (2/3) com B variando 1/2/3/4. Assim, alterar os metadados muda o resultado esperado. A também é testada com sinais positivos/negativos; todas as saídas são verificadas.

### Vetores

#### FP32 na RDNA4: seleção de variantes

O padrão `--fp32-kernel auto` compara o kernel original (`reference`) com seis variantes: 8/16/32 cadeias de acumuladores, cada uma escalar (`w1`) ou com duas componentes (`w2`). As novas variantes compartilham dois operandos FP32 carregados antes do loop e executam 16 FMA por cadeia e iteração, sem alternância de sinais no loop. Todas as operações permanecem em FP32, sem fast-math. Cada FMA escalar conta duas operações; cada FMA de duas componentes conta quatro. `c32w2`, por exemplo, possui 64 acumuladores escalares independentes.

A seleção usa 5 amostras em cada uma das três configurações de blocos, por variante. Depois escolhe variante/blocos pela melhor mediana e executa **novas amostras** na configuração fixa: a quantidade solicitada pelo usuário. Somente essa rodada final aparece na tabela principal. O custo da seleção não integra o tempo das amostras. Fontes, assembly e resultados da seleção ficam em `FP32_tuning/`; o JSON principal também registra as variantes avaliadas, a escolha e métricas do assembly. Falhas de candidatos são registradas e excluídas da seleção.

As entradas normais são A=B=1. O maior acumulador medido é `16*65536+31`, exatamente representável em FP32. Antes da medição também são testados A negativo e B=2. Cada saída é verificada, inclusive após cada configuração medida. A auditoria confere o número de FMA escalares, contando as duas partes de instruções duais, e rejeita novas variantes com armazenamento privado indicado no assembly. Não remove esperas de dependência inseridas pelo compilador.

Na GUI, **FP32 RDNA4** oferece `auto` e `reference`. Na linha de comando também é possível escolher uma variante:

```powershell
.\run.ps1 -Device amd:0 -Modes FP32_VECTOR -Samples 50 -Fp32Kernel auto -NonInteractive
.\run.ps1 -Device amd:0 -Modes FP32_VECTOR -Samples 50 -Fp32Kernel reference -NonInteractive
.\run.ps1 -Device amd:0 -Modes FP32_VECTOR -Samples 50 -Fp32Kernel c16w2 -NonInteractive
```

A seleção automática adicional está habilitada apenas para AMD gfx12. Outras arquiteturas e fornecedores mantêm seus kernels existentes. O cancelamento da GUI continua aguardando o formato atual, incluindo sua seleção de variantes. Alcançar um throughput alto não certifica um pico universal: o clock efetivo não é coletado e a utilização depende do kernel.

O kernel FP32 de referência e os demais kernels vetoriais alternam FMA com operandos de sinais opostos para manter os acumuladores limitados, evitando saturação/perda de incremento em FP16. Um teste assimétrico altera o segundo multiplicador e verifica a diferença acumulada. Na RDNA4, FP32 de referência usa pares `v_dual_fmac_f32` e FP16 usa `v_pk_fma_f16`. Cada par FP32 conta quatro operações por thread; cada FMA de duas componentes FP16 também conta quatro. O programa confere o número de instruções esperado no assembly. A emissão dual, isoladamente, não garante atingir o pico teórico.

- Verifica todas as saídas com entradas +1 e -1 antes de medir; verifica todas as saídas novamente após cada configuração. É uma validação dirigida do kernel, não um teste exaustivo dos formatos.
- AMD: confere a contagem exata de instruções nativas esperada para cada kernel no assembly. NVIDIA: emite PTX explícito; o JIT deve aceitá-lo e a validação numérica deve passar. O executável não inspeciona SASS.
- Aquece cada configuração por pelo menos 0,3 segundo e mede 21 amostras por padrão com eventos de GPU. Compilação, preparação e cópias de validação ficam fora do intervalo. Cargas/stores do kernel ficam dentro.
- Busca amostras de aproximadamente 8 ms, limitando iterações. Para a RX 9070 e Intel usa 256/512/1.024 blocos; os demais caminhos usam 16/64/256. Um kernel medido acima de 250 ms interrompe aquele formato após terminar a chamada.
- Escolhe a configuração pela melhor mediana. Console, CSV e resumo apresentam mínimo, máximo, média aritmética e mediana das taxas dessa mesma configuração; o JSON guarda estatísticas de tempo, de throughput e todas as amostras de todas as configurações. Máximo de throughput corresponde ao menor tempo. A média das taxas difere de operações divididas pelo tempo médio. Com quantidade par de amostras, a mediana das taxas também pode diferir ligeiramente de operações divididas pela mediana dos tempos (fórmula das versões anteriores). Resultados de DP4A, DOT8 e Tensor/WMMA devem ser identificados pelo caminho; não representam a mesma utilização de hardware.
- O máximo observado é a melhor amostra deste kernel, não o pico teórico do fabricante. Um kernel limitado por dependências ou escalonamento pode ficar abaixo da referência em todas as amostras; trocar mediana por máximo não elimina essa limitação. Extremos também são mais sensíveis a variações e ruído de medição.
- Não altera clocks, tensão, plano de energia ou limites do driver. Não mede temperatura, consumo ou clocks.

Cada execução cria uma pasta `results/gpu-...` com `RESUMO.md`, `results.json`, `results.csv` e os kernels `.hip`/`.s`, `.ptx` ou `.cl` (Intel). Execuções antigas são preservadas.

## Validação desta versão

Ampliação de formatos e GUI: 19 testes de host passaram. Os novos kernels AMD passaram em 31 casos de compilação/auditoria para gfx1201, gfx1036, gfx1100, gfx90a e gfx942, conforme os caminhos disponíveis. Os 36 casos novos NVIDIA foram aceitos pelo `ptxas` 12.9, incluindo os quatro formatos block-scale para SM120a. Isso verifica compilação, não execução NVIDIA. Na RX 9070 XT foram executados e validados os 17 testes disponíveis, com 5 amostras por configuração (FP32 em modo reference nesta regressão), incluindo FP64, BF16, INT32, INT16 e INT4 vetoriais e BF16 matricial. Relatório: `results/gpu-20260912-141645-600478/RESUMO.md`. FP64 também executou na iGPU. Os novos kernels OpenCL FP64 e INT32 executaram e validaram na AMD como verificação de transporte/dados, sem validar XMX ou desempenho Intel. Não há NVIDIA, Intel Arc ou AMD CDNA neste computador.

A GUI foi testada com execuções reais nas duas GPUs: seleção preservada ao alternar lista única/abas Vetor e Matriz, resultados separados mesmo ao trocar a aba durante a medição, pasta/cabeçalho por GPU e área de resultados visível no tamanho mínimo da janela.

Otimização FP32: 16 testes de host passaram. Na RX 9070 XT, a referência e as seis variantes passaram nas auditorias/validações. A seleção automática foi seguida por 50 amostras independentes da vencedora e comparada com uma nova execução de 50 amostras por configuração da referência. Medianas observadas: 51,627 TFLOPS (c32w2) e 26,587 TFLOPS (referência). O seletor FP32 da GUI e seus limites de layout também foram verificados.

Interface gráfica: testadas detecção de GPUs, troca entre RX 9070 XT/iGPU e filtragem dos testes, execução real de dois formatos na iGPU, atualização da tabela, cancelamento antes da primeira medição e cancelamento durante a execução preservando dois resultados concluídos. Verificados os limites de tamanho dos controles da janela. Os 15 testes de regressão do backend também passaram.

Extensão Intel/menu: 15 testes de host passaram, incluindo layouts XMX para subgrupos 8/16, contagem de operações, capacidades ausentes e seleção filtrada. O transporte OpenCL, compilação dos kernels vetoriais, eventos e validação numérica foram executados na RX 9070 XT como teste do backend. **Não há uma Intel Arc neste computador: os kernels XMX ainda não foram compilados/executados por um driver Intel.** O suporte Intel é implementado, mas precisa de validação em Arc A/B real. A execução OpenCL na AMD não valida XMX nem desempenho Intel.

Executado e validado numericamente na RX 9070 XT (FP8/INT8/INT4) e na iGPU gfx1036 (INT8 DOT4 e INT4 DOT8). Testado menu com um formato, execução com dois formatos, parser e rejeição de formatos inválidos.

Kernels AMD INT8/INT4 também compilados para gfx1100, gfx1103, gfx1200 e gfx1030. Ausência de DOT em gfx900 confirmada pelo compilador.

Os 20 casos NVIDIA gerados foram aceitos pelo montador oficial `ptxas` 12.9 para sm_61, sm_75, sm_80, sm_89, sm_90 e sm_120, conforme os formatos disponíveis. **Não houve execução em hardware NVIDIA neste computador.** O caminho JIT, a validação numérica e os tempos NVIDIA ainda precisam ser confirmados em uma placa real. A validação de compilação não certifica desempenho ou cobertura de todas as placas.

Na ampliação do menu, os 11 formatos foram executados na RX 9070 XT com 50 amostras por configuração, passando na validação numérica. Outros 33 casos de compilação NVIDIA para os novos formatos foram aceitos pelo `ptxas` 12.9, cobrindo as arquiteturas acima conforme o suporte. O backend usa `mma.sp` convencional; não implementa variantes `ordered_metadata`/WMMA mais recentes para buscar o máximo de cada geração NVIDIA.

Testes sem GPU: `python -m unittest test_selection.py`.

## Referências

- [AMD WMMA RDNA 4](https://gpuopen.com/learn/wmma-guide-amd-rdna-4-gpus-part-2/)
- [LLVM AMDGPU intrinsics](https://github.com/llvm/llvm-project/blob/main/clang/include/clang/Basic/BuiltinsAMDGPU.td)
- [NVIDIA PTX ISA — DP4A e MMA](https://docs.nvidia.com/cuda/parallel-thread-execution/)
- [Intel OpenCL: matrizes XMX e assinaturas por subgrupo](https://registry.khronos.org/OpenCL/extensions/intel/cl_intel_subgroup_matrix_multiply_accumulate.html)
- [Intel OpenCL: consulta e tamanho obrigatório de subgrupo](https://registry.khronos.org/OpenCL/extensions/intel/cl_intel_required_subgroup_size.html)

O benchmark AVX-512 da CPU permanece em sua pasta separada.
