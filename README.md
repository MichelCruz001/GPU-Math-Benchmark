# GPU Math Benchmark — AMD, NVIDIA and Intel Arc

The application uses a custom chip icon, included in `assets/chip.ico` (16–256 px) and `assets/chip.png`. Machine-specific Windows shortcuts are not included in the repository; launch with **INICIAR.cmd** or create your own shortcut.

Requires Windows x64, 64-bit Python 3.10+ with Tkinter, and a compatible GPU driver. No pip packages or vendor SDKs are required to run the benchmark. The launcher uses Python from PATH when the optional bundled runtime is not present. Run host tests with `python -m unittest test_selection.py test_i18n.py`.

Hardware support depends on the implemented instruction path and driver. NVIDIA kernels were checked with the PTX assembler but have not been run on NVIDIA hardware here; Intel XMX and AMD CDNA also require hardware validation. INT2 is reserved but not implemented. This measures kernel throughput, not guaranteed vendor peak performance. Consult the coverage table in **LEIA-ME.md** before comparing results.

Open **INICIAR.cmd** to launch the desktop interface. English is the default language. Use **Language** at the top right to select **English**, **Português (Brasil)** or **简体中文**.

Select a GPU, choose the supported tests, set 5–200 samples per configuration and click **Start benchmark**. Use **Separate Vector and Matrix** to group tests. Each GPU has a separate results tab. Language changes take effect immediately, including during a run, without clearing selections or results. A new session starts in English.

The GUI labels, test descriptions, status messages and dialogs are translated. Technical logs, detailed driver/compatibility diagnostics, the command-line interface and saved reports retain their original language. Format IDs and units remain unchanged. **Open results** opens the folder associated with the selected GPU results tab.

See [LEIA-ME.md](LEIA-ME.md) for the full format coverage, hardware limitations and benchmark methodology (Portuguese).

## 简体中文

打开 **INICIAR.cmd** 启动图形界面。默认语言为英语。在右上角的 **Language** 下拉菜单中选择 **简体中文**。

选择 GPU 和兼容的测试项目，将每种配置的采样次数设为 5–200，然后点击 **开始测试**。可使用 **按向量和矩阵分类** 整理测试列表。每个 GPU 都有独立的结果选项卡。运行过程中也可切换语言，已有选择和结果不会丢失。重新打开程序时恢复英语。

界面文字、测试说明、状态消息和对话框已翻译。技术日志、驱动及兼容性详情、命令行界面和保存的报告保留原始语言。格式标识符和单位保持不变。**打开结果** 会打开当前 GPU 结果选项卡对应的文件夹。
