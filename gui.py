"""Simple Tk desktop front end; benchmark runs in a separate process."""
import json
import os
from pathlib import Path
import queue
import subprocess
import sys
import uuid
import threading
import tkinter as tk
from tkinter import ttk, messagebox
from tkinter.scrolledtext import ScrolledText
from gpu_bench import inventory, available_modes, describe, unit_for
from more_formats import category
from i18n import LANGUAGES, DEFAULT_LANGUAGE, translate

ROOT=Path(__file__).resolve().parent

class LocalizedVar(tk.StringVar):
    def __init__(self,owner,value):
        self.owner=owner
        super().__init__(master=owner.root,value=owner.t(value))
    def set(self,value): super().set(self.owner.t(value))


class BenchmarkGUI:
    def __init__(self,root):
        self.root=root; self.events=queue.Queue(); self.devices=[]; self.variables={}
        self.busy=False; self.scanning=False; self.generation=0; self.folder=None
        self.cancel_path=None; self.closing=False; self.selected_count=0
        self.result_views={}; self.run_gpu_id=None; self.checkbuttons=[]
        self.language=DEFAULT_LANGUAGE
        root.title('Benchmark GPU'); root.geometry('1100x900'); root.minsize(950,820)
        icon=ROOT/'assets'/'chip.ico'
        if os.name=='nt' and icon.is_file():
            root.iconbitmap(str(icon))
            root.iconbitmap(default=str(icon))
        style=ttk.Style(); style.theme_use('clam')
        style.configure('.',font=('Segoe UI',10))
        style.configure('Title.TLabel',font=('Segoe UI',21,'bold'))
        style.configure('Treeview',rowheight=29)
        main=ttk.Frame(root,padding=20); main.pack(fill='both',expand=True)
        language_bar=ttk.Frame(main); language_bar.pack(fill='x')
        self.language_choice=tk.StringVar(value='English')
        self.language_box=ttk.Combobox(language_bar,textvariable=self.language_choice,values=list(LANGUAGES),state='readonly',width=20)
        self.language_box.pack(side='right'); self.language_box.bind('<<ComboboxSelected>>',lambda e:self.change_language())
        ttk.Label(language_bar,text='Idioma').pack(side='right',padx=8)
        ttk.Label(main,text='Benchmark GPU',style='Title.TLabel').pack(anchor='w')
        ttk.Label(main,text='AMD · NVIDIA · Intel Arc   |   Medições nativas de cálculo').pack(anchor='w',pady=(0,16))
        top=ttk.Frame(main); top.pack(fill='x')
        ttk.Label(top,text='Placa de vídeo').grid(row=0,column=0,sticky='w')
        ttk.Label(top,text='Amostras por configuração').grid(row=0,column=1,sticky='w',padx=12)
        self.device=ttk.Combobox(top,state='readonly'); self.device.grid(row=1,column=0,sticky='ew',pady=5)
        self.device.bind('<<ComboboxSelected>>',lambda e:self.check_modes())
        self.samples=tk.StringVar(value='50')
        self.sample_box=ttk.Spinbox(top,from_=5,to=200,textvariable=self.samples,width=10)
        self.sample_box.grid(row=1,column=1,sticky='w',padx=12)
        self.refresh=ttk.Button(top,text='Atualizar GPUs',command=self.discover); self.refresh.grid(row=1,column=2)
        top.columnconfigure(0,weight=1)
        options=ttk.Frame(main); options.pack(fill='x',pady=(5,10))
        ttk.Label(options,text='INT8 NVIDIA').pack(side='left')
        self.int8=tk.StringVar(value='auto'); self.int8_box=ttk.Combobox(options,textvariable=self.int8,values=['auto','dp4a','tensor'],width=9,state='disabled')
        self.int8_box.pack(side='left',padx=(8,22)); self.int8_box.bind('<<ComboboxSelected>>',lambda e:self.check_modes())
        ttk.Label(options,text='INT4 AMD RDNA4 · K').pack(side='left')
        self.int4=tk.StringVar(value='32'); self.int4_box=ttk.Combobox(options,textvariable=self.int4,values=['32','16'],width=5,state='disabled')
        self.int4_box.pack(side='left',padx=8); self.int4_box.bind('<<ComboboxSelected>>',lambda e:self.check_modes())
        ttk.Label(options,text='FP32 RDNA4').pack(side='left',padx=(12,5))
        self.fp32=tk.StringVar(value='auto')
        self.fp32_box=ttk.Combobox(options,textvariable=self.fp32,values=['auto','reference'],width=10,state='disabled')
        self.fp32_box.pack(side='left')
        self.tests=ttk.LabelFrame(main,text='Testes disponíveis',padding=10); self.tests.pack(fill='x')
        self.separate=tk.BooleanVar(value=True)
        ttk.Checkbutton(self.tests,text='Separar em Vetor e Matriz',variable=self.separate,command=self.render_tests).pack(anchor='w',pady=(0,5))
        self.checks=ttk.Frame(self.tests); self.checks.pack(fill='x')
        selection=ttk.Frame(self.tests); selection.pack(fill='x',pady=(8,0))
        self.all_button=ttk.Button(selection,text='Selecionar todos',command=lambda:self.select_all(True)); self.all_button.pack(side='left')
        self.none_button=ttk.Button(selection,text='Limpar seleção',command=lambda:self.select_all(False)); self.none_button.pack(side='left',padx=8)
        self.reasons=ttk.Button(selection,text='Compatibilidade',command=self.show_reasons); self.reasons.pack(side='right')
        self.hidden={}
        actions=ttk.Frame(main); actions.pack(fill='x',pady=12)
        self.start_button=ttk.Button(actions,text='Iniciar benchmark',command=self.start,state='disabled'); self.start_button.pack(side='left')
        self.cancel_button=ttk.Button(actions,text='Cancelar',command=self.cancel,state='disabled'); self.cancel_button.pack(side='left',padx=8)
        self.open_button=ttk.Button(actions,text='Abrir resultados',command=self.open_results,state='disabled'); self.open_button.pack(side='right')
        self.status=LocalizedVar(self,'Detectando GPUs…'); ttk.Label(main,textvariable=self.status).pack(anchor='w')
        self.progress=ttk.Progressbar(main,mode='determinate'); self.progress.pack(fill='x',pady=(6,12))
        self.result_caption=LocalizedVar(self,'Resultados da execução')
        ttk.Label(main,textvariable=self.result_caption).pack(anchor='w',pady=(0,5))
        tabs=ttk.Notebook(main); tabs.pack(fill='both',expand=True)
        result_frame=ttk.Frame(tabs); tabs.add(result_frame,text='Resultados')
        self.result_tabs=ttk.Notebook(result_frame); self.result_tabs.pack(fill='both',expand=True)
        self.result_tabs.bind('<<NotebookTabChanged>>',lambda e:self.result_tab_changed())
        self.table=None
        self.log=ScrolledText(tabs,wrap='word',font=('Consolas',9),state='disabled'); tabs.add(self.log,text='Registro da execução')
        ttk.Label(main,text='Estatísticas da configuração com melhor mediana. Esparsos: taxa equivalente da matriz expandida.',wraplength=930).pack(anchor='w',pady=(10,0))
        root.protocol('WM_DELETE_WINDOW',self.close)
        self.change_language()
        root.after(100,self.poll); self.discover()

    def t(self,text): return translate(text,self.language)

    def change_language(self):
        self.language=LANGUAGES[self.language_choice.get()]
        self.root.title(self.t('Benchmark GPU'))
        style=ttk.Style()
        family='Microsoft YaHei UI' if self.language=='zh-CN' else 'Segoe UI'
        style.configure('.',font=(family,10)); style.configure('Title.TLabel',font=(family,21,'bold'))
        def visit(widget):
            options=widget.keys()
            if 'text' in options and ('textvariable' not in options or not widget.cget('textvariable')):
                widget.configure(text=self.t(widget.cget('text')))
            if isinstance(widget,ttk.Notebook):
                for tab in widget.tabs(): widget.tab(tab,text=self.t(widget.tab(tab,'text')))
            if isinstance(widget,ttk.Treeview):
                for column in widget['columns']: widget.heading(column,text=self.t(widget.heading(column,'text')))
                for row in widget.get_children():
                    values=list(widget.item(row,'values'))
                    if len(values)>1: values[1]=self.t(values[1]); widget.item(row,values=values)
            for child in widget.winfo_children(): visit(child)
        visit(self.root)
        self.status.set(self.status.get()); self.result_caption.set(self.result_caption.get())

    def append_log(self,text):
        self.log.configure(state='normal'); self.log.insert('end',text+'\n'); self.log.see('end'); self.log.configure(state='disabled')

    def ensure_gpu_tab(self,info):
        key=info['id']+'|'+info['name']
        if key not in self.result_views:
            frame=ttk.Frame(self.result_tabs)
            self.result_tabs.add(frame,text=info['name']+' ('+info['id']+')')
            columns=('mode','unit','min','max','mean','median')
            table=ttk.Treeview(frame,columns=columns,show='headings',height=6)
            for col,label in zip(columns,('Teste','Unidade','Mínimo','Máximo','Média','Mediana')):
                table.heading(col,text=self.t(label)); table.column(col,width=245 if col=='mode' else 100,anchor='w' if col=='mode' else 'e')
            scroll=ttk.Scrollbar(frame,orient='vertical',command=table.yview); table.configure(yscrollcommand=scroll.set)
            scroll.pack(side='right',fill='y'); table.pack(fill='both',expand=True)
            self.result_views[key]={'frame':frame,'table':table,'folder':None,'caption':info['name']+' · ainda sem resultados'}
            if self.table is None: self.table=table
        return key

    def result_tab_changed(self):
        selected=self.result_tabs.select()
        view=next((v for v in self.result_views.values() if str(v['frame'])==selected),None)
        if view:
            self.result_caption.set(view['caption'])
            self.open_button.configure(state='normal' if view['folder'] else 'disabled')

    def render_tests(self):
        for widget in self.checks.winfo_children(): widget.destroy()
        self.checkbuttons=[]
        if not self.variables: return
        info=self.devices[self.device.current()].copy(); info['int8_path']=self.int8.get()
        groups=('Vetor','Matriz') if self.separate.get() else ('Todos',)
        notebook=ttk.Notebook(self.checks); notebook.pack(fill='x')
        for group in groups:
            modes=[m for m in self.variables if group=='Todos' or category(m,info)==group]
            frame=ttk.Frame(notebook); notebook.add(frame,text=self.t(f'{group} ({len(modes)})'))
            canvas=tk.Canvas(frame,height=120,highlightthickness=0)
            scroll=ttk.Scrollbar(frame,orient='vertical',command=canvas.yview); canvas.configure(yscrollcommand=scroll.set)
            scroll.pack(side='right',fill='y'); canvas.pack(side='left',fill='x',expand=True)
            contents=ttk.Frame(canvas); window=canvas.create_window((0,0),window=contents,anchor='nw')
            contents.bind('<Configure>',lambda e,c=canvas:c.configure(scrollregion=c.bbox('all')))
            canvas.bind('<Configure>',lambda e,c=canvas,w=window:c.itemconfigure(w,width=e.width))
            if not modes: ttk.Label(contents,text=self.t('Nenhum teste desta categoria disponível.')).pack(anchor='w',padx=8,pady=8)
            for index,mode in enumerate(modes):
                button=ttk.Checkbutton(contents,text=self.t(describe(mode)),variable=self.variables[mode],command=self.controls)
                button.grid(row=index//2,column=index%2,sticky='w',padx=8,pady=4)
                button.bind('<MouseWheel>',lambda e,c=canvas:c.yview_scroll(-int(e.delta/120),'units'))
                self.checkbuttons.append(button)
        self.controls()

    def controls(self):
        locked=self.busy or self.scanning
        self.device.configure(state='disabled' if locked else 'readonly')
        self.refresh.configure(state='disabled' if locked else 'normal')
        self.sample_box.configure(state='disabled' if locked else 'normal')
        info=self.devices[self.device.current()] if self.devices and self.device.current()>=0 else {}
        self.int8_box.configure(state='readonly' if not locked and info.get('vendor')=='NVIDIA' else 'disabled')
        self.int4_box.configure(state='readonly' if not locked and info.get('arch','').startswith('gfx12') else 'disabled')
        self.fp32_box.configure(state='readonly' if not locked and info.get('arch','').startswith('gfx12') else 'disabled')
        for widget in self.checkbuttons: widget.configure(state='disabled' if locked else 'normal')
        for widget in (self.all_button,self.none_button): widget.configure(state='disabled' if locked else 'normal')
        self.start_button.configure(state='normal' if not locked and any(v.get() for v in self.variables.values()) else 'disabled')

    def discover(self):
        self.scanning=True; self.controls(); self.status.set('Detectando GPUs…')
        def work():
            try: self.events.put(('inventory',inventory()))
            except Exception as exc: self.events.put(('error',str(exc)))
        threading.Thread(target=work,daemon=True).start()

    def check_modes(self):
        if self.device.current()<0: return
        self.scanning=True; self.generation+=1; generation=self.generation
        self.controls(); self.status.set('Verificando os testes compatíveis…')
        info=self.devices[self.device.current()]; int8=self.int8.get(); int4=int(self.int4.get())
        def work():
            try: self.events.put(('modes',(generation,available_modes(info,int8,int4,probe_amd=True))))
            except Exception as exc: self.events.put(('error',str(exc)))
        threading.Thread(target=work,daemon=True).start()

    def select_all(self,value):
        for var in self.variables.values(): var.set(value)
        self.controls()

    def show_reasons(self):
        text=self.t('Todos os testes estão disponíveis.') if not self.hidden else '\n\n'.join(self.t(f'Requisitos não atendidos para {m}. Detalhes técnicos originais:')+'\n'+r for m,r in self.hidden.items())
        messagebox.showinfo(self.t('Compatibilidade'),text,parent=self.root)

    def start(self):
        if self.busy or self.scanning: return
        try:
            samples=int(self.samples.get())
            if not 5<=samples<=200: raise ValueError()
        except ValueError:
            messagebox.showerror(self.t('Amostras'),self.t('Informe um número inteiro entre 5 e 200.'),parent=self.root); return
        modes=[m for m,v in self.variables.items() if v.get()]
        if not modes: return
        info=self.devices[self.device.current()]
        self.run_gpu_id=self.ensure_gpu_tab(info)
        view=self.result_views[self.run_gpu_id]; self.table=view['table']; self.result_tabs.select(view['frame'])
        try:
            (ROOT/'results').mkdir(exist_ok=True)
        except OSError as exc:
            messagebox.showerror(self.t('Resultados'),str(exc),parent=self.root); return
        self.cancel_path=ROOT/'results'/('.cancel-'+uuid.uuid4().hex)
        executable=Path(sys.executable)
        if executable.name.lower()=='pythonw.exe': executable=executable.with_name('python.exe')
        args=[str(executable),'-u',str(ROOT/'gpu_bench.py'),'--device',info['id'],'--samples',str(samples),
              '--int8-path',self.int8.get(),'--amd-int4-k',self.int4.get(),'--fp32-kernel',self.fp32.get(),'--cancel-file',str(self.cancel_path),'--modes',*modes]
        self.busy=True; self.folder=None; self.selected_count=len(modes); self.controls()
        self.open_button.configure(state='disabled'); self.cancel_button.configure(state='normal')
        self.table.delete(*self.table.get_children()); self.progress.configure(value=0,maximum=len(modes))
        self.append_log('\n=== Nova execução: '+info['name']+' ===')
        self.result_caption.set('Resultados · '+info['name']+' · '+str(samples)+' amostras por configuração')
        view['caption']=self.result_caption.get(); view['folder']=None
        self.status.set('Iniciando benchmark…')
        def work():
            try:
                env=os.environ.copy(); env['PYTHONIOENCODING']='utf-8'
                with subprocess.Popen(args,cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                                      text=True,encoding='utf-8',errors='replace',env=env,
                                      creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0)) as process:
                    for line in process.stdout: self.events.put(('line',line.rstrip()))
                    self.events.put(('done',process.wait()))
            except Exception as exc:
                self.events.put(('line','Falha ao iniciar: '+str(exc))); self.events.put(('done',1))
        threading.Thread(target=work,daemon=True).start()

    def read_results(self):
        if not self.folder: return
        path=self.folder/'results.json'
        if not path.exists(): return
        try: report=json.loads(path.read_text(encoding='utf-8'))
        except (OSError,ValueError) as exc: self.append_log('Não foi possível ler resultados: '+str(exc)); return
        view=self.result_views[self.run_gpu_id]; table=view['table']
        table.delete(*table.get_children())
        for result in report['results']:
            if result['status']=='ok':
                row=result['best_configuration']; values=[result['mode'],unit_for(result['mode']),*[f'{row[k+"_tops"]:.3f}' for k in ('min','max','mean','median')]]
            else: values=[result['mode'],self.t(result['status']),'—','—','—','—']
            table.insert('', 'end',values=values)
        count=len(report['results']); self.progress.configure(value=count)
        if not self.cancel_path or not self.cancel_path.exists(): self.status.set(f'{count} de {self.selected_count} testes concluídos')

    def cancel(self):
        if self.busy and self.cancel_path:
            try: self.cancel_path.touch()
            except OSError as exc:
                self.closing=False; messagebox.showerror(self.t('Cancelamento'),str(exc),parent=self.root); return
            self.cancel_button.configure(state='disabled')
            self.status.set('Cancelando após o teste atual. Os resultados concluídos serão preservados.')

    def open_results(self):
        selected=self.result_tabs.select()
        view=next((v for v in self.result_views.values() if str(v['frame'])==selected),None)
        if view and view['folder']:
            try: os.startfile(str(view['folder']))
            except OSError as exc: messagebox.showerror(self.t('Resultados'),str(exc),parent=self.root)

    def close(self):
        if self.busy: self.closing=True; self.cancel()
        else: self.root.destroy()

    def poll(self):
        try:
            while True:
                kind,data=self.events.get_nowait()
                if kind=='inventory':
                    self.devices,diagnostics=data
                    for info in self.devices: self.ensure_gpu_tab(info)
                    for diagnostic in diagnostics: self.append_log(diagnostic)
                    self.device.configure(values=[d['name']+' — '+d['id'] for d in self.devices])
                    if self.devices: self.device.current(0); self.check_modes()
                    else:
                        self.variables={}
                        self.checkbuttons=[]
                        for widget in self.checks.winfo_children(): widget.destroy()
                        self.scanning=False; self.status.set('Nenhuma GPU detectada. Consulte o registro e os drivers.'); self.controls()
                elif kind=='modes':
                    generation,(available,self.hidden)=data
                    if generation!=self.generation: continue
                    for widget in self.checks.winfo_children(): widget.destroy()
                    self.variables={mode:tk.BooleanVar(value=True) for mode in available}
                    self.render_tests()
                    self.scanning=False; self.status.set(f'Pronto · {len(available)} testes disponíveis'); self.controls()
                elif kind=='error':
                    self.scanning=False; self.status.set('Falha ao detectar capacidades. Consulte o registro.'); self.append_log(data)
                    self.checkbuttons=[]
                    for widget in self.checks.winfo_children(): widget.destroy()
                    self.variables={}; self.controls()
                elif kind=='line':
                    self.append_log(data)
                    if data.startswith('Pasta de resultados:'):
                        self.folder=Path(data.partition(':')[2].strip()); self.open_button.configure(state='normal')
                        self.result_views[self.run_gpu_id]['folder']=self.folder; self.result_tab_changed()
                    elif data.startswith('Formato concluido:'): self.read_results()
                elif kind=='done':
                    self.read_results(); self.busy=False; self.cancel_button.configure(state='disabled'); self.controls()
                    self.status.set('Concluído. Resultados salvos.' if data==0 else ('Cancelado. Resultados parciais salvos.' if data==130 else 'Execução com falha. Consulte o registro.'))
                    try: self.cancel_path.unlink(missing_ok=True)
                    except OSError as exc: self.append_log('Não foi possível remover sinal de cancelamento: '+str(exc))
                    self.cancel_path=None
                    if self.closing: self.root.destroy(); return
        except queue.Empty: pass
        self.root.after(100,self.poll)


def main():
    if os.name=='nt':
        try:
            import ctypes
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID('GPU.Benchmark.Desktop')
        except (OSError,AttributeError): pass
        try:
            import ctypes
            ctypes.windll.shcore.SetProcessDpiAwareness(1)
        except (OSError,AttributeError): pass
    root=tk.Tk(); BenchmarkGUI(root); root.mainloop()

if __name__=='__main__': main()
