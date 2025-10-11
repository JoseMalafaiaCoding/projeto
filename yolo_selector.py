import tkinter as tk
from tkinter import ttk, messagebox

class YOLOSelector:
    def __init__(self, master):
        self.master = master
        self.master.title("Seleção de Classes YOLO")
        largura_tela = self.master.winfo_screenwidth()
        altura_tela = self.master.winfo_screenheight()
        x = (largura_tela // 2) - (400 // 2)
        y = (altura_tela // 2) - (400 // 2)
        self.master.geometry(f"400x400+{x}+{y}")
        # self.master.geometry("400x300")
        self.all_classes = [
            "earplugs"
            ,"half_mask"
            ,"hard_hat"
            ,"no_earplugs"
            ,"no_half_mask"
            ,"no_hard_hat"
            ,"no_safety_boots"
            ,"no_safety_glasses"
            ,"no_safety_gloves"
            ,"no_safety_vest"
            ,"safety_boots"
            ,"safety_glasses"
            ,"safety_gloves"
            ,"safety_vest"
        ]
        self.selected_classes = []

        # Label título
        ttk.Label(master, text="Selecione as classes para exibir:", font=("Segoe UI", 12)).pack(pady=10)

        # Combobox (lista suspensa)
        self.combo = ttk.Combobox(master, values=self.all_classes, state="readonly")
        self.combo.pack(pady=5)
        self.combo.set("Escolha uma classe...")

        # Botão para adicionar classe
        ttk.Button(master, text="Adicionar Classe", command=self.add_class).pack(pady=5)

        # Lista de classes selecionadas
        self.listbox = tk.Listbox(master, height=8, width=30)
        self.listbox.pack(pady=10)

        # Botão para remover item selecionado
        ttk.Button(master, text="Remover Selecionada", command=self.remove_class).pack(pady=5)

        # Botão Adicionar Todas
        ttk.Button(master, text="Adicionar Todas", command=self.add_all).pack(pady=10)

        # Botão confirmar
        ttk.Button(master, text="Confirmar Seleção", command=self.confirm_selection).pack(pady=10)

    def add_class(self):
        selected = self.combo.get()
        if selected in self.all_classes and selected not in self.selected_classes:
            self.selected_classes.append(selected)
            self.listbox.insert(tk.END, selected)
        elif selected in self.selected_classes:
            messagebox.showinfo("Aviso", f"A classe '{selected}' já foi adicionada.")
        else:
            messagebox.showwarning("Atenção", "Selecione uma classe válida.")

    def add_all(self):
        self.selected_classes = [
            "earplugs"
            ,"half_mask"
            ,"hard_hat"
            ,"no_earplugs"
            ,"no_half_mask"
            ,"no_hard_hat"
            ,"no_safety_boots"
            ,"no_safety_glasses"
            ,"no_safety_gloves"
            ,"no_safety_vest"
            ,"safety_boots"
            ,"safety_glasses"
            ,"safety_gloves"
            ,"safety_vest"
        ]
        messagebox.showinfo("Classes Selecionadas", f"{', '.join(self.selected_classes)}")
        self.master.destroy()
        

    def remove_class(self):
        selection = self.listbox.curselection()
        if selection:
            index = selection[0]
            removed = self.listbox.get(index)
            self.selected_classes.remove(removed)
            self.listbox.delete(index)
        else:
            messagebox.showinfo("Aviso", "Nenhuma classe selecionada para remover.")

    def confirm_selection(self):
        if not self.selected_classes:
            messagebox.showwarning("Aviso", "Nenhuma classe selecionada.")
        else:
            messagebox.showinfo("Classes Selecionadas", f"{', '.join(self.selected_classes)}")
            self.master.destroy()
            return
